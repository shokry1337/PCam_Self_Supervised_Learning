"""
evaluate.py — Frozen Linear Probing & End-to-End Fine-Tuning Evaluation

Implements both downstream evaluation protocols on the low-data PCam regime (Nc=100):
  1. Frozen Linear Probing: StandardScaler + 5-fold CV LogisticRegressionCV fitted
     strictly on each 200-image training draw, evaluated across 10 draws.
  2. End-to-End Fine-Tuning: Unfrozen ResNet-34 trained for 100 epochs (Adam lr=1e-3,
     weight_decay=1e-3, batch_size=64), matching Kupferschmidt et al. (MIDL 2021).

All reported standard deviations use ddof=1 (sample SD across labeled draws).
"""

from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image

from data_loader import LabeledPatchDataset, set_seed
from models import BinaryClassifier


@torch.no_grad()
def extract_representations(
    model: nn.Module,
    images: Sequence[Image.Image],
    labels: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts frozen 512-dim backbone representations for a labeled dataset."""
    model.eval()
    dataset = LabeledPatchDataset(images, labels, mean=mean, std=std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings, y_true = [], []
    for x_batch, y_batch in loader:
        h = model.get_representation(x_batch.to(device))
        embeddings.append(h.cpu().numpy())
        y_true.append(y_batch.numpy())

    return np.concatenate(embeddings, axis=0), np.concatenate(y_true, axis=0)


def run_scaled_cv_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Fits StandardScaler + 5-fold CV LogisticRegressionCV on training features."""
    import warnings

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 10),
            cv=5,
            max_iter=2000,
            solver="lbfgs",
            scoring="roc_auc",
        )),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred) * 100.0
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return acc, auc, fpr, tpr


def evaluate_linear_probe_models(
    model_specs: Dict[str, Tuple[nn.Module, Sequence[float], Sequence[float]]],
    labeled_seeds: Dict[int, Tuple[List[Image.Image], List[int]]],
    test_images: Sequence[Image.Image],
    test_labels: Sequence[int],
    device: torch.device,
) -> Dict[str, dict]:
    """Evaluates all models via frozen linear probing across all labeled draws."""
    results: Dict[str, dict] = {}
    fpr_grid = np.linspace(0.0, 1.0, 200)
    n_draws = len(labeled_seeds)

    print(f"\n[Protocol 1] Frozen Linear Probe ({n_draws} draws of Nc=100, {len(test_images)} test patches)...")
    for name, (model, mean, std) in model_specs.items():
        X_test, y_test = extract_representations(model, test_images, test_labels, mean, std, device)
        accs, aucs, tprs = [], [], []

        for seed, (tr_imgs, tr_lbls) in labeled_seeds.items():
            X_tr, y_tr = extract_representations(model, tr_imgs, tr_lbls, mean, std, device)
            acc, auc, fpr, tpr = run_scaled_cv_probe(X_tr, y_tr, X_test, y_test)
            accs.append(acc)
            aucs.append(auc)
            interp_tpr = np.interp(fpr_grid, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        ddof = 1 if len(accs) > 1 else 0
        results[name] = {
            "accs": accs,
            "aucs": aucs,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs, ddof=ddof)),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs, ddof=ddof)),
            "fpr_grid": fpr_grid,
            "mean_tpr": np.mean(tprs, axis=0),
        }
        print(
            f"  {name:<26} | Acc: {results[name]['mean_acc']:5.1f} ± {results[name]['std_acc']:.1f}% "
            f"| AUC: {results[name]['mean_auc']:.3f} ± {results[name]['std_auc']:.3f}"
        )

    return results


def fine_tune_single_run(
    backbone_state_dict: Dict[str, torch.Tensor],
    train_images: Sequence[Image.Image],
    train_labels: Sequence[int],
    test_images: Sequence[Image.Image],
    test_labels: Sequence[int],
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    seed: int = 42,
) -> Tuple[float, float]:
    """End-to-end fine-tuning matching Kupferschmidt et al. (MIDL 2021) downstream settings."""
    set_seed(seed)
    clf = BinaryClassifier(backbone_state_dict).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_ds = LabeledPatchDataset(train_images, train_labels, mean=mean, std=std)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)

    clf.train()
    for _ in range(epochs):
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(clf(x_b), y_b)
            loss.backward()
            optimizer.step()

    clf.eval()
    test_ds = LabeledPatchDataset(test_images, test_labels, mean=mean, std=std)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    probs, y_true = [], []
    with torch.no_grad():
        for x_b, y_b in test_loader:
            logits = clf(x_b.to(device))
            p1 = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.append(p1)
            y_true.append(y_b.numpy())

    y_prob = np.concatenate(probs)
    y_t = np.concatenate(y_true)
    acc = accuracy_score(y_t, (y_prob >= 0.5).astype(int)) * 100.0
    auc = roc_auc_score(y_t, y_prob)
    return acc, auc


def evaluate_finetune_models(
    model_specs: Dict[str, Tuple[nn.Module, Sequence[float], Sequence[float]]],
    labeled_seeds: Dict[int, Tuple[List[Image.Image], List[int]]],
    test_images: Sequence[Image.Image],
    test_labels: Sequence[int],
    device: torch.device,
    finetune_seeds: Sequence[int] = (42, 101, 202),
    epochs: int = 100,
) -> Dict[str, dict]:
    """Evaluates all initializations via 100-epoch end-to-end fine-tuning."""
    results: Dict[str, dict] = {}
    print(f"\n[Protocol 2] End-to-End Fine-Tuning ({epochs} epochs, {len(finetune_seeds)} draws of Nc=100)...")

    for name, (model, mean, std) in model_specs.items():
        state = model.backbone.state_dict()
        accs, aucs = [], []
        for s in finetune_seeds:
            tr_imgs, tr_lbls = labeled_seeds[s]
            acc, auc = fine_tune_single_run(
                state, tr_imgs, tr_lbls, test_images, test_labels,
                mean=mean, std=std, device=device, epochs=epochs, seed=s,
            )
            accs.append(acc)
            aucs.append(auc)

        ddof = 1 if len(accs) > 1 else 0
        results[name] = {
            "accs": accs,
            "aucs": aucs,
            "mean_acc": float(np.mean(accs)),
            "std_acc": float(np.std(accs, ddof=ddof)),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs, ddof=ddof)),
        }
        print(
            f"  {name:<26} | Acc: {results[name]['mean_acc']:5.1f} ± {results[name]['std_acc']:.1f}% "
            f"| AUC: {results[name]['mean_auc']:.3f} ± {results[name]['std_auc']:.3f}"
        )

    return results


def print_paired_differences(
    probe_results: Dict[str, dict],
    pairs: Sequence[Tuple[str, str]],
) -> None:
    """Prints paired per-draw differences (Model A - Model B) across identical labeled draws."""
    print("\nPaired Linear-Probe Differences across identical draws (mean ± SD, ddof=1):")
    for a, b in pairs:
        d_acc = np.array(probe_results[a]["accs"]) - np.array(probe_results[b]["accs"])
        d_auc = np.array(probe_results[a]["aucs"]) - np.array(probe_results[b]["aucs"])
        wins = int(np.sum(d_auc > 0))
        print(
            f"  {a} vs {b}: "
            f"ΔAcc = {d_acc.mean():+.1f} ± {d_acc.std(ddof=1):.1f}% | "
            f"ΔAUC = {d_auc.mean():+.3f} ± {d_auc.std(ddof=1):.3f} ({wins}/{len(d_auc)} draws higher AUC)"
        )


def plot_mean_roc_curves(
    probe_results: Dict[str, dict],
    save_path: str = "roc_curves_comparison.png",
) -> None:
    """Plots mean ROC curves averaged across all linear-probe draws."""
    plt.figure(figsize=(8.5, 6), dpi=150)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.45, label="Chance (AUC = 0.500)")

    palette = {
        "Random Init (BN-cal)": "#7f7f7f",
        "ImageNet (pretrained)": "#1f77b4",
        "RotNet (no aug)": "#d62728",
        "RotNet (+ crop/jitter)": "#ff7f0e",
        "SimCLR (no rot/flip)": "#9467bd",
        "SimCLR (+ rot/flip)": "#2ca02c",
    }

    n_draws = len(next(iter(probe_results.values()))["accs"])
    for name, r in probe_results.items():
        color = palette.get(name, None)
        plt.plot(
            r["fpr_grid"],
            r["mean_tpr"],
            label=f"{name} (Mean AUC = {r['mean_auc']:.3f})",
            color=color,
            linewidth=2.0,
        )

    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(f"PCam Frozen Linear Probe — Mean ROC Curves ({n_draws} draws of $N_c=100$)", fontsize=12)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nSaved mean ROC plot to: {save_path}")
