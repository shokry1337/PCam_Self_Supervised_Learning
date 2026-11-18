"""
train_ssl.py — Self-Supervised Pre-Training Routines & Diagnostics

Implements seeded pre-training for:
  - RotNet (4-way rotation classification, with or without crop/color augmentation).
  - Mini-SimCLR (vectorized NT-Xent loss, with or without discrete rotations/flips).
  - Held-out 4-way rotation accuracy check on unseen test patches.
  - BatchNorm running-statistic calibration for the frozen random-init baseline.
"""

from typing import List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from data_loader import (
    RotNetBaseDataset,
    SimCLRDataset,
    create_rotnet_batch_4way,
    set_seed,
)
from models import RotNetModel, SimCLRModel


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss (Chen et al., 2020)."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        b = z_i.shape[0]
        representations = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        mask = torch.eye(2 * b, dtype=torch.bool, device=z_i.device)
        similarity_matrix.masked_fill_(mask, -float("inf"))

        target_labels = torch.cat([
            torch.arange(b, 2 * b, device=z_i.device),
            torch.arange(0, b, device=z_i.device),
        ])
        return F.cross_entropy(similarity_matrix, target_labels)


def train_rotnet(
    unlabeled_images: Sequence[Image.Image],
    mean: Sequence[float],
    std: Sequence[float],
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    use_aug: bool = False,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[RotNetModel, List[float]]:
    """Trains RotNet on unlabeled patches using 4-way batch rotation (effective batch = 4 * batch_size)."""
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RotNetModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    dataset = RotNetBaseDataset(unlabeled_images, mean=mean, std=std, use_aug=use_aug)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=gen)

    tag = "RotNet (+ crop/jitter)" if use_aug else "RotNet (no aug)"
    print(f"\nPre-training {tag} ({epochs} epochs, {len(unlabeled_images)} patches, eff_batch={batch_size * 4})...")
    loss_history: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for raw_batch in loader:
            raw_batch = raw_batch.to(device)
            batch_x, batch_y = create_rotnet_batch_4way(raw_batch)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(batches, 1)
        loss_history.append(avg_loss)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f}")

    return model, loss_history


def train_simclr(
    unlabeled_images: Sequence[Image.Image],
    mean: Sequence[float],
    std: Sequence[float],
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    temperature: float = 0.5,
    use_rotation: bool = True,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[SimCLRModel, List[float]]:
    """Trains Mini-SimCLR with concatenated dual views (effective batch = 2 * batch_size)."""
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimCLRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = NTXentLoss(temperature=temperature)

    dataset = SimCLRDataset(unlabeled_images, mean=mean, std=std, use_rotation=use_rotation)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, generator=gen)

    tag = "SimCLR (+ rot/flip)" if use_rotation else "SimCLR (no rot/flip)"
    print(f"\nPre-training {tag} ({epochs} epochs, {len(unlabeled_images)} patches, eff_batch={batch_size * 2})...")
    loss_history: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for view1, view2 in loader:
            b = view1.shape[0]
            # Single concatenated forward pass so BatchNorm sees all 2*B views together
            views = torch.cat([view1, view2], dim=0).to(device)

            optimizer.zero_grad()
            _, z = model(views)
            z1, z2 = z[:b], z[b:]
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(batches, 1)
        loss_history.append(avg_loss)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | NT-Xent Loss: {avg_loss:.4f}")

    return model, loss_history


@torch.no_grad()
def evaluate_heldout_rotation_accuracy(
    rotnet_model: RotNetModel,
    test_images: Sequence[Image.Image],
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    batch_size: int = 64,
) -> float:
    """Measures 4-way rotation prediction accuracy (%) on unseen holdout test patches (chance = 25%)."""
    rotnet_model.eval()
    dataset = RotNetBaseDataset(test_images, mean=mean, std=std, use_aug=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0
    for raw_batch in loader:
        raw_batch = raw_batch.to(device)
        batch_x, batch_y = create_rotnet_batch_4way(raw_batch)
        preds = rotnet_model(batch_x).argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.numel()

    return (correct / max(total, 1)) * 100.0


@torch.no_grad()
def calibrate_batchnorm(
    model: nn.Module,
    unlabeled_images: Sequence[Image.Image],
    mean: Sequence[float],
    std: Sequence[float],
    device: torch.device,
    batch_size: int = 128,
) -> None:
    """Updates BatchNorm running mean/var over unlabeled patches without modifying any weights."""
    model.train()
    dataset = RotNetBaseDataset(unlabeled_images, mean=mean, std=std, use_aug=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for x_batch in loader:
        model.backbone(x_batch.to(device))
    model.eval()
