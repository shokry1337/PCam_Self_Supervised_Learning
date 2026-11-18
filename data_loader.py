"""
data_loader.py — PatchCamelyon (PCam) Data Loading, Seeding, and Augmentations

Streams shuffled subsets from the Hugging Face PatchCamelyon mirror
('1aurent/PatchCamelyon', official train and test splits), computes per-channel
mean/std from the unlabeled pre-training subset, and provides ablation-ready
augmentation pipelines for RotNet and Mini-SimCLR.
"""

import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 42) -> None:
    """Sets Python, NumPy, and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_channel_stats(images: Sequence[Image.Image]) -> Tuple[List[float], List[float]]:
    """Computes per-channel RGB mean and std across a list of PIL images in [0, 1] (float64 accumulation)."""
    arr = np.stack([np.asarray(img, dtype=np.float32) / 255.0 for img in images], axis=0)
    mean = arr.mean(axis=(0, 1, 2), dtype=np.float64).tolist()
    std = arr.std(axis=(0, 1, 2), dtype=np.float64).tolist()
    return mean, std


class RandomDiscreteRotation:
    """Randomly rotates a PIL image by 0, 90, 180, or 270 degrees."""

    def __init__(self, angles: Sequence[int] = (0, 90, 180, 270)):
        self.angles = tuple(angles)

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.choice(self.angles)
        return transforms.functional.rotate(img, angle)


class SimCLRAugmentation:
    """Produces two independently augmented views of an input patch.

    Args:
        mean: Per-channel RGB normalization mean.
        std: Per-channel RGB normalization std.
        img_size: Spatial resolution (96 for PCam).
        use_rotation: If True, includes discrete 90-degree rotations and flips.
            Set to False for the no-rotation ablation.
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        img_size: int = 96,
        use_rotation: bool = True,
    ):
        ops: List = [transforms.RandomResizedCrop(size=img_size, scale=(0.6, 1.0))]
        if use_rotation:
            ops.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                RandomDiscreteRotation((0, 90, 180, 270)),
            ])
        ops.extend([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ])
        self.transform = transforms.Compose(ops)

    def __call__(self, x: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transform(x), self.transform(x)


class RotNetBaseDataset(Dataset):
    """Dataset yielding base tensors before 4-way batch rotation.

    Args:
        images: Unlabeled PIL images.
        mean: Per-channel RGB normalization mean.
        std: Per-channel RGB normalization std.
        use_aug: If True, applies the same crop and color jitter used in SimCLR
            before the 4-way rotation step (augmentation-controlled ablation).
    """

    def __init__(
        self,
        images: Sequence[Image.Image],
        mean: Sequence[float],
        std: Sequence[float],
        use_aug: bool = False,
    ):
        self.images = images
        ops: List = []
        if use_aug:
            ops.extend([
                transforms.RandomResizedCrop(size=96, scale=(0.6, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            ])
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ])
        self.transform = transforms.Compose(ops)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(self.images[idx])


class SimCLRDataset(Dataset):
    """Dataset returning dual augmented views for contrastive pre-training."""

    def __init__(
        self,
        images: Sequence[Image.Image],
        mean: Sequence[float],
        std: Sequence[float],
        use_rotation: bool = True,
    ):
        self.images = images
        self.augment = SimCLRAugmentation(mean=mean, std=std, img_size=96, use_rotation=use_rotation)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.augment(self.images[idx])


class LabeledPatchDataset(Dataset):
    """Dataset for downstream binary classification (0 = Normal, 1 = Tumor)."""

    def __init__(
        self,
        images: Sequence[Image.Image],
        labels: Sequence[int],
        mean: Sequence[float],
        std: Sequence[float],
    ):
        self.images = images
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.transform(self.images[idx]), int(self.labels[idx])


def create_rotnet_batch_4way(images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expands a mini-batch of B images into 4*B images rotated by 0, 90, 180, 270 degrees."""
    b = images.shape[0]
    r0 = images
    r90 = torch.rot90(images, k=1, dims=[-2, -1])
    r180 = torch.rot90(images, k=2, dims=[-2, -1])
    r270 = torch.rot90(images, k=3, dims=[-2, -1])

    batch_x = torch.cat([r0, r90, r180, r270], dim=0)
    batch_y = torch.cat([
        torch.zeros(b, dtype=torch.long, device=images.device),
        torch.ones(b, dtype=torch.long, device=images.device),
        torch.full((b,), 2, dtype=torch.long, device=images.device),
        torch.full((b,), 3, dtype=torch.long, device=images.device),
    ])
    return batch_x, batch_y


def load_pcam_subsets(
    num_unlabeled: int = 4000,
    nc_labeled: int = 100,
    num_test: int = 5000,
    seeds: Optional[Sequence[int]] = None,
    stream_seed: int = 42,
    shuffle_buffer: int = 10_000,
    candidate_per_class: int = 1500,
) -> Dict:
    """Streams shuffled PCam subsets from '1aurent/PatchCamelyon' (official train/test splits).

    Raises immediately if Hugging Face dataset loading fails (no silent synthetic fallback).

    Returns a dictionary with:
      - 'unlabeled': list of PIL images (labels stripped; class balance logged).
      - 'unlabeled_pos_rate': fraction of positive patches in the unlabeled pool.
      - 'mean', 'std': per-channel RGB stats computed strictly on the unlabeled pool.
      - 'labeled_seeds': {seed: (images, labels)} each with nc_labeled per class.
      - 'test': (test_images, test_labels) balanced sample from official test split.
    """
    if num_test % 2 != 0:
        raise ValueError(f"num_test must be even for balanced sampling, got {num_test}.")

    if seeds is None:
        seeds = (42, 101, 202, 303, 404, 505, 606, 707, 808, 909)

    from datasets import load_dataset

    print("Streaming shuffled subsets from Hugging Face ('1aurent/PatchCamelyon')...")
    ds_train = load_dataset("1aurent/PatchCamelyon", split="train", streaming=True).shuffle(
        seed=stream_seed, buffer_size=shuffle_buffer
    )
    ds_test = load_dataset("1aurent/PatchCamelyon", split="test", streaming=True).shuffle(
        seed=stream_seed, buffer_size=shuffle_buffer
    )

    train_iter = iter(ds_train)

    print(f"Sampling {num_unlabeled} unlabeled patches from shuffled train split...")
    unlabeled_images: List[Image.Image] = []
    unlabeled_pos = 0
    for _ in range(num_unlabeled):
        item = next(train_iter)
        unlabeled_images.append(item["image"].convert("RGB"))
        unlabeled_pos += int(item["label"])
    pos_rate = unlabeled_pos / num_unlabeled
    print(f"  Unlabeled pool class balance (logged before stripping labels): {pos_rate * 100:.1f}% positive")

    mean, std = compute_channel_stats(unlabeled_images)
    print(f"  Computed unlabeled RGB mean={[round(m, 4) for m in mean]}, std={[round(s, 4) for s in std]}")

    print(f"Collecting {candidate_per_class}/class candidate pool for {len(seeds)} labeled draws (Nc={nc_labeled})...")
    cand_0: List[Image.Image] = []
    cand_1: List[Image.Image] = []
    while len(cand_0) < candidate_per_class or len(cand_1) < candidate_per_class:
        item = next(train_iter)
        lbl = int(item["label"])
        if lbl == 0 and len(cand_0) < candidate_per_class:
            cand_0.append(item["image"].convert("RGB"))
        elif lbl == 1 and len(cand_1) < candidate_per_class:
            cand_1.append(item["image"].convert("RGB"))

    labeled_seeds: Dict[int, Tuple[List[Image.Image], List[int]]] = {}
    for s in seeds:
        rng = random.Random(s)
        idx0 = rng.sample(range(len(cand_0)), nc_labeled)
        idx1 = rng.sample(range(len(cand_1)), nc_labeled)
        s_imgs = [cand_0[i] for i in idx0] + [cand_1[i] for i in idx1]
        s_lbls = [0] * nc_labeled + [1] * nc_labeled
        combined = list(zip(s_imgs, s_lbls))
        rng.shuffle(combined)
        shuff_imgs, shuff_lbls = zip(*combined)
        labeled_seeds[s] = (list(shuff_imgs), list(shuff_lbls))

    print(f"Sampling {num_test} balanced holdout test patches from shuffled official test split...")
    test_0: List[Image.Image] = []
    test_1: List[Image.Image] = []
    half_test = num_test // 2
    test_iter = iter(ds_test)
    while len(test_0) < half_test or len(test_1) < half_test:
        item = next(test_iter)
        lbl = int(item["label"])
        if lbl == 0 and len(test_0) < half_test:
            test_0.append(item["image"].convert("RGB"))
        elif lbl == 1 and len(test_1) < half_test:
            test_1.append(item["image"].convert("RGB"))

    test_combined = list(zip(test_0 + test_1, [0] * half_test + [1] * half_test))
    random.Random(stream_seed).shuffle(test_combined)
    test_imgs, test_lbls = zip(*test_combined)

    return {
        "unlabeled": unlabeled_images,
        "unlabeled_pos_rate": pos_rate,
        "mean": mean,
        "std": std,
        "labeled_seeds": labeled_seeds,
        "test": (list(test_imgs), list(test_lbls)),
    }
