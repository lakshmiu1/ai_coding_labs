"""In-memory FashionMNIST Dataset with a seeded train/validation split.

Unlike fashion_mnist.py, which wraps a torchvision dataset and decodes PIL
images per item, this module copies the raw tensors once and augments on
tensors. Normalization statistics are computed from the training split only.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import v2

CLASS_NAMES = (
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
)


class FashionMNISTTensorDataset(Dataset):
    """Holds uint8 images in memory and applies a transform per access.

    Images are stored as [N, 1, 28, 28] so the augmentation pipeline sees a
    channel dimension; labels are int64 and ready for cross entropy.
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        transform: v2.Transform | None = None,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError("images and labels must have the same length")
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


def build_transform(mean: float, std: float, augment: bool) -> v2.Transform:
    """Flip only when augmenting; scale to float and standardize always.

    Horizontal flips are label preserving here: every FashionMNIST class is a
    garment or accessory whose mirror image belongs to the same class.
    """
    steps: list[v2.Transform] = []
    if augment:
        steps.append(v2.RandomHorizontalFlip(p=0.5))
    steps.append(v2.ToDtype(torch.float32, scale=True))  # uint8 [0, 255] -> [0, 1]
    steps.append(v2.Normalize(mean=(mean,), std=(std,)))
    return v2.Compose(steps)


def _seed_worker(worker_id: int) -> None:
    """Give each worker a distinct, run-reproducible seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _subset(source: datasets.FashionMNIST, indices: Sequence[int] | None = None):
    images = source.data.unsqueeze(1)  # [N, 28, 28] -> [N, 1, 28, 28]
    labels = source.targets.to(torch.int64)
    if indices is not None:
        selection = torch.as_tensor(indices, dtype=torch.long)
        images, labels = images[selection], labels[selection]
    return images.contiguous(), labels.contiguous()


def build_dataloaders(
    root: str = "./data",
    batch_size: int = 64,
    validation_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    download: bool = True,
    include_test: bool = False,
) -> dict[str, DataLoader]:
    """Split the official training set and return the loaders by name.

    Returns "train" and "validation" loaders, plus "test" (the official test
    set, evaluation transform) when include_test is set. The seed fixes split
    membership, shuffling, and augmentation across runs.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")

    source = datasets.FashionMNIST(root=root, train=True, download=download)
    validation_size = int(len(source) * validation_fraction)
    if not 0 < validation_size < len(source):
        raise ValueError("split must leave at least one sample in each set")

    permutation = torch.randperm(len(source), generator=torch.Generator().manual_seed(seed))
    validation_indices = permutation[:validation_size]
    train_indices = permutation[validation_size:]

    train_images, train_labels = _subset(source, train_indices)
    validation_images, validation_labels = _subset(source, validation_indices)

    # Statistics come from the training split alone, so validation and test
    # data never leak into the preprocessing.
    scaled = train_images.to(torch.float32).div_(255.0)
    mean, std = scaled.mean().item(), scaled.std().item()
    del scaled

    loaders = {
        "train": _loader(
            FashionMNISTTensorDataset(
                train_images, train_labels, build_transform(mean, std, augment=True)
            ),
            batch_size, num_workers, seed, shuffle=True,
        ),
        "validation": _loader(
            FashionMNISTTensorDataset(
                validation_images, validation_labels, build_transform(mean, std, augment=False)
            ),
            batch_size, num_workers, seed, shuffle=False,
        ),
    }
    if include_test:
        test_source = datasets.FashionMNIST(root=root, train=False, download=download)
        loaders["test"] = _loader(
            FashionMNISTTensorDataset(
                *_subset(test_source), build_transform(mean, std, augment=False)
            ),
            batch_size, num_workers, seed, shuffle=False,
        )
    return loaders


def _loader(
    dataset: Dataset, batch_size: int, num_workers: int, seed: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=torch.Generator().manual_seed(seed),
    )


if __name__ == "__main__":
    # The main guard keeps num_workers > 0 safe on macOS and Windows.
    torch.manual_seed(42)
    loaders = build_dataloaders(include_test=True)
    for name, loader in loaders.items():
        print(f"{name:>10}: {len(loader.dataset):>6} samples, {len(loader):>4} batches")

    images, labels = next(iter(loaders["train"]))
    print(f"\nbatch images {tuple(images.shape)} {images.dtype}, "
          f"labels {tuple(labels.shape)} {labels.dtype}")
    print(f"normalized range [{images.min():.2f}, {images.max():.2f}], "
          f"mean {images.mean():.3f}, std {images.std():.3f}")
    print("first labels:", [CLASS_NAMES[label] for label in labels[:4].tolist()])
