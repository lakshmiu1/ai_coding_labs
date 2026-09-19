"""Custom FashionMNIST dataset and reproducible train/validation loaders."""

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class FashionMNISTDataset(Dataset):
    """An indexed view of FashionMNIST with its own transform pipeline.

    The source dataset must have no transforms so views can independently
    augment training images and leave validation images deterministic.
    """

    def __init__(
        self,
        source: datasets.FashionMNIST,
        indices: Sequence[int],
        augment: bool = False,
    ) -> None:
        if source.transform is not None or source.target_transform is not None:
            raise ValueError("source must have no transforms")
        self.source = source
        self.indices = tuple(indices)
        steps = [transforms.RandomHorizontalFlip(p=0.5)] if augment else []
        steps.extend([
            transforms.ToTensor(),
            # Map grayscale pixels from [0, 1] to [-1, 1].
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])
        self.transform = transforms.Compose(steps)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.source[self.indices[index]]
        return self.transform(image), int(label)


def create_dataloaders(
    root: str = "./data",
    batch_size: int = 64,
    validation_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Split the official training set; keep the official test set untouched.

    The seed fixes split membership and loader shuffling. Call
    torch.manual_seed(seed) as well to reproduce augmentation when using
    num_workers=0; DataLoader seeds Torch RNGs in worker processes otherwise.
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
        raise ValueError("split must contain at least one sample in each set")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(source), generator=generator).tolist()
    train_dataset = FashionMNISTDataset(source, indices[validation_size:], augment=True)
    validation_dataset = FashionMNISTDataset(source, indices[:validation_size])

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        generator=torch.Generator().manual_seed(seed),
        **common,
    )
    return train_loader, validation_loader


if __name__ == "__main__":
    # The main guard also makes multiprocessing safe on macOS/Windows.
    torch.manual_seed(42)
    train_loader, validation_loader = create_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"Train: {len(train_loader.dataset)}, validation: {len(validation_loader.dataset)}")
    print(f"Images: {images.shape}, labels: {labels.shape}")
