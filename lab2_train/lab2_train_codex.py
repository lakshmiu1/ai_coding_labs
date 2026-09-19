"""Train a small FashionMNIST CNN and plot training/validation loss.

Install: python3 -m pip install torch torchvision matplotlib
Run:     python3 lab2_train_codex.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots even on machines without a display.
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 x 28 -> 14 x 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 x 14 -> 7 x 7
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),  # Raw logits for CrossEntropyLoss.
        )

    def forward(self, images):
        return self.layers(images)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            # Weight by batch size so the final smaller batch counts correctly.
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    return total_loss / total_samples


def main():
    torch.manual_seed(42)
    output_dir = Path(__file__).resolve().parent
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Fixed constants map [0, 1] to [-1, 1]; no statistics or leakage.
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    dataset = datasets.FashionMNIST(
        root=str(output_dir / "data"), train=True,
        download=True, transform=transform,
    )
    # Keep the official test set untouched for final evaluation.
    train_dataset, validation_dataset = random_split(
        dataset, [48000, 12000], generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        generator=torch.Generator().manual_seed(42), num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=64, shuffle=False, num_workers=0,
    )

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses, validation_losses = [], []
    epochs = 5
    print(f"Device: {device}")
    print(f"Train: {len(train_dataset)}, validation: {len(validation_dataset)}")

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss = run_epoch(model, validation_loader, criterion, device)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Training loss: {train_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f}"
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, epochs + 1), train_losses, marker="o", label="Training")
    ax.plot(range(1, epochs + 1), validation_losses, marker="o", label="Validation")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss",
           title="FashionMNIST: training vs. validation loss")
    ax.set_xticks(range(1, epochs + 1))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = output_dir / "loss_curves.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Loss plot saved to {plot_path}")


if __name__ == "__main__":
    main()
