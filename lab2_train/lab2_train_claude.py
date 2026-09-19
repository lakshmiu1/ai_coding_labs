"""Train a small CNN on FashionMNIST and plot train vs. validation loss.

Reuses the lab 1 data pipeline, so normalization statistics stay fitted on the
training split alone -- validation data never reaches the preprocessing.

    python3 lab2_train_claude.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

# Lab 1 lives in a sibling directory; resolve it from this file so the labs
# folder can move as a unit.
LAB1_DIR = Path(__file__).resolve().parent.parent / "lab1_data_pipeline"
sys.path.insert(0, str(LAB1_DIR))
try:
    from lab1_data_claude import build_dataloaders
except ImportError as error:  # pragma: no cover - setup guidance only
    raise SystemExit(f"Could not import the lab 1 pipeline from {LAB1_DIR}: {error}")

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 42
DATA_ROOT = str(LAB1_DIR / "data")  # Reuse the copy lab 1 already downloaded.
PLOT_PATH = Path(__file__).resolve().parent / "loss_curves.png"


class SmallCNN(nn.Module):
    """Two convolution blocks, each halving resolution, then a linear head.

    BatchNorm keeps the blocks trainable at lr=1e-3 without warmup; dropout
    after each pool is what makes the train/validation gap worth plotting.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Dropout(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(2 * dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one pass over the training split; return mean loss per sample."""
    model.train()
    running_loss, seen = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        # Weight by batch size: the final batch is usually short, so a plain
        # mean over batches would quietly overweight it.
        running_loss += loss.item() * labels.size(0)
        seen += labels.size(0)
    return running_loss / seen


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return (mean loss per sample, accuracy) with dropout/BN in eval mode."""
    model.eval()
    running_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        running_loss += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return running_loss / seen, correct / seen


def plot_loss_curves(
    train_losses: list[float],
    validation_losses: list[float],
    output_path: Path,
    mode: str = "light",
) -> None:
    """Save a two-series line chart of the per-epoch loss curves."""
    # Validated categorical slots 1 and 2 (blue, orange) on the matching surface.
    theme = {
        "light": dict(
            surface="#fcfcfb", train="#2a78d6", validation="#eb6834",
            primary="#0b0b0b", secondary="#52514e", muted="#898781",
            grid="#e1e0d9", axis="#c3c2b7",
        ),
        "dark": dict(
            surface="#1a1a19", train="#3987e5", validation="#d95926",
            primary="#ffffff", secondary="#c3c2b7", muted="#898781",
            grid="#2c2c2a", axis="#383835",
        ),
    }[mode]

    epochs = range(1, len(train_losses) + 1)
    figure, axes = plt.subplots(figsize=(8, 5), dpi=160)
    figure.patch.set_facecolor(theme["surface"])
    axes.set_facecolor(theme["surface"])

    series = (
        ("Training loss", train_losses, theme["train"]),
        ("Validation loss", validation_losses, theme["validation"]),
    )
    for label, values, color in series:
        axes.plot(
            epochs, values,
            color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
            marker="o", markersize=8,
            # A 2px ring in the surface color keeps markers legible where the
            # two curves cross.
            markeredgecolor=theme["surface"], markeredgewidth=2,
            label=label, zorder=3,
        )

    # Direct end labels supplement the legend, but only when the curves have
    # separated enough that the labels stay attached to the right lines.
    span = max(max(train_losses), max(validation_losses)) - min(
        min(train_losses), min(validation_losses)
    )
    final_gap = abs(train_losses[-1] - validation_losses[-1])
    if span > 0 and final_gap / span > 0.08:
        for label, values, _ in series:
            axes.annotate(
                f"{values[-1]:.3f}",
                xy=(len(values), values[-1]),
                xytext=(10, 0), textcoords="offset points",
                va="center", fontsize=10, fontweight="semibold",
                # Text wears ink tokens; the colored marker beside it carries
                # identity.
                color=theme["secondary"],
            )

    axes.set_title(
        "Training vs. validation loss", loc="left", pad=16,
        fontsize=14, fontweight="semibold", color=theme["primary"],
    )
    axes.set_xlabel("Epoch", fontsize=11, color=theme["secondary"], labelpad=10)
    axes.set_ylabel("Cross-entropy loss", fontsize=11, color=theme["secondary"], labelpad=10)
    axes.set_xticks(list(epochs))
    axes.set_xlim(0.85, len(train_losses) + 0.45)  # Room for the end labels.
    axes.tick_params(colors=theme["muted"], labelsize=10, length=0)

    axes.grid(axis="y", color=theme["grid"], linewidth=1, linestyle="-", zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(theme["axis"])
        axes.spines[side].set_linewidth(1)

    legend = axes.legend(
        frameon=False, loc="upper right", fontsize=10, handlelength=1.6, borderpad=0,
    )
    for text in legend.get_texts():
        text.set_color(theme["secondary"])

    figure.tight_layout()
    figure.savefig(output_path, facecolor=theme["surface"], bbox_inches="tight")
    print(f"\nSaved loss curves to {output_path}")
    plt.show()


def main() -> None:
    torch.manual_seed(SEED)
    device = select_device()
    print(f"Device: {device}")

    loaders = build_dataloaders(root=DATA_ROOT, batch_size=BATCH_SIZE, seed=SEED)
    train_loader, validation_loader = loaders["train"], loaders["validation"]
    print(
        f"Train: {len(train_loader.dataset):,} samples · "
        f"Validation: {len(validation_loader.dataset):,} samples"
    )

    model = SmallCNN().to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: SmallCNN, {parameters:,} trainable parameters\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss, validation_accuracy = evaluate(
            model, validation_loader, criterion, device
        )
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        print(
            f"Epoch {epoch}/{EPOCHS}  "
            f"train loss {train_loss:.4f}  "
            f"val loss {validation_loss:.4f}  "
            f"val acc {validation_accuracy:.2%}  "
            f"({time.perf_counter() - started:.1f}s)"
        )

    best = min(range(EPOCHS), key=lambda i: validation_losses[i])
    print(f"\nBest validation loss: {validation_losses[best]:.4f} at epoch {best + 1}")

    plot_loss_curves(train_losses, validation_losses, PLOT_PATH)


if __name__ == "__main__":
    # The main guard keeps num_workers > 0 safe on macOS and Windows.
    main()
