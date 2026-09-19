FashionMNIST data pipeline
=========================

Install dependencies and run the example (the first run downloads FashionMNIST):

```sh
python3 -m pip install -r requirements.txt
python3 fashion_mnist.py
```

Use the loaders in training code:

```python
import torch
from fashion_mnist import create_dataloaders

torch.manual_seed(42)
train_loader, val_loader = create_dataloaders(batch_size=64, seed=42)

for images, labels in train_loader:
    # images: float32 [batch, 1, 28, 28]; labels: int64 [batch]
    pass  # Forward pass, loss, backward pass, optimizer step.
```

The default seeded random split uses 48,000 training and 12,000 validation
examples from the official training set. The official test set is reserved
for final evaluation. Training applies random horizontal flips with probability
0.5. Both splits convert images to tensors and normalize pixels to [-1, 1]
using mean=0.5 and std=0.5. Validation has no random augmentation and no shuffle.
Each split has its own transform pipeline while sharing the underlying data.

Set `download=False` when using an existing local copy. `num_workers=0` works
in notebooks; when increasing it in a script, create and iterate loaders under
an `if __name__ == "__main__":` guard.

API reference: [torchvision FashionMNIST](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html).
