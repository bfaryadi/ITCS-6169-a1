"""Data loading, augmentation, and training for the scene recognition CNN.

Usage:
    python train.py                       # notebook defaults (120 epochs)
    python train.py --epochs 40 --lr 1e-3
    python train.py --visualize           # also save a grid of augmented samples

Writes best.pt, last.pt, history.json and loss/accuracy plots to --out-dir.
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import TNet, count_parameters

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SEED = 0
IMG_SIZE = 128
BATCH_SIZE = 32
VAL_FRACTION = 0.20
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# Reproducibility and device
# ----------------------------------------------------------------------------
def set_random_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """CUDA if available, else DirectML (AMD GPU on Windows), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml

        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")


# ----------------------------------------------------------------------------
# Mixing augmentations
# ----------------------------------------------------------------------------
def mixup_data(x, y, alpha=0.2):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    H, W = x.size(2), x.size(3)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)

    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))  # correct lam for the clipped box
    return x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Works for both MixUp and CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def random_grayscale_rgb(img):
    # Augment only on training; validation/test should stay clean.
    if random.random() < 0.5:
        return img.convert("L").convert("RGB")
    return img


def build_transforms(img_size=IMG_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.Lambda(random_grayscale_rgb),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    # Validation/test preprocessing: no randomness.
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform


class TransformSubset(torch.utils.data.Dataset):
    """Applies a transform to a Subset so train/val can use different pipelines."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_dataloaders(data_root, seed=SEED, batch_size=BATCH_SIZE,
                        val_fraction=VAL_FRACTION, img_size=IMG_SIZE):
    """Returns train_loader, val_loader, test_loader, class_names.

    The train/val split depends only on `seed`, so evaluate.py can reproduce
    the exact same validation set by passing the same seed.
    """
    data_root = Path(data_root)
    train_transform, eval_transform = build_transforms(img_size)

    # Load base dataset once so we can split by index.
    base_train_dataset = datasets.ImageFolder(data_root / "train")
    class_names = base_train_dataset.classes

    val_size = int(round(len(base_train_dataset) * val_fraction))
    train_size = len(base_train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        base_train_dataset, [train_size, val_size], generator=generator
    )

    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, eval_transform)
    test_dataset = datasets.ImageFolder(data_root / "test", transform=eval_transform)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=pin)

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Total training images: {len(base_train_dataset)}")
    print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)} | "
            f"Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, class_names


# ----------------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()  # used instead of inference_mode for DirectML compatibility
def evaluate(model, loader, device):
    """Returns (loss, accuracy) over a loader."""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_model(model, train_loader, val_loader, optimizer, scheduler, device,
                epochs=20, mixup_alpha=0.0, cutmix_alpha=1.0, cutmix_prob=0.0):
    """Trains the model; returns (model with best weights loaded, history,
    best_state, last_state, best_val_acc)."""
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    best_state = _cpu_state(model)
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Pick at most one mixing method per batch
            use_cutmix = cutmix_prob > 0 and np.random.rand() < cutmix_prob
            use_mixup = (not use_cutmix) and mixup_alpha > 0

            if use_cutmix:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha)
            elif use_mixup:
                images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=mixup_alpha)

            logits = model(images)
            if use_cutmix or use_mixup:
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_seen += labels.size(0)

        train_loss = total_loss / total_seen
        val_loss, val_acc = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if scheduler is not None:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = _cpu_state(model)

        elapsed = time.time() - start
        print(f"Epoch {epoch:02d}/{epochs} | "
                f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
                f"val acc {val_acc:.4f} | {elapsed:.1f}s")

    last_state = _cpu_state(model)
    model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    return model, history, best_state, last_state, best_val_acc


# ----------------------------------------------------------------------------
# Plots / checkpoints
# ----------------------------------------------------------------------------
def visualize_batch(loader, class_names, out_path):
    """Save a grid of augmented training images."""
    images, labels = next(iter(loader))
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for ax, image, label in zip(axes.flat, images[:8], labels[:8]):
        image = (image * std + mean).clamp(0, 1).permute(1, 2, 0)
        ax.imshow(image.numpy())
        ax.set_title(class_names[label.item()])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_history(history, out_dir):
    out_dir = Path(out_dir)

    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=120)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(history["val_acc"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "val_accuracy_curve.png", dpi=120)
    plt.close()


def save_checkpoint(path, state, class_names, seed, config, val_acc=None):
    torch.save({
        "model_state": state,
        "class_names": class_names,
        "seed": seed,
        "config": config,
        "best_val_acc": val_acc,
    }, path)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train the scene recognition CNN.")
    p.add_argument("--data-root", default="data", help="folder containing train/ and test/")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--mixup-alpha", type=float, default=0.4)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)
    p.add_argument("--cutmix-prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--visualize", action="store_true",
                    help="save a grid of augmented training samples")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)
    device = get_device()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")

    train_loader, val_loader, _, class_names = build_dataloaders(args.data_root, seed=args.seed)

    if args.visualize:
        visualize_batch(train_loader, class_names, out_dir / "train_samples.png")

    set_random_seed(args.seed)
    model = TNet(num_classes=len(class_names)).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model, history, best_state, last_state, best_val_acc = train_model(
        model, train_loader, val_loader, optimizer, scheduler, device,
        epochs=args.epochs, mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha, cutmix_prob=args.cutmix_prob,
    )

    config = vars(args)
    save_checkpoint(out_dir / "best.pt", best_state, class_names, args.seed, config, best_val_acc)
    save_checkpoint(out_dir / "last.pt", last_state, class_names, args.seed, config, history["val_acc"][-1])
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f)
    plot_history(history, out_dir)
    print(f"Saved checkpoints, history, and plots to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
