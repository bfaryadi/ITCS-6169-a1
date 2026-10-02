"""Evaluate trained checkpoints: loss/accuracy, per-class report, confusion matrix.

Usage:
    python evaluate.py                          # best.pt and last.pt on val + test
    python evaluate.py --checkpoint best        # only the best-validation checkpoint
    python evaluate.py --splits val             # validation only (does not touch test)

The train/val split is rebuilt from the seed stored in the checkpoint, so the
validation set matches the one used during training.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import TNet
from train import build_dataloaders, evaluate, get_device


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels)
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def per_class_report(model, loader, class_names, device, title="", save_path=None):
    preds, labels = get_predictions(model, loader, device)
    n = len(class_names)

    # Confusion matrix: rows = true class, cols = predicted class
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    support = cm.sum(axis=1)
    per_class_acc = cm.diagonal() / np.maximum(support, 1)

    print(f"{title} overall accuracy: {(preds == labels).mean():.4f}\n")
    print(f'{"Class":<14}{"Acc":>7}{"N":>6}   Most confused with')
    for i in np.argsort(per_class_acc):  # worst classes first
        off = cm[i].copy()
        off[i] = 0
        confused = class_names[off.argmax()] if off.sum() > 0 else "-"
        print(f"{class_names[i]:<14}{per_class_acc[i]:>7.3f}{support[i]:>6}   {confused}")

    # Confusion matrix plot (row-normalized)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm / np.maximum(support[:, None], 1), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title} confusion matrix")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=120)
        print(f"\nSaved confusion matrix to {save_path}")
    plt.close(fig)
    return cm, per_class_acc


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained scene recognition CNN.")
    p.add_argument("--data-root", default="data", help="folder containing train/ and test/")
    p.add_argument("--ckpt-dir", default="checkpoints", help="folder with best.pt / last.pt")
    p.add_argument("--checkpoint", choices=["best", "last", "both"], default="both")
    p.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    p.add_argument("--seed", type=int, default=None,
                    help="override the seed stored in the checkpoint (changes the val split)")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    names = ["best", "last"] if args.checkpoint == "both" else [args.checkpoint]
    device = get_device()
    print(f"Using device: {device}")

    ckpts = {n: torch.load(ckpt_dir / f"{n}.pt", map_location="cpu") for n in names}
    first = next(iter(ckpts.values()))
    seed = args.seed if args.seed is not None else first["seed"]
    class_names = first["class_names"]

    _, val_loader, test_loader, _ = build_dataloaders(args.data_root, seed=seed)
    loaders = {"val": val_loader, "test": test_loader}

    for name, ckpt in ckpts.items():
        model = TNet(num_classes=len(class_names))
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)

        for split in args.splits:
            title = f"{name} / {split}"
            print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
            loss, acc = evaluate(model, loaders[split], device)
            print(f"{split} loss: {loss:.4f}")
            print(f"{split} accuracy: {acc:.4f}\n")
            per_class_report(
                model, loaders[split], class_names, device, title=title,
                save_path=ckpt_dir / f"confusion_{name}_{split}.png",
            )


if __name__ == "__main__":
    main()
