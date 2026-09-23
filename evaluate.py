from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.model import SceneCNN, get_transform


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained scene recognition model")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        try:
            import torch_directml

            device = torch_directml.device()
        except Exception:
            device = torch.device("cpu")

    test_root = Path(args.data_root) / "test"
    test_dataset = datasets.ImageFolder(test_root, transform=get_transform(args.img_size, use_color=False))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SceneCNN(num_classes=len(test_dataset.classes))
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    loss, acc = evaluate(model, test_loader, device)
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
