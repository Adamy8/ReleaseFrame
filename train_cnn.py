# How to run:
# python train_cnn.py --dataset dataset --epochs 5 --batch-size 32 --lr 1e-4
# If you already have a frame index CSV: add --index-csv path/to/frame_index.csv.
# Outputs release_cnn.pth in the working directory (configurable via --save-path).

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms


# ---- Data utilities --------------------------------------------------------
@dataclass
class FrameRecord:
    video_id: str
    frame_path: Path
    label: float


def build_frame_index(dataset_root: Path, index_csv: Optional[Path] = None) -> Path:
    """
    Ensure a CSV exists with columns: video_id,frame_path,label.
    If index_csv is provided and exists, it is returned. Otherwise, build it
    by pairing frames with labels.npy inside each video directory.
    """
    if index_csv is None:
        index_csv = dataset_root / "frame_index.csv"

    if index_csv.exists():
        return index_csv

    rows: list[list[str]] = []
    for video_dir in sorted(dataset_root.glob("video_*")):
        if not video_dir.is_dir():
            continue
        labels_path = video_dir / "labels.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels.npy in {video_dir}")

        labels = np.load(labels_path)
        frame_files = sorted(video_dir.glob("frame_*.jpg"))
        if len(frame_files) != len(labels):
            raise RuntimeError(
                f"Frame/label length mismatch in {video_dir}: "
                f"{len(frame_files)} frames vs {len(labels)} labels"
            )

        for frame_path, label in zip(frame_files, labels):
            rows.append([video_dir.name, str(frame_path), float(label)])

    with index_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "frame_path", "label"])
        writer.writerows(rows)

    return index_csv


class ReleaseFrameDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        index_csv: Optional[Path] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        dataset_root = dataset_root.resolve()
        index_csv = build_frame_index(dataset_root, index_csv)

        self.records: list[FrameRecord] = []
        with index_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame_path = Path(row["frame_path"])
                label = float(row["label"])
                self.records.append(FrameRecord(row["video_id"], frame_path, label))

        if not self.records:
            raise RuntimeError(f"No records loaded from {index_csv}")

        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec.frame_path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(rec.label, dtype=torch.float32)
        return img, label


# ---- Model -----------------------------------------------------------------
def build_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


# ---- Training / evaluation -------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int = 50,
) -> float:
    model.train()
    running_loss = 0.0
    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # shape (batch, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % log_interval == 0:
            avg = running_loss / log_interval
            print(f"  step {step:05d}: loss={avg:.4f}")
            running_loss = 0.0

    return running_loss


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).squeeze(1)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total if total > 0 else 0.0


# ---- CLI / main ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train release-frame CNN (single-frame).")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Path to dataset root")
    parser.add_argument("--index-csv", type=Path, help="Optional precomputed frame index CSV")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data for validation")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-path", type=Path, default=Path("release_cnn.pth"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = ReleaseFrameDataset(args.dataset, args.index_csv)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_one_epoch(model, train_loader, criterion, optimizer, device, args.log_interval)
        val_acc = evaluate(model, val_loader, device) if len(val_ds) > 0 else 0.0
        print(f"  Val accuracy: {val_acc:.4f}")

    torch.save(model.state_dict(), args.save_path)
    print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
    main()
