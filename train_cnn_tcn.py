# To Run:
# Create splits only: 
    # python train_cnn_tcn.py --dataset dataset --make-splits --seed 42
# Train: 
    # python train_cnn_tcn.py --dataset dataset --epochs 25 --batch-size 4 --device cuda

# Inference stats run automatically on val each epoch and on test after training; 
# adjust --freeze-epochs, --lr, --window, --stride, and --save-path as needed.

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# --------------------------------------------------------------------------- #
# Utility helpers                                                            #
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_video_dirs(dataset_root: Path) -> list[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("video_")])


def ensure_splits(
    dataset_root: Path, split_dir: Path, seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> tuple[Path, Path, Path]:
    split_dir.mkdir(parents=True, exist_ok=True)
    train_file = split_dir / "train.txt"
    val_file = split_dir / "val.txt"
    test_file = split_dir / "test.txt"

    if train_file.exists() and val_file.exists() and test_file.exists():
        return train_file, val_file, test_file

    videos = [p.name for p in list_video_dirs(dataset_root)]
    if not videos:
        raise RuntimeError(f"No video_* folders found in {dataset_root}")

    rng = random.Random(seed)
    rng.shuffle(videos)

    n_total = len(videos)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_videos = videos[:n_train]
    val_videos = videos[n_train : n_train + n_val]
    test_videos = videos[n_train + n_val :]

    for path, items in [
        (train_file, train_videos),
        (val_file, val_videos),
        (test_file, test_videos),
    ]:
        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(items))

    return train_file, val_file, test_file


def read_split_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


# --------------------------------------------------------------------------- #
# Dataset                                                                    #
# --------------------------------------------------------------------------- #
def default_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def window_start_indices(num_frames: int, window: int, stride: int) -> list[int]:
    starts = list(range(0, max(1, num_frames - window + 1), stride))
    last_start = num_frames - window
    if last_start not in starts and last_start >= 0:
        starts.append(last_start)
    return sorted(set(starts))


@dataclass
class VideoMeta:
    video_id: str
    frame_paths: list[Path]
    labels: np.ndarray

    @property
    def num_frames(self) -> int:
        return len(self.frame_paths)


@dataclass
class WindowSpec:
    video_id: str
    start: int


class ReleaseFrameWindowDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        video_ids: Sequence[str],
        window: int = 64,
        stride: int = 8,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.transform = transform or default_transform()
        self.window = window
        self.stride = stride

        self.videos: dict[str, VideoMeta] = {}
        self.windows: list[WindowSpec] = []

        for video_id in video_ids:
            video_dir = self.dataset_root / video_id
            labels_path = video_dir / "labels.npy"
            frame_paths = sorted(video_dir.glob("frame_*.jpg"))
            if not frame_paths:
                raise RuntimeError(f"No frames found in {video_dir}")
            if not labels_path.exists():
                raise RuntimeError(f"Missing labels.npy in {video_dir}")
            labels = np.load(labels_path)
            if len(labels) != len(frame_paths):
                raise RuntimeError(f"Frame/label length mismatch in {video_dir}")

            meta = VideoMeta(video_id=video_id, frame_paths=frame_paths, labels=labels.astype(np.float32))
            self.videos[video_id] = meta

            for start in window_start_indices(meta.num_frames, window, stride):
                end = start + window
                if end <= meta.num_frames:
                    self.windows.append(WindowSpec(video_id=video_id, start=start))

        if not self.windows:
            raise RuntimeError("No training windows constructed; check dataset contents.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        spec = self.windows[idx]
        meta = self.videos[spec.video_id]
        frames: list[torch.Tensor] = []
        for frame_path in meta.frame_paths[spec.start : spec.start + self.window]:
            img = Image.open(frame_path).convert("RGB")
            frames.append(self.transform(img))
        frame_tensor = torch.stack(frames, dim=0)  # (T, C, H, W)
        labels = torch.from_numpy(meta.labels[spec.start : spec.start + self.window])
        return frame_tensor, labels


# --------------------------------------------------------------------------- #
# Model: ResNet18 backbone + Temporal Conv Net                               #
# --------------------------------------------------------------------------- #
class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        # Use "same" padding for odd kernels so temporal length is preserved.
        padding = ((kernel_size - 1) // 2) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return out + x


class TemporalConvNet(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, num_layers: int, kernel_size: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            dilation = 2**i
            layers.append(TemporalBlock(in_ch, hidden_channels, kernel_size, dilation, dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReleaseModel(nn.Module):
    def __init__(self, hidden_channels: int = 256, num_layers: int = 3, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        modules = list(backbone.children())[:-1]  # remove FC
        self.backbone = nn.Sequential(*modules)
        feature_dim = backbone.fc.in_features
        self.tcn = TemporalConvNet(feature_dim, hidden_channels, num_layers, kernel_size, dropout)
        self.head = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feats = self.backbone(x).view(b, t, -1)  # (B, T, feature_dim)
        feats = feats.permute(0, 2, 1)  # (B, feature_dim, T)
        tcn_out = self.tcn(feats)
        logits = self.head(tcn_out).squeeze(1)  # (B, T)
        return logits


# --------------------------------------------------------------------------- #
# Training & evaluation                                                      #
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: ReleaseModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    grad_clip: float = 1.0,
    log_interval: int = 20,
) -> float:
    model.train()
    scaler = GradScaler(enabled=use_amp)
    running_loss = 0.0

    for step, (frames, labels) in enumerate(loader, start=1):
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(frames)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if step % log_interval == 0:
            avg = running_loss / log_interval
            print(f"  step {step:05d}: loss={avg:.4f}")
            running_loss = 0.0

    return running_loss


@torch.inference_mode()
def evaluate_videos(
    model: ReleaseModel,
    dataset_root: Path,
    video_ids: Sequence[str],
    device: torch.device,
    transform: transforms.Compose,
    window: int,
    stride: int,
    batch_size: int,
) -> dict:
    model.eval()
    all_errors: list[float] = []
    within_1 = within_2 = within_3 = 0

    for video_id in video_ids:
        video_dir = dataset_root / video_id
        labels = np.load(video_dir / "labels.npy").astype(np.float32)
        frame_paths = sorted(video_dir.glob("frame_*.jpg"))
        frames = [transform(Image.open(p).convert("RGB")) for p in frame_paths]
        frame_tensor = torch.stack(frames, dim=0)  # (F, C, H, W)
        num_frames = frame_tensor.shape[0]
        starts = window_start_indices(num_frames, window, stride)

        sum_probs = torch.zeros(num_frames, device=device)
        counts = torch.zeros(num_frames, device=device)

        batch: list[torch.Tensor] = []
        batch_starts: list[int] = []
        for start in starts:
            batch.append(frame_tensor[start : start + window])
            batch_starts.append(start)
            if len(batch) == batch_size:
                probs = torch.sigmoid(model(torch.stack(batch, dim=0).to(device)))
                for p, s in zip(probs, batch_starts):
                    sum_probs[s : s + window] += p
                    counts[s : s + window] += 1
                batch.clear()
                batch_starts.clear()

        if batch:
            probs = torch.sigmoid(model(torch.stack(batch, dim=0).to(device)))
            for p, s in zip(probs, batch_starts):
                sum_probs[s : s + window] += p
                counts[s : s + window] += 1

        avg_probs = (sum_probs / counts.clamp_min(1)).cpu().numpy()

        true_release = next((i for i, v in enumerate(labels) if v >= 0.5), len(labels) - 1)
        pred_release = next((i for i, v in enumerate(avg_probs) if v >= 0.5), int(np.argmax(avg_probs)))

        error = abs(pred_release - true_release)
        all_errors.append(error)
        within_1 += int(error <= 1)
        within_2 += int(error <= 2)
        within_3 += int(error <= 3)

    mean_error = float(np.mean(all_errors)) if all_errors else 0.0
    median_error = float(np.median(all_errors)) if all_errors else 0.0
    total = len(all_errors) if all_errors else 1

    return {
        "mean_error": mean_error,
        "median_error": median_error,
        "within_1": within_1 / total,
        "within_2": within_2 / total,
        "within_3": within_3 / total,
        "num_videos": total if all_errors else 0,
    }


# --------------------------------------------------------------------------- #
# Main                                                                       
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 + TCN for release frame detection.")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"), help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (default: auto)")
    parser.add_argument("--make-splits", action="store_true", help="Only (re)create splits and exit")
    parser.add_argument("--split-dir", type=Path, default=Path("splits"))
    parser.add_argument("--freeze-epochs", type=int, default=2, help="Freeze CNN backbone for first N epochs")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-path", type=Path, default=Path("release_cnn_tcn.pth"))
    parser.add_argument("--best-path", type=Path, default=Path("best_release_cnn_tcn.pth"))
    parser.add_argument("--last-path", type=Path, default=Path("last_release_cnn_tcn.pth"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    dataset_root = args.dataset.resolve()
    split_dir = args.split_dir.resolve()
    train_split, val_split, test_split = ensure_splits(dataset_root, split_dir, args.seed)

    if args.make_splits:
        print(f"Splits written to {split_dir}")
        return

    train_ids = read_split_file(train_split)
    val_ids = read_split_file(val_split)
    test_ids = read_split_file(test_split)

    transform = default_transform()
    train_ds = ReleaseFrameWindowDataset(dataset_root, train_ids, window=args.window, stride=args.stride, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    model = ReleaseModel(hidden_channels=256, num_layers=3, kernel_size=7, dropout=0.1).to(device)
    model.freeze_backbone()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_score = float("inf")
    best_saved = False

    try:
        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs} (device={device})")
            if epoch > args.freeze_epochs:
                model.unfreeze_backbone()
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                use_amp,
                grad_clip=1.0,
                log_interval=args.log_interval,
            )

            if val_ids:
                val_metrics = evaluate_videos(
                    model,
                    dataset_root,
                    val_ids,
                    device,
                    transform,
                    window=args.window,
                    stride=args.stride,
                    batch_size=args.batch_size,
                )
                mean_err = val_metrics["mean_error"]
                print(
                    "  Val | "
                    f"mean={val_metrics['mean_error']:.2f} "
                    f"median={val_metrics['median_error']:.2f} "
                    f"±1={val_metrics['within_1']:.2%} "
                    f"±2={val_metrics['within_2']:.2%} "
                    f"±3={val_metrics['within_3']:.2%}"
                )
                if mean_err < best_score:
                    best_score = mean_err
                    torch.save(model.state_dict(), args.best_path)
                    best_saved = True
                    print(f"  Saved new best checkpoint to {args.best_path}")
    except KeyboardInterrupt:
        torch.save(model.state_dict(), args.last_path)
        print(f"\nTraining interrupted; last checkpoint saved to {args.last_path}")
        return
    except Exception:
        torch.save(model.state_dict(), args.last_path)
        print(f"\nException during training; last checkpoint saved to {args.last_path}")
        raise

    # Normal completion: always save last and ensure a best checkpoint exists.
    torch.save(model.state_dict(), args.last_path)
    print(f"Saved last checkpoint to {args.last_path}")
    if not best_saved:
        torch.save(model.state_dict(), args.best_path)
        print(f"No validation improvements tracked; saved current model as best to {args.best_path}")

    # Optional legacy save path.
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Saved model to {args.save_path}")

    if test_ids:
        test_metrics = evaluate_videos(
            model,
            dataset_root,
            test_ids,
            device,
            transform,
            window=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
        )
        print(
            "Test set results | "
            f"mean={test_metrics['mean_error']:.2f} "
            f"median={test_metrics['median_error']:.2f} "
            f"±1={test_metrics['within_1']:.2%} "
            f"±2={test_metrics['within_2']:.2%} "
            f"±3={test_metrics['within_3']:.2%}"
        )


if __name__ == "__main__":
    main()
