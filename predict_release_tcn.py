# To run:
# python predict_release_tcn.py --video dodgers.mp4 --model best_model.pth --min-frames 10

# Optional:
# --threshold 0.8 --min-frames 5 --window 64 --stride 8 --batch-size 4 --device cuda --output my_out.mp4

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from train_cnn_tcn import ReleaseModel, default_transform, window_start_indices

RELEASE_THRESHOLD = 0.8
RELEASE_MIN_FRAMES = 5


def build_model(weights_path: Path, device: torch.device) -> ReleaseModel:
    model = ReleaseModel(hidden_channels=256, num_layers=3, kernel_size=7, dropout=0.1)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_frame(frame_bgr, transform):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    return transform(pil_img)


@torch.inference_mode()
def compute_probabilities(
    video_path: Path,
    model: ReleaseModel,
    device: torch.device,
    transform,
    window: int = 64,
    stride: int = 8,
    batch_size: int = 4,
) -> list[float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(preprocess_frame(frame, transform))
    cap.release()

    if not frames:
        raise RuntimeError("No frames found in video.")

    frame_tensor = torch.stack(frames, dim=0)  # (F, C, H, W)
    num_frames = frame_tensor.shape[0]
    starts = window_start_indices(num_frames, window, stride)

    sum_probs = torch.zeros(num_frames, device=device)
    counts = torch.zeros(num_frames, device=device)

    batch: list[torch.Tensor] = []
    batch_starts: list[int] = []
    for start in starts:
        end = start + window
        if end > num_frames:
            continue
        batch.append(frame_tensor[start:end])
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

    avg_probs = (sum_probs / counts.clamp_min(1)).cpu().tolist()
    return [float(p) for p in avg_probs]


def choose_release_frame(probs: list[float], threshold: float, min_frames: int) -> int:
    if not probs:
        raise RuntimeError("No probabilities to choose from.")

    arr = np.array(probs, dtype=np.float32)
    max_prob = float(arr.max())
    window = max(1, min_frames)

    for i in range(arr.size):
        if arr[i] >= threshold:
            end_idx = min(i + window, arr.size)
            if np.all(arr[i:end_idx] >= threshold):
                return int(i)

    max_indices = np.flatnonzero(arr >= max_prob - 1e-6)
    return int(max_indices[0])


def annotate_video(video_path: Path, probs: list[float], release_frame_idx: int, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for annotation: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        text = f"Prob: {probs[frame_idx]:.3f}"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if frame_idx == release_frame_idx:
            cv2.rectangle(frame, (5, 5), (width - 5, height - 5), (0, 255, 0), 6)
            cv2.putText(
                frame,
                "Release frame",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict release frame using ResNet18+TCN model.")
    parser.add_argument("--video", type=Path, required=True, help="Path to input video")
    parser.add_argument("--model", type=Path, default=Path("best_release_cnn_tcn.pth"), help="Path to trained weights")
    parser.add_argument("--output", type=Path, help="Path to save annotated video (mp4)")
    parser.add_argument("--threshold", type=float, default=RELEASE_THRESHOLD, help="Probability threshold")
    parser.add_argument("--min-frames", type=int, default=RELEASE_MIN_FRAMES, help="Consecutive frames above threshold")
    parser.add_argument("--window", type=int, default=64, help="Sliding window size (frames)")
    parser.add_argument("--stride", type=int, default=8, help="Sliding window stride (frames)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference windows")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (default: auto)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model, device)
    transform = default_transform()

    probs = compute_probabilities(
        args.video,
        model,
        device,
        transform,
        window=args.window,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    release_frame_idx = choose_release_frame(probs, threshold=args.threshold, min_frames=args.min_frames)
    raw_prob_at_release = probs[release_frame_idx]

    output_path = args.output or args.video.with_name(f"annotated_{args.video.stem}_tcn.mp4")
    annotate_video(args.video, probs, release_frame_idx, output_path)

    print(f"Predicted release frame: {release_frame_idx} (raw_prob={raw_prob_at_release:.3f})")
    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    main()
