# Run to evaluate:
# python evaluate_release_accuracy.py --dataset test_production --model best_model.pth --output test_production/release_accuracy.md --device cuda

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from predict_release_tcn import build_model, compute_probabilities
from train_cnn_tcn import default_transform


def find_release_frame_first_one(probs: Iterable[float]) -> int:
    """
    Return the first frame index whose probability hits 1.0 or higher.
    Falls back to the max-probability frame if no value reaches 1.0.
    """
    for idx, p in enumerate(probs):
        if p >= 1.0:
            return idx
    arr = np.asarray(list(probs), dtype=np.float32)
    if arr.size == 0:
        raise RuntimeError("No probabilities provided.")
    return int(np.argmax(arr))


def load_annotations(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def write_markdown(
    output_path: Path,
    rows: list[tuple[str, int, int, int]],
    accuracy_pct: float,
    mae: float,
    model_path: Path,
    dataset_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# Release frame accuracy\n\n")
        f.write(f"- Model: `{model_path}`\n")
        f.write(f"- Dataset: `{dataset_path}`\n")
        f.write("- Release frame rule: first frame with probability >= 1.0; fallback to max probability.\n\n")
        f.write("| video | annotated | predicted | abs diff |\n")
        f.write("| --- | --- | --- | --- |\n")
        for video_id, annotated, predicted, diff in rows:
            f.write(f"| {video_id} | {annotated} | {predicted} | {diff} |\n")
        f.write("\n")
        f.write(f"Overall exact-match accuracy: {accuracy_pct:.2f}%\n\n")
        f.write(f"Mean absolute frame error: {mae:.2f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate release-frame accuracy on a dataset.")
    parser.add_argument("--dataset", type=Path, default=Path("test_production"), help="Path to dataset folder")
    parser.add_argument("--model", type=Path, default=Path("best_model.pth"), help="Path to trained weights")
    parser.add_argument("--output", type=Path, default=Path("release_accuracy.md"), help="Markdown report path")
    parser.add_argument("--window", type=int, default=64, help="Sliding window size (frames)")
    parser.add_argument("--stride", type=int, default=8, help="Sliding window stride (frames)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference windows")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda (default: auto)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    videos_dir = args.dataset / "test_videos"
    annotations_path = args.dataset / "release_annotations.json"

    annotations = load_annotations(annotations_path)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, device)
    transform = default_transform()

    rows: list[tuple[str, int, int, int]] = []
    matches = 0
    abs_errors: list[int] = []

    for video_path in sorted(videos_dir.glob("*")):
        if not video_path.is_file():
            continue
        video_id = video_path.stem
        if video_id not in annotations:
            continue

        probs = compute_probabilities(
            video_path,
            model,
            device,
            transform,
            window=args.window,
            stride=args.stride,
            batch_size=args.batch_size,
        )
        predicted_idx = find_release_frame_first_one(probs)
        annotated_idx = annotations[video_id]
        diff = abs(predicted_idx - annotated_idx)

        rows.append((video_id, annotated_idx, predicted_idx, diff))
        abs_errors.append(diff)
        if predicted_idx == annotated_idx:
            matches += 1

    total = len(rows)
    accuracy_pct = 100.0 * matches / total if total else 0.0
    mae = float(np.mean(abs_errors)) if abs_errors else 0.0

    write_markdown(args.output, rows, accuracy_pct, mae, args.model, args.dataset)
    print(f"Wrote report with {total} videos to {args.output}")


if __name__ == "__main__":
    main()
