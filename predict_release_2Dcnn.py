# python predit_release.py --video path/to/input.mp4

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

# logic: The release frame = the FIRST frame where probability exceeds a threshold and stays above it for K frames.

RELEASE_THRESHOLD = 0.8  # between 0.0 and 1.0
RELEASE_MIN_FRAMES = 5   # K; number of consecutive frames that must stay above threshold


def build_model(weights_path: Path, device: torch.device) -> torch.nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 1)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def preprocess_frame(frame_bgr, transform):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    return transform(pil_img).unsqueeze(0)


def compute_probabilities(
    video_path: Path, model: torch.nn.Module, device: torch.device, transform
) -> list[float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    probs: list[float] = []
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            tensor = preprocess_frame(frame, transform).to(device)
            logit = model(tensor).squeeze(1)
            prob = torch.sigmoid(logit)[0].item()
            probs.append(prob)

    cap.release()
    return probs


def choose_release_frame(probs: list[float], threshold: float, min_frames: int) -> int:
    """
    Pick the earliest frame whose probability crosses `threshold` and stays above it
    for `min_frames` consecutive frames. Falls back to the max if none satisfy.
    """
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


def predict_video(video_path: Path, model_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_path, device)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    probs = compute_probabilities(video_path, model, device, transform)
    if not probs:
        raise RuntimeError("No frames found in video.")

    release_frame_idx = choose_release_frame(probs, threshold=RELEASE_THRESHOLD, min_frames=RELEASE_MIN_FRAMES)
    raw_prob_at_release = probs[release_frame_idx]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for annotation: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    out_path = video_path.with_name(f"annotated_{video_path.stem}.mp4")
    writer = cv2.VideoWriter(
        str(out_path),
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

    print(
        f"Predicted release frame: {release_frame_idx} "
        f"(raw_prob={raw_prob_at_release:.3f})"
    )
    print(f"Annotated video saved to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict release frame probability per video frame.")
    parser.add_argument("--video", type=Path, required=True, help="Path to input video")
    parser.add_argument("--model", type=Path, default=Path("best_model.pth"), help="Path to trained weights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predict_video(args.video, args.model)


if __name__ == "__main__":
    main()
