# python predit_release.py --video path/to/input.mp4

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from torchvision import models, transforms
from PIL import Image


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


def predict_video(video_path: Path, model_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_path, device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

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

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    best_prob = -1.0
    best_frame_idx = -1
    frame_idx = 0

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            tensor = preprocess_frame(frame, transform).to(device)
            logit = model(tensor).squeeze(1)
            prob = torch.sigmoid(logit)[0].item()

            if prob > best_prob:
                best_prob = prob
                best_frame_idx = frame_idx

            text = f"Prob: {prob:.3f}"
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
            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()

    print(f"Predicted release frame: {best_frame_idx} (prob={best_prob:.3f})")
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
