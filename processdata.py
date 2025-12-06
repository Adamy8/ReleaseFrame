# after download annotations/ from our Google Drive, you could use this file to process data.
# 1. python processdata.py rename
# 2. python processdata.py build-dataset    // will put data into format for CNN training


from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# ---- Renaming helpers -----------------------------------------------------
def build_mapping(clips_dir: Path) -> dict[str, str]:
    """Return mapping of existing clip stems to sequential videos### names."""
    clips = sorted(p for p in clips_dir.glob("*") if p.is_file())
    if not clips:
        raise RuntimeError(f"No clips found in {clips_dir}")

    width = max(3, len(str(len(clips))))
    mapping: dict[str, str] = {}
    for idx, clip in enumerate(clips, start=1):
        new_stem = f"videos{idx:0{width}d}"
        mapping[clip.stem] = new_stem
    return mapping


def validate(mapping: dict[str, str], annotations: dict[str, int]) -> None:
    """Ensure clips and annotations match before mutating anything."""
    clip_keys = set(mapping.keys())
    annotation_keys = set(annotations.keys())

    missing_in_annotations = clip_keys - annotation_keys
    missing_on_disk = annotation_keys - clip_keys
    if missing_in_annotations or missing_on_disk:
        raise RuntimeError(
            "Mismatch between clips and annotation keys\n"
            f"Missing in annotations: {sorted(missing_in_annotations)}\n"
            f"Missing on disk: {sorted(missing_on_disk)}"
        )

    # Prevent collisions if a new name already exists.
    new_names = set(mapping.values())
    if len(new_names) != len(mapping):
        raise RuntimeError("New clip names would collide; aborting.")


def rename_clips(clips_dir: Path, mapping: dict[str, str]) -> None:
    """Rename clip files according to mapping."""
    for clip in sorted(clips_dir.glob("*")):
        if not clip.is_file():
            continue
        new_name = mapping[clip.stem] + clip.suffix
        target = clip.with_name(new_name)
        if target.exists():
            raise RuntimeError(f"Target {target} already exists; aborting.")
        clip.rename(target)


def rewrite_annotations(annotations_path: Path, mapping: dict[str, str]) -> None:
    """Rewrite JSON with updated clip names, keeping values the same."""
    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)

    updated = {mapping[old]: value for old, value in annotations.items()}
    backup_path = annotations_path.with_suffix(annotations_path.suffix + ".bak")
    annotations_path.replace(backup_path)

    with annotations_path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Updated annotations written to {annotations_path}")
    print(f"Original annotations backed up to {backup_path}")


# ---- Dataset building helpers ---------------------------------------------
@dataclass
class VideoMeta:
    dataset_id: str
    source_id: str
    num_frames: int
    fps: float | None
    release_frame: int
    release_frame_used: int


def generate_labels(num_frames: int, release_frame: int) -> np.ndarray:
    """
    Build label vector per frame:
      0 before release frame,
      frame-2 -> 0.3, frame-1 -> 0.4, release frame -> 0.9,
      frames after release -> 1.0.
    """
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.float32)

    rf = min(max(release_frame, 0), num_frames - 1)
    labels = np.zeros((num_frames,), dtype=np.float32)

    if rf - 2 >= 0:
        labels[rf - 2] = 0.3
    if rf - 1 >= 0:
        labels[rf - 1] = 0.4

    labels[rf] = 0.9
    if rf + 1 < num_frames:
        labels[rf + 1 :] = 1.0
    return labels


def write_meta(meta_path: Path, rows: Iterable[VideoMeta]) -> None:
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["dataset_id", "source_id", "num_frames", "fps", "release_frame", "release_frame_used"]
        )
        for row in rows:
            writer.writerow(
                [
                    row.dataset_id,
                    row.source_id,
                    row.num_frames,
                    f"{row.fps:.3f}" if row.fps is not None else "",
                    row.release_frame,
                    row.release_frame_used,
                ]
            )


def ensure_empty_or_create(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def extract_video(
    video_path: Path, output_dir: Path, release_frame: int, frame_pad: int = 4
) -> VideoMeta:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps_value = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_value if fps_value and fps_value > 0 else None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_path = output_dir / f"frame_{frame_idx:0{frame_pad}d}.jpg"
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f"Failed to write frame {frame_idx} for {video_path}")
        frame_idx += 1

    cap.release()
    labels = generate_labels(frame_idx, release_frame)
    labels_path = output_dir / "labels.npy"
    np.save(labels_path, labels)

    rf_used = min(max(release_frame, 0), max(frame_idx - 1, 0))
    return VideoMeta(
        dataset_id=output_dir.name,
        source_id=video_path.stem,
        num_frames=frame_idx,
        fps=fps,
        release_frame=release_frame,
        release_frame_used=rf_used,
    )


def build_dataset(
    clips_dir: Path,
    annotations_path: Path,
    dataset_dir: Path,
    one_indexed_release: bool = False,
) -> None:
    with annotations_path.open("r", encoding="utf-8") as f:
        annotations: dict[str, int] = json.load(f)

    ensure_empty_or_create(dataset_dir)

    meta_rows: list[VideoMeta] = []
    for idx, (video_id, release_frame) in enumerate(sorted(annotations.items()), start=1):
        adjusted_release = release_frame - 1 if one_indexed_release else release_frame
        dataset_id = f"video_{idx:03d}"
        video_file = clips_dir / f"{video_id}.mp4"
        if not video_file.exists():
            raise RuntimeError(f"Missing video file for {video_id}: {video_file}")

        video_out_dir = dataset_dir / dataset_id
        video_out_dir.mkdir(parents=True, exist_ok=True)

        meta = extract_video(
            video_file,
            video_out_dir,
            adjusted_release,
            frame_pad=4,
        )
        meta_rows.append(meta)
        print(f"Processed {video_id} -> {dataset_id} ({meta.num_frames} frames)")

    write_meta(dataset_dir / "meta.csv", meta_rows)
    print(f"Wrote {len(meta_rows)} entries to {dataset_dir/'meta.csv'}")


# ---- CLI ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotation utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rename_parser = subparsers.add_parser("rename", help="Rename clips and update annotations.")
    rename_parser.add_argument("--clips-dir", type=Path, default=Path("annotations/clips"))
    rename_parser.add_argument(
        "--annotations", type=Path, default=Path("annotations/release_annotations.json")
    )

    dataset_parser = subparsers.add_parser("build-dataset", help="Export frames + labels.")
    dataset_parser.add_argument("--clips-dir", type=Path, default=Path("annotations/clips"))
    dataset_parser.add_argument(
        "--annotations", type=Path, default=Path("annotations/release_annotations.json")
    )
    dataset_parser.add_argument("--output", type=Path, default=Path("dataset"))
    dataset_parser.add_argument(
        "--one-indexed-release",
        action="store_true",
        help="Set if annotation release frame indices are 1-based.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    if args.command == "rename":
        clips_dir = (repo_root / args.clips_dir).resolve()
        annotations_path = (repo_root / args.annotations).resolve()

        with annotations_path.open("r", encoding="utf-8") as f:
            annotations = json.load(f)

        mapping = build_mapping(clips_dir)
        validate(mapping, annotations)

        rename_clips(clips_dir, mapping)
        rewrite_annotations(annotations_path, mapping)
        print("Renamed clips and updated release_annotations.json")
        return

    if args.command == "build-dataset":
        clips_dir = (repo_root / args.clips_dir).resolve()
        annotations_path = (repo_root / args.annotations).resolve()
        dataset_dir = (repo_root / args.output).resolve()

        build_dataset(
            clips_dir=clips_dir,
            annotations_path=annotations_path,
            dataset_dir=dataset_dir,
            one_indexed_release=args.one_indexed_release,
        )
        print("Dataset built successfully.")


if __name__ == "__main__":
    main()
