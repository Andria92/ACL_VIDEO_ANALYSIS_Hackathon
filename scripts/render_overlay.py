"""Render skeleton-overlay video for visual pose quality control."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acl_motion.video.roi import BBox
from acl_motion.visualisation.overlay import (
    DEFAULT_POSE_DISPLAY_CONFIDENCE_THRESHOLD,
    draw_pose_overlay,
)


def main() -> int:
    args = parse_args()

    import cv2
    import pandas as pd

    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pose_df = read_pose_table(Path(args.pose), pd)
    rows_by_frame = {
        int(frame_index): group.to_dict(orient="records")
        for frame_index, group in pose_df.groupby("frame_index")
    }

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps / args.every_n, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Could not open output video writer: {output_path}")

    roi = BBox.from_string(args.roi) if args.roi else None
    start_frame = args.start_frame
    end_frame = args.end_frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current = start_frame
    written = 0
    try:
        while True:
            if end_frame is not None and current > end_frame:
                break
            ok, frame = capture.read()
            if not ok:
                break
            if (current - start_frame) % args.every_n == 0:
                frame_rows = rows_by_frame.get(current, [])
                landmarks = {row["landmark_name"]: row for row in frame_rows}
                bbox = roi or bbox_from_rows(frame_rows)
                timestamp_ms = (
                    float(frame_rows[0]["timestamp_ms"])
                    if frame_rows and frame_rows[0].get("timestamp_ms") is not None
                    else (current / fps) * 1000
                )
                frame_label = f"frame {current} | {timestamp_ms:.1f} ms"
                writer.write(
                    draw_pose_overlay(
                        frame,
                        landmarks,
                        bbox=bbox,
                        frame_label=frame_label,
                        confidence_threshold=args.confidence_threshold,
                    )
                )
                written += 1
            current += 1
    finally:
        capture.release()
        writer.release()

    print(f"Wrote {written} overlay frames to {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--pose", required=True, help="Pose .parquet or .csv file.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--roi", default=None, help="Optional x,y,width,height override.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--every-n", type=int, default=1)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_POSE_DISPLAY_CONFIDENCE_THRESHOLD,
    )
    args = parser.parse_args()
    if args.every_n <= 0:
        parser.error("--every-n must be positive.")
    return args


def read_pose_table(path: Path, pd):
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Pose path must end in .parquet or .csv.")


def bbox_from_rows(rows: list[dict]) -> BBox | None:
    if not rows:
        return None
    first = rows[0]
    values = [
        first.get("target_bbox_x"),
        first.get("target_bbox_y"),
        first.get("target_bbox_width"),
        first.get("target_bbox_height"),
    ]
    if any(value is None for value in values):
        return None
    try:
        return BBox(float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
