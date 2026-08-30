"""Render a skeleton overlay colored by Milestone 2 QC state."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pandas as pd

from acl_motion.visualisation.qc_overlay import draw_qc_pose_overlay


def main() -> int:
    args = parse_args()
    processed = pd.read_parquet(args.processed_pose)
    frame_quality = pd.read_csv(args.frame_quality).set_index("frame_index")
    rows_by_frame = {
        int(frame_index): group.to_dict(orient="records")
        for frame_index, group in processed.groupby("frame_index")
    }

    video_path = Path(args.video)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    intermediate_path = output_path.with_name(
        f"{output_path.stem}.mp4v-intermediate.mp4"
    )
    writer = cv2.VideoWriter(
        str(intermediate_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / args.every_n,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open output video writer: {output_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    current = args.start_frame
    written = 0
    try:
        while True:
            if args.end_frame is not None and current > args.end_frame:
                break
            ok, frame = capture.read()
            if not ok:
                break
            if (current - args.start_frame) % args.every_n == 0:
                rows = rows_by_frame.get(current, [])
                status = (
                    frame_quality.loc[current, "frame_status"]
                    if current in frame_quality.index
                    else "TARGET_NOT_FOUND"
                )
                timestamp_ms = rows[0]["timestamp_ms"] if rows else (current / fps) * 1000
                label = f"frame {current} | {timestamp_ms:.1f} ms"
                writer.write(draw_qc_pose_overlay(frame, rows, status, label))
                written += 1
            current += 1
    finally:
        capture.release()
        writer.release()
    _encode_browser_compatible_video(intermediate_path, output_path)
    print(f"Wrote {written} QC overlay frames to {output_path}")
    return 0


def _encode_browser_compatible_video(source: Path, output: Path) -> None:
    """Convert OpenCV's intermediate stream to browser-compatible H.264."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        source.unlink(missing_ok=True)
        raise ValueError("ffmpeg is required to create browser-compatible skeleton videos.")
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise ValueError(
                "Could not encode a browser-compatible skeleton video: "
                + (result.stderr.strip() or "ffmpeg failed")
            )
    finally:
        source.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--processed-pose", required=True)
    parser.add_argument("--frame-quality", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--every-n", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
