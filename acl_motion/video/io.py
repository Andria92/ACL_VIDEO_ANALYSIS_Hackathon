"""Video metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Basic video metadata used for traceability."""

    file_path: Path
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float


def read_video_metadata(path: str | Path) -> VideoMetadata:
    """Read video metadata with OpenCV."""

    import cv2

    file_path = Path(path)
    capture = cv2.VideoCapture(str(file_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {file_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()

    duration_seconds = frame_count / fps if fps > 0 else 0.0
    return VideoMetadata(
        file_path=file_path,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
    )
