"""Manual target-athlete region of interest helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned image-space bounding box in pixels."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("BBox width and height must be positive.")

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> BBox:
        """Create a box from two corners."""

        return cls(x=x1, y=y1, width=x2 - x1, height=y2 - y1)

    @classmethod
    def from_string(cls, value: str) -> BBox:
        """Parse CLI ROI strings in 'x,y,width,height' format."""

        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError("ROI must use x,y,width,height format.")
        try:
            x, y, width, height = (float(part) for part in parts)
        except ValueError as exc:
            raise ValueError("ROI values must be numeric.") from exc
        return cls(x=x, y=y, width=width, height=height)

    @property
    def x2(self) -> float:
        return self.x + self.width

    @property
    def y2(self) -> float:
        return self.y + self.height

    def pad(self, fraction: float) -> BBox:
        """Return a symmetrically padded box."""

        if fraction < 0:
            raise ValueError("Padding fraction cannot be negative.")
        dx = self.width * fraction
        dy = self.height * fraction
        return BBox(
            x=self.x - dx,
            y=self.y - dy,
            width=self.width + 2 * dx,
            height=self.height + 2 * dy,
        )

    def clamp(self, image_width: int, image_height: int) -> BBox:
        """Clamp the box to image boundaries."""

        x1 = min(max(self.x, 0), image_width)
        y1 = min(max(self.y, 0), image_height)
        x2 = min(max(self.x2, 0), image_width)
        y2 = min(max(self.y2, 0), image_height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("ROI does not overlap the image.")
        return BBox.from_xyxy(x1, y1, x2, y2)

    def as_int_xyxy(self) -> tuple[int, int, int, int]:
        """Return integer crop coordinates."""

        x1 = round(self.x)
        y1 = round(self.y)
        x2 = round(self.x2)
        y2 = round(self.y2)
        return x1, y1, x2, y2

    def to_dict(self, prefix: str = "bbox") -> dict[str, float]:
        """Return flat dict columns for tabular exports."""

        return {
            f"{prefix}_x": self.x,
            f"{prefix}_y": self.y,
            f"{prefix}_width": self.width,
            f"{prefix}_height": self.height,
        }


@dataclass(frozen=True, slots=True)
class RoiKeyframe:
    """Manual ROI at a specific frame."""

    frame_index: int
    bbox: BBox


@dataclass(frozen=True, slots=True)
class RoiTimeline:
    """Linearly interpolated manual ROI keyframes."""

    keyframes: tuple[RoiKeyframe, ...]

    def __post_init__(self) -> None:
        if not self.keyframes:
            raise ValueError("RoiTimeline requires at least one keyframe.")
        frames = [keyframe.frame_index for keyframe in self.keyframes]
        if frames != sorted(frames):
            raise ValueError("ROI keyframes must be sorted by frame_index.")
        if len(set(frames)) != len(frames):
            raise ValueError("ROI keyframes must not contain duplicate frames.")

    @classmethod
    def from_csv(cls, path: str | Path) -> RoiTimeline:
        """Load keyframes from CSV columns: frame_index,x,y,width,height."""

        keyframes: list[RoiKeyframe] = []
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"frame_index", "x", "y", "width", "height"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"ROI keyframe CSV missing columns: {sorted(missing)}")
            for row in reader:
                keyframes.append(
                    RoiKeyframe(
                        frame_index=int(row["frame_index"]),
                        bbox=BBox(
                            x=float(row["x"]),
                            y=float(row["y"]),
                            width=float(row["width"]),
                            height=float(row["height"]),
                        ),
                    )
                )
        return cls(tuple(sorted(keyframes, key=lambda keyframe: keyframe.frame_index)))

    def bbox_for_frame(self, frame_index: int) -> BBox:
        """Return the manual ROI for a frame, interpolating between keyframes."""

        if frame_index <= self.keyframes[0].frame_index:
            return self.keyframes[0].bbox
        if frame_index >= self.keyframes[-1].frame_index:
            return self.keyframes[-1].bbox

        previous = self.keyframes[0]
        for next_keyframe in self.keyframes[1:]:
            if frame_index <= next_keyframe.frame_index:
                span = next_keyframe.frame_index - previous.frame_index
                fraction = (frame_index - previous.frame_index) / span if span else 0.0
                return _interpolate_bbox(previous.bbox, next_keyframe.bbox, fraction)
            previous = next_keyframe
        return self.keyframes[-1].bbox


def _interpolate_bbox(start: BBox, end: BBox, fraction: float) -> BBox:
    return BBox(
        x=start.x + (end.x - start.x) * fraction,
        y=start.y + (end.y - start.y) * fraction,
        width=start.width + (end.width - start.width) * fraction,
        height=start.height + (end.height - start.height) * fraction,
    )
