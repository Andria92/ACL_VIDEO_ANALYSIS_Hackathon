"""Skeleton overlay rendering for visual pose QC."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from acl_motion.video.roi import BBox

MEDIAPIPE_SKELETON: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_foot_index"),
)


def draw_pose_overlay(
    frame: Any,
    landmarks: Mapping[str, Mapping[str, Any]],
    *,
    bbox: BBox | None = None,
    frame_label: str | None = None,
    confidence_threshold: float = 0.0,
) -> Any:
    """Draw ROI, skeleton, and landmarks onto a video frame."""

    import cv2

    output = frame.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox.as_int_xyxy()
        cv2.rectangle(output, (x1, y1), (x2, y2), (60, 200, 255), 2)

    for start, end in MEDIAPIPE_SKELETON:
        point_a = _visible_point(landmarks.get(start), confidence_threshold)
        point_b = _visible_point(landmarks.get(end), confidence_threshold)
        if point_a is not None and point_b is not None:
            cv2.line(output, point_a, point_b, (255, 180, 80), 2, lineType=cv2.LINE_AA)

    for landmark in landmarks.values():
        point = _visible_point(landmark, confidence_threshold)
        if point is None:
            continue
        confidence = landmark.get("confidence")
        color = _confidence_color(confidence)
        cv2.circle(output, point, 4, color, -1, lineType=cv2.LINE_AA)

    if frame_label:
        cv2.putText(
            output,
            frame_label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    return output


def _visible_point(
    landmark: Mapping[str, Any] | None,
    confidence_threshold: float,
) -> tuple[int, int] | None:
    if not landmark or not bool(landmark.get("observed")):
        return None
    confidence = landmark.get("confidence")
    if confidence is not None and confidence < confidence_threshold:
        return None
    x_px = landmark.get("x_px")
    y_px = landmark.get("y_px")
    if x_px is None or y_px is None:
        return None
    return round(float(x_px)), round(float(y_px))


def _confidence_color(confidence: float | None) -> tuple[int, int, int]:
    if confidence is None:
        return (220, 220, 220)
    if confidence >= 0.75:
        return (70, 220, 70)
    if confidence >= 0.45:
        return (40, 220, 255)
    return (60, 90, 255)
