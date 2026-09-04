"""Skeleton overlay rendering for visual pose QC."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from acl_motion.video.roi import BBox

DEFAULT_POSE_DISPLAY_CONFIDENCE_THRESHOLD = 0.25
PROVISIONAL_POSE_DISPLAY_CONFIDENCE_THRESHOLD = 0.45

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
    confidence_threshold: float = DEFAULT_POSE_DISPLAY_CONFIDENCE_THRESHOLD,
) -> Any:
    """Draw a confidence-aware ROI and skeleton onto a video frame."""

    import cv2

    output = frame.copy()
    if bbox is not None:
        x1, y1, x2, y2 = bbox.as_int_xyxy()
        cv2.rectangle(output, (x1, y1), (x2, y2), (60, 200, 255), 2)

    for start, end in MEDIAPIPE_SKELETON:
        landmark_a = landmarks.get(start)
        landmark_b = landmarks.get(end)
        point_a = _visible_point(landmark_a, confidence_threshold)
        point_b = _visible_point(landmark_b, confidence_threshold)
        if point_a is not None and point_b is not None:
            if _provisional_landmark(landmark_a) or _provisional_landmark(landmark_b):
                _draw_dashed_line(output, point_a, point_b, (40, 190, 255))
            else:
                cv2.line(
                    output,
                    point_a,
                    point_b,
                    (255, 180, 80),
                    2,
                    lineType=cv2.LINE_AA,
                )

    for landmark in landmarks.values():
        point = _visible_point(landmark, confidence_threshold)
        if point is None:
            continue
        confidence = landmark.get("confidence")
        color = _confidence_color(confidence)
        thickness = 1 if _provisional_landmark(landmark) else -1
        cv2.circle(output, point, 4, color, thickness, lineType=cv2.LINE_AA)

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
    if (
        not landmark
        or not bool(landmark.get("observed"))
        or bool(landmark.get("rejected"))
    ):
        return None
    confidence = landmark.get("confidence")
    if confidence is not None and confidence < confidence_threshold:
        return None
    x_px = landmark.get("x_px")
    y_px = landmark.get("y_px")
    if x_px is None or y_px is None:
        return None
    return round(float(x_px)), round(float(y_px))


def _provisional_landmark(landmark: Mapping[str, Any] | None) -> bool:
    if not landmark:
        return True
    confidence = landmark.get("confidence")
    if confidence is None or confidence < PROVISIONAL_POSE_DISPLAY_CONFIDENCE_THRESHOLD:
        return True
    status = str(
        landmark.get("processing_status")
        or landmark.get("landmark_status")
        or ""
    ).upper()
    return status in {
        "INTERPOLATED",
        "LOW_CONFIDENCE",
        "TEMPORAL_OUTLIER",
        "IDENTITY_UNCERTAIN",
    }


def _draw_dashed_line(
    image: Any,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    *,
    dash_length: float = 7.0,
) -> None:
    import cv2

    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    if distance == 0:
        return
    segment_count = max(math.ceil(distance / dash_length), 1)
    for segment in range(0, segment_count, 2):
        start_fraction = segment / segment_count
        end_fraction = min((segment + 1) / segment_count, 1.0)
        dash_start = (
            round(start[0] + (end[0] - start[0]) * start_fraction),
            round(start[1] + (end[1] - start[1]) * start_fraction),
        )
        dash_end = (
            round(start[0] + (end[0] - start[0]) * end_fraction),
            round(start[1] + (end[1] - start[1]) * end_fraction),
        )
        cv2.line(image, dash_start, dash_end, color, 1, lineType=cv2.LINE_AA)


def _confidence_color(confidence: float | None) -> tuple[int, int, int]:
    if confidence is None:
        return (220, 220, 220)
    if confidence >= 0.75:
        return (70, 220, 70)
    if confidence >= 0.45:
        return (40, 220, 255)
    return (60, 90, 255)
