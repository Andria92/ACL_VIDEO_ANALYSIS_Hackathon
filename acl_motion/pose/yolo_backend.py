"""Ultralytics YOLO pose backend.

This is a pragmatic Milestone 1 fallback when MediaPipe is unavailable in the
runtime. It exposes COCO-style 2D pose landmarks: shoulders, elbows, wrists,
hips, knees, ankles, and face landmarks. It does not emit heels or foot-index
landmarks.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from acl_motion.pose.base import PoseBackend
from acl_motion.pose.models import Landmark, PoseFrame, QualityFlag, QualityFlagCode
from acl_motion.video.roi import BBox

COCO_POSE_LANDMARKS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class YoloPoseBackend(PoseBackend):
    """Ultralytics YOLO pose wrapper emitting canonical PoseFrame rows."""

    name = "yolo"
    model_name = "Ultralytics YOLO pose"

    def __init__(
        self,
        *,
        model_path: str | Path,
        confidence_threshold: float = 0.25,
        detection_confidence_threshold: float | None = None,
        device: str = "cpu",
        selection_strategy: str = "largest",
        image_size: int = 640,
        iou_threshold: float = 0.7,
        temporal_max_gap_frames: int = 12,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO backend requires 'ultralytics'. "
                "Install project dependencies with: python -m pip install -e '.[dev]'"
            ) from exc

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"YOLO pose model not found: {model_file}. "
                "Run scripts/download_yolo_pose_model.py or pass --model-path."
            )

        detection_threshold = (
            confidence_threshold
            if detection_confidence_threshold is None
            else detection_confidence_threshold
        )
        for label, value in (
            ("confidence_threshold", confidence_threshold),
            ("detection_confidence_threshold", detection_threshold),
            ("iou_threshold", iou_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be between 0 and 1.")
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        if temporal_max_gap_frames < 0:
            raise ValueError("temporal_max_gap_frames cannot be negative.")
        if selection_strategy not in {"largest", "center", "temporal"}:
            raise ValueError(
                "selection_strategy must be 'largest', 'center', or 'temporal'."
            )

        self._model_path = model_file
        self._model = YOLO(str(model_file))
        self._confidence_threshold = confidence_threshold
        self._detection_confidence_threshold = detection_threshold
        self._device = device
        self._selection_strategy = selection_strategy
        self._image_size = int(image_size)
        self._iou_threshold = float(iou_threshold)
        self._temporal_max_gap_frames = int(temporal_max_gap_frames)
        self._previous_selected_bbox_xyxy: np.ndarray | None = None
        self._previous_selected_frame_index: int | None = None
        self._last_selection_metadata: dict[str, Any] = {}

    @property
    def landmark_names(self) -> tuple[str, ...]:
        return COCO_POSE_LANDMARKS

    def extract_frame(
        self,
        image: Any,
        roi: BBox | None = None,
        *,
        frame_index: int,
        timestamp_ms: float,
        source_id: str,
    ) -> PoseFrame:
        """Extract raw YOLO pose landmarks from one BGR video frame."""

        image_height, image_width = image.shape[:2]
        target_bbox = roi.clamp(image_width, image_height) if roi is not None else None
        crop = image
        crop_x = 0.0
        crop_y = 0.0
        if target_bbox is not None:
            x1, y1, x2, y2 = target_bbox.as_int_xyxy()
            crop = image[y1:y2, x1:x2]
            crop_x = float(x1)
            crop_y = float(y1)

        if crop.size == 0:
            flag = QualityFlag(
                code=QualityFlagCode.ROI_OUTSIDE_IMAGE,
                message="ROI produced an empty crop.",
                frame_index=frame_index,
            )
            return PoseFrame(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                source_id=source_id,
                backend=self.name,
                target_bbox=target_bbox,
                landmarks=self._empty_landmarks(flag),
                quality_flags=(flag,),
                metadata={"image_width": image_width, "image_height": image_height},
            )

        results = self._model.predict(
            source=crop,
            verbose=False,
            conf=self._detection_confidence_threshold,
            device=self._device,
            imgsz=self._image_size,
            iou=self._iou_threshold,
        )
        result = results[0]
        selected_index = self._select_pose_index(
            result,
            crop_width=crop.shape[1],
            crop_height=crop.shape[0],
            crop_x=crop_x,
            crop_y=crop_y,
            frame_index=frame_index,
        )
        if selected_index is None:
            flag = QualityFlag(
                code=QualityFlagCode.TARGET_NOT_FOUND,
                message="YOLO pose did not return a person pose in this frame.",
                frame_index=frame_index,
            )
            frame_flags = (flag,)
            landmarks = self._empty_landmarks(flag)
        else:
            frame_flags = ()
            landmarks = self._convert_landmarks(
                result=result,
                selected_index=selected_index,
                crop_x=crop_x,
                crop_y=crop_y,
                image_width=image_width,
                image_height=image_height,
            )

        return PoseFrame(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            source_id=source_id,
            backend=self.name,
            target_bbox=target_bbox,
            landmarks=landmarks,
            quality_flags=frame_flags,
            metadata={
                "image_width": image_width,
                "image_height": image_height,
                "crop_width": crop.shape[1],
                "crop_height": crop.shape[0],
                "model_name": self.model_name,
                "model_path": str(self._model_path),
                "selected_pose_index": selected_index,
                "selection_strategy": self._selection_strategy,
                "image_size": self._image_size,
                "detection_confidence_threshold": self._detection_confidence_threshold,
                "landmark_confidence_threshold": self._confidence_threshold,
                "iou_threshold": self._iou_threshold,
                "temporal_max_gap_frames": self._temporal_max_gap_frames,
                "pose_count": self._pose_count(result),
                "landmark_schema": "COCO-17",
                **self._last_selection_metadata,
            },
        )

    def _empty_landmarks(self, flag: QualityFlag) -> dict[str, Landmark]:
        return {
            name: Landmark(
                name=name,
                x_px=None,
                y_px=None,
                x_norm=None,
                y_norm=None,
                confidence=None,
                visibility=None,
                presence=None,
                observed=False,
                quality_flags=(flag,),
            )
            for name in COCO_POSE_LANDMARKS
        }

    def reset_tracking(self) -> None:
        """Forget temporal target history after a discontinuity or unavailable interval."""

        self._previous_selected_bbox_xyxy = None
        self._previous_selected_frame_index = None
        self._last_selection_metadata = {}

    def _select_pose_index(
        self,
        result: Any,
        *,
        crop_width: int,
        crop_height: int,
        crop_x: float = 0.0,
        crop_y: float = 0.0,
        frame_index: int | None = None,
    ) -> int | None:
        self._last_selection_metadata = {}
        if result.keypoints is None or result.keypoints.xy is None:
            return None
        keypoints_xy = result.keypoints.xy.cpu().numpy()
        if keypoints_xy.shape[0] == 0:
            return None
        if result.boxes is None or result.boxes.xyxy is None or len(result.boxes.xyxy) == 0:
            return 0
        boxes = result.boxes.xyxy.cpu().numpy()
        if self._selection_strategy == "center":
            centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
            centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
            distances = (centers_x - crop_width / 2) ** 2 + (centers_y - crop_height / 2) ** 2
            selected_index = int(np.argmin(distances))
        elif self._selection_strategy == "temporal":
            selected_index = self._select_temporal_pose_index(
                result,
                boxes=boxes,
                crop_width=crop_width,
                crop_height=crop_height,
                crop_x=crop_x,
                crop_y=crop_y,
                frame_index=frame_index,
            )
        else:
            areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(
                boxes[:, 3] - boxes[:, 1], 0
            )
            selected_index = int(np.argmax(areas))
        self._last_selection_metadata.update(
            _competing_pose_metadata(boxes, selected_index)
        )
        return selected_index

    def _select_temporal_pose_index(
        self,
        result: Any,
        *,
        boxes: np.ndarray,
        crop_width: int,
        crop_height: int,
        crop_x: float,
        crop_y: float,
        frame_index: int | None,
    ) -> int:
        global_boxes = boxes.astype(float, copy=True)
        global_boxes[:, [0, 2]] += crop_x
        global_boxes[:, [1, 3]] += crop_y

        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
        center_distance = np.sqrt(
            (centers_x - crop_width / 2) ** 2
            + (centers_y - crop_height / 2) ** 2
        )
        half_diagonal = max(math.hypot(crop_width, crop_height) / 2, 1.0)
        center_score = np.clip(1.0 - center_distance / half_diagonal, 0.0, 1.0)

        areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(
            boxes[:, 3] - boxes[:, 1], 0
        )
        maximum_area = max(float(np.max(areas)), 1.0)
        area_score = np.sqrt(np.clip(areas / maximum_area, 0.0, 1.0))
        pose_score = self._candidate_pose_confidence(result, len(boxes))

        previous_box = getattr(self, "_previous_selected_bbox_xyxy", None)
        previous_frame = getattr(self, "_previous_selected_frame_index", None)
        max_gap = getattr(self, "_temporal_max_gap_frames", 12)
        frame_gap = (
            frame_index - previous_frame
            if frame_index is not None and previous_frame is not None
            else None
        )
        continuity_available = (
            previous_box is not None
            and frame_gap is not None
            and 0 < frame_gap <= max_gap
        )
        if continuity_available:
            continuity_score = np.asarray(
                [_bbox_iou(candidate, previous_box) for candidate in global_boxes]
            )
            scores = (
                0.50 * continuity_score
                + 0.30 * center_score
                + 0.10 * area_score
                + 0.10 * pose_score
            )
        else:
            continuity_score = np.zeros(len(boxes), dtype=float)
            scores = 0.65 * center_score + 0.15 * area_score + 0.20 * pose_score

        selected_index = int(np.argmax(scores))
        self._previous_selected_bbox_xyxy = global_boxes[selected_index].copy()
        self._previous_selected_frame_index = frame_index
        self._last_selection_metadata = {
            "target_selection_score": float(scores[selected_index]),
            "target_continuity_iou": float(continuity_score[selected_index]),
            "target_continuity_used": bool(continuity_available),
            "target_selection_frame_gap": frame_gap,
            "selected_box_touches_crop_edge": _box_touches_crop_edge(
                boxes[selected_index], crop_width=crop_width, crop_height=crop_height
            ),
        }
        return selected_index

    @staticmethod
    def _candidate_pose_confidence(result: Any, candidate_count: int) -> np.ndarray:
        if result.keypoints is None or result.keypoints.conf is None:
            return np.full(candidate_count, 0.5, dtype=float)
        confidence = result.keypoints.conf.cpu().numpy()
        if confidence.ndim != 2 or confidence.shape[0] != candidate_count:
            return np.full(candidate_count, 0.5, dtype=float)
        with np.errstate(invalid="ignore"):
            scores = np.nanmedian(confidence, axis=1)
        return np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)

    def _pose_count(self, result: Any) -> int:
        if result.keypoints is None or result.keypoints.xy is None:
            return 0
        return int(result.keypoints.xy.shape[0])

    def _convert_landmarks(
        self,
        *,
        result: Any,
        selected_index: int,
        crop_x: float,
        crop_y: float,
        image_width: int,
        image_height: int,
    ) -> dict[str, Landmark]:
        keypoints_xy = result.keypoints.xy.cpu().numpy()[selected_index]
        keypoint_conf = None
        if result.keypoints.conf is not None:
            keypoint_conf = result.keypoints.conf.cpu().numpy()[selected_index]

        converted: dict[str, Landmark] = {}
        for index, name in enumerate(COCO_POSE_LANDMARKS):
            x_value = float(keypoints_xy[index, 0])
            y_value = float(keypoints_xy[index, 1])
            confidence = float(keypoint_conf[index]) if keypoint_conf is not None else None
            finite = all(math.isfinite(value) for value in (x_value, y_value))
            observed = finite and (confidence is None or confidence > 0)
            x_px = crop_x + x_value if observed else None
            y_px = crop_y + y_value if observed else None
            flags = ()
            if confidence is not None and confidence < self._confidence_threshold:
                flags = (
                    QualityFlag(
                        code=QualityFlagCode.LOW_LANDMARK_CONFIDENCE,
                        message="Landmark confidence is below backend threshold.",
                        landmark_name=name,
                    ),
                )

            converted[name] = Landmark(
                name=name,
                x_px=x_px,
                y_px=y_px,
                x_norm=(x_px / image_width) if x_px is not None and image_width else None,
                y_norm=(y_px / image_height) if y_px is not None and image_height else None,
                confidence=confidence,
                visibility=None,
                presence=confidence,
                observed=observed,
                backend_specific_metadata={"landmark_schema": "COCO-17"},
                quality_flags=flags,
            )
        return converted


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    first_area = max(float(first[2] - first[0]), 0.0) * max(
        float(first[3] - first[1]), 0.0
    )
    second_area = max(float(second[2] - second[0]), 0.0) * max(
        float(second[3] - second[1]), 0.0
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _box_touches_crop_edge(
    box: np.ndarray,
    *,
    crop_width: int,
    crop_height: int,
    margin_fraction: float = 0.02,
) -> bool:
    margin_x = crop_width * margin_fraction
    margin_y = crop_height * margin_fraction
    return bool(
        box[0] <= margin_x
        or box[1] <= margin_y
        or box[2] >= crop_width - margin_x
        or box[3] >= crop_height - margin_y
    )


def _competing_pose_metadata(boxes: np.ndarray, selected_index: int) -> dict[str, Any]:
    """Describe nearby person detections without declaring target identity valid.

    Detection confidence is not identity evidence.  Continuous overlap and proximity
    values are therefore preserved for the quality layer and human review rather
    than being converted into a backend-level pass/fail decision.
    """

    selected = boxes[selected_index].astype(float)
    competitors = np.delete(boxes.astype(float), selected_index, axis=0)
    if len(competitors) == 0:
        return {
            "competing_pose_count": 0,
            "competing_pose_max_iou": 0.0,
            "competing_pose_min_center_distance_fraction": None,
        }

    selected_center = np.asarray(
        [(selected[0] + selected[2]) / 2.0, (selected[1] + selected[3]) / 2.0]
    )
    selected_diagonal = max(
        math.hypot(selected[2] - selected[0], selected[3] - selected[1]),
        1.0,
    )
    competitor_centers = np.column_stack(
        (
            (competitors[:, 0] + competitors[:, 2]) / 2.0,
            (competitors[:, 1] + competitors[:, 3]) / 2.0,
        )
    )
    center_distances = np.linalg.norm(competitor_centers - selected_center, axis=1)
    overlaps = np.asarray([_bbox_iou(selected, other) for other in competitors])
    return {
        "competing_pose_count": len(competitors),
        "competing_pose_max_iou": float(np.max(overlaps)),
        "competing_pose_min_center_distance_fraction": float(
            np.min(center_distances) / selected_diagonal
        ),
    }
