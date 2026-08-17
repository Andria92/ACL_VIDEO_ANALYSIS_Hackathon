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
        device: str = "cpu",
        selection_strategy: str = "largest",
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

        self._model_path = model_file
        self._model = YOLO(str(model_file))
        self._confidence_threshold = confidence_threshold
        self._device = device
        if selection_strategy not in {"largest", "center"}:
            raise ValueError("selection_strategy must be 'largest' or 'center'.")
        self._selection_strategy = selection_strategy

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
            conf=self._confidence_threshold,
            device=self._device,
        )
        result = results[0]
        selected_index = self._select_pose_index(result, crop_width=crop.shape[1], crop_height=crop.shape[0])
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
                "pose_count": self._pose_count(result),
                "landmark_schema": "COCO-17",
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

    def _select_pose_index(self, result: Any, *, crop_width: int, crop_height: int) -> int | None:
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
            return int(np.argmin(distances))
        areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
        return int(np.argmax(areas))

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
