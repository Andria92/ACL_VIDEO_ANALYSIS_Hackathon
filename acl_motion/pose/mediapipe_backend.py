"""MediaPipe Pose Landmarker backend.

MediaPipe details stay behind the PoseBackend contract. Primary measurements are
only image-plane x/y coordinates; model-estimated world coordinates are preserved
as experimental metadata for traceability and are not used by Milestone 1.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from acl_motion.pose.base import PoseBackend
from acl_motion.pose.models import Landmark, PoseFrame, QualityFlag, QualityFlagCode
from acl_motion.video.roi import BBox


class MediaPipePoseBackend(PoseBackend):
    """MediaPipe Pose Landmarker wrapper emitting the canonical PoseFrame schema."""

    name = "mediapipe"
    model_name = "MediaPipe Pose Landmarker"

    def __init__(
        self,
        *,
        model_path: str | Path,
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe backend requires 'mediapipe' and 'opencv-python'. "
                "Install project dependencies with: python -m pip install -e '.[dev]'"
            ) from exc

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"MediaPipe Pose Landmarker model not found: {model_file}. "
                "Run scripts/download_mediapipe_pose_model.py or pass --model-path."
            )

        self._cv2 = cv2
        self._mp = mp
        self._vision = vision
        self._model_path = model_file
        self._landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(model_file),
                    delegate=BaseOptions.Delegate.CPU,
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=num_poses,
                min_pose_detection_confidence=min_pose_detection_confidence,
                min_pose_presence_confidence=min_pose_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_segmentation_masks=False,
            )
        )
        self._landmark_names = tuple(member.name.lower() for member in vision.PoseLandmark)

    @property
    def landmark_names(self) -> tuple[str, ...]:
        return self._landmark_names

    def extract_frame(
        self,
        image: Any,
        roi: BBox | None = None,
        *,
        frame_index: int,
        timestamp_ms: float,
        source_id: str,
    ) -> PoseFrame:
        """Extract raw pose landmarks from one BGR video frame."""

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

        crop_height, crop_width = crop.shape[:2]
        rgb = self._cv2.cvtColor(crop, self._cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        results = self._landmarker.detect_for_video(mp_image, round(timestamp_ms))

        frame_flags: tuple[QualityFlag, ...] = ()
        if not results.pose_landmarks:
            flag = QualityFlag(
                code=QualityFlagCode.TARGET_NOT_FOUND,
                message="Pose model did not return landmarks for this frame.",
                frame_index=frame_index,
            )
            frame_flags = (flag,)
            landmarks = self._empty_landmarks(flag)
        else:
            landmarks = self._convert_landmarks(
                results=results,
                crop_width=crop_width,
                crop_height=crop_height,
                image_width=image_width,
                image_height=image_height,
                crop_x=crop_x,
                crop_y=crop_y,
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
                "crop_width": crop_width,
                "crop_height": crop_height,
                "model_name": self.model_name,
                "model_path": str(self._model_path),
            },
        )

    def close(self) -> None:
        self._landmarker.close()

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
            for name in self._landmark_names
        }

    def _convert_landmarks(
        self,
        *,
        results: Any,
        crop_width: int,
        crop_height: int,
        image_width: int,
        image_height: int,
        crop_x: float,
        crop_y: float,
    ) -> dict[str, Landmark]:
        converted: dict[str, Landmark] = {}
        raw_landmarks = results.pose_landmarks[0]
        world_landmarks = results.pose_world_landmarks[0] if results.pose_world_landmarks else None
        for index, raw_landmark in enumerate(raw_landmarks):
            name = self._landmark_names[index]
            x_crop = raw_landmark.x * crop_width
            y_crop = raw_landmark.y * crop_height
            x_px = crop_x + x_crop
            y_px = crop_y + y_crop
            finite = all(math.isfinite(value) for value in (x_px, y_px))
            in_image = 0 <= x_px <= image_width and 0 <= y_px <= image_height
            flags = ()
            if finite and not in_image:
                flags = (
                    QualityFlag(
                        code=QualityFlagCode.LANDMARK_OUTSIDE_IMAGE,
                        message="Landmark lies outside the full image bounds.",
                        landmark_name=name,
                    ),
                )
            metadata: dict[str, Any] = {}
            if world_landmarks is not None:
                world = world_landmarks[index]
                metadata = {
                    "experimental_world_x": world.x,
                    "experimental_world_y": world.y,
                    "experimental_world_z": world.z,
                }
            converted[name] = Landmark(
                name=name,
                x_px=x_px if finite else None,
                y_px=y_px if finite else None,
                x_norm=(x_px / image_width) if finite and image_width else None,
                y_norm=(y_px / image_height) if finite and image_height else None,
                confidence=getattr(raw_landmark, "presence", None)
                or getattr(raw_landmark, "visibility", None),
                visibility=getattr(raw_landmark, "visibility", None),
                presence=getattr(raw_landmark, "presence", None),
                observed=finite,
                backend_specific_metadata=metadata,
                quality_flags=flags,
            )
        return converted
