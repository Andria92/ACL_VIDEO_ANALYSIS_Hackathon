"""Pose backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

from acl_motion.pose.models import PoseFrame
from acl_motion.video.roi import BBox


class PoseBackend(ABC):
    """Abstract interface for replaceable pose-estimation backends."""

    name: str
    model_name: str

    @property
    def landmark_names(self) -> tuple[str, ...]:
        """Canonical landmark names emitted by this backend, if known."""

        return ()

    @abstractmethod
    def extract_frame(
        self,
        image: Any,
        roi: BBox | None = None,
        *,
        frame_index: int,
        timestamp_ms: float,
        source_id: str,
    ) -> PoseFrame:
        """Extract one raw pose frame from an image and optional target ROI."""

    def close(self) -> None:
        """Release backend resources."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
