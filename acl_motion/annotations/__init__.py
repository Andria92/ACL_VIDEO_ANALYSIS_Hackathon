"""Human annotation infrastructure for ACL Movement Analytics Lab."""

from acl_motion.annotations.models import (
    ANNOTATION_UI_VERSION,
    AnnotationCase,
    AnnotationProvenance,
    EventConfidence,
    HumanAnnotationSession,
    MovementWindowAnnotation,
    OperatorFlag,
    RoiKeyframeAnnotation,
    TargetAcceptedIntervalAnnotation,
    TargetUnavailableIntervalAnnotation,
)
from acl_motion.annotations.movement_window import (
    infer_movement_start_frame,
    migrate_session_to_movement_window,
)

__all__ = [
    "ANNOTATION_UI_VERSION",
    "AnnotationCase",
    "AnnotationProvenance",
    "EventConfidence",
    "HumanAnnotationSession",
    "MovementWindowAnnotation",
    "OperatorFlag",
    "RoiKeyframeAnnotation",
    "TargetAcceptedIntervalAnnotation",
    "TargetUnavailableIntervalAnnotation",
    "infer_movement_start_frame",
    "migrate_session_to_movement_window",
]
