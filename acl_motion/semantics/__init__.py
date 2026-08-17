"""Deterministic semantic movement interpretation layer."""

from acl_motion.semantics.bilateral import (
    BilateralHkaSummary,
    compute_bilateral_hka_summary,
)
from acl_motion.semantics.builder import build_movement_observations
from acl_motion.semantics.metric_explorer import (
    MetricVisualisationSpec,
    SelectionMode,
    build_metric_explorer_payload,
    metric_statistics,
    selection_statistics,
)
from acl_motion.semantics.models import MovementObservation
from acl_motion.semantics.path import (
    CameraMotionConfig,
    PathAnalysisConfig,
    compensate_projected_path,
    direction_change_summary,
    estimate_background_camera_motion,
    path_quality_summary,
)
from acl_motion.semantics.phases import (
    MovementPhase,
    PhaseEvidenceStatus,
    PhaseSegmentationConfig,
    segment_movement_phases,
)
from acl_motion.semantics.vocabulary import (
    MovementDescriptorDefinition,
    MovementVocabularyConfig,
    ObservableMovementDescription,
    build_controlled_movement_vocabulary,
    build_observable_movement_description_payload,
)

__all__ = [
    "BilateralHkaSummary",
    "CameraMotionConfig",
    "MetricVisualisationSpec",
    "MovementDescriptorDefinition",
    "MovementObservation",
    "MovementPhase",
    "MovementVocabularyConfig",
    "ObservableMovementDescription",
    "PathAnalysisConfig",
    "PhaseEvidenceStatus",
    "PhaseSegmentationConfig",
    "SelectionMode",
    "build_controlled_movement_vocabulary",
    "build_metric_explorer_payload",
    "build_movement_observations",
    "build_observable_movement_description_payload",
    "compensate_projected_path",
    "compute_bilateral_hka_summary",
    "direction_change_summary",
    "estimate_background_camera_motion",
    "metric_statistics",
    "path_quality_summary",
    "segment_movement_phases",
    "selection_statistics",
]
