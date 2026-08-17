"""Case-level movement-signature contracts for future cohort analytics."""

from acl_motion.signatures.case_signature import (
    SIGNATURE_VERSION,
    CaseMovementSignature,
    build_case_movement_signature,
    build_clustering_feature_registry,
)

__all__ = [
    "SIGNATURE_VERSION",
    "CaseMovementSignature",
    "build_case_movement_signature",
    "build_clustering_feature_registry",
]
