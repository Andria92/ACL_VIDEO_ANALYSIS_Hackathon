"""Feature grouping registry for MovementProfile outputs."""

from __future__ import annotations

from acl_motion.profiles.models import BodyRegion, FeatureFamily


def classify_feature(feature_name: str) -> tuple[BodyRegion, FeatureFamily]:
    """Return body-region and feature-family metadata for a canonical feature."""

    if "bilateral" in feature_name or feature_name.startswith(("injured_", "contralateral_")):
        return BodyRegion.BILATERAL, FeatureFamily.BILATERAL
    if "wrist" in feature_name:
        return BodyRegion.UPPER_BODY, FeatureFamily.WRIST_GEOMETRY
    if "hka" in feature_name or "knee" in feature_name:
        family = FeatureFamily.KNEE_GEOMETRY if "knee" in feature_name else FeatureFamily.LOWER_LIMB_GEOMETRY
        return BodyRegion.LOWER_LIMB, family
    if any(token in feature_name for token in ("trunk", "hip_line", "shoulder_line", "pelvis")):
        return BodyRegion.TRUNK_PELVIS, FeatureFamily.TRUNK_PELVIS
    if "elbow" in feature_name or "upper_arm" in feature_name:
        return BodyRegion.UPPER_BODY, FeatureFamily.UPPER_LIMB
    return BodyRegion.WHOLE_BODY, FeatureFamily.LOWER_LIMB_GEOMETRY
