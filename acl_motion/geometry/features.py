"""Milestone 3 framewise projected whole-body geometry features."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from acl_motion.cases.models import InjurySide
from acl_motion.geometry.angles import angle_2d, wrapped_angle_difference_deg
from acl_motion.geometry.distances import (
    midpoint,
    normalized_distance,
    signed_point_line_distance,
)
from acl_motion.geometry.normalisation import (
    NormalisationReference,
    compute_body_scale_reference,
)
from acl_motion.geometry.orientation import segment_orientation_2d

GEOMETRY_VERSION = "m3_geometry_primitives_v1"
FEATURE_SET_VERSION = "m3_projected_geometry_core_v1"
VIEW_SUITABILITY_NOT_ASSESSED = "not_assessed_m3_generic_projected_geometry"
INVALID_TARGET_STATUSES = {
    "TARGET_IDENTITY_UNCERTAIN",
    "TARGET_NOT_FOUND",
    "INVALID_TRACK_SEGMENT",
}
FORBIDDEN_CLINICAL_LABELS = (
    "knee_valgus",
    "knee_varus",
    "knee_flexion",
    "spinal_rotation",
    "lumbar_rotation",
    "medial_knee",
    "fppa",
)


class FeatureStatus(StrEnum):
    """Feature-level support state."""

    SUPPORTED = "SUPPORTED"
    INSUFFICIENT_LANDMARKS = "INSUFFICIENT_LANDMARKS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INVALID_TARGET_FRAME = "INVALID_TARGET_FRAME"
    UNSUPPORTED_VIEW = "UNSUPPORTED_VIEW"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"


@dataclass(frozen=True, slots=True)
class LandmarkInput:
    """Processed coordinate input for one landmark in one frame."""

    landmark_name: str
    point: tuple[float, float] | None
    coordinate_source: str
    landmark_status: str
    processing_status: str
    interpolated: bool
    smoothed: bool
    rejected: bool


@dataclass(frozen=True, slots=True)
class FeatureResult:
    """Structured framewise geometry result with evidence provenance."""

    case_id: str
    source_id: str
    frame_index: int
    source_frame_index: int
    analysis_frame_index: int
    timestamp_ms: float
    feature_name: str
    feature_value: float
    unit: str
    status: str
    quality_status: str
    landmarks_used: tuple[str, ...]
    completeness: float
    observed: bool
    input_interpolated: bool
    input_smoothed: bool
    view_suitability: str
    rejection_reason: str
    frame_status: str
    metadata: dict

    def to_row(self) -> dict:
        row = asdict(self)
        row["landmarks_used"] = list(self.landmarks_used)
        return row


def compute_geometry_features(
    processed_pose: pd.DataFrame,
    *,
    injured_side: InjurySide | str = InjurySide.UNKNOWN,
    normalisation_reference: NormalisationReference | None = None,
) -> tuple[pd.DataFrame, NormalisationReference]:
    """Compute M3 framewise generic projected geometry from processed pose."""

    _assert_scientific_names()
    side = InjurySide(injured_side)
    normalisation = normalisation_reference or compute_body_scale_reference(processed_pose)
    results: list[FeatureResult] = []

    for _, frame in processed_pose.groupby("frame_index", sort=True):
        frame_results = _compute_frame_features(frame, normalisation)
        results.extend(frame_results)
        if side in {InjurySide.LEFT, InjurySide.RIGHT}:
            results.extend(_compute_laterality_features(frame_results, side))

    feature_df = pd.DataFrame([result.to_row() for result in results])
    return feature_df, normalisation


def build_feature_completeness(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize supported/unavailable frames for each feature."""

    rows: list[dict] = []
    for feature_name, group in feature_df.groupby("feature_name", sort=True):
        supported = group["status"].eq(FeatureStatus.SUPPORTED.value)
        unsupported = group[~supported]
        primary_reason = ""
        if not unsupported.empty:
            reasons = unsupported["rejection_reason"].replace("", np.nan).dropna()
            primary_reason = str(reasons.mode().iloc[0]) if not reasons.empty else ""
        rows.append(
            {
                "feature_name": feature_name,
                "supported_frames": int(supported.sum()),
                "relevant_frames": len(group),
                "completeness": float(supported.mean()) if len(group) else 0.0,
                "unsupported_frames": int((~supported).sum()),
                "primary_rejection_reason": primary_reason,
            }
        )
    return pd.DataFrame(rows).sort_values("feature_name").reset_index(drop=True)


def _compute_frame_features(
    frame: pd.DataFrame,
    normalisation: NormalisationReference,
) -> list[FeatureResult]:
    context = _frame_context(frame)
    landmarks = {name: _landmark_input(frame, name) for name in frame["landmark_name"].unique()}
    results: list[FeatureResult] = []

    for side in ("left", "right"):
        hip = f"{side}_hip"
        knee = f"{side}_knee"
        ankle = f"{side}_ankle"
        shoulder = f"{side}_shoulder"
        elbow = f"{side}_elbow"
        wrist = f"{side}_wrist"
        results.extend(
            [
                _feature(
                    context,
                    landmarks,
                    (hip, knee, ankle),
                    f"{side}_hka_angle_2d_deg",
                    "deg",
                    lambda pts, _, hip=hip, knee=knee, ankle=ankle: angle_2d(
                        pts[hip], pts[knee], pts[ankle]
                    ),
                    normalisation,
                ),
                _feature(
                    context,
                    landmarks,
                    (hip, knee, ankle),
                    f"{side}_knee_line_deviation_2d",
                    "px",
                    lambda pts, _, hip=hip, knee=knee, ankle=ankle: signed_point_line_distance(
                        pts[knee], pts[hip], pts[ankle]
                    ),
                    normalisation,
                ),
                _feature(
                    context,
                    landmarks,
                    (hip, knee, ankle),
                    f"{side}_knee_line_deviation_normalized",
                    "body_scale",
                    lambda pts, scale_value, hip=hip, knee=knee, ankle=ankle: _divide(
                        signed_point_line_distance(pts[knee], pts[hip], pts[ankle]),
                        scale_value,
                    ),
                    normalisation,
                    requires_scale=True,
                ),
                _feature(
                    context,
                    landmarks,
                    (knee, ankle),
                    f"{side}_knee_ankle_x_offset_normalized",
                    "body_scale",
                    lambda pts, scale_value, knee=knee, ankle=ankle: _divide(
                        pts[knee][0] - pts[ankle][0], scale_value
                    ),
                    normalisation,
                    requires_scale=True,
                ),
                _feature(
                    context,
                    landmarks,
                    (knee, ankle),
                    f"{side}_knee_ankle_distance_normalized",
                    "body_scale",
                    lambda pts, scale_value, knee=knee, ankle=ankle: normalized_distance(
                        pts[knee], pts[ankle], scale_value
                    ),
                    normalisation,
                    requires_scale=True,
                ),
                _feature(
                    context,
                    landmarks,
                    (shoulder, elbow, wrist),
                    f"{side}_elbow_angle_2d_deg",
                    "deg",
                    lambda pts, _, shoulder=shoulder, elbow=elbow, wrist=wrist: angle_2d(
                        pts[shoulder], pts[elbow], pts[wrist]
                    ),
                    normalisation,
                ),
                _feature(
                    context,
                    landmarks,
                    (shoulder, elbow),
                    f"{side}_upper_arm_orientation_2d_deg",
                    "deg",
                    lambda pts, _, shoulder=shoulder, elbow=elbow: segment_orientation_2d(
                        pts[shoulder], pts[elbow]
                    ),
                    normalisation,
                ),
                _feature(
                    context,
                    landmarks,
                    (wrist, "left_hip", "right_hip"),
                    f"{side}_wrist_pelvis_x_offset_normalized",
                    "body_scale",
                    lambda pts, scale_value, wrist=wrist: _divide(
                        pts[wrist][0] - _midpoint_or_nan(pts["left_hip"], pts["right_hip"])[0],
                        scale_value,
                    ),
                    normalisation,
                    requires_scale=True,
                ),
                _feature(
                    context,
                    landmarks,
                    (wrist, "left_hip", "right_hip"),
                    f"{side}_wrist_pelvis_distance_normalized",
                    "body_scale",
                    lambda pts, scale_value, wrist=wrist: normalized_distance(
                        pts[wrist],
                        _midpoint_or_nan(pts["left_hip"], pts["right_hip"]),
                        scale_value,
                    ),
                    normalisation,
                    requires_scale=True,
                ),
            ]
        )

    results.extend(
        [
            _feature(
                context,
                landmarks,
                ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
                "projected_trunk_axis_angle_deg",
                "deg",
                lambda pts, _: segment_orientation_2d(
                    _midpoint_or_nan(pts["left_hip"], pts["right_hip"]),
                    _midpoint_or_nan(pts["left_shoulder"], pts["right_shoulder"]),
                ),
                normalisation,
            ),
            _feature(
                context,
                landmarks,
                ("left_hip", "right_hip"),
                "projected_hip_line_angle_deg",
                "deg",
                lambda pts, _: segment_orientation_2d(pts["left_hip"], pts["right_hip"]),
                normalisation,
            ),
            _feature(
                context,
                landmarks,
                ("left_shoulder", "right_shoulder"),
                "projected_shoulder_line_angle_deg",
                "deg",
                lambda pts, _: segment_orientation_2d(
                    pts["left_shoulder"], pts["right_shoulder"]
                ),
                normalisation,
            ),
            _feature(
                context,
                landmarks,
                ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
                "projected_shoulder_pelvis_x_offset_px",
                "px",
                lambda pts, _: _midpoint_or_nan(
                    pts["left_shoulder"], pts["right_shoulder"]
                )[0]
                - _midpoint_or_nan(pts["left_hip"], pts["right_hip"])[0],
                normalisation,
            ),
            _feature(
                context,
                landmarks,
                ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
                "projected_shoulder_pelvis_x_offset_normalized",
                "body_scale",
                lambda pts, scale_value: _divide(
                    _midpoint_or_nan(pts["left_shoulder"], pts["right_shoulder"])[0]
                    - _midpoint_or_nan(pts["left_hip"], pts["right_hip"])[0],
                    scale_value,
                ),
                normalisation,
                requires_scale=True,
            ),
            _feature(
                context,
                landmarks,
                ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
                "projected_shoulder_pelvis_orientation_difference_deg",
                "deg",
                lambda pts, _: wrapped_angle_difference_deg(
                    segment_orientation_2d(pts["left_shoulder"], pts["right_shoulder"]),
                    segment_orientation_2d(pts["left_hip"], pts["right_hip"]),
                ),
                normalisation,
            ),
        ]
    )
    return results


def _feature(
    context: dict,
    landmarks: dict[str, LandmarkInput],
    required: tuple[str, ...],
    name: str,
    unit: str,
    compute: Callable[[dict[str, tuple[float, float]], float], float],
    normalisation: NormalisationReference,
    *,
    requires_scale: bool = False,
) -> FeatureResult:
    available = {
        landmark: landmarks.get(landmark)
        for landmark in required
        if landmarks.get(landmark) is not None and landmarks[landmark].point is not None
    }
    completeness = len(available) / len(required) if required else 1.0
    if context["frame_status"] in INVALID_TARGET_STATUSES:
        return _unavailable_result(
            context,
            required,
            name,
            unit,
            FeatureStatus.INVALID_TARGET_FRAME,
            completeness,
            f"Frame status is {context['frame_status']}.",
            landmarks,
            normalisation,
        )
    if len(available) != len(required):
        status, reason = _missing_status_reason(required, landmarks)
        return _unavailable_result(
            context,
            required,
            name,
            unit,
            status,
            completeness,
            reason,
            landmarks,
            normalisation,
        )
    interpolated_outliers = [
        landmark
        for landmark in required
        if landmarks[landmark].interpolated
        and landmarks[landmark].landmark_status == "TEMPORAL_OUTLIER"
    ]
    if interpolated_outliers:
        return _unavailable_result(
            context,
            required,
            name,
            unit,
            FeatureStatus.INVALID_GEOMETRY,
            completeness,
            (
                "Required landmarks were interpolated after temporal-outlier rejection: "
                f"{interpolated_outliers}."
            ),
            landmarks,
            normalisation,
        )
    if requires_scale and not normalisation.is_available:
        return _unavailable_result(
            context,
            required,
            name,
            unit,
            FeatureStatus.INVALID_GEOMETRY,
            completeness,
            "Body-scale normalisation reference is unavailable.",
            landmarks,
            normalisation,
        )

    points = {landmark: available[landmark].point for landmark in required}
    value = compute(points, normalisation.reference_value_px)
    if not np.isfinite(value):
        return _unavailable_result(
            context,
            required,
            name,
            unit,
            FeatureStatus.INVALID_GEOMETRY,
            completeness,
            "Geometry primitive returned NaN.",
            landmarks,
            normalisation,
        )
    used_inputs = tuple(available[landmark] for landmark in required)
    return FeatureResult(
        **context,
        feature_name=name,
        feature_value=float(value),
        unit=unit,
        status=FeatureStatus.SUPPORTED.value,
        quality_status=_quality_status(context, required, landmarks),
        landmarks_used=required,
        completeness=1.0,
        observed=True,
        input_interpolated=any(item.interpolated for item in used_inputs),
        input_smoothed=any(item.coordinate_source == "smoothed" or item.smoothed for item in used_inputs),
        view_suitability=VIEW_SUITABILITY_NOT_ASSESSED,
        rejection_reason="",
        metadata=_feature_metadata(required, landmarks, normalisation),
    )


def _compute_laterality_features(
    frame_results: list[FeatureResult],
    injured_side: InjurySide,
) -> list[FeatureResult]:
    side_map = {
        "injured": injured_side.value,
        "contralateral": InjurySide.LEFT.value
        if injured_side == InjurySide.RIGHT
        else InjurySide.RIGHT.value,
    }
    by_name = {result.feature_name: result for result in frame_results}
    results: list[FeatureResult] = []

    for base, unit in (
        ("hka_angle_2d_deg", "deg"),
        ("elbow_angle_2d_deg", "deg"),
    ):
        injured = by_name[f"{side_map['injured']}_{base}"]
        contralateral = by_name[f"{side_map['contralateral']}_{base}"]
        results.append(_alias_result(injured, f"injured_{base}", side_map))
        results.append(_alias_result(contralateral, f"contralateral_{base}", side_map))
        results.extend(
            _difference_results(
                injured,
                contralateral,
                feature_name=f"{base.split('_angle')[0]}_projected_bilateral_difference_deg",
                absolute_feature_name=(
                    f"{base.split('_angle')[0]}_projected_bilateral_absolute_difference_deg"
                ),
                unit=unit,
                side_map=side_map,
            )
        )

    for base, unit in (
        ("knee_line_deviation_2d", "px"),
        ("knee_line_deviation_normalized", "body_scale"),
    ):
        injured = by_name[f"{side_map['injured']}_{base}"]
        contralateral = by_name[f"{side_map['contralateral']}_{base}"]
        suffix = "" if base.endswith("_2d") else "_normalized"
        results.extend(
            _difference_results(
                injured,
                contralateral,
                feature_name=f"knee_line_deviation{suffix}_bilateral_difference",
                absolute_feature_name=f"knee_line_deviation{suffix}_bilateral_absolute_difference",
                unit=unit,
                side_map=side_map,
            )
        )
    return results


def _alias_result(source: FeatureResult, feature_name: str, side_map: dict[str, str]) -> FeatureResult:
    metadata = dict(source.metadata)
    metadata.update({"laterality_mapping": side_map, "source_feature_name": source.feature_name})
    return FeatureResult(
        **{
            **asdict(source),
            "feature_name": feature_name,
            "metadata": metadata,
        }
    )


def _difference_results(
    injured: FeatureResult,
    contralateral: FeatureResult,
    *,
    feature_name: str,
    absolute_feature_name: str,
    unit: str,
    side_map: dict[str, str],
) -> list[FeatureResult]:
    context = _result_context(injured)
    landmarks = tuple(dict.fromkeys((*injured.landmarks_used, *contralateral.landmarks_used)))
    metadata = {
        "laterality_mapping": side_map,
        "injured_feature_name": injured.feature_name,
        "contralateral_feature_name": contralateral.feature_name,
    }
    if (
        injured.status == FeatureStatus.SUPPORTED.value
        and contralateral.status == FeatureStatus.SUPPORTED.value
    ):
        value = injured.feature_value - contralateral.feature_value
        status = FeatureStatus.SUPPORTED.value
        reason = ""
    else:
        value = float("nan")
        status = _combined_unavailable_status(injured, contralateral)
        reason = _combined_rejection_reason(injured, contralateral)
    base = FeatureResult(
        **context,
        feature_name=feature_name,
        feature_value=float(value),
        unit=unit,
        status=status,
        quality_status=f"{injured.quality_status} | {contralateral.quality_status}",
        landmarks_used=landmarks,
        completeness=min(injured.completeness, contralateral.completeness),
        observed=status == FeatureStatus.SUPPORTED.value,
        input_interpolated=injured.input_interpolated or contralateral.input_interpolated,
        input_smoothed=injured.input_smoothed or contralateral.input_smoothed,
        view_suitability=VIEW_SUITABILITY_NOT_ASSESSED,
        rejection_reason=reason,
        metadata=metadata,
    )
    absolute = FeatureResult(
        **{
            **asdict(base),
            "feature_name": absolute_feature_name,
            "feature_value": abs(value) if np.isfinite(value) else float("nan"),
        }
    )
    return [base, absolute]


def _unavailable_result(
    context: dict,
    required: tuple[str, ...],
    name: str,
    unit: str,
    status: FeatureStatus,
    completeness: float,
    reason: str,
    landmarks: dict[str, LandmarkInput],
    normalisation: NormalisationReference,
) -> FeatureResult:
    return FeatureResult(
        **context,
        feature_name=name,
        feature_value=float("nan"),
        unit=unit,
        status=status.value,
        quality_status=_quality_status(context, required, landmarks),
        landmarks_used=required,
        completeness=float(completeness),
        observed=False,
        input_interpolated=any(
            landmarks[name].interpolated for name in required if name in landmarks
        ),
        input_smoothed=any(landmarks[name].smoothed for name in required if name in landmarks),
        view_suitability=VIEW_SUITABILITY_NOT_ASSESSED,
        rejection_reason=reason,
        metadata=_feature_metadata(required, landmarks, normalisation),
    )


def _frame_context(frame: pd.DataFrame) -> dict:
    first = frame.iloc[0]
    frame_index = int(first["frame_index"])
    return {
        "case_id": str(first.get("case_id", "")),
        "source_id": str(first.get("source_id", "")),
        "frame_index": frame_index,
        "source_frame_index": int(first.get("source_frame_index", frame_index)),
        "analysis_frame_index": int(first.get("analysis_frame_index", frame_index)),
        "timestamp_ms": float(first["timestamp_ms"]),
        "frame_status": str(first.get("frame_status", "")),
    }


def _landmark_input(frame: pd.DataFrame, landmark_name: str) -> LandmarkInput:
    rows = frame[frame["landmark_name"].eq(landmark_name)]
    if rows.empty:
        return LandmarkInput(
            landmark_name=landmark_name,
            point=None,
            coordinate_source="missing",
            landmark_status="MISSING",
            processing_status="MISSING",
            interpolated=False,
            smoothed=False,
            rejected=True,
        )
    row = rows.iloc[0]
    point: tuple[float, float] | None = None
    source = "missing"
    if pd.notna(row.get("smoothed_x")) and pd.notna(row.get("smoothed_y")):
        point = float(row["smoothed_x"]), float(row["smoothed_y"])
        source = "smoothed"
    elif pd.notna(row.get("clean_x")) and pd.notna(row.get("clean_y")):
        point = float(row["clean_x"]), float(row["clean_y"])
        source = "clean"
    return LandmarkInput(
        landmark_name=landmark_name,
        point=point,
        coordinate_source=source,
        landmark_status=str(row.get("landmark_status", "MISSING")),
        processing_status=str(row.get("processing_status", "MISSING")),
        interpolated=bool(row.get("interpolated", False)),
        smoothed=bool(row.get("smoothed", False)),
        rejected=bool(row.get("rejected", True)),
    )


def _missing_status_reason(
    required: tuple[str, ...],
    landmarks: dict[str, LandmarkInput],
) -> tuple[FeatureStatus, str]:
    missing = [
        name
        for name in required
        if name not in landmarks or landmarks[name].point is None
    ]
    statuses = {landmarks[name].landmark_status for name in missing if name in landmarks}
    if "LOW_CONFIDENCE" in statuses:
        return FeatureStatus.LOW_CONFIDENCE, f"Required landmarks low confidence: {missing}."
    if "TEMPORAL_OUTLIER" in statuses:
        return FeatureStatus.INVALID_GEOMETRY, f"Required landmarks temporal outlier: {missing}."
    return FeatureStatus.INSUFFICIENT_LANDMARKS, f"Required landmarks unavailable: {missing}."


def _quality_status(
    context: dict,
    required: tuple[str, ...],
    landmarks: dict[str, LandmarkInput],
) -> str:
    landmark_bits = [
        f"{name}:{landmarks[name].landmark_status if name in landmarks else 'MISSING'}"
        for name in required
    ]
    return f"frame:{context['frame_status']} | " + ",".join(landmark_bits)


def _feature_metadata(
    required: tuple[str, ...],
    landmarks: dict[str, LandmarkInput],
    normalisation: NormalisationReference,
) -> dict:
    return {
        "geometry_version": GEOMETRY_VERSION,
        "feature_set_version": FEATURE_SET_VERSION,
        "coordinate_source_by_landmark": {
            name: landmarks[name].coordinate_source if name in landmarks else "missing"
            for name in required
        },
        "processing_status_by_landmark": {
            name: landmarks[name].processing_status if name in landmarks else "MISSING"
            for name in required
        },
        "landmark_status_by_landmark": {
            name: landmarks[name].landmark_status if name in landmarks else "MISSING"
            for name in required
        },
        "normalisation": normalisation.to_metadata(),
    }


def _result_context(result: FeatureResult) -> dict:
    return {
        "case_id": result.case_id,
        "source_id": result.source_id,
        "frame_index": result.frame_index,
        "source_frame_index": result.source_frame_index,
        "analysis_frame_index": result.analysis_frame_index,
        "timestamp_ms": result.timestamp_ms,
        "frame_status": result.frame_status,
    }


def _combined_unavailable_status(a: FeatureResult, b: FeatureResult) -> str:
    for status in (
        FeatureStatus.INVALID_TARGET_FRAME.value,
        FeatureStatus.UNSUPPORTED_VIEW.value,
        FeatureStatus.INVALID_GEOMETRY.value,
        FeatureStatus.LOW_CONFIDENCE.value,
        FeatureStatus.INSUFFICIENT_LANDMARKS.value,
    ):
        if a.status == status or b.status == status:
            return status
    return FeatureStatus.INSUFFICIENT_LANDMARKS.value


def _combined_rejection_reason(a: FeatureResult, b: FeatureResult) -> str:
    reasons = [reason for reason in (a.rejection_reason, b.rejection_reason) if reason]
    return " | ".join(reasons) if reasons else "Required bilateral features unavailable."


def _midpoint_or_nan(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    value = midpoint(a, b)
    if value is None:
        return float("nan"), float("nan")
    return value


def _divide(value: float, scale: float) -> float:
    if not np.isfinite(value) or not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float(value / scale)


def _assert_scientific_names() -> None:
    names = _declared_feature_names()
    forbidden = [
        forbidden for forbidden in FORBIDDEN_CLINICAL_LABELS for name in names if forbidden in name
    ]
    if forbidden:
        raise ValueError(f"M3 feature names contain forbidden clinical labels: {forbidden}")


def _declared_feature_names() -> tuple[str, ...]:
    base = (
        "hka_angle_2d_deg",
        "knee_line_deviation_2d",
        "knee_line_deviation_normalized",
        "knee_ankle_x_offset_normalized",
        "knee_ankle_distance_normalized",
        "elbow_angle_2d_deg",
        "upper_arm_orientation_2d_deg",
        "wrist_pelvis_x_offset_normalized",
        "wrist_pelvis_distance_normalized",
    )
    side_names = tuple(f"{side}_{name}" for side in ("left", "right") for name in base)
    return (
        *side_names,
        "projected_trunk_axis_angle_deg",
        "projected_hip_line_angle_deg",
        "projected_shoulder_line_angle_deg",
        "projected_shoulder_pelvis_x_offset_px",
        "projected_shoulder_pelvis_x_offset_normalized",
        "projected_shoulder_pelvis_orientation_difference_deg",
        "injured_hka_angle_2d_deg",
        "contralateral_hka_angle_2d_deg",
        "hka_projected_bilateral_difference_deg",
        "hka_projected_bilateral_absolute_difference_deg",
        "injured_elbow_angle_2d_deg",
        "contralateral_elbow_angle_2d_deg",
        "elbow_projected_bilateral_difference_deg",
        "elbow_projected_bilateral_absolute_difference_deg",
        "knee_line_deviation_bilateral_difference",
        "knee_line_deviation_bilateral_absolute_difference",
        "knee_line_deviation_normalized_bilateral_difference",
        "knee_line_deviation_normalized_bilateral_absolute_difference",
    )
