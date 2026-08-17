"""Build semantic movement observations from existing human Movement Profiles."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from acl_motion.semantics.bilateral import BilateralHkaSummary, compute_bilateral_hka_summary
from acl_motion.semantics.models import MovementObservation, ObservationEvidenceStatus

SEMANTIC_CATEGORIES = (
    "movement_path",
    "hip_knee_ankle_chain",
    "hip_thigh",
    "trunk_pelvis",
    "upper_body",
    "bilateral_limb_relationship",
    "movement_timing",
    "evidence",
)


def build_movement_observations(
    *,
    case_id: str,
    dynamic_df: pd.DataFrame,
    case_summary: pd.DataFrame,
    path_summary: dict,
    bilateral_summary: BilateralHkaSummary | None = None,
) -> list[MovementObservation]:
    """Build deterministic semantic observations for one human case."""

    bilateral = bilateral_summary or compute_bilateral_hka_summary(dynamic_df)
    observations = [
        _movement_path_observation(case_id, path_summary),
        _speed_observation(case_id, path_summary),
        _feature_change_observation(
            case_id,
            dynamic_df,
            case_summary,
            category="hip_knee_ankle_chain",
            feature_name="injured_hka_angle_2d_deg",
            title="Injured-Side Knee Chain",
            label="projected hip-knee-ankle angle",
        ),
        _feature_change_observation(
            case_id,
            dynamic_df,
            case_summary,
            category="hip_knee_ankle_chain",
            feature_name="contralateral_hka_angle_2d_deg",
            title="Contralateral Knee Chain",
            label="projected hip-knee-ankle angle",
        ),
        _knee_ankle_observation(
            case_id,
            dynamic_df,
            case_summary,
            "right_knee_ankle_x_offset_normalized",
            "Injured Knee Relative to Foot",
        ),
        _knee_ankle_observation(
            case_id,
            dynamic_df,
            case_summary,
            "left_knee_ankle_x_offset_normalized",
            "Contralateral Knee Relative to Foot",
        ),
        _unavailable_observation(
            case_id,
            "hip_thigh",
            "Hip / Thigh Geometry",
            "Dedicated projected hip/thigh descriptors are not part of the current M3 feature set.",
            ("projected_thigh_orientation",),
        ),
        _feature_change_observation(
            case_id,
            dynamic_df,
            case_summary,
            category="trunk_pelvis",
            feature_name="projected_trunk_axis_angle_deg",
            title="Trunk Orientation",
            label="projected trunk-axis orientation",
        ),
        _feature_change_observation(
            case_id,
            dynamic_df,
            case_summary,
            category="trunk_pelvis",
            feature_name="projected_hip_line_angle_deg",
            title="Pelvic Orientation",
            label="projected hip-line orientation",
        ),
        _feature_change_observation(
            case_id,
            dynamic_df,
            case_summary,
            category="trunk_pelvis",
            feature_name="projected_shoulder_pelvis_orientation_difference_deg",
            title="Shoulder-Pelvis Relationship",
            label="projected shoulder-pelvis orientation difference",
        ),
        _feature_change_observation(
            case_id,
            dynamic_df,
            case_summary,
            category="upper_body",
            feature_name="right_elbow_angle_2d_deg",
            title="Injured-Side Elbow Configuration",
            label="projected elbow angle",
        ),
        _feature_change_observation(
            case_id,
            dynamic_df,
            case_summary,
            category="upper_body",
            feature_name="right_upper_arm_orientation_2d_deg",
            title="Injured-Side Arm Orientation",
            label="projected upper-arm orientation",
        ),
        _bilateral_relationship_observation(case_id, bilateral),
        _bilateral_window_observation(case_id, bilateral, "final_500ms"),
        _timing_observation(case_id, case_summary),
        _unavailable_observation(
            case_id,
            "evidence",
            "Foot / Ankle Orientation",
            (
                "Foot/ankle orientation requires heel, toe, or foot-index landmarks and is not "
                "available in the current YOLO COCO-17 analysis."
            ),
            ("ankle_angle", "foot_progression"),
        ),
    ]
    return observations


def observations_by_category(observations: list[MovementObservation]) -> dict[str, list[dict]]:
    """Group observations for the Results UI."""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for observation in observations:
        grouped[observation.category].append(observation.to_dict())
    return dict(grouped)


def write_observations_json(
    observations: list[MovementObservation],
    path: str | Path,
    *,
    metadata: dict | None = None,
) -> Path:
    """Write semantic observations to disk."""

    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata or {},
        "observations": [observation.to_dict() for observation in observations],
        "categories": observations_by_category(observations),
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return output


def _feature_change_observation(
    case_id: str,
    dynamic_df: pd.DataFrame,
    case_summary: pd.DataFrame,
    *,
    category: str,
    feature_name: str,
    title: str,
    label: str,
) -> MovementObservation:
    rows = _supported_rows(dynamic_df, feature_name)
    summary = _summary_row(case_summary, feature_name)
    if rows.empty or summary is None:
        return _unavailable_observation(
            case_id,
            category,
            title,
            f"{label} was unavailable under the current quality rules.",
            (feature_name,),
        )
    rows = rows.sort_values("movement_end_relative_ms")
    start = float(rows.iloc[0]["feature_value"])
    end = float(rows.iloc[-1]["feature_value"])
    change = end - start
    peak_row = rows.loc[rows["feature_value"].abs().idxmax()]
    evidence_status = _evidence_from_quality(str(summary["quality_category"]))
    unit = str(rows.iloc[0].get("unit", ""))
    return MovementObservation(
        observation_id=f"{category}.{feature_name}",
        case_id=case_id,
        category=category,
        title=title,
        plain_language_summary=(
            f"{title} changed by {change:.1f} {unit} across supported samples "
            f"of the human Movement Window."
        ),
        technical_feature_names=(feature_name,),
        unit=unit,
        start_value=start,
        end_value=end,
        change=float(change),
        peak_value=float(peak_row["feature_value"]),
        peak_source_frame=int(peak_row["source_frame_index"]),
        peak_movement_relative_ms=float(peak_row["movement_end_relative_ms"]),
        evidence_status=evidence_status,
        evidence_completeness=float(summary["geometry_completeness"]),
        quality_reasons=(_reason(summary),),
        source_frames=tuple(int(item) for item in rows["source_frame_index"]),
        technical_explanation=f"{label}; generic projected 2D descriptor.",
    )


def _knee_ankle_observation(
    case_id: str,
    dynamic_df: pd.DataFrame,
    case_summary: pd.DataFrame,
    feature_name: str,
    title: str,
) -> MovementObservation:
    observation = _feature_change_observation(
        case_id,
        dynamic_df,
        case_summary,
        category="hip_knee_ankle_chain",
        feature_name=feature_name,
        title=title,
        label="projected knee-ankle relationship",
    )
    return MovementObservation(
        **{
            **observation.to_dict(),
            "plain_language_summary": (
                observation.plain_language_summary
                + " The current distal reference is the ankle because YOLO COCO-17 does not "
                "provide heel/toe landmarks."
                if observation.evidence_status != ObservationEvidenceStatus.UNAVAILABLE
                else observation.plain_language_summary
            ),
            "technical_explanation": (
                "Knee Relative to Foot uses a projected knee-ankle relationship in V1; "
                "it is not a true foot-axis measurement."
            ),
        }
    )


def _movement_path_observation(case_id: str, path_summary: dict) -> MovementObservation:
    direction = path_summary.get("direction_change", {})
    if path_summary.get("overall_status") != "SUPPORTED" or direction.get("evidence_status") != "SUPPORTED":
        return _unavailable_observation(
            case_id,
            "movement_path",
            "Movement Direction",
            path_summary.get("reason")
            or direction.get("reason")
            or "Camera-compensated projected movement path was unavailable.",
            ("projected_movement_heading_deg",),
        )
    change = float(direction["projected_change_of_direction_angle_deg"])
    return MovementObservation(
        observation_id="movement_path.projected_direction_change",
        case_id=case_id,
        category="movement_path",
        title="Movement Direction",
        plain_language_summary=(
            f"Camera-compensated projected direction changed by {change:.1f} degrees "
            "across the supported body-center path."
        ),
        technical_feature_names=("projected_change_of_direction_angle_deg",),
        value=change,
        unit="deg",
        evidence_status="SUPPORTED",
        evidence_completeness=None,
        source_frames=(
            int(direction["source_frame_start"]),
            int(direction["source_frame_end"]),
        ),
        technical_explanation=direction["technical_explanation"],
        metadata=direction,
    )


def _speed_observation(case_id: str, path_summary: dict) -> MovementObservation:
    speed_change = path_summary.get("projected_speed_change_final_500ms", {})
    if path_summary.get("overall_status") != "SUPPORTED" or speed_change.get("evidence_status") != "SUPPORTED":
        return _unavailable_observation(
            case_id,
            "movement_path",
            "Projected Speed Pattern",
            speed_change.get("reason") or "Projected speed pattern was unavailable.",
            ("normalized_projected_movement_speed",),
        )
    change = float(speed_change["projected_speed_change"])
    return MovementObservation(
        observation_id="movement_path.projected_speed_pattern",
        case_id=case_id,
        category="movement_path",
        title="Projected Speed Pattern",
        plain_language_summary=(
            f"Final-500-ms body-scale-normalized projected speed changed by {change:.2f} "
            "projected body units per second relative to earlier supported samples."
        ),
        technical_feature_names=("normalized_projected_movement_speed",),
        value=change,
        unit="body-scale units/s",
        evidence_status="SUPPORTED",
        quality_reasons=("Projected speed is not true speed in m/s.",),
        technical_explanation="Camera-compensated projected body-center speed, normalized by body scale.",
        metadata=speed_change,
    )


def _bilateral_relationship_observation(
    case_id: str,
    summary: BilateralHkaSummary,
) -> MovementObservation:
    if summary.evidence_status != "SUPPORTED":
        return _unavailable_observation(
            case_id,
            "bilateral_limb_relationship",
            "Bilateral Knee-Chain Relationship",
            summary.pattern_explanation,
            ("hka_projected_bilateral_difference_deg",),
        )
    return MovementObservation(
        observation_id="bilateral_limb_relationship.projected_hka_relationship",
        case_id=case_id,
        category="bilateral_limb_relationship",
        title="Bilateral Knee-Chain Relationship",
        plain_language_summary=(
            f"Mean absolute injured-vs-contralateral projected HKA relationship was "
            f"{summary.mean_absolute_hka_bilateral_difference_deg:.1f} deg. Peak absolute "
            f"relationship was {summary.peak_absolute_hka_bilateral_difference_deg:.1f} deg "
            f"at {summary.time_peak_absolute_hka_bilateral_difference_ms:.1f} ms before Movement End."
        ),
        technical_feature_names=(
            "hka_projected_bilateral_difference_deg",
            "hka_projected_bilateral_absolute_difference_deg",
        ),
        value=summary.mean_absolute_hka_bilateral_difference_deg,
        unit="deg",
        start_value=summary.start_signed_difference_deg,
        end_value=summary.end_signed_difference_deg,
        change=summary.signed_change_deg,
        peak_value=summary.peak_absolute_hka_bilateral_difference_deg,
        peak_source_frame=summary.source_frame_peak_absolute_hka_bilateral_difference,
        peak_movement_relative_ms=summary.time_peak_absolute_hka_bilateral_difference_ms,
        evidence_status="SUPPORTED",
        evidence_completeness=None,
        quality_reasons=(summary.pattern_explanation,),
        source_frames=(summary.source_frame_peak_absolute_hka_bilateral_difference,),
        technical_explanation=(
            "Signed relationship is injured HKA minus contralateral HKA. Absolute relationship "
            "is descriptive magnitude, not a clinical score."
        ),
        metadata=summary.to_dict(),
    )


def _bilateral_window_observation(
    case_id: str,
    summary: BilateralHkaSummary,
    window_name: str,
) -> MovementObservation:
    window = next((item for item in summary.window_summaries if item.window_name == window_name), None)
    if window is None or window.evidence_status != "SUPPORTED":
        return _unavailable_observation(
            case_id,
            "bilateral_limb_relationship",
            "Final-Window Bilateral Relationship",
            "Insufficient supported samples in the requested Movement-End-relative window.",
            ("hka_projected_bilateral_difference_deg",),
        )
    return MovementObservation(
        observation_id=f"bilateral_limb_relationship.{window_name}",
        case_id=case_id,
        category="bilateral_limb_relationship",
        title="Final-Window Bilateral Relationship",
        plain_language_summary=(
            f"During the {window_name.replace('_', ' ')}, mean absolute projected HKA "
            f"relationship was {window.mean_absolute_difference_deg:.1f} deg."
        ),
        technical_feature_names=("hka_projected_bilateral_difference_deg",),
        value=window.mean_absolute_difference_deg,
        unit="deg",
        start_value=window.signed_start_deg,
        end_value=window.signed_end_deg,
        change=window.signed_change_deg,
        evidence_status="SUPPORTED",
        source_frames=(),
        technical_explanation="Final-window summary uses only supported Movement-End-relative samples.",
        metadata=window.to_dict(),
    )


def _timing_observation(case_id: str, case_summary: pd.DataFrame) -> MovementObservation:
    ready = case_summary[case_summary["peak_robust_dynamic_rate"].notna()].copy()
    if ready.empty:
        return _unavailable_observation(
            case_id,
            "movement_timing",
            "Timing of Largest Supported Changes",
            "No supported robust dynamic extrema were available.",
            (),
        )
    row = ready.loc[ready["peak_robust_dynamic_rate"].abs().idxmax()]
    return MovementObservation(
        observation_id="movement_timing.peak_supported_dynamic_change",
        case_id=case_id,
        category="movement_timing",
        title="Timing of Largest Supported Change",
        plain_language_summary=(
            f"The largest supported robust rate among current displayed descriptors occurred "
            f"{row['peak_robust_rate_event_relative_ms']:.1f} ms before Movement End."
        ),
        technical_feature_names=(str(row["feature_name"]),),
        value=float(row["peak_robust_dynamic_rate"]),
        unit="feature units/s",
        peak_value=float(row["peak_robust_dynamic_rate"]),
        peak_source_frame=int(row["peak_robust_rate_source_frame_index"]),
        peak_movement_relative_ms=float(row["peak_robust_rate_event_relative_ms"]),
        evidence_status=_evidence_from_quality(str(row["quality_category"])),
        evidence_completeness=float(row["dynamic_completeness"]),
        source_frames=(int(row["peak_robust_rate_source_frame_index"]),),
        technical_explanation="Robust local dynamic estimate from M4.1; not a clinical rate threshold.",
    )


def _unavailable_observation(
    case_id: str,
    category: str,
    title: str,
    reason: str,
    technical_feature_names: tuple[str, ...],
) -> MovementObservation:
    return MovementObservation(
        observation_id=f"{category}.{title.lower().replace(' ', '_').replace('/', '_')}",
        case_id=case_id,
        category=category,
        title=title,
        plain_language_summary="Unavailable under the current evidence rules.",
        technical_feature_names=technical_feature_names,
        evidence_status="UNAVAILABLE",
        quality_reasons=(reason,),
        technical_explanation=reason,
    )


def _supported_rows(dynamic_df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    return dynamic_df[
        dynamic_df["feature_name"].eq(feature_name)
        & dynamic_df["feature_status"].eq("SUPPORTED")
        & dynamic_df["feature_value"].notna()
    ].copy()


def _summary_row(case_summary: pd.DataFrame, feature_name: str):
    rows = case_summary[case_summary["feature_name"].eq(feature_name)]
    return None if rows.empty else rows.iloc[0]


def _evidence_from_quality(quality_category: str) -> ObservationEvidenceStatus:
    if quality_category == "SUPPORTED":
        return ObservationEvidenceStatus.SUPPORTED
    if quality_category == "LIMITED":
        return ObservationEvidenceStatus.LIMITED
    return ObservationEvidenceStatus.UNAVAILABLE


def _reason(summary_row) -> str:
    reason = summary_row.get("primary_rejection_reason")
    if pd.notna(reason) and str(reason):
        return str(reason)
    return str(summary_row.get("eligibility_reason", ""))
