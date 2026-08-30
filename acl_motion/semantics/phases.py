"""Within-case movement phase segmentation for the human Movement Window."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from acl_motion.geometry.angles import wrapped_angle_difference_deg

PHASE_SEGMENTATION_VERSION = "m5_10_case_smart_evidence_interval_v5"
SEGMENTATION_EXCLUDED_FEATURE_PATTERNS = {
    "knee_ankle_distance_normalized": (
        "diagnostic-only projected segment-length / foreshortening measure"
    ),
}


class PhaseEvidenceStatus(StrEnum):
    """Transparent phase-level evidence state."""

    GOOD = "GOOD"
    MODERATE = "MODERATE"
    LIMITED = "LIMITED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class PhaseSegmentationConfig:
    """Configurable quality and segmentation parameters for movement phases."""

    minimum_geometry_completeness: float = 0.70
    minimum_supported_fraction: float = 0.55
    minimum_dynamic_supported_fraction: float = 0.40
    minimum_continuous_supported_ms: float = 300.0
    minimum_eligible_descriptors: int = 4
    enable_partial_window_segmentation: bool = True
    minimum_evidence_interval_duration_ms: float = 300.0
    minimum_partial_window_duration_ms: float = 500.0
    minimum_partial_window_fraction: float = 0.15
    minimum_phase_duration_ms: float = 300.0
    minimum_boundary_separation_ms: float = 300.0
    smoothing_ms: float = 166.0
    boundary_threshold_mad_multiplier: float = 0.75
    minimum_boundary_score: float = 0.20
    minimum_sustained_shift: float = 0.35
    max_user_phases: int = 6
    enable_hierarchical_refinement: bool = True
    refinement_min_duration_ms: float = 1200.0
    refinement_max_depth: int = 2
    refinement_min_child_phases: int = 2
    max_refined_phases: int = 6
    epsilon: float = 1e-9

    def to_dict(self) -> dict:
        """Return a JSON-ready config record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class MovementPhase:
    """A contiguous interval inside one human-defined Movement Window."""

    phase_id: str
    case_id: str
    source_id: str
    phase_index: int
    start_frame: int
    end_frame: int
    start_timestamp_ms: float | None
    end_timestamp_ms: float | None
    start_relative_ms: float | None
    end_relative_ms: float | None
    duration_ms: float
    title: str
    segmentation_features: tuple[str, ...]
    change_score_summary: dict[str, Any]
    category_summaries: dict[str, dict]
    phase_observations: tuple[str, ...]
    evidence_summary: dict[str, Any]
    notable_extrema: tuple[dict, ...]
    source_frames: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        forbidden = (
            "abnormal",
            "pathological",
            "dangerous",
            "risky",
            "high-risk",
            "critical plant",
            "acl-causing",
            "injury transition",
            "injury phase",
        )
        text = " ".join(
            [
                self.title,
                " ".join(self.phase_observations),
                " ".join(str(item.get("summary", "")) for item in self.category_summaries.values()),
            ]
        ).lower()
        if any(term in text for term in forbidden):
            raise ValueError("MovementPhase language must remain neutral and non-clinical.")

    def to_dict(self) -> dict:
        """Return a JSON-ready movement phase."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PhaseSegmentationResult:
    """Full output of within-case phase segmentation."""

    case_id: str
    source_id: str
    status: str
    sequence_summary: str
    phases: tuple[MovementPhase, ...]
    transitions: tuple[dict, ...]
    eligible_descriptors: tuple[dict, ...]
    excluded_descriptors: tuple[dict, ...]
    change_signal: pd.DataFrame
    frame_map: pd.DataFrame
    metadata: dict[str, Any]

    def to_json_dict(self) -> dict:
        """Return the serializable phase-story payload."""

        presentation_mode = str(self.metadata.get("presentation_mode") or "")
        evidence_interval = presentation_mode == "SUPPORTED_EVIDENCE_INTERVAL"
        return {
            "metadata": self.metadata,
            "case_id": self.case_id,
            "source_id": self.source_id,
            "status": self.status,
            "presentation_mode": presentation_mode or None,
            "sequence_summary": self.sequence_summary,
            "phase_count": 0 if evidence_interval else len(self.phases),
            "internal_segment_count": len(self.phases),
            "supported_evidence_interval": (
                _supported_evidence_interval_payload(self.phases[0])
                if evidence_interval and self.phases
                else None
            ),
            "phases": [phase.to_dict() for phase in self.phases],
            "transitions": list(self.transitions),
            "eligible_descriptors": list(self.eligible_descriptors),
            "excluded_descriptors": list(self.excluded_descriptors),
        }


def _supported_evidence_interval_payload(segment: MovementPhase) -> dict:
    payload = segment.to_dict()
    for key in ("phase_id", "phase_index", "title"):
        payload.pop(key, None)
    payload["interval_id"] = "supported_evidence_interval"
    return payload


def segment_movement_phases(
    *,
    case_id: str,
    source_id: str,
    dynamic_df: pd.DataFrame,
    case_summary: pd.DataFrame,
    path_df: pd.DataFrame,
    movement_window: Any,
    config: PhaseSegmentationConfig | None = None,
) -> PhaseSegmentationResult:
    """Segment the human Movement Window into data-supported movement phases.

    The algorithm uses only supported projected geometry, supported robust
    dynamic rates, and camera-compensated path descriptors. Unsupported values
    are left missing; they are never converted to zero movement.
    """

    cfg = config or PhaseSegmentationConfig()
    start_frame, end_frame, duration_ms = _movement_window_bounds(movement_window, dynamic_df)
    frame_timing = _frame_timing(dynamic_df, start_frame, end_frame)
    fps = _fps_from_timing(frame_timing)
    min_phase_frames = max(2, int(np.ceil(cfg.minimum_phase_duration_ms / 1000.0 * fps)))
    min_separation_frames = max(2, int(np.ceil(cfg.minimum_boundary_separation_ms / 1000.0 * fps)))
    min_continuous_frames = max(
        2,
        int(np.ceil(cfg.minimum_continuous_supported_ms / 1000.0 * fps)),
    )
    descriptors, excluded = _build_descriptor_table(
        dynamic_df,
        case_summary,
        path_df,
        frame_timing,
        cfg,
        min_continuous_frames=min_continuous_frames,
    )
    if len(descriptors.metadata) < cfg.minimum_eligible_descriptors:
        partial_result = _partial_window_result(
            case_id=case_id,
            source_id=source_id,
            dynamic_df=dynamic_df,
            case_summary=case_summary,
            path_df=path_df,
            frame_timing=frame_timing,
            full_window=(start_frame, end_frame, duration_ms),
            cfg=cfg,
            global_excluded=excluded,
        )
        if partial_result is not None:
            return partial_result
        frame_map = frame_timing.copy()
        frame_map["phase_id"] = None
        frame_map["phase_index"] = None
        frame_map["phase_title"] = "Phase segmentation unavailable"
        frame_map["change_score"] = np.nan
        metadata = _metadata(cfg, fps, start_frame, end_frame, duration_ms)
        metadata["presentation_mode"] = "PHASES_WITHHELD"
        metadata["phase_decision_rationale"] = _phase_decision_rationale(
            "INSUFFICIENT_CONTINUOUS_MULTIVARIATE_EVIDENCE"
        )
        return PhaseSegmentationResult(
            case_id=case_id,
            source_id=source_id,
            status="INSUFFICIENT_EVIDENCE_FOR_PHASE_SEGMENTATION",
            sequence_summary=(
                "Phase segmentation is unavailable because too few supported movement "
                "descriptors met the configured evidence rules."
            ),
            phases=(),
            transitions=(),
            eligible_descriptors=tuple(descriptors.metadata),
            excluded_descriptors=tuple(excluded),
            change_signal=frame_map.copy(),
            frame_map=frame_map,
            metadata=metadata,
        )

    standardized = _standardize_descriptors(descriptors.values, descriptors.metadata, cfg)
    change_signal = _change_signal(
        frame_timing,
        standardized,
        descriptors.metadata,
        cfg,
        smoothing_window_frames=_odd_window(cfg.smoothing_ms, fps),
    )
    boundaries, transitions = _select_boundaries(
        change_signal,
        standardized,
        descriptors.metadata,
        cfg,
        min_phase_frames=min_phase_frames,
        min_separation_frames=min_separation_frames,
    )
    phases = _build_phases(
        case_id=case_id,
        source_id=source_id,
        dynamic_df=dynamic_df,
        path_df=path_df,
        frame_timing=frame_timing,
        change_signal=change_signal,
        descriptors=descriptors.metadata,
        boundaries=boundaries,
    )
    initial_phases = tuple(phases)
    initial_transitions = tuple(_transition_payload(item, phases) for item in transitions)
    refinement_records: tuple[dict, ...] = ()
    if cfg.enable_hierarchical_refinement and phases:
        phases, transitions, refinement_records = _refine_phases_hierarchically(
            case_id=case_id,
            source_id=source_id,
            dynamic_df=dynamic_df,
            case_summary=case_summary,
            path_df=path_df,
            frame_timing=frame_timing,
            global_change_signal=change_signal,
            descriptors=descriptors.metadata,
            initial_phases=initial_phases,
            initial_transitions=initial_transitions,
            cfg=cfg,
        )
    else:
        transitions = initial_transitions
    frame_map = _phase_frame_map(frame_timing, phases, change_signal)
    is_phase_sequence = len(phases) >= 2 and bool(transitions)
    sequence_summary = (
        _sequence_summary(duration_ms, phases, transitions)
        if is_phase_sequence
        else _evidence_interval_summary(duration_ms, phases[0])
    )
    metadata = _metadata(cfg, fps, start_frame, end_frame, duration_ms)
    metadata["presentation_mode"] = (
        "PHASE_SEQUENCE" if is_phase_sequence else "SUPPORTED_EVIDENCE_INTERVAL"
    )
    metadata["phase_decision_rationale"] = _phase_decision_rationale(
        "SUPPORTED_MULTIVARIATE_TRANSITION"
        if is_phase_sequence
        else "NO_SUPPORTED_MULTIVARIATE_TRANSITION"
    )
    if not is_phase_sequence:
        metadata["analysis_scope"] = {
            "type": "FULL_MOVEMENT_WINDOW",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": len(frame_timing),
            "duration_ms": duration_ms,
            "movement_window_fraction": 1.0,
            "position_in_movement_window": "FULL_WINDOW",
            "includes_annotated_movement_end": True,
            "selection_rule": (
                "The complete Movement Window met measurement-support rules, but no "
                "sustained multivariate transition met the phase-boundary rule."
            ),
        }
    metadata["refinement"] = {
        "enabled": bool(cfg.enable_hierarchical_refinement),
        "original_phase_count": len(initial_phases),
        "refined_phase_count": len(phases),
        "refined": len(phases) != len(initial_phases),
        "records": list(refinement_records),
    }
    if not is_phase_sequence:
        supported = frame_map["phase_id"].notna()
        frame_map.loc[supported, "phase_index"] = np.nan
        frame_map.loc[supported, "phase_title"] = "Supported Evidence Interval"
    return PhaseSegmentationResult(
        case_id=case_id,
        source_id=source_id,
        status=("SUPPORTED" if is_phase_sequence else "SUPPORTED_EVIDENCE_INTERVAL"),
        sequence_summary=sequence_summary,
        phases=tuple(phases),
        transitions=transitions,
        eligible_descriptors=tuple(descriptors.metadata),
        excluded_descriptors=tuple(excluded),
        change_signal=change_signal,
        frame_map=frame_map,
        metadata=metadata,
    )


def _partial_window_result(
    *,
    case_id: str,
    source_id: str,
    dynamic_df: pd.DataFrame,
    case_summary: pd.DataFrame,
    path_df: pd.DataFrame,
    frame_timing: pd.DataFrame,
    full_window: tuple[int, int, float],
    cfg: PhaseSegmentationConfig,
    global_excluded: tuple[dict, ...],
) -> PhaseSegmentationResult | None:
    """Return a labelled partial-window result for one strong continuous block.

    This path is attempted only after the complete Movement Window fails the
    descriptor rule. It never fills gaps and it never changes the complete-window
    coverage recorded in the Movement Profile.
    """

    if not cfg.enable_partial_window_segmentation or frame_timing.empty:
        return None
    candidate = _partial_window_candidate(dynamic_df, frame_timing, cfg)
    if candidate is None:
        return None
    local_summary = _case_summary_for_window(
        case_summary,
        dynamic_df,
        candidate["start_frame"],
        candidate["end_frame"],
    )
    local_cfg = replace(
        cfg,
        enable_partial_window_segmentation=False,
    )
    local_result = segment_movement_phases(
        case_id=case_id,
        source_id=source_id,
        dynamic_df=dynamic_df,
        case_summary=local_summary,
        path_df=path_df,
        movement_window={
            "movement_start_frame": candidate["start_frame"],
            "movement_end_frame": candidate["end_frame"],
            "duration_ms": candidate["duration_ms"],
        },
        config=local_cfg,
    )
    if local_result.status not in {"SUPPORTED", "SUPPORTED_EVIDENCE_INTERVAL"} or not local_result.phases:
        return None
    local_is_phase_sequence = (
        len(local_result.phases) >= 2 and bool(local_result.transitions)
    )
    if (
        local_is_phase_sequence
        and candidate["duration_ms"] < cfg.minimum_partial_window_duration_ms
    ):
        return None

    phases = tuple(
        replace(
            phase,
            metadata={
                **phase.metadata,
                "analysis_scope": "PARTIAL_MOVEMENT_WINDOW",
                "full_movement_window_start_frame": full_window[0],
                "full_movement_window_end_frame": full_window[1],
            },
        )
        for phase in local_result.phases
    )
    eligible = tuple(
        {
            **item,
            "analysis_scope": "PARTIAL_MOVEMENT_WINDOW",
            "partial_window_start_frame": candidate["start_frame"],
            "partial_window_end_frame": candidate["end_frame"],
        }
        for item in local_result.eligible_descriptors
    )
    full_change = _expand_partial_table(
        frame_timing,
        local_result.change_signal,
    )
    full_map = _phase_frame_map(frame_timing, list(phases), full_change)
    outside = full_map["phase_id"].isna()
    full_map.loc[outside, "phase_title"] = "Outside supported partial evidence block"
    metadata = dict(local_result.metadata)
    metadata["movement_window"] = {
        "movement_start_frame": full_window[0],
        "movement_end_frame": full_window[1],
        "duration_ms": full_window[2],
    }
    metadata["configuration"] = cfg.to_dict()
    metadata["analysis_scope"] = {
        "type": "PARTIAL_MOVEMENT_WINDOW",
        "start_frame": candidate["start_frame"],
        "end_frame": candidate["end_frame"],
        "frame_count": candidate["frame_count"],
        "duration_ms": candidate["duration_ms"],
        "movement_window_fraction": candidate["movement_window_fraction"],
        "position_in_movement_window": _scope_position(
            candidate["start_frame"],
            candidate["end_frame"],
            full_window[0],
            full_window[1],
        ),
        "includes_annotated_movement_end": (
            candidate["end_frame"] >= full_window[1]
        ),
        "minimum_supported_descriptors_per_frame": candidate[
            "minimum_supported_descriptors_per_frame"
        ],
        "minimum_evidence_interval_duration_ms": (
            cfg.minimum_evidence_interval_duration_ms
        ),
        "minimum_phase_sequence_window_duration_ms": (
            cfg.minimum_partial_window_duration_ms
        ),
        "selection_rule": (
            "Longest continuous block meeting the configured descriptor-count, "
            "duration, and Movement Window fraction safeguards."
        ),
        "outside_scope_rule": (
            "Frames outside this block remain unsegmented and are not silently filled."
        ),
    }
    is_phase_sequence = len(phases) >= 2 and bool(local_result.transitions)
    metadata["presentation_mode"] = (
        "PHASE_SEQUENCE" if is_phase_sequence else "SUPPORTED_EVIDENCE_INTERVAL"
    )
    metadata["phase_decision_rationale"] = _phase_decision_rationale(
        "SUPPORTED_MULTIVARIATE_TRANSITION"
        if is_phase_sequence
        else "NO_SUPPORTED_MULTIVARIATE_TRANSITION"
    )
    if not is_phase_sequence:
        supported = full_map["phase_id"].notna()
        full_map.loc[supported, "phase_index"] = np.nan
        full_map.loc[supported, "phase_title"] = "Supported Evidence Interval"
    if is_phase_sequence:
        summary = (
            f"Partial-window phase analysis covers source frames {candidate['start_frame']} to "
            f"{candidate['end_frame']} ({candidate['duration_ms'] / 1000.0:.2f} seconds; "
            f"{candidate['movement_window_fraction'] * 100.0:.1f}% of the human Movement "
            "Window). Evidence outside this continuous block remains unsegmented. "
            f"{local_result.sequence_summary}"
        )
    else:
        summary = (
            f"A supported evidence interval covers source frames {candidate['start_frame']} to "
            f"{candidate['end_frame']} ({candidate['duration_ms'] / 1000.0:.2f} seconds; "
            f"{candidate['movement_window_fraction'] * 100.0:.1f}% of the human Movement "
            "Window). No sustained multivariate transition met the phase-boundary rule, "
            "so this interval is not presented as a phase sequence. Evidence outside this "
            "continuous block remains unsegmented. The interval does not establish injury timing."
        )
    return PhaseSegmentationResult(
        case_id=case_id,
        source_id=source_id,
        status=(
            "SUPPORTED_PARTIAL_WINDOW"
            if is_phase_sequence
            else "SUPPORTED_EVIDENCE_INTERVAL"
        ),
        sequence_summary=summary,
        phases=phases,
        transitions=local_result.transitions,
        eligible_descriptors=eligible,
        excluded_descriptors=global_excluded,
        change_signal=full_change,
        frame_map=full_map,
        metadata=metadata,
    )


def _partial_window_candidate(
    dynamic_df: pd.DataFrame,
    frame_timing: pd.DataFrame,
    cfg: PhaseSegmentationConfig,
) -> dict[str, Any] | None:
    feature_rows = dynamic_df[
        dynamic_df["feature_status"].eq("SUPPORTED")
        & dynamic_df["feature_value"].notna()
        & ~dynamic_df["feature_name"].astype(str).map(
            lambda name: bool(_segmentation_exclusion_reason(name))
        )
    ]
    supported_counts = feature_rows.groupby("source_frame_index")["feature_name"].nunique()
    timing = frame_timing[["source_frame_index", "timestamp_ms"]].copy()
    timing["supported_descriptor_count"] = (
        timing["source_frame_index"].map(supported_counts).fillna(0).astype(int)
    )
    timing["candidate"] = timing["supported_descriptor_count"].ge(
        cfg.minimum_eligible_descriptors
    )
    runs: list[pd.DataFrame] = []
    current_indices: list[int] = []
    previous_frame: int | None = None
    for index, row in timing.iterrows():
        frame = int(row["source_frame_index"])
        if bool(row["candidate"]) and (
            previous_frame is None or frame == previous_frame + 1
        ):
            current_indices.append(index)
        elif bool(row["candidate"]):
            if current_indices:
                runs.append(timing.loc[current_indices])
            current_indices = [index]
        else:
            if current_indices:
                runs.append(timing.loc[current_indices])
            current_indices = []
        previous_frame = frame
    if current_indices:
        runs.append(timing.loc[current_indices])
    if not runs:
        return None

    candidates = []
    for run in runs:
        frame_count = len(run)
        duration_ms = float(run["timestamp_ms"].iloc[-1] - run["timestamp_ms"].iloc[0])
        fraction = frame_count / len(frame_timing)
        if duration_ms < cfg.minimum_evidence_interval_duration_ms:
            continue
        if fraction < cfg.minimum_partial_window_fraction:
            continue
        candidates.append(
            {
                "start_frame": int(run["source_frame_index"].iloc[0]),
                "end_frame": int(run["source_frame_index"].iloc[-1]),
                "frame_count": frame_count,
                "duration_ms": duration_ms,
                "movement_window_fraction": float(fraction),
                "minimum_supported_descriptors_per_frame": int(
                    run["supported_descriptor_count"].min()
                ),
            }
        )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item["duration_ms"],
            item["frame_count"],
            -item["start_frame"],
        ),
    )


def _scope_position(
    start_frame: int,
    end_frame: int,
    movement_start_frame: int,
    movement_end_frame: int,
) -> str:
    if start_frame <= movement_start_frame and end_frame >= movement_end_frame:
        return "FULL_WINDOW"
    span = max(1, movement_end_frame - movement_start_frame)
    midpoint_fraction = (
        ((start_frame + end_frame) / 2.0) - movement_start_frame
    ) / span
    if midpoint_fraction < 1.0 / 3.0:
        return "EARLY"
    if midpoint_fraction < 2.0 / 3.0:
        return "MIDDLE"
    return "LATE"


def _case_summary_for_window(
    case_summary: pd.DataFrame,
    dynamic_df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
) -> pd.DataFrame:
    local = case_summary.copy()
    window_rows = dynamic_df[
        dynamic_df["source_frame_index"].astype(int).between(
            start_frame, end_frame, inclusive="both"
        )
    ]
    for index, row in local.iterrows():
        feature_rows = window_rows[
            window_rows["feature_name"].eq(row["feature_name"])
        ]
        if feature_rows.empty:
            local.loc[index, "geometry_completeness"] = 0.0
            local.loc[index, "dynamic_completeness"] = 0.0
            local.loc[index, "quality_category"] = "UNAVAILABLE"
            continue
        geometry_supported = feature_rows["feature_status"].eq("SUPPORTED")
        dynamic_eligible = geometry_supported & feature_rows["dynamic_status"].ne(
            "NOT_DYNAMIC_FEATURE"
        )
        robust_supported = dynamic_eligible & feature_rows["dynamic_status"].eq(
            "SUPPORTED"
        )
        local.loc[index, "geometry_completeness"] = float(geometry_supported.mean())
        local.loc[index, "dynamic_completeness"] = (
            float(robust_supported.sum() / dynamic_eligible.sum())
            if dynamic_eligible.any()
            else 0.0
        )
        local.loc[index, "quality_category"] = (
            "SUPPORTED" if geometry_supported.any() else "UNAVAILABLE"
        )
    return local


def _expand_partial_table(
    frame_timing: pd.DataFrame,
    partial: pd.DataFrame,
) -> pd.DataFrame:
    timing_columns = set(frame_timing.columns)
    extra_columns = [
        column
        for column in partial.columns
        if column not in timing_columns and column != "source_frame_index"
    ]
    return frame_timing.merge(
        partial[["source_frame_index", *extra_columns]],
        on="source_frame_index",
        how="left",
    )


def write_phase_json(result: PhaseSegmentationResult, path: str | Path) -> Path:
    """Write phase-story JSON to disk."""

    import json

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_ready(result.to_json_dict()), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


@dataclass(frozen=True, slots=True)
class _DescriptorTable:
    values: pd.DataFrame
    metadata: tuple[dict, ...]


def _movement_window_bounds(movement_window: Any, dynamic_df: pd.DataFrame) -> tuple[int, int, float]:
    if hasattr(movement_window, "movement_start_frame"):
        start_frame = int(movement_window.movement_start_frame)
        end_frame = int(movement_window.movement_end_frame)
        duration_ms = float(getattr(movement_window, "duration_ms", np.nan))
    else:
        start_frame = int(movement_window["movement_start_frame"])
        end_frame = int(movement_window["movement_end_frame"])
        duration_ms = float(movement_window.get("duration_ms", np.nan))
    if not np.isfinite(duration_ms):
        timing = _frame_timing(dynamic_df, start_frame, end_frame)
        duration_ms = float(timing["timestamp_ms"].max() - timing["timestamp_ms"].min())
    return start_frame, end_frame, duration_ms


def _frame_timing(dynamic_df: pd.DataFrame, start_frame: int, end_frame: int) -> pd.DataFrame:
    columns = [
        "source_frame_index",
        "analysis_frame_index",
        "timestamp_ms",
        "movement_elapsed_ms",
        "movement_end_relative_ms",
    ]
    timing = (
        dynamic_df[columns]
        .drop_duplicates("source_frame_index")
        .sort_values("source_frame_index")
        .reset_index(drop=True)
    )
    timing = timing[
        timing["source_frame_index"].astype(int).between(start_frame, end_frame, inclusive="both")
    ].copy()
    return timing.reset_index(drop=True)


def _fps_from_timing(frame_timing: pd.DataFrame) -> float:
    deltas = frame_timing["timestamp_ms"].astype(float).diff().dropna()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return 30.0
    return float(1000.0 / deltas.median())


def _build_descriptor_table(
    dynamic_df: pd.DataFrame,
    case_summary: pd.DataFrame,
    path_df: pd.DataFrame,
    frame_timing: pd.DataFrame,
    cfg: PhaseSegmentationConfig,
    *,
    min_continuous_frames: int,
) -> tuple[_DescriptorTable, tuple[dict, ...]]:
    values = frame_timing[["source_frame_index", "movement_end_relative_ms"]].copy()
    metadata: list[dict] = []
    excluded: list[dict] = []
    descriptor_columns: list[str] = []
    for descriptor_name, source, family, unit, series in _path_descriptors(path_df, frame_timing):
        record = _descriptor_record(
            descriptor_name,
            source=source,
            family=family,
            unit=unit,
            series=series,
            frame_count=len(frame_timing),
            min_continuous_frames=min_continuous_frames,
            cfg=cfg,
        )
        if record["eligible"]:
            values[descriptor_name] = series.to_numpy(dtype=float)
            descriptor_columns.append(descriptor_name)
            metadata.append(record["metadata"])
        else:
            excluded.append(record["metadata"])

    summary_by_feature = case_summary.set_index("feature_name")
    for feature_name in sorted(dynamic_df["feature_name"].dropna().unique()):
        if feature_name not in summary_by_feature.index:
            continue
        diagnostic_reason = _segmentation_exclusion_reason(feature_name)
        if diagnostic_reason:
            for kind in ("geometry", "robust_dynamic_rate"):
                excluded.append(
                    {
                        "feature_name": feature_name,
                        "family": _feature_family(
                            feature_name,
                            str(summary_by_feature.loc[feature_name].get("body_region", "")),
                        ),
                        "kind": kind,
                        "unit": _summary_unit(feature_name, kind),
                        "body_region": str(summary_by_feature.loc[feature_name].get("body_region", "")),
                        "quality_category": str(
                            summary_by_feature.loc[feature_name].get("quality_category", "")
                        ),
                        "geometry_completeness": float(
                            summary_by_feature.loc[feature_name].get("geometry_completeness", 0.0)
                        ),
                        "dynamic_completeness": float(
                            summary_by_feature.loc[feature_name].get("dynamic_completeness", 0.0)
                        ),
                        "supported_fraction": 0.0,
                        "max_continuous_supported_frames": 0,
                        "robust_scale": 0.0,
                        "exclusion_reason": diagnostic_reason,
                        "descriptor_name": f"{kind}:{feature_name}",
                    }
                )
            continue
        summary = summary_by_feature.loc[feature_name]
        feature_rows = dynamic_df[
            dynamic_df["feature_name"].eq(feature_name)
            & dynamic_df["source_frame_index"].isin(frame_timing["source_frame_index"])
        ]
        geometry_series = _feature_value_series(feature_rows, frame_timing)
        geometry_record = _feature_descriptor_record(
            feature_name,
            kind="geometry",
            summary=summary,
            series=geometry_series,
            frame_count=len(frame_timing),
            min_continuous_frames=min_continuous_frames,
            cfg=cfg,
        )
        if geometry_record["eligible"]:
            column = f"geometry:{feature_name}"
            values[column] = geometry_series.to_numpy(dtype=float)
            descriptor_columns.append(column)
            metadata.append({**geometry_record["metadata"], "descriptor_name": column})
        else:
            excluded.append({**geometry_record["metadata"], "descriptor_name": f"geometry:{feature_name}"})

        rate_series = _dynamic_rate_series(feature_rows, frame_timing)
        dynamic_record = _feature_descriptor_record(
            feature_name,
            kind="robust_dynamic_rate",
            summary=summary,
            series=rate_series,
            frame_count=len(frame_timing),
            min_continuous_frames=min_continuous_frames,
            cfg=cfg,
        )
        if dynamic_record["eligible"]:
            column = f"dynamic_rate:{feature_name}"
            values[column] = rate_series.to_numpy(dtype=float)
            descriptor_columns.append(column)
            metadata.append({**dynamic_record["metadata"], "descriptor_name": column})
        else:
            excluded.append(
                {**dynamic_record["metadata"], "descriptor_name": f"dynamic_rate:{feature_name}"}
            )
    return _DescriptorTable(values=values[["source_frame_index", *descriptor_columns]], metadata=tuple(metadata)), tuple(excluded)


def _path_descriptors(
    path_df: pd.DataFrame,
    frame_timing: pd.DataFrame,
) -> list[tuple[str, str, str, str, pd.Series]]:
    path = path_df[path_df["source_frame_index"].isin(frame_timing["source_frame_index"])].copy()
    path = path.set_index("source_frame_index")
    index = frame_timing["source_frame_index"].astype(int)
    status = path.reindex(index)["path_status"].eq("SUPPORTED")
    descriptors = []
    for column, unit in (
        ("projected_heading_deg", "deg"),
        ("normalized_projected_speed_per_s", "body-scale units/s"),
    ):
        series = path.reindex(index)[column].where(status).reset_index(drop=True)
        descriptors.append(
            (
                f"path:{column}",
                column,
                "movement_path",
                unit,
                pd.to_numeric(series, errors="coerce"),
            )
        )
    return descriptors


def _feature_value_series(feature_rows: pd.DataFrame, frame_timing: pd.DataFrame) -> pd.Series:
    supported = feature_rows[
        feature_rows["feature_status"].eq("SUPPORTED") & feature_rows["feature_value"].notna()
    ]
    series = supported.set_index("source_frame_index")["feature_value"]
    return pd.to_numeric(
        series.reindex(frame_timing["source_frame_index"].astype(int)).reset_index(drop=True),
        errors="coerce",
    )


def _dynamic_rate_series(feature_rows: pd.DataFrame, frame_timing: pd.DataFrame) -> pd.Series:
    supported = feature_rows[
        feature_rows["dynamic_status"].eq("SUPPORTED") & feature_rows["robust_dynamic_rate"].notna()
    ]
    series = supported.set_index("source_frame_index")["robust_dynamic_rate"]
    return pd.to_numeric(
        series.reindex(frame_timing["source_frame_index"].astype(int)).reset_index(drop=True),
        errors="coerce",
    )


def _descriptor_record(
    descriptor_name: str,
    *,
    source: str,
    family: str,
    unit: str,
    series: pd.Series,
    frame_count: int,
    min_continuous_frames: int,
    cfg: PhaseSegmentationConfig,
) -> dict:
    supported_fraction = float(series.notna().mean()) if frame_count else 0.0
    max_run = _max_finite_run(series)
    variance = _robust_scale(series, cfg.epsilon)
    eligible = (
        supported_fraction >= cfg.minimum_supported_fraction
        and max_run >= min_continuous_frames
        and variance > cfg.epsilon
    )
    reason = ""
    if not eligible:
        reason = _exclusion_reason(
            supported_fraction,
            max_run,
            variance,
            cfg.minimum_supported_fraction,
            min_continuous_frames,
            cfg.epsilon,
        )
    return {
        "eligible": eligible,
        "metadata": {
            "descriptor_name": descriptor_name,
            "source": source,
            "feature_name": source,
            "family": family,
            "kind": "movement_path",
            "unit": unit,
            "supported_fraction": supported_fraction,
            "max_continuous_supported_frames": int(max_run),
            "robust_scale": float(variance),
            "exclusion_reason": reason,
        },
    }


def _feature_descriptor_record(
    feature_name: str,
    *,
    kind: str,
    summary: pd.Series,
    series: pd.Series,
    frame_count: int,
    min_continuous_frames: int,
    cfg: PhaseSegmentationConfig,
) -> dict:
    supported_fraction = float(series.notna().mean()) if frame_count else 0.0
    max_run = _max_finite_run(series)
    variance = _robust_scale(series, cfg.epsilon)
    geometry_completeness = float(summary.get("geometry_completeness", 0.0))
    dynamic_completeness = float(summary.get("dynamic_completeness", 0.0))
    quality_category = str(summary.get("quality_category", ""))
    unit = _summary_unit(feature_name, kind)
    minimum_fraction = (
        cfg.minimum_dynamic_supported_fraction
        if kind == "robust_dynamic_rate"
        else cfg.minimum_supported_fraction
    )
    eligible = (
        geometry_completeness >= cfg.minimum_geometry_completeness
        and quality_category != "UNAVAILABLE"
        and supported_fraction >= minimum_fraction
        and max_run >= min_continuous_frames
        and variance > cfg.epsilon
    )
    if kind == "robust_dynamic_rate":
        eligible = eligible and dynamic_completeness >= cfg.minimum_dynamic_supported_fraction
    reason = ""
    if not eligible:
        reason = _feature_exclusion_reason(
            geometry_completeness,
            supported_fraction,
            dynamic_completeness,
            max_run,
            variance,
            minimum_fraction,
            min_continuous_frames,
            cfg,
            kind,
        )
    return {
        "eligible": eligible,
        "metadata": {
            "feature_name": feature_name,
            "family": _feature_family(feature_name, str(summary.get("body_region", ""))),
            "kind": kind,
            "unit": unit,
            "body_region": str(summary.get("body_region", "")),
            "quality_category": quality_category,
            "geometry_completeness": geometry_completeness,
            "dynamic_completeness": dynamic_completeness,
            "supported_fraction": supported_fraction,
            "max_continuous_supported_frames": int(max_run),
            "robust_scale": float(variance),
            "exclusion_reason": reason,
        },
    }


def _summary_unit(feature_name: str, kind: str) -> str:
    if kind == "robust_dynamic_rate":
        return "feature units/s"
    if feature_name.endswith("_deg"):
        return "deg"
    if feature_name.endswith("_normalized"):
        return "body-scale units"
    if feature_name.endswith("_px"):
        return "px"
    return ""


def _feature_family(feature_name: str, body_region: str) -> str:
    if (
        ("bilateral" in feature_name or "injured_" in feature_name or "contralateral_" in feature_name)
        and ("hka" in feature_name or "knee" in feature_name)
    ):
        return "bilateral_limb_relationship"
    if any(token in feature_name for token in ("hka", "knee_ankle", "knee_line")):
        return "hip_knee_ankle_chain"
    if any(token in feature_name for token in ("trunk", "hip_line", "shoulder_line", "pelvis")):
        return "trunk_pelvis"
    if any(token in feature_name for token in ("elbow", "upper_arm", "wrist")):
        return "upper_body"
    return body_region or "other"


def _segmentation_exclusion_reason(feature_name: str) -> str:
    for pattern, reason in SEGMENTATION_EXCLUDED_FEATURE_PATTERNS.items():
        if pattern in feature_name:
            return reason
    return ""


def _exclusion_reason(
    supported_fraction: float,
    max_run: int,
    variance: float,
    minimum_fraction: float,
    min_continuous_frames: int,
    epsilon: float,
) -> str:
    if supported_fraction < minimum_fraction:
        return "supported fraction below segmentation requirement"
    if max_run < min_continuous_frames:
        return "continuous supported interval below segmentation requirement"
    if variance <= epsilon:
        return "insufficient within-window variation"
    return "did not meet segmentation evidence rules"


def _feature_exclusion_reason(
    geometry_completeness: float,
    supported_fraction: float,
    dynamic_completeness: float,
    max_run: int,
    variance: float,
    minimum_fraction: float,
    min_continuous_frames: int,
    cfg: PhaseSegmentationConfig,
    kind: str,
) -> str:
    if geometry_completeness < cfg.minimum_geometry_completeness:
        return "geometry completeness below segmentation requirement"
    if supported_fraction < minimum_fraction:
        return "supported fraction below segmentation requirement"
    if kind == "robust_dynamic_rate" and dynamic_completeness < cfg.minimum_dynamic_supported_fraction:
        return "robust dynamic completeness below segmentation requirement"
    return _exclusion_reason(
        supported_fraction,
        max_run,
        variance,
        minimum_fraction,
        min_continuous_frames,
        cfg.epsilon,
    )


def _max_finite_run(series: pd.Series) -> int:
    max_run = 0
    current = 0
    for finite in pd.notna(series):
        if finite:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _robust_scale(series: pd.Series, epsilon: float) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(valid) < 2:
        return 0.0
    median = float(valid.median())
    mad = float((valid - median).abs().median()) * 1.4826
    if mad > epsilon:
        return mad
    iqr = float(valid.quantile(0.75) - valid.quantile(0.25)) / 1.349
    if iqr > epsilon:
        return iqr
    std = float(valid.std())
    return std if std > epsilon else 0.0


def _standardize_descriptors(
    descriptor_values: pd.DataFrame,
    metadata: tuple[dict, ...],
    cfg: PhaseSegmentationConfig,
) -> pd.DataFrame:
    standardized = descriptor_values[["source_frame_index"]].copy()
    for item in metadata:
        name = item["descriptor_name"]
        values = pd.to_numeric(descriptor_values[name], errors="coerce").astype(float)
        if item["unit"] == "deg" and (
            "orientation" in item.get("feature_name", "")
            or "heading" in item.get("feature_name", "")
            or "line_angle" in item.get("feature_name", "")
        ):
            values = _unwrap_degree_series(values)
        valid = values.dropna()
        center = float(valid.median())
        scale = _robust_scale(valid, cfg.epsilon)
        standardized[name] = (values - center) / scale if scale > cfg.epsilon else np.nan
        item["robust_center"] = center
        item["standardization_scale"] = scale
    return standardized


def _unwrap_degree_series(series: pd.Series) -> pd.Series:
    output = series.copy()
    valid = output.dropna()
    if len(valid) < 2:
        return output
    output.loc[valid.index] = np.degrees(np.unwrap(np.radians(valid.astype(float).to_numpy())))
    return output


def _change_signal(
    frame_timing: pd.DataFrame,
    standardized: pd.DataFrame,
    metadata: tuple[dict, ...],
    cfg: PhaseSegmentationConfig,
    *,
    smoothing_window_frames: int,
) -> pd.DataFrame:
    output = frame_timing.copy()
    names = [item["descriptor_name"] for item in metadata]
    families = {item["descriptor_name"]: item["family"] for item in metadata}
    diffs = standardized[names].astype(float).diff().abs()
    finite = diffs.notna()
    output["contributing_descriptors"] = finite.sum(axis=1).astype(int)
    weighted_sum = diffs.fillna(0.0).sum(axis=1)
    weights = finite.sum(axis=1).replace(0, np.nan)
    output["change_score"] = weighted_sum / weights
    for family in sorted(set(families.values())):
        family_names = [name for name, item_family in families.items() if item_family == family]
        family_diffs = diffs[family_names]
        family_weights = family_diffs.notna().sum(axis=1).replace(0, np.nan)
        output[f"family_{family}_contribution"] = family_diffs.fillna(0.0).sum(axis=1) / family_weights
        output[f"family_{family}_contributors"] = family_diffs.notna().sum(axis=1).astype(int)
    output["smoothed_change_score"] = _smooth_without_bridging(
        output["change_score"],
        smoothing_window_frames,
    )
    output["candidate_boundary"] = False
    output["selected_boundary"] = False
    output["sustained_shift_score"] = np.nan
    output["boundary_threshold"] = _boundary_threshold(output["smoothed_change_score"], cfg)
    return output


def _smooth_without_bridging(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series.copy()
    output = pd.Series(np.nan, index=series.index, dtype=float)
    finite = series.notna()
    group_id = (finite.ne(finite.shift(fill_value=False))).cumsum()
    for indices in series[finite].groupby(group_id[finite]).groups.values():
        run = series.loc[indices]
        min_periods = max(1, min(len(run), window // 2 + 1))
        output.loc[indices] = run.rolling(window, center=True, min_periods=min_periods).mean()
    return output


def _boundary_threshold(series: pd.Series, cfg: PhaseSegmentationConfig) -> float:
    valid = series.dropna().astype(float)
    if valid.empty:
        return float("inf")
    median = float(valid.median())
    mad = float((valid - median).abs().median()) * 1.4826
    return max(cfg.minimum_boundary_score, median + cfg.boundary_threshold_mad_multiplier * mad)


def _select_boundaries(
    change_signal: pd.DataFrame,
    standardized: pd.DataFrame,
    metadata: tuple[dict, ...],
    cfg: PhaseSegmentationConfig,
    *,
    min_phase_frames: int,
    min_separation_frames: int,
) -> tuple[tuple[int, ...], tuple[dict, ...]]:
    threshold = float(change_signal["boundary_threshold"].dropna().iloc[0])
    candidates: list[dict] = []
    scores = change_signal["smoothed_change_score"].to_numpy(dtype=float)
    frames = change_signal["source_frame_index"].astype(int).to_numpy()
    for index, score in enumerate(scores):
        if not np.isfinite(score) or score < threshold:
            continue
        if index < min_phase_frames or index > len(scores) - min_phase_frames - 1:
            continue
        left = max(0, index - min_separation_frames)
        right = min(len(scores), index + min_separation_frames + 1)
        local = scores[left:right]
        if np.nanmax(local) > score + cfg.epsilon:
            continue
        sustained = _sustained_shift_score(
            standardized,
            metadata,
            index,
            min_phase_frames=min_phase_frames,
        )
        change_signal.loc[index, "candidate_boundary"] = True
        change_signal.loc[index, "sustained_shift_score"] = sustained
        if sustained < cfg.minimum_sustained_shift:
            continue
        candidates.append(
            {
                "index": index,
                "source_frame_index": int(frames[index]),
                "timestamp_ms": _optional_float(change_signal.loc[index, "timestamp_ms"]),
                "movement_end_relative_ms": _optional_float(
                    change_signal.loc[index, "movement_end_relative_ms"]
                ),
                "score": float(score),
                "sustained_shift_score": float(sustained),
                "feature_family_contributions": _family_contributions_at(change_signal.iloc[index]),
            }
        )
    selected: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True):
        if len(selected) >= cfg.max_user_phases - 1:
            break
        if all(abs(candidate["index"] - item["index"]) >= min_separation_frames for item in selected):
            selected.append(candidate)
    selected = sorted(selected, key=lambda item: item["index"])
    selected = _remove_micro_phase_boundaries(selected, len(change_signal), min_phase_frames)
    for candidate in selected:
        change_signal.loc[candidate["index"], "selected_boundary"] = True
    return tuple(int(item["source_frame_index"]) for item in selected), tuple(selected)


def _sustained_shift_score(
    standardized: pd.DataFrame,
    metadata: tuple[dict, ...],
    index: int,
    *,
    min_phase_frames: int,
) -> float:
    names = [item["descriptor_name"] for item in metadata]
    pre = standardized.loc[max(0, index - min_phase_frames) : index - 1, names]
    post = standardized.loc[index : index + min_phase_frames - 1, names]
    shifts = []
    for name in names:
        pre_values = pre[name].dropna()
        post_values = post[name].dropna()
        if len(pre_values) < max(2, min_phase_frames // 3):
            continue
        if len(post_values) < max(2, min_phase_frames // 3):
            continue
        shifts.append(abs(float(post_values.median() - pre_values.median())))
    if not shifts:
        return 0.0
    return float(np.mean(shifts))


def _family_contributions_at(row: pd.Series) -> dict:
    contributions = {}
    for column, value in row.items():
        if column.startswith("family_") and column.endswith("_contribution"):
            family = column.removeprefix("family_").removesuffix("_contribution")
            if pd.notna(value):
                contributions[family] = float(value)
    total = sum(contributions.values())
    if total <= 0:
        return contributions
    return {
        key: {"score": value, "fraction": value / total}
        for key, value in sorted(contributions.items(), key=lambda item: item[1], reverse=True)
    }


def _remove_micro_phase_boundaries(
    selected: list[dict],
    frame_count: int,
    min_phase_frames: int,
) -> list[dict]:
    boundaries = list(selected)
    while boundaries:
        indices = [0, *[item["index"] for item in boundaries], frame_count]
        durations = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        if min(durations) >= min_phase_frames:
            break
        shortest = int(np.argmin(durations))
        if shortest == 0:
            remove_at = 0
        elif shortest >= len(boundaries):
            remove_at = len(boundaries) - 1
        else:
            left_score = boundaries[shortest - 1]["score"]
            right_score = boundaries[shortest]["score"]
            remove_at = shortest - 1 if left_score <= right_score else shortest
        boundaries.pop(remove_at)
    return boundaries


def _build_phases(
    *,
    case_id: str,
    source_id: str,
    dynamic_df: pd.DataFrame,
    path_df: pd.DataFrame,
    frame_timing: pd.DataFrame,
    change_signal: pd.DataFrame,
    descriptors: tuple[dict, ...],
    boundaries: tuple[int, ...],
    lineage_by_start: dict[int, dict] | None = None,
) -> list[MovementPhase]:
    starts = [int(frame_timing["source_frame_index"].iloc[0]), *list(boundaries)]
    ends = [
        boundary - 1 for boundary in boundaries
    ] + [int(frame_timing["source_frame_index"].iloc[-1])]
    phases = []
    for index, (start, end) in enumerate(zip(starts, ends, strict=True), start=1):
        phase_frames = frame_timing[
            frame_timing["source_frame_index"].astype(int).between(start, end, inclusive="both")
        ].copy()
        phase_change = change_signal[
            change_signal["source_frame_index"].astype(int).between(start, end, inclusive="both")
        ]
        category_summaries = _phase_category_summaries(dynamic_df, path_df, start, end)
        evidence_summary = _phase_evidence_summary(phase_frames, phase_change, descriptors, category_summaries)
        observations = _phase_observations(category_summaries)
        title = _phase_title(category_summaries, phase_change, index, len(starts))
        lineage = (lineage_by_start or {}).get(start, {})
        phases.append(
            MovementPhase(
                phase_id=f"phase_{index}",
                case_id=case_id,
                source_id=source_id,
                phase_index=index,
                start_frame=start,
                end_frame=end,
                start_timestamp_ms=_optional_float(phase_frames["timestamp_ms"].iloc[0]),
                end_timestamp_ms=_optional_float(phase_frames["timestamp_ms"].iloc[-1]),
                start_relative_ms=_optional_float(phase_frames["movement_end_relative_ms"].iloc[0]),
                end_relative_ms=_optional_float(phase_frames["movement_end_relative_ms"].iloc[-1]),
                duration_ms=float(
                    phase_frames["timestamp_ms"].iloc[-1] - phase_frames["timestamp_ms"].iloc[0]
                ),
                title=title,
                segmentation_features=tuple(item["descriptor_name"] for item in descriptors),
                change_score_summary=_change_summary(phase_change),
                category_summaries=category_summaries,
                phase_observations=tuple(observations),
                evidence_summary=evidence_summary,
                notable_extrema=tuple(_notable_extrema(dynamic_df, start, end)),
                source_frames=tuple(int(item) for item in phase_frames["source_frame_index"]),
                metadata={"phase_segmentation_version": PHASE_SEGMENTATION_VERSION, **lineage},
            )
        )
    return phases


def _refine_phases_hierarchically(
    *,
    case_id: str,
    source_id: str,
    dynamic_df: pd.DataFrame,
    case_summary: pd.DataFrame,
    path_df: pd.DataFrame,
    frame_timing: pd.DataFrame,
    global_change_signal: pd.DataFrame,
    descriptors: tuple[dict, ...],
    initial_phases: tuple[MovementPhase, ...],
    initial_transitions: tuple[dict, ...],
    cfg: PhaseSegmentationConfig,
) -> tuple[list[MovementPhase], tuple[dict, ...], tuple[dict, ...]]:
    local_records: list[dict] = []
    local_boundaries: dict[int, dict] = {}
    lineage_by_start: dict[int, dict] = {}
    for phase in initial_phases:
        phase_boundaries, phase_records = _local_refinement_boundaries(
            case_id=case_id,
            source_id=source_id,
            dynamic_df=dynamic_df,
            case_summary=case_summary,
            path_df=path_df,
            phase=phase,
            cfg=cfg,
            depth=1,
            root_phase=phase,
        )
        local_records.extend(phase_records)
        local_boundaries.update({int(item["source_frame_index"]): item for item in phase_boundaries})

    initial_boundaries = {int(item["transition_frame"]): item for item in initial_transitions}
    all_boundaries = sorted({*initial_boundaries.keys(), *local_boundaries.keys()})
    if len(all_boundaries) + 1 > cfg.max_refined_phases:
        keep_local_count = max(0, cfg.max_refined_phases - 1 - len(initial_boundaries))
        strongest_local = sorted(
            local_boundaries.values(),
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )[:keep_local_count]
        local_boundaries = {int(item["source_frame_index"]): item for item in strongest_local}
        all_boundaries = sorted({*initial_boundaries.keys(), *local_boundaries.keys()})
        local_records.append(
            {
                "status": "TRUNCATED_BY_MAX_REFINED_PHASES",
                "max_refined_phases": cfg.max_refined_phases,
                "retained_local_boundaries": sorted(local_boundaries),
            }
        )

    for phase_start, phase_end in _ranges_from_boundaries(
        int(frame_timing["source_frame_index"].iloc[0]),
        int(frame_timing["source_frame_index"].iloc[-1]),
        tuple(all_boundaries),
    ):
        parent = next(
            (
                phase
                for phase in initial_phases
                if phase.start_frame <= phase_start and phase_end <= phase.end_frame
            ),
            None,
        )
        if parent and (parent.start_frame != phase_start or parent.end_frame != phase_end):
            lineage_by_start[phase_start] = {
                "parent_phase_id": parent.phase_id,
                "parent_phase_index": parent.phase_index,
                "parent_phase_start_frame": parent.start_frame,
                "parent_phase_end_frame": parent.end_frame,
                "refined_from_parent_phase": True,
            }

    phases = _build_phases(
        case_id=case_id,
        source_id=source_id,
        dynamic_df=dynamic_df,
        path_df=path_df,
        frame_timing=frame_timing,
        change_signal=global_change_signal,
        descriptors=descriptors,
        boundaries=tuple(all_boundaries),
        lineage_by_start=lineage_by_start,
    )
    transition_records = []
    for boundary in all_boundaries:
        source = local_boundaries.get(boundary) or initial_boundaries.get(boundary)
        transition_records.append(_transition_payload(source, phases))
    return phases, tuple(transition_records), tuple(local_records)


def _local_refinement_boundaries(
    *,
    case_id: str,
    source_id: str,
    dynamic_df: pd.DataFrame,
    case_summary: pd.DataFrame,
    path_df: pd.DataFrame,
    phase: MovementPhase,
    cfg: PhaseSegmentationConfig,
    depth: int,
    root_phase: MovementPhase,
) -> tuple[list[dict], list[dict]]:
    if depth > cfg.refinement_max_depth:
        return [], []
    if phase.duration_ms < cfg.refinement_min_duration_ms:
        return [], [
            {
                "status": "SKIPPED_SHORT_PHASE",
                "phase_id": phase.phase_id,
                "start_frame": phase.start_frame,
                "end_frame": phase.end_frame,
                "duration_ms": phase.duration_ms,
                "refinement_depth": depth,
            }
        ]
    local_cfg = replace(cfg, enable_hierarchical_refinement=False)
    local_result = segment_movement_phases(
        case_id=case_id,
        source_id=source_id,
        dynamic_df=dynamic_df,
        case_summary=case_summary,
        path_df=path_df,
        movement_window={
            "movement_start_frame": phase.start_frame,
            "movement_end_frame": phase.end_frame,
            "duration_ms": phase.duration_ms,
        },
        config=local_cfg,
    )
    record = {
        "status": "REVIEWED",
        "phase_id": phase.phase_id,
        "root_phase_id": root_phase.phase_id,
        "start_frame": phase.start_frame,
        "end_frame": phase.end_frame,
        "duration_ms": phase.duration_ms,
        "refinement_depth": depth,
        "local_phase_count": len(local_result.phases),
        "local_transition_frames": [int(item["transition_frame"]) for item in local_result.transitions],
        "eligible_descriptor_count": len(local_result.eligible_descriptors),
    }
    if local_result.status != "SUPPORTED" or len(local_result.phases) < cfg.refinement_min_child_phases:
        record["status"] = "NO_SUPPORTED_INTERNAL_BOUNDARY"
        return [], [record]
    boundaries: list[dict] = []
    records = [record]
    for transition in local_result.transitions:
        boundary = {
            "index": transition.get("index"),
            "source_frame_index": int(transition["transition_frame"]),
            "timestamp_ms": transition.get("transition_timestamp"),
            "movement_end_relative_ms": transition.get("movement_end_relative_ms"),
            "score": float(transition["change_score"]),
            "sustained_shift_score": float(transition["sustained_shift_score"]),
            "feature_family_contributions": transition["feature_family_contributions"],
            "refinement_depth": depth,
            "parent_phase_id": root_phase.phase_id,
            "local_parent_phase_id": phase.phase_id,
            "parent_phase_start_frame": root_phase.start_frame,
            "parent_phase_end_frame": root_phase.end_frame,
            "local_parent_phase_start_frame": phase.start_frame,
            "local_parent_phase_end_frame": phase.end_frame,
            "root_phase_id": root_phase.phase_id,
            "boundary_source": "LOCAL_RESTANDARDIZED_PHASE_REVIEW",
        }
        boundaries.append(boundary)
    if depth < cfg.refinement_max_depth:
        for child in local_result.phases:
            child_boundaries, child_records = _local_refinement_boundaries(
                case_id=case_id,
                source_id=source_id,
                dynamic_df=dynamic_df,
                case_summary=case_summary,
                path_df=path_df,
                phase=child,
                cfg=cfg,
                depth=depth + 1,
                root_phase=root_phase,
            )
            boundaries.extend(child_boundaries)
            records.extend(child_records)
    return boundaries, records


def _ranges_from_boundaries(
    start_frame: int,
    end_frame: int,
    boundaries: tuple[int, ...],
) -> list[tuple[int, int]]:
    starts = [start_frame, *list(boundaries)]
    ends = [boundary - 1 for boundary in boundaries] + [end_frame]
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _phase_category_summaries(
    dynamic_df: pd.DataFrame,
    path_df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
) -> dict[str, dict]:
    summaries = {}
    movement_path = _movement_path_summary(path_df, start_frame, end_frame)
    if movement_path:
        summaries["movement_path"] = movement_path
    hka = _hka_chain_summary(dynamic_df, start_frame, end_frame)
    if hka:
        summaries["hip_knee_ankle_chain"] = hka
    bilateral = _bilateral_phase_summary(dynamic_df, start_frame, end_frame)
    if bilateral:
        summaries["bilateral_limb_relationship"] = bilateral
    trunk = _trunk_pelvis_summary(dynamic_df, start_frame, end_frame)
    if trunk:
        summaries["trunk_pelvis"] = trunk
    upper = _upper_body_summary(dynamic_df, start_frame, end_frame)
    if upper:
        summaries["upper_body"] = upper
    summaries["movement_timing"] = {
        "evidence_status": PhaseEvidenceStatus.GOOD.value,
        "summary": f"Phase interval spans source frames {start_frame} to {end_frame}.",
        "metrics": {"start_frame": start_frame, "end_frame": end_frame},
    }
    return summaries


def _movement_path_summary(path_df: pd.DataFrame, start_frame: int, end_frame: int) -> dict | None:
    rows = path_df[
        path_df["source_frame_index"].astype(int).between(start_frame, end_frame, inclusive="both")
        & path_df["path_status"].eq("SUPPORTED")
    ].copy()
    if len(rows) < 3:
        return None
    heading_rows = rows[rows["projected_heading_deg"].notna()]
    speed_rows = rows[rows["normalized_projected_speed_per_s"].notna()]
    metrics: dict[str, Any] = {
        "supported_samples": len(rows),
        "source_frames": [int(item) for item in rows["source_frame_index"]],
    }
    parts = []
    if len(heading_rows) >= 2:
        heading_change = wrapped_angle_difference_deg(
            float(heading_rows["projected_heading_deg"].iloc[-1]),
            float(heading_rows["projected_heading_deg"].iloc[0]),
        )
        metrics["mean_projected_heading_deg"] = _circular_mean_deg(
            heading_rows["projected_heading_deg"]
        )
        metrics["heading_change_deg"] = heading_change
        if abs(heading_change) >= 20.0:
            parts.append(
                f"Camera-compensated projected heading changed by {heading_change:.1f} degrees."
            )
        else:
            parts.append("Projected travel direction was relatively stable.")
    if len(speed_rows) >= 2:
        speed_change = float(
            speed_rows["normalized_projected_speed_per_s"].iloc[-1]
            - speed_rows["normalized_projected_speed_per_s"].iloc[0]
        )
        metrics["mean_normalized_projected_speed_per_s"] = float(
            speed_rows["normalized_projected_speed_per_s"].mean()
        )
        metrics["speed_change_normalized_per_s"] = speed_change
        parts.append(
            f"Body-scale-normalized projected speed changed by {speed_change:.2f} units/s."
        )
    return {
        "evidence_status": _evidence_from_fraction(len(rows) / max(end_frame - start_frame + 1, 1)),
        "summary": " ".join(parts) or "Camera-compensated path evidence was supported.",
        "metrics": metrics,
    }


def _bilateral_phase_summary(dynamic_df: pd.DataFrame, start_frame: int, end_frame: int) -> dict | None:
    signed = _supported_feature(dynamic_df, "hka_projected_bilateral_difference_deg", start_frame, end_frame)
    absolute = _supported_feature(
        dynamic_df,
        "hka_projected_bilateral_absolute_difference_deg",
        start_frame,
        end_frame,
    )
    if len(signed) < 3 or len(absolute) < 3:
        return None
    start_value = float(signed["feature_value"].iloc[0])
    end_value = float(signed["feature_value"].iloc[-1])
    abs_start = abs(start_value)
    abs_end = abs(end_value)
    abs_change = abs_end - abs_start
    peak = absolute.loc[absolute["feature_value"].idxmax()]
    relationship = _bilateral_relationship_label(start_value, end_value, abs_change)
    dynamic_supported = signed[
        signed["dynamic_status"].eq("SUPPORTED") & signed["robust_dynamic_rate"].notna()
    ]
    metrics = {
        "mean_signed_hka_difference_deg": float(signed["feature_value"].mean()),
        "mean_absolute_hka_difference_deg": float(absolute["feature_value"].mean()),
        "maximum_absolute_hka_difference_deg": float(peak["feature_value"]),
        "time_of_maximum_within_phase_ms": _optional_float(peak.get("movement_end_relative_ms")),
        "source_frame_of_maximum": int(peak["source_frame_index"]),
        "signed_difference_start_deg": start_value,
        "signed_difference_end_deg": end_value,
        "signed_difference_change_deg": end_value - start_value,
        "absolute_difference_change_deg": abs_change,
        "relationship_pattern": relationship,
        "robust_change_rate_median": (
            float(dynamic_supported["robust_dynamic_rate"].median())
            if len(dynamic_supported) >= 3
            else None
        ),
    }
    return {
        "evidence_status": _evidence_from_fraction(len(signed) / max(end_frame - start_frame + 1, 1)),
        "summary": f"The projected bilateral HKA relationship was {relationship}.",
        "metrics": metrics,
    }


def _hka_chain_summary(dynamic_df: pd.DataFrame, start_frame: int, end_frame: int) -> dict | None:
    features = {
        "injured_hka_angle_2d_deg": "injured-side projected HKA",
        "contralateral_hka_angle_2d_deg": "contralateral projected HKA",
    }
    metrics = {}
    parts = []
    for feature_name, label in features.items():
        summary = _single_feature_phase_summary(dynamic_df, feature_name, start_frame, end_frame)
        if summary is None:
            continue
        metrics[feature_name] = summary
        parts.append(f"{label} changed by {summary['change']:.1f} degrees.")
    if not metrics:
        return None
    evidence_fraction = np.mean([item["supported_fraction"] for item in metrics.values()])
    return {
        "evidence_status": _evidence_from_fraction(float(evidence_fraction)),
        "summary": " ".join(parts),
        "metrics": metrics,
    }


def _trunk_pelvis_summary(dynamic_df: pd.DataFrame, start_frame: int, end_frame: int) -> dict | None:
    features = {
        "projected_trunk_axis_angle_deg": "projected trunk axis",
        "projected_hip_line_angle_deg": "projected hip line",
        "projected_shoulder_line_angle_deg": "projected shoulder line",
        "projected_shoulder_pelvis_orientation_difference_deg": (
            "projected shoulder-pelvis orientation difference"
        ),
    }
    return _multi_feature_category_summary(
        dynamic_df,
        features,
        start_frame,
        end_frame,
        fallback="Supported trunk and pelvis descriptors were available in this phase.",
    )


def _upper_body_summary(dynamic_df: pd.DataFrame, start_frame: int, end_frame: int) -> dict | None:
    features = {
        "left_elbow_angle_2d_deg": "left projected elbow angle",
        "right_elbow_angle_2d_deg": "right projected elbow angle",
        "left_upper_arm_orientation_2d_deg": "left projected upper-arm orientation",
        "right_upper_arm_orientation_2d_deg": "right projected upper-arm orientation",
    }
    summary = _multi_feature_category_summary(
        dynamic_df,
        features,
        start_frame,
        end_frame,
        fallback="Supported upper-body descriptors were available in this phase.",
    )
    if summary is None:
        return None
    informative = [
        item for item in summary["metrics"].values() if abs(float(item.get("change") or 0.0)) >= 10.0
    ]
    if not informative:
        summary["summary"] = "Upper-body descriptors were supported but showed smaller projected changes."
    return summary


def _multi_feature_category_summary(
    dynamic_df: pd.DataFrame,
    features: dict[str, str],
    start_frame: int,
    end_frame: int,
    *,
    fallback: str,
) -> dict | None:
    metrics = {}
    parts = []
    for feature_name, label in features.items():
        summary = _single_feature_phase_summary(dynamic_df, feature_name, start_frame, end_frame)
        if summary is None:
            continue
        metrics[feature_name] = summary
        if abs(summary["change"]) >= 8.0:
            parts.append(f"{label} changed by {summary['change']:.1f} {summary['unit']}.")
    if not metrics:
        return None
    evidence_fraction = np.mean([item["supported_fraction"] for item in metrics.values()])
    return {
        "evidence_status": _evidence_from_fraction(float(evidence_fraction)),
        "summary": " ".join(parts) or fallback,
        "metrics": metrics,
    }


def _single_feature_phase_summary(
    dynamic_df: pd.DataFrame,
    feature_name: str,
    start_frame: int,
    end_frame: int,
) -> dict | None:
    rows = _supported_feature(dynamic_df, feature_name, start_frame, end_frame)
    if len(rows) < 3:
        return None
    start_value = float(rows["feature_value"].iloc[0])
    end_value = float(rows["feature_value"].iloc[-1])
    change = _feature_change_value(feature_name, start_value, end_value)
    dynamic_supported = rows[rows["dynamic_status"].eq("SUPPORTED") & rows["robust_dynamic_rate"].notna()]
    return {
        "start_value": start_value,
        "end_value": end_value,
        "change": change,
        "mean": float(rows["feature_value"].mean()),
        "range": float(rows["feature_value"].max() - rows["feature_value"].min()),
        "unit": str(rows["unit"].iloc[0] if "unit" in rows else ""),
        "supported_samples": len(rows),
        "supported_fraction": float(len(rows) / max(end_frame - start_frame + 1, 1)),
        "robust_rate_median": (
            float(dynamic_supported["robust_dynamic_rate"].median())
            if len(dynamic_supported) >= 3
            else None
        ),
        "source_frame_start": int(rows["source_frame_index"].iloc[0]),
        "source_frame_end": int(rows["source_frame_index"].iloc[-1]),
    }


def _supported_feature(
    dynamic_df: pd.DataFrame,
    feature_name: str,
    start_frame: int,
    end_frame: int,
) -> pd.DataFrame:
    return dynamic_df[
        dynamic_df["feature_name"].eq(feature_name)
        & dynamic_df["source_frame_index"].astype(int).between(start_frame, end_frame, inclusive="both")
        & dynamic_df["feature_status"].eq("SUPPORTED")
        & dynamic_df["feature_value"].notna()
    ].sort_values("source_frame_index")


def _bilateral_relationship_label(start_value: float, end_value: float, abs_change: float) -> str:
    if start_value == 0 or np.sign(start_value) != np.sign(end_value):
        return "sign crossing"
    if abs_change >= 8.0:
        return "increasing difference"
    if abs_change <= -8.0:
        return "decreasing difference"
    return "relatively stable"


def _feature_change_value(feature_name: str, start_value: float, end_value: float) -> float:
    if feature_name.endswith("_deg") and any(
        token in feature_name
        for token in (
            "orientation",
            "trunk_axis",
            "hip_line",
            "shoulder_line",
            "upper_arm",
        )
    ):
        return wrapped_angle_difference_deg(end_value, start_value)
    return end_value - start_value


def _phase_evidence_summary(
    phase_frames: pd.DataFrame,
    phase_change: pd.DataFrame,
    descriptors: tuple[dict, ...],
    category_summaries: dict[str, dict],
) -> dict:
    finite_change_fraction = float(phase_change["change_score"].notna().mean()) if len(phase_change) else 0.0
    category_statuses = {
        key: value["evidence_status"] for key, value in category_summaries.items()
    }
    usable_categories = [
        key for key, value in category_statuses.items() if value in {"GOOD", "MODERATE", "LIMITED"}
    ]
    descriptor_count = len(descriptors)
    status = PhaseEvidenceStatus.LIMITED
    if descriptor_count >= 10 and finite_change_fraction >= 0.70:
        status = PhaseEvidenceStatus.GOOD
    elif descriptor_count >= 6 and finite_change_fraction >= 0.45:
        status = PhaseEvidenceStatus.MODERATE
    elif finite_change_fraction <= 0:
        status = PhaseEvidenceStatus.UNAVAILABLE
    limitation = "No primary local limitation."
    if finite_change_fraction < 0.70:
        limitation = "Some frames or descriptors were unavailable inside this phase."
    return {
        "evidence_status": status.value,
        "geometry_evidence": status.value,
        "dynamic_evidence": "MODERATE" if finite_change_fraction >= 0.45 else "LIMITED",
        "eligible_descriptor_count": descriptor_count,
        "finite_change_fraction": finite_change_fraction,
        "usable_categories": usable_categories,
        "major_limitation": limitation,
        "frame_count": len(phase_frames),
    }


def _phase_observations(category_summaries: dict[str, dict]) -> list[str]:
    observations = []
    for key in (
        "movement_path",
        "hip_knee_ankle_chain",
        "bilateral_limb_relationship",
        "trunk_pelvis",
        "upper_body",
    ):
        summary = category_summaries.get(key)
        if summary and summary.get("summary"):
            observations.append(str(summary["summary"]))
    return observations


def _phase_title(
    category_summaries: dict[str, dict],
    phase_change: pd.DataFrame,
    phase_index: int,
    phase_count: int,
) -> str:
    if phase_index == phase_count and phase_count > 1:
        return "Final Observable Movement"
    path = category_summaries.get("movement_path", {}).get("metrics", {})
    heading_change = abs(float(path.get("heading_change_deg") or 0.0))
    bilateral = category_summaries.get("bilateral_limb_relationship", {}).get("metrics", {})
    abs_change = float(bilateral.get("absolute_difference_change_deg") or 0.0)
    if abs_change >= 10.0:
        return "Increasing Bilateral Difference"
    if abs_change <= -10.0:
        return "Decreasing Bilateral Difference"
    if heading_change >= 90.0:
        return "Directional Transition"
    trunk = category_summaries.get("trunk_pelvis", {}).get("metrics", {})
    if any(abs(float(item.get("change") or 0.0)) >= 15.0 for item in trunk.values()):
        return "Whole-Body Reorientation"
    if heading_change >= 25.0:
        return "Directional Transition"
    score = phase_change["smoothed_change_score"].dropna()
    if not score.empty and float(score.mean()) >= float(score.quantile(0.75)):
        return "Large Multidimensional Change"
    if category_summaries:
        return "Relatively Stable Movement"
    return f"Movement Phase {phase_index}"


def _change_summary(phase_change: pd.DataFrame) -> dict:
    scores = phase_change["smoothed_change_score"].dropna()
    if scores.empty:
        return {
            "mean_change_score": None,
            "maximum_change_score": None,
            "maximum_change_frame": None,
            "supported_change_samples": 0,
        }
    max_index = scores.idxmax()
    return {
        "mean_change_score": float(scores.mean()),
        "maximum_change_score": float(scores.max()),
        "maximum_change_frame": int(phase_change.loc[max_index, "source_frame_index"]),
        "supported_change_samples": len(scores),
    }


def _notable_extrema(dynamic_df: pd.DataFrame, start_frame: int, end_frame: int) -> list[dict]:
    notable = []
    for feature_name in (
        "hka_projected_bilateral_absolute_difference_deg",
        "projected_trunk_axis_angle_deg",
        "right_elbow_angle_2d_deg",
    ):
        rows = _supported_feature(dynamic_df, feature_name, start_frame, end_frame)
        if rows.empty:
            continue
        row = rows.loc[rows["feature_value"].abs().idxmax()]
        notable.append(
            {
                "feature_name": feature_name,
                "source_frame_index": int(row["source_frame_index"]),
                "movement_end_relative_ms": _optional_float(row.get("movement_end_relative_ms")),
                "value": float(row["feature_value"]),
                "unit": str(row.get("unit", "")),
            }
        )
    return notable


def _transition_payload(transition: dict, phases: list[MovementPhase]) -> dict:
    frame = int(transition.get("source_frame_index", transition.get("transition_frame")))
    before = next((phase.phase_id for phase in phases if phase.end_frame == frame - 1), None)
    after = next((phase.phase_id for phase in phases if phase.start_frame == frame), None)
    contributors = _top_contributors(transition["feature_family_contributions"])
    return {
        "transition_frame": frame,
        "transition_timestamp": transition.get("timestamp_ms"),
        "movement_end_relative_ms": transition.get("movement_end_relative_ms"),
        "from_phase_id": before,
        "to_phase_id": after,
        "change_score": transition.get("score", transition.get("change_score")),
        "sustained_shift_score": transition["sustained_shift_score"],
        "feature_family_contributions": transition["feature_family_contributions"],
        "dominant_feature_families": contributors,
        "boundary_source": transition.get("boundary_source", "GLOBAL_MOVEMENT_SEGMENTATION"),
        "refinement_depth": transition.get("refinement_depth"),
        "parent_phase_id": transition.get("parent_phase_id"),
        "parent_phase_start_frame": transition.get("parent_phase_start_frame"),
        "parent_phase_end_frame": transition.get("parent_phase_end_frame"),
        "local_parent_phase_id": transition.get("local_parent_phase_id"),
        "local_parent_phase_start_frame": transition.get("local_parent_phase_start_frame"),
        "local_parent_phase_end_frame": transition.get("local_parent_phase_end_frame"),
        "evidence": (
            "This transition was selected from a sustained multivariate movement-change "
            "peak inside the human Movement Window."
        ),
    }


def _top_contributors(contributions: dict) -> list[str]:
    return [
        str(key)
        for key, value in sorted(
            contributions.items(),
            key=lambda item: item[1].get("fraction", 0.0),
            reverse=True,
        )[:3]
    ]


def _phase_frame_map(
    frame_timing: pd.DataFrame,
    phases: list[MovementPhase],
    change_signal: pd.DataFrame,
) -> pd.DataFrame:
    frame_map = frame_timing.copy()
    frame_map["phase_id"] = None
    frame_map["phase_index"] = np.nan
    frame_map["phase_title"] = None
    for phase in phases:
        mask = frame_map["source_frame_index"].astype(int).between(
            phase.start_frame,
            phase.end_frame,
            inclusive="both",
        )
        frame_map.loc[mask, "phase_id"] = phase.phase_id
        frame_map.loc[mask, "phase_index"] = phase.phase_index
        frame_map.loc[mask, "phase_title"] = phase.title
    change_columns = [
        "source_frame_index",
        "change_score",
        "smoothed_change_score",
        "contributing_descriptors",
        "candidate_boundary",
        "selected_boundary",
        "sustained_shift_score",
    ]
    return frame_map.merge(change_signal[change_columns], on="source_frame_index", how="left")


def _sequence_summary(
    duration_ms: float,
    phases: list[MovementPhase],
    transitions: tuple[dict, ...],
) -> str:
    duration_s = duration_ms / 1000.0
    if not phases:
        return "Phase segmentation is unavailable for this human Movement Window."
    if transitions:
        strongest = max(transitions, key=lambda item: item["change_score"])
        to_phase = strongest.get("to_phase_id", "the next phase")
        contributors = ", ".join(_readable_family(item) for item in strongest["dominant_feature_families"])
        return (
            f"The {duration_s:.2f}-second observable movement contained {len(phases)} "
            f"data-supported movement phases. The largest multidimensional movement change "
            f"occurred entering {to_phase.replace('_', ' ').title()} at source frame "
            f"{strongest['transition_frame']}. The strongest contributors were {contributors}."
        )
    return (
        f"The {duration_s:.2f}-second observable movement was represented as one "
        "data-supported movement phase because no sustained multivariate transition met "
        "the configured evidence threshold."
    )


def _evidence_interval_summary(
    duration_ms: float,
    interval: MovementPhase,
) -> str:
    return (
        f"A {duration_ms / 1000.0:.2f}-second supported evidence interval spans source "
        f"frames {interval.start_frame} to {interval.end_frame}. No sustained multivariate "
        "transition met the configured phase-boundary rule, so this result is not presented "
        "as a phase sequence. It describes observable measurements only and does not "
        "identify injury timing."
    )


def _readable_family(family: str) -> str:
    return {
        "movement_path": "movement path",
        "hip_knee_ankle_chain": "Hip-Knee-Ankle chain",
        "bilateral_limb_relationship": "bilateral limb relationship",
        "trunk_pelvis": "trunk and pelvis",
        "upper_body": "upper body",
    }.get(family, family.replace("_", " "))


def _evidence_from_fraction(value: float) -> str:
    if value >= 0.75:
        return PhaseEvidenceStatus.GOOD.value
    if value >= 0.55:
        return PhaseEvidenceStatus.MODERATE.value
    if value > 0:
        return PhaseEvidenceStatus.LIMITED.value
    return PhaseEvidenceStatus.UNAVAILABLE.value


def _circular_mean_deg(values: pd.Series) -> float:
    radians = np.radians(pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy())
    if len(radians) == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())))


def _odd_window(smoothing_ms: float, fps: float) -> int:
    frames = max(1, round(smoothing_ms / 1000.0 * fps))
    return frames if frames % 2 == 1 else frames + 1


def _phase_decision_rationale(reason_code: str) -> dict[str, str]:
    """Return the saved, user-facing rationale for the rule-based phase decision."""

    method_note = (
        "This decision was made by a deterministic rule-based procedure. No AI or "
        "generative model was used."
    )
    if reason_code == "SUPPORTED_MULTIVARIATE_TRANSITION":
        return {
            "decision": "PHASE_SEQUENCE_PRODUCED",
            "reason_code": reason_code,
            "explanation": (
                "The rule-based phase procedure produced a phase sequence because a sustained "
                "multivariate transition satisfied the configured phase-boundary safeguards."
            ),
            "safety_rationale": (
                "A boundary is shown only when several supported movement measurements change "
                "together and the changed pattern persists."
            ),
            "evidence_preserved": (
                "The supported measurements, boundary evidence, and analysis scope remain "
                "available for human review."
            ),
            "method_note": method_note,
        }
    if reason_code == "NO_SUPPORTED_MULTIVARIATE_TRANSITION":
        return {
            "decision": "PHASES_NOT_PRODUCED",
            "reason_code": reason_code,
            "explanation": (
                "The rule-based phase procedure did not produce distinct phases because no "
                "sustained multivariate transition satisfied the configured phase-boundary rule."
            ),
            "safety_rationale": (
                "Withholding a phase sequence avoids imposing a before/after structure or "
                "implying an injury moment that the supported measurements do not establish."
            ),
            "evidence_preserved": (
                "The supported framewise measurements and evidence interval remain visible, "
                "including their scope and evidence gaps."
            ),
            "method_note": method_note,
        }
    if reason_code == "INSUFFICIENT_CONTINUOUS_MULTIVARIATE_EVIDENCE":
        return {
            "decision": "PHASES_NOT_PRODUCED",
            "reason_code": reason_code,
            "explanation": (
                "The rule-based phase procedure did not produce phases because too few supported "
                "movement descriptors formed a continuous block that satisfied the configured "
                "evidence safeguards."
            ),
            "safety_rationale": (
                "Withholding phases avoids turning unsupported values or fragmented evidence "
                "into a seemingly complete movement story."
            ),
            "evidence_preserved": (
                "Available measurements and frame-level support reasons remain visible for "
                "human inspection."
            ),
            "method_note": method_note,
        }
    raise ValueError(f"Unsupported phase decision rationale reason: {reason_code}")


def _metadata(
    cfg: PhaseSegmentationConfig,
    fps: float,
    start_frame: int,
    end_frame: int,
    duration_ms: float,
) -> dict:
    return {
        "phase_segmentation_version": PHASE_SEGMENTATION_VERSION,
        "movement_window": {
            "movement_start_frame": start_frame,
            "movement_end_frame": end_frame,
            "duration_ms": duration_ms,
        },
        "estimated_fps": fps,
        "standardization_method": "within-window median and MAD robust z-score",
        "change_score_method": (
            "mean absolute frame-to-frame change across supported standardized descriptors"
        ),
        "temporal_smoothing": (
            "centered rolling mean applied only within finite change-score runs"
        ),
        "boundary_detection_method": (
            "sustained local peaks above robust threshold with minimum phase duration"
        ),
        "micro_phase_merge_rule": (
            "candidate boundaries that would create intervals below the minimum duration "
            "are removed, preferring the stronger adjacent boundary"
        ),
        "configuration": cfg.to_dict(),
    }


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if np.isnan(number) or np.isinf(number) else number
    return value
