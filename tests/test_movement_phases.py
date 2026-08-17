from __future__ import annotations

import numpy as np
import pandas as pd

from acl_motion.semantics.phases import (
    MovementPhase,
    PhaseSegmentationConfig,
    segment_movement_phases,
)


def test_stable_sequence_creates_one_phase() -> None:
    dynamic = _dynamic_df(
        {
            "feature_a": [i * 0.03 for i in range(24)],
            "feature_b": [1.0 + i * 0.02 for i in range(24)],
        }
    )

    result = _segment(dynamic)

    assert result.status == "SUPPORTED"
    assert len(result.phases) == 1
    assert result.transitions == ()


def test_one_clear_sustained_transition_creates_two_phases() -> None:
    dynamic = _dynamic_df(
        {
            "feature_a": [0.0] * 12 + [5.0] * 12,
            "feature_b": [1.0] * 12 + [6.0] * 12,
        }
    )

    result = _segment(dynamic)

    assert len(result.phases) == 2
    assert len(result.transitions) == 1
    assert 10 <= result.transitions[0]["transition_frame"] <= 13


def test_two_clear_sustained_transitions_create_three_phases() -> None:
    dynamic = _dynamic_df(
        {
            "feature_a": [0.0] * 10 + [5.0] * 10 + [-3.0] * 10,
            "feature_b": [1.0] * 10 + [6.0] * 10 + [-2.0] * 10,
        }
    )

    result = _segment(dynamic)

    assert len(result.phases) == 3
    assert [item["transition_frame"] for item in result.transitions] == [10, 20]


def test_single_frame_spike_does_not_create_phase_boundary() -> None:
    values = [0.0] * 24
    values[12] = 10.0
    dynamic = _dynamic_df({"feature_a": values, "feature_b": [item * 0.5 for item in values]})

    result = _segment(dynamic)

    assert len(result.phases) == 1
    assert result.transitions == ()


def test_missing_interval_does_not_create_artificial_transition() -> None:
    base = [i * 0.03 for i in range(28)]
    values = [value if index not in {11, 12, 13, 14} else np.nan for index, value in enumerate(base)]
    dynamic = _dynamic_df({"feature_a": values, "feature_b": [value * 0.5 for value in values]})

    result = _segment(dynamic)

    assert len(result.phases) == 1
    assert result.transitions == ()


def test_short_micro_segment_is_merged_or_rejected() -> None:
    dynamic = _dynamic_df(
        {
            "feature_a": [0.0] * 10 + [4.0] * 2 + [8.0] * 14,
            "feature_b": [1.0] * 10 + [5.0] * 2 + [9.0] * 14,
        }
    )

    result = _segment(dynamic)

    assert len(result.phases) <= 2
    assert all(phase.duration_ms >= 200.0 for phase in result.phases)


def test_unsupported_feature_does_not_contribute_to_change_score() -> None:
    dynamic = _dynamic_df(
        {
            "feature_a": [i * 0.03 for i in range(24)],
            "feature_b": [1.0 + i * 0.02 for i in range(24)],
            "unsupported_large_feature": [0.0] * 12 + [10_000.0] * 12,
        },
        unsupported_features={"unsupported_large_feature"},
    )

    result = _segment(dynamic)

    assert len(result.phases) == 1
    assert all(
        item["feature_name"] != "unsupported_large_feature"
        for item in result.eligible_descriptors
    )


def test_large_unit_feature_does_not_dominate_after_standardization() -> None:
    dynamic = _dynamic_df(
        {
            "small_feature": [0.0] * 12 + [1.0] * 12,
            "large_unit_feature": [0.0] * 12 + [10_000.0] * 12,
        }
    )

    result = _segment(dynamic)

    assert len(result.phases) == 2
    assert max(result.change_signal["smoothed_change_score"].dropna()) < 3.0
    scales = {item["feature_name"]: item["standardization_scale"] for item in result.eligible_descriptors}
    assert scales["large_unit_feature"] > scales["small_feature"]


def test_long_homogeneous_phase_is_not_forced_to_split() -> None:
    dynamic = _dynamic_df(
        {
            "feature_a": [i * 0.01 for i in range(40)],
            "feature_b": [1.0 + i * 0.01 for i in range(40)],
        }
    )

    result = _segment(dynamic)

    assert len(result.phases) == 1
    assert result.metadata["refinement"]["refined"] is False


def test_hierarchical_refinement_splits_long_phase_with_internal_transition() -> None:
    dynamic = _dynamic_df(
        {
            "feature_a": [0.0] * 10 + [4.0] * 10 + [8.0] * 14,
            "feature_b": [1.0] * 10 + [5.0] * 10 + [9.0] * 14,
        }
    )

    result = segment_movement_phases(
        case_id="synthetic_case",
        source_id="synthetic_view",
        dynamic_df=dynamic,
        case_summary=_case_summary(dynamic),
        path_df=_path_df(dynamic),
        movement_window={
            "movement_start_frame": 0,
            "movement_end_frame": 33,
            "duration_ms": 3300.0,
        },
        config=PhaseSegmentationConfig(
            minimum_geometry_completeness=0.50,
            minimum_supported_fraction=0.45,
            minimum_dynamic_supported_fraction=0.45,
            minimum_continuous_supported_ms=300.0,
            minimum_eligible_descriptors=2,
            minimum_phase_duration_ms=300.0,
            minimum_boundary_separation_ms=300.0,
            smoothing_ms=100.0,
            minimum_boundary_score=0.15,
            minimum_sustained_shift=0.40,
            max_user_phases=2,
            refinement_min_duration_ms=900.0,
            refinement_max_depth=2,
            max_refined_phases=4,
        ),
    )

    assert result.metadata["refinement"]["original_phase_count"] <= 2
    assert result.metadata["refinement"]["refined"] is True
    assert len(result.phases) >= 3
    assert any(
        transition["boundary_source"] == "LOCAL_RESTANDARDIZED_PHASE_REVIEW"
        for transition in result.transitions
    )


def test_phase_language_rejects_non_neutral_terms() -> None:
    try:
        MovementPhase(
            phase_id="phase_1",
            case_id="case",
            source_id="source",
            phase_index=1,
            start_frame=0,
            end_frame=10,
            start_timestamp_ms=0.0,
            end_timestamp_ms=1000.0,
            start_relative_ms=-1000.0,
            end_relative_ms=0.0,
            duration_ms=1000.0,
            title="Dangerous phase",
            segmentation_features=(),
            change_score_summary={},
            category_summaries={},
            phase_observations=(),
            evidence_summary={},
            notable_extrema=(),
            source_frames=tuple(range(11)),
        )
    except ValueError:
        return
    raise AssertionError("MovementPhase accepted non-neutral wording")


def _segment(dynamic_df: pd.DataFrame):
    return segment_movement_phases(
        case_id="synthetic_case",
        source_id="synthetic_view",
        dynamic_df=dynamic_df,
        case_summary=_case_summary(dynamic_df),
        path_df=_path_df(dynamic_df),
        movement_window={
            "movement_start_frame": int(dynamic_df["source_frame_index"].min()),
            "movement_end_frame": int(dynamic_df["source_frame_index"].max()),
            "duration_ms": float(dynamic_df["timestamp_ms"].max() - dynamic_df["timestamp_ms"].min()),
        },
        config=PhaseSegmentationConfig(
            minimum_geometry_completeness=0.50,
            minimum_supported_fraction=0.45,
            minimum_dynamic_supported_fraction=0.45,
            minimum_continuous_supported_ms=300.0,
            minimum_eligible_descriptors=2,
            minimum_phase_duration_ms=300.0,
            minimum_boundary_separation_ms=300.0,
            smoothing_ms=100.0,
            minimum_boundary_score=0.15,
            minimum_sustained_shift=0.40,
            max_user_phases=6,
        ),
    )


def _dynamic_df(
    features: dict[str, list[float]],
    *,
    unsupported_features: set[str] | None = None,
) -> pd.DataFrame:
    unsupported = unsupported_features or set()
    rows = []
    frame_count = len(next(iter(features.values())))
    for feature_name, values in features.items():
        for frame, value in enumerate(values):
            missing = pd.isna(value)
            unsupported_feature = feature_name in unsupported
            rows.append(
                {
                    "case_id": "synthetic_case",
                    "source_id": "synthetic_view",
                    "view_id": "synthetic_view",
                    "source_frame_index": frame,
                    "analysis_frame_index": frame,
                    "timestamp_ms": frame * 100.0,
                    "movement_elapsed_ms": frame * 100.0,
                    "movement_end_relative_ms": (frame - frame_count + 1) * 100.0,
                    "feature_name": feature_name,
                    "feature_value": value,
                    "unit": "",
                    "feature_status": (
                        "LOW_CONFIDENCE"
                        if unsupported_feature or missing
                        else "SUPPORTED"
                    ),
                    "dynamic_status": "SUPPORTED" if not unsupported_feature and not missing else "MISSING_FEATURE",
                    "robust_dynamic_rate": 0.0 if not unsupported_feature and not missing else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _case_summary(dynamic_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_name, rows_for_feature in dynamic_df.groupby("feature_name"):
        supported = rows_for_feature["feature_status"].eq("SUPPORTED")
        rows.append(
            {
                "feature_name": feature_name,
                "body_region": "lower_limb",
                "geometry_completeness": float(supported.mean()),
                "dynamic_completeness": float(supported.mean()),
                "quality_category": "SUPPORTED" if supported.any() else "UNAVAILABLE",
            }
        )
    return pd.DataFrame(rows)


def _path_df(dynamic_df: pd.DataFrame) -> pd.DataFrame:
    frames = (
        dynamic_df[
            [
                "source_frame_index",
                "timestamp_ms",
                "movement_elapsed_ms",
                "movement_end_relative_ms",
            ]
        ]
        .drop_duplicates("source_frame_index")
        .sort_values("source_frame_index")
    )
    return frames.assign(
        projected_heading_deg=np.nan,
        normalized_projected_speed_per_s=np.nan,
        path_status="UNAVAILABLE",
    )
