from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acl_motion.signatures import build_case_movement_signature, build_clustering_feature_registry


def test_registry_keeps_path_disabled_until_validated() -> None:
    registry = build_clustering_feature_registry(path_enabled=False)

    path_rows = registry[registry["family"].eq("Movement Path")]

    assert not path_rows["enabled_for_clustering"].any()


def test_case_signature_uses_supported_values_and_normalized_timing() -> None:
    dynamic_df = _dynamic_df()
    movement_story = {
        "phases": [{"phase_id": "phase_1"}, {"phase_id": "phase_2"}],
        "transitions": [
            {
                "transition_frame": 3,
                "change_score": 2.5,
                "movement_end_relative_ms": -200.0,
            }
        ],
    }
    movement_window = {
        "movement_start_frame": 0,
        "movement_end_frame": 4,
        "duration_ms": 1000.0,
    }

    signature = build_case_movement_signature(
        case_id="case",
        source_id="source",
        dynamic_df=dynamic_df,
        path_df=pd.DataFrame(),
        movement_story=movement_story,
        movement_window=movement_window,
        path_quality_summary={"overall_status": "UNAVAILABLE"},
    )

    long = signature.long_table.set_index("feature_name")
    assert long.loc["injured_hka_median_whole", "value"] == pytest.approx(12.0)
    assert bool(long.loc["injured_hka_median_whole", "eligible_for_future_clustering"])
    assert long.loc["injured_hka_median_whole", "view_suitability"] == (
        "SUPPORTED_GENERIC_PROJECTED_VIEW"
    )
    assert long.loc["injured_hka_median_whole", "missingness_fraction"] == pytest.approx(0.0)
    assert long.loc["path_heading_change_whole", "excluded_by"] == "evidence"
    assert long.loc["path_heading_change_whole", "view_suitability"] == "PATH_QA_REQUIRED"
    assert long.loc["strongest_transition_timing_normalized", "value"] == pytest.approx(0.8)
    assert "case_id" in signature.matrix_preview.columns
    assert "path_heading_change_whole" not in signature.matrix_preview.columns


def test_signature_missing_window_remains_unavailable_not_zero() -> None:
    signature = build_case_movement_signature(
        case_id="case",
        source_id="source",
        dynamic_df=_dynamic_df(),
        path_df=pd.DataFrame(),
        movement_story={"phases": [], "transitions": []},
        movement_window={"movement_start_frame": 0, "movement_end_frame": 4, "duration_ms": 1000.0},
        path_quality_summary={"overall_status": "UNAVAILABLE"},
    )

    row = signature.long_table.set_index("feature_name").loc[
        "hka_bilateral_abs_median_final_500ms"
    ]
    phase_count = signature.long_table.set_index("feature_name").loc["phase_count_supported"]

    assert np.isnan(row["value"])
    assert not bool(row["eligible_for_future_clustering"])
    assert row["excluded_by"] == "evidence"
    assert phase_count["evidence"] == "UNAVAILABLE"
    assert "Phase segmentation unavailable" in phase_count["exclusion_reason"]


def test_signature_accepts_human_movement_duration_key_for_normalized_timing() -> None:
    signature = build_case_movement_signature(
        case_id="case",
        source_id="source",
        dynamic_df=_dynamic_df(),
        path_df=pd.DataFrame(),
        movement_story={
            "phases": [{"phase_id": "phase_1"}, {"phase_id": "phase_2"}],
            "transitions": [
                {
                    "transition_frame": 3,
                    "change_score": 2.5,
                    "movement_end_relative_ms": -200.0,
                }
            ],
        },
        movement_window={
            "movement_start_frame": 0,
            "movement_end_frame": 4,
            "movement_duration_ms": 1000.0,
        },
        path_quality_summary={"overall_status": "UNAVAILABLE"},
    )

    long = signature.long_table.set_index("feature_name")

    assert long.loc["injured_hka_start_end_change_whole", "normalized_movement_progress"] == (
        pytest.approx(1.0)
    )
    assert long.loc["strongest_transition_timing_normalized", "value"] == pytest.approx(0.8)


def test_signature_uses_wrap_aware_orientation_range_and_change() -> None:
    dynamic_df = _dynamic_df()
    mask = dynamic_df["feature_name"].eq("projected_trunk_axis_angle_deg")
    dynamic_df.loc[mask, "feature_value"] = [179.0, -179.0, 178.0, -178.0, 177.0]

    signature = build_case_movement_signature(
        case_id="case",
        source_id="source",
        dynamic_df=dynamic_df,
        path_df=pd.DataFrame(),
        movement_story={"phases": [], "transitions": []},
        movement_window={"movement_start_frame": 0, "movement_end_frame": 4, "duration_ms": 1000.0},
        path_quality_summary={"overall_status": "UNAVAILABLE"},
    )

    long = signature.long_table.set_index("feature_name")
    assert long.loc["trunk_axis_range_whole", "value"] == pytest.approx(5.0)
    assert long.loc["trunk_axis_change_whole", "value"] == pytest.approx(-2.0)


def _dynamic_df() -> pd.DataFrame:
    rows = []
    frames = [0, 1, 2, 3, 4]
    times = [-1000.0, -750.0, -500.0, -250.0, 0.0]
    metrics = {
        "injured_hka_angle_2d_deg": [10.0, 11.0, 12.0, 13.0, 14.0],
        "contralateral_hka_angle_2d_deg": [20.0, 20.5, 21.0, 21.5, 22.0],
        "hka_projected_bilateral_absolute_difference_deg": [10.0, 9.5, np.nan, np.nan, np.nan],
        "hka_projected_bilateral_difference_deg": [-10.0, -9.5, -9.0, -8.5, -8.0],
        "projected_trunk_axis_angle_deg": [40.0, 42.0, 44.0, 46.0, 48.0],
        "projected_shoulder_pelvis_orientation_difference_deg": [5.0, 7.0, 6.0, 8.0, 10.0],
        "right_upper_arm_orientation_2d_deg": [1.0, 2.0, 3.0, 4.0, 5.0],
        "elbow_projected_bilateral_absolute_difference_deg": [2.0, 2.5, 3.0, 3.5, 4.0],
    }
    for feature_name, values in metrics.items():
        for frame, time_ms, value in zip(frames, times, values, strict=True):
            supported = np.isfinite(value)
            rows.append(
                {
                    "case_id": "case",
                    "source_id": "source",
                    "feature_name": feature_name,
                    "source_frame_index": frame,
                    "timestamp_ms": frame * 250.0,
                    "movement_elapsed_ms": frame * 250.0,
                    "movement_end_relative_ms": time_ms,
                    "feature_value": value,
                    "feature_status": "SUPPORTED" if supported else "INSUFFICIENT_LANDMARKS",
                }
            )
    return pd.DataFrame(rows)
