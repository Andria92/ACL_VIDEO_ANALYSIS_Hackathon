from __future__ import annotations

import pandas as pd
import pytest

from acl_motion.semantics.metric_explorer import (
    SelectionMode,
    build_metric_explorer_payload,
    metric_statistics,
    selection_statistics,
)


def test_metric_statistics_exclude_unsupported_values() -> None:
    rows = _metric_rows([1.0, None, 4.0, 7.0], ["SUPPORTED", "UNAVAILABLE", "SUPPORTED", "SUPPORTED"])

    stats = metric_statistics(rows)

    assert stats["supported_n"] == 3
    assert stats["relevant_n"] == 4
    assert stats["mean"] == pytest.approx(4.0)
    assert stats["median"] == pytest.approx(4.0)
    assert stats["minimum"] == 1.0
    assert stats["maximum"] == 7.0
    assert stats["range"] == 6.0
    assert stats["q1"] == pytest.approx(2.5)
    assert stats["q3"] == pytest.approx(5.5)
    assert stats["iqr"] == pytest.approx(3.0)
    assert stats["absolute_change"] == pytest.approx(6.0)
    assert stats["total_absolute_change"] == pytest.approx(6.0)
    assert stats["minimum_frame"] == 0
    assert stats["maximum_frame"] == 3


def test_selection_statistics_for_single_and_five_frame_window() -> None:
    rows = _metric_rows([1.0, None, 4.0, 7.0, 10.0], ["SUPPORTED", "UNAVAILABLE", "SUPPORTED", "SUPPORTED", "SUPPORTED"])

    single = selection_statistics(rows, mode=SelectionMode.SINGLE_FRAME, frame=1)
    window = selection_statistics(
        rows,
        mode=SelectionMode.FIVE_FRAME_WINDOW,
        start_frame=0,
        end_frame=4,
    )

    assert single["current_value"] is None
    assert single["evidence_state"] == "UNAVAILABLE"
    assert window["supported_n"] == 4
    assert window["mean"] == pytest.approx(5.5)
    assert len(window["frames"]) == 5


@pytest.mark.parametrize(
    ("metric_name", "values", "angle_type", "raw_change", "canonical_change"),
    [
        ("injured_hka_angle_2d_deg", [158.6, 85.5], "internal", -73.1, -73.1),
        ("projected_trunk_axis_angle_deg", [-175.0, 178.0], "directed", 353.0, -7.0),
        ("projected_hip_line_angle_deg", [-0.1, 173.9], "axis", 174.0, -6.0),
    ],
)
def test_metric_statistics_add_canonical_display_change_without_altering_existing_change(
    metric_name,
    values,
    angle_type,
    raw_change,
    canonical_change,
) -> None:
    rows = _metric_rows(values, ["SUPPORTED", "SUPPORTED"])
    rows["metric_name"] = metric_name
    rows["unit"] = "deg"

    stats = metric_statistics(rows)

    assert stats["start_value"] == pytest.approx(values[0])
    assert stats["end_value"] == pytest.approx(values[1])
    assert stats["change"] == pytest.approx(raw_change)
    assert stats["raw_start_angle"] == pytest.approx(values[0])
    assert stats["raw_end_angle"] == pytest.approx(values[1])
    assert stats["angle_type"] == angle_type
    assert stats["canonical_signed_change"] == pytest.approx(canonical_change)
    assert stats["canonical_absolute_change"] == pytest.approx(abs(canonical_change))


def test_visualisation_registry_covers_every_metric() -> None:
    dynamic = _dynamic_df()
    path = _path_df()
    movement_story = {
        "phases": [
            {"phase_id": "phase_1", "phase_index": 1, "title": "Phase 1", "start_frame": 0, "end_frame": 4},
            {"phase_id": "phase_2", "phase_index": 2, "title": "Phase 2", "start_frame": 5, "end_frame": 9},
        ]
    }

    payload = build_metric_explorer_payload(
        dynamic_df=dynamic,
        path_df=path,
        movement_story=movement_story,
    )

    assert payload["selection_modes"] == [mode.value for mode in SelectionMode]
    assert payload["metrics"]
    assert "injured_hka_angle_2d_deg" in payload["angular_metric_names"]
    assert "dynamic_rate:injured_hka_angle_2d_deg" not in payload["angular_metric_names"]
    assert payload["metrics"]["injured_hka_angle_2d_deg"]["angular"] is True
    assert payload["metrics"]["injured_hka_angle_2d_deg"]["angle_type"] == "internal"
    assert payload["metrics"]["dynamic_rate:injured_hka_angle_2d_deg"]["angular"] is False
    assert payload["metrics"]["injured_hka_angle_2d_deg"]["display_label"] == "Injured HKA angle"
    assert "change" in payload["angular_heatmap"]
    bilateral_metrics = [
        item["metric_name"]
        for item in payload["categories"]["bilateral_limb_relationship"][:4]
    ]
    assert bilateral_metrics == [
        "injured_hka_angle_2d_deg",
        "contralateral_hka_angle_2d_deg",
        "hka_projected_bilateral_difference_deg",
        "hka_projected_bilateral_absolute_difference_deg",
    ]
    assert payload["categories"]["bilateral_limb_relationship"][2]["display_label"] == (
        "Injured-contralateral signed HKA difference"
    )
    for metric_name, spec in payload["metrics"].items():
        assert spec["preferred_visualisation"] or spec["no_visualisation_reason"], metric_name
        assert metric_name in payload["series"]
        assert metric_name in payload["whole_movement_statistics"]
        assert metric_name in payload["phase_statistics"]


def _metric_rows(values: list[float | None], statuses: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric_name": "synthetic_metric",
                "source_frame_index": frame,
                "analysis_frame_index": frame,
                "timestamp_ms": frame * 33.3,
                "movement_elapsed_ms": frame * 33.3,
                "movement_end_relative_ms": (frame - len(values) + 1) * 33.3,
                "value": value,
                "unit": "unit",
                "evidence_status": status,
                "quality_reason": "",
            }
            for frame, (value, status) in enumerate(zip(values, statuses, strict=True))
        ]
    )


def _dynamic_df() -> pd.DataFrame:
    rows = []
    for feature_name in (
        "injured_hka_angle_2d_deg",
        "contralateral_hka_angle_2d_deg",
        "hka_projected_bilateral_difference_deg",
        "hka_projected_bilateral_absolute_difference_deg",
    ):
        for frame in range(10):
            value = 100.0 + frame
            if feature_name == "contralateral_hka_angle_2d_deg":
                value = 96.0 + frame
            elif feature_name in {
                "hka_projected_bilateral_difference_deg",
                "hka_projected_bilateral_absolute_difference_deg",
            }:
                value = 4.0
            rows.append(
                {
                    "feature_name": feature_name,
                    "source_frame_index": frame,
                    "analysis_frame_index": frame,
                    "timestamp_ms": frame * 33.3,
                    "movement_elapsed_ms": frame * 33.3,
                    "movement_end_relative_ms": (frame - 9) * 33.3,
                    "feature_value": value,
                    "feature_status": "SUPPORTED",
                    "unit": "deg",
                    "rejection_reason": "",
                    "robust_dynamic_rate": float(frame),
                    "dynamic_status": "SUPPORTED",
                    "robust_dynamic_unit": "deg/s",
                    "dynamic_rejection_reason": "",
                }
            )
    return pd.DataFrame(rows)


def _path_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_frame_index": frame,
                "timestamp_ms": frame * 33.3,
                "movement_elapsed_ms": frame * 33.3,
                "movement_end_relative_ms": (frame - 9) * 33.3,
                "compensated_x": float(frame),
                "compensated_y": float(frame) / 2,
                "projected_heading_deg": float(frame * 3),
                "normalized_projected_speed_per_s": float(frame) / 10,
                "path_status": "SUPPORTED",
                "path_rejection_reason": "",
            }
            for frame in range(10)
        ]
    )
