from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from acl_motion.semantics.bilateral import (
    absolute_difference,
    compute_bilateral_hka_summary,
    signed_difference,
)
from acl_motion.semantics.models import MovementObservation
from acl_motion.semantics.path import (
    PathAnalysisConfig,
    compensate_projected_path,
    direction_change_summary,
    enforce_path_validation,
    validate_projected_path,
)


def test_bilateral_signed_and_absolute_relationship() -> None:
    assert signed_difference(120.0, 100.0) == 20.0
    assert signed_difference(90.0, 120.0) == -30.0
    assert absolute_difference(90.0, 120.0) == 30.0


def test_bilateral_peak_timing_and_final_window_summary() -> None:
    dynamic_df = _bilateral_df([-5.0, 8.0, 18.0, -22.0, -30.0])

    summary = compute_bilateral_hka_summary(dynamic_df)

    assert summary.evidence_status == "SUPPORTED"
    assert summary.peak_absolute_hka_bilateral_difference_deg == 30.0
    assert summary.source_frame_peak_absolute_hka_bilateral_difference == 4
    assert summary.time_peak_absolute_hka_bilateral_difference_ms == 0.0
    final_500 = next(item for item in summary.window_summaries if item.window_name == "final_500ms")
    assert final_500.evidence_status == "SUPPORTED"
    assert final_500.mean_absolute_difference_deg == pytest.approx((8 + 18 + 22 + 30) / 4)
    assert final_500.signed_change_deg == -38.0


def test_missing_bilateral_limb_returns_unavailable() -> None:
    dynamic_df = _bilateral_df([1.0, 2.0])
    dynamic_df = dynamic_df[dynamic_df["feature_name"].eq("hka_projected_bilateral_difference_deg")]

    summary = compute_bilateral_hka_summary(dynamic_df)

    assert summary.evidence_status == "UNAVAILABLE"
    assert summary.mean_absolute_hka_bilateral_difference_deg is None


def test_pure_camera_pan_compensates_to_zero_target_motion() -> None:
    centers = _centers([(0, 0), (5, 0), (10, 0)])
    camera = _camera_motion([0, 5, 5])

    path = compensate_projected_path(
        centers,
        camera,
        scale_reference_px=1.0,
        config=PathAnalysisConfig(minimum_path_samples=2),
    )

    supported = path[path["path_status"].eq("SUPPORTED")]
    assert supported["compensated_x"].iloc[-1] == pytest.approx(0.0)
    assert supported["compensated_y"].iloc[-1] == pytest.approx(0.0)


def test_target_movement_plus_camera_pan_preserves_target_component() -> None:
    centers = _centers([(0, 0), (6, 0), (12, 0)])
    camera = _camera_motion([0, 5, 5])

    path = compensate_projected_path(
        centers,
        camera,
        scale_reference_px=1.0,
        config=PathAnalysisConfig(minimum_path_samples=2),
    )

    assert path[path["path_status"].eq("SUPPORTED")]["compensated_x"].iloc[-1] == pytest.approx(2.0)


def test_direction_change_from_synthetic_path() -> None:
    centers = _centers([(0, 0), (1, 0), (2, 0), (2, -1), (2, -2), (2, -3)])
    camera = _camera_motion([0, 0, 0, 0, 0, 0])
    path = compensate_projected_path(
        centers,
        camera,
        scale_reference_px=1.0,
        config=PathAnalysisConfig(minimum_path_samples=3),
    )

    summary = direction_change_summary(
        path,
        config=PathAnalysisConfig(
            minimum_path_samples=3,
            direction_window_fraction=0.4,
            minimum_direction_window_samples=2,
        ),
    )

    assert summary["evidence_status"] == "SUPPORTED"
    assert abs(summary["projected_change_of_direction_angle_deg"]) >= 80.0


def test_camera_compensation_failure_returns_unavailable_path_status() -> None:
    centers = _centers([(0, 0), (5, 0), (10, 0)])
    camera = _camera_motion([0, 5, 5], status="LOW_BACKGROUND_FEATURE_COUNT")

    path = compensate_projected_path(
        centers,
        camera,
        scale_reference_px=1.0,
        config=PathAnalysisConfig(minimum_path_samples=2),
    )

    assert "LOW_BACKGROUND_FEATURE_COUNT" in set(path["path_status"])
    assert not path["path_status"].eq("SUPPORTED").any()


def test_invalid_frame_gap_starts_new_path_segment_without_bridge() -> None:
    centers = _centers([(0, 0), (1, 0), (4, 0), (5, 0)])
    centers.loc[2:, "source_frame_index"] += 1
    camera = _camera_motion([0, 0, 0, 0, 0])

    path = compensate_projected_path(
        centers,
        camera,
        scale_reference_px=1.0,
        config=PathAnalysisConfig(minimum_path_samples=2),
    )

    supported = path[path["path_status"].eq("SUPPORTED")]
    assert supported["path_segment_id"].nunique() == 2
    second_start = supported[supported["path_segment_id"].eq("path_segment_002")].iloc[0]
    assert pd.isna(second_start["compensated_dx"])


def test_affine_camera_motion_recovers_target_motion_under_zoom() -> None:
    centers = _centers([(10, 0), (23, 0)])
    translation_camera = _camera_motion([0, 0])
    affine_camera = _camera_motion([0, 0])
    affine_camera.loc[1, ["camera_motion_method", "affine_a", "affine_d"]] = [
        "partial_affine_ransac_sparse_flow",
        2.0,
        2.0,
    ]

    translation_path = compensate_projected_path(
        centers,
        translation_camera,
        scale_reference_px=1.0,
        config=PathAnalysisConfig(minimum_path_samples=2),
    )
    affine_path = compensate_projected_path(
        centers,
        affine_camera,
        scale_reference_px=1.0,
        config=PathAnalysisConfig(minimum_path_samples=2),
    )

    assert translation_path["compensated_x"].iloc[-1] == pytest.approx(13.0)
    assert affine_path["compensated_x"].iloc[-1] == pytest.approx(3.0)


def test_path_validation_can_withhold_wrong_player_jump() -> None:
    path = compensate_projected_path(
        _centers([(0, 0), (1, 0), (2, 0), (80, 0), (81, 0), (82, 0)]),
        _camera_motion([0, 0, 0, 0, 0, 0]),
        scale_reference_px=1.0,
        config=PathAnalysisConfig(minimum_path_samples=3),
    )

    validation = validate_projected_path(
        path,
        config=PathAnalysisConfig(minimum_path_samples=3, maximum_step_to_median_ratio=4.0),
    )
    enforced = enforce_path_validation(path, validation)

    assert validation["validation_status"] == "QA_REQUIRED"
    assert not enforced["path_status"].eq("SUPPORTED").any()


def test_semantic_outputs_remain_task_neutral() -> None:
    payload = json.loads(
        Path("data/semantic/human/christen_press_movement_observations.json").read_text()
    )
    text = json.dumps(payload).lower()

    for forbidden in ("abnormal", "pathological", "dangerous", "risky"):
        assert forbidden not in text


def test_movement_observation_rejects_non_neutral_terms() -> None:
    with pytest.raises(ValueError):
        MovementObservation(
            observation_id="x",
            case_id="case",
            category="evidence",
            title="Dangerous movement",
            plain_language_summary="not allowed",
        )


def _bilateral_df(signed_values: list[float]) -> pd.DataFrame:
    rows = []
    times = [-1000.0, -500.0, -250.0, -100.0, 0.0]
    for frame, (time, value) in enumerate(zip(times, signed_values, strict=False)):
        for feature_name, feature_value in (
            ("hka_projected_bilateral_difference_deg", value),
            ("hka_projected_bilateral_absolute_difference_deg", abs(value)),
        ):
            rows.append(
                {
                    "feature_name": feature_name,
                    "feature_status": "SUPPORTED",
                    "feature_value": feature_value,
                    "source_frame_index": frame,
                    "movement_end_relative_ms": time,
                }
            )
    return pd.DataFrame(rows)


def _centers(points: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_frame_index": i,
                "timestamp_ms": i * 100.0,
                "movement_elapsed_ms": i * 100.0,
                "movement_end_relative_ms": (i - len(points) + 1) * 100.0,
                "center_x": x,
                "center_y": y,
                "center_source": "pelvis_midpoint",
                "center_status": "SUPPORTED",
            }
            for i, (x, y) in enumerate(points)
        ]
    )


def _camera_motion(dx_values: list[float], status: str = "SUPPORTED") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_frame_index": i,
                "background_dx_px": dx,
                "background_dy_px": 0.0,
                "background_feature_count": 100 if status == "SUPPORTED" else 0,
                "camera_motion_status": "INSUFFICIENT_EVIDENCE" if i == 0 else status,
                "camera_motion_residual_px": 0.0,
                "camera_motion_method": "translation_median_sparse_flow",
                "affine_a": 1.0,
                "affine_b": 0.0,
                "affine_tx": dx,
                "affine_c": 0.0,
                "affine_d": 1.0,
                "affine_ty": 0.0,
            }
            for i, dx in enumerate(dx_values)
        ]
    )
