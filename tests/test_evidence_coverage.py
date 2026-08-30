from __future__ import annotations

import numpy as np
import pandas as pd

from acl_motion.analytics.evidence_coverage import build_geometry_coverage_evidence


def test_geometry_coverage_keeps_window_and_reviewed_denominators_separate() -> None:
    dynamic = _dynamic(
        {
            "feature_a": [1.0, 2.0, np.nan, np.nan, np.nan, np.nan],
            "feature_b": [1.0, np.nan, np.nan, 4.0, 5.0, np.nan],
        }
    )
    quality = _quality(6, accepted={0, 1}, valid={0, 1, 3, 4})

    evidence = build_geometry_coverage_evidence(dynamic, quality)

    assert evidence["movement_window_coverage"] == 5 / 12
    assert evidence["reviewed_frame_geometry_yield"] == 3 / 4
    assert evidence["target_present_geometry_yield"] == 5 / 8
    feature_a = next(
        item for item in evidence["per_feature"] if item["feature_name"] == "feature_a"
    )
    assert feature_a["movement_window_coverage"] == 2 / 6
    assert feature_a["reviewed_frame_yield"] == 1.0
    assert feature_a["target_present_yield"] == 2 / 4


def test_sparse_review_cannot_masquerade_as_complete_window_coverage() -> None:
    values = [1.0, 2.0, 3.0] + [np.nan] * 17
    evidence = build_geometry_coverage_evidence(
        _dynamic({"feature_a": values}),
        _quality(20, accepted={0, 1, 2}, valid={0, 1, 2}),
    )

    assert evidence["reviewed_frame_geometry_yield"] == 1.0
    assert evidence["movement_window_coverage"] == 3 / 20
    assert evidence["best_continuous_feature"]["longest_continuous_supported"][
        "frame_count"
    ] == 3


def test_no_human_review_reports_unavailable_reviewed_yield() -> None:
    evidence = build_geometry_coverage_evidence(
        _dynamic({"feature_a": [1.0, 2.0, np.nan]}),
        _quality(3, accepted=set(), valid={0, 1}),
    )

    assert evidence["reviewed_frame_geometry_yield"] is None
    assert evidence["best_reviewed_feature"] is None


def _dynamic(features: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for feature_name, values in features.items():
        for frame, value in enumerate(values):
            rows.append(
                {
                    "source_frame_index": frame,
                    "timestamp_ms": frame * 100.0,
                    "feature_name": feature_name,
                    "feature_value": value,
                    "feature_status": "SUPPORTED" if pd.notna(value) else "LOW_CONFIDENCE",
                }
            )
    return pd.DataFrame(rows)


def _quality(
    frame_count: int,
    *,
    accepted: set[int],
    valid: set[int],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_frame_index": frame,
                "frame_status": "VALID_TARGET" if frame in valid else "TARGET_NOT_FOUND",
                "valid_target_frame": frame in valid,
                "human_target_accepted": frame in accepted,
                "human_target_unavailable": False,
                "observed_landmark_count": 17 if frame in valid else 0,
            }
            for frame in range(frame_count)
        ]
    )
