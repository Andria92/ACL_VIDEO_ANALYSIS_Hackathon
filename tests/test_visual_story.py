from __future__ import annotations

import pandas as pd

from acl_motion.semantics.visual_story import build_movement_visual_story


def test_salience_prefers_supported_distinct_movement_families() -> None:
    story = build_movement_visual_story(
        movement_story=_movement_story(),
        metric_explorer=_metric_explorer(),
        processed_pose=_pose(),
        laterality_mapping={"injured": "right", "contralateral": "left"},
    )

    phase_two = story["phases"][1]
    categories = [item["category"] for item in phase_two["observations"]]

    assert categories[0] == "movement_path"
    assert "bilateral_limb_relationship" in categories
    assert "upper_body" not in categories
    assert all("score_components" in item for item in phase_two["observations"])
    assert "other_observations" in phase_two
    assert all("support" in item for item in phase_two["observations"])


def test_phase_snapshots_include_intermediate_temporal_positions() -> None:
    story = build_movement_visual_story(
        movement_story=_movement_story(),
        metric_explorer=_metric_explorer(),
        processed_pose=_pose(),
        laterality_mapping={"injured": "right", "contralateral": "left"},
    )

    phase_two = story["phases"][1]
    labels = [item["label"] for item in phase_two["snapshot_frames"]]

    assert labels[0] == "Phase start"
    assert labels[-1] == "Phase end"
    assert "Mid-phase" in labels
    assert len(labels) == 3


def test_snapshot_strip_uses_nearest_supported_pose_frame() -> None:
    story = build_movement_visual_story(
        movement_story=_movement_story(),
        metric_explorer=_metric_explorer(),
        processed_pose=_pose(frame_status_by_source={6: "TARGET_IDENTITY_UNCERTAIN"}),
    )

    phase_two = story["phases"][1]
    first_snapshot = phase_two["snapshot_frames"][0]

    assert first_snapshot["requested_source_frame_index"] == 6
    assert first_snapshot["source_frame_index"] == 7
    assert first_snapshot["selected_nearest_supported"] is True


def test_technical_feature_count_does_not_make_unsupported_family_salient() -> None:
    movement_story = _movement_story()
    unsupported = {
        f"synthetic_upper_metric_{index}": {
            "start_value": 0,
            "end_value": 100,
            "change": 100,
        }
        for index in range(30)
    }
    movement_story["phases"][1]["category_summaries"]["upper_body"] = {
        "evidence_status": "UNAVAILABLE",
        "summary": "Unsupported upper-body values.",
        "metrics": unsupported,
    }

    story = build_movement_visual_story(
        movement_story=movement_story,
        metric_explorer=_metric_explorer(),
        processed_pose=_pose(),
    )

    categories = [item["category"] for item in story["phases"][1]["observations"]]
    assert "upper_body" not in categories


def _movement_story() -> dict:
    return {
        "sequence_summary": "Synthetic observable sequence.",
        "metadata": {"movement_window": {"movement_start_frame": 0, "movement_end_frame": 10, "duration_ms": 333.0}},
        "transitions": [
            {
                "to_phase_id": "phase_2",
                "dominant_feature_families": ["movement_path", "bilateral_limb_relationship"],
                "feature_family_contributions": {
                    "movement_path": {"fraction": 0.65},
                    "bilateral_limb_relationship": {"fraction": 0.25},
                    "upper_body": {"fraction": 0.10},
                },
            }
        ],
        "phases": [
            {
                "phase_id": "phase_1",
                "phase_index": 1,
                "title": "Opening",
                "start_frame": 0,
                "end_frame": 5,
                "duration_ms": 166.0,
                "category_summaries": {
                    "movement_path": {
                        "evidence_status": "GOOD",
                        "summary": "Small path change.",
                        "metrics": {"heading_change_deg": 4.0, "speed_change_normalized_per_s": 0.1},
                    },
                    "bilateral_limb_relationship": {
                        "evidence_status": "GOOD",
                        "summary": "Stable bilateral relation.",
                        "metrics": {
                            "signed_difference_start_deg": 1.0,
                            "signed_difference_end_deg": 2.0,
                            "signed_difference_change_deg": 1.0,
                            "absolute_difference_change_deg": 1.0,
                            "maximum_absolute_hka_difference_deg": 2.0,
                        },
                    },
                    "upper_body": {
                        "evidence_status": "GOOD",
                        "summary": "Small upper-body change.",
                        "metrics": {
                            "left_elbow_angle_2d_deg": {
                                "start_value": 10.0,
                                "end_value": 11.0,
                                "change": 1.0,
                            }
                        },
                    },
                },
            },
            {
                "phase_id": "phase_2",
                "phase_index": 2,
                "title": "Directional change",
                "start_frame": 6,
                "end_frame": 10,
                "duration_ms": 166.0,
                "evidence_summary": {"evidence_status": "GOOD"},
                "category_summaries": {
                    "movement_path": {
                        "evidence_status": "GOOD",
                        "summary": "Large path change.",
                        "metrics": {"heading_change_deg": 90.0, "speed_change_normalized_per_s": 2.0},
                    },
                    "bilateral_limb_relationship": {
                        "evidence_status": "GOOD",
                        "summary": "Bilateral relation changed.",
                        "metrics": {
                            "signed_difference_start_deg": 1.0,
                            "signed_difference_end_deg": 25.0,
                            "signed_difference_change_deg": 24.0,
                            "absolute_difference_change_deg": 24.0,
                            "maximum_absolute_hka_difference_deg": 25.0,
                        },
                    },
                    "upper_body": {
                        "evidence_status": "GOOD",
                        "summary": "Weak upper-body change.",
                        "metrics": {
                            "left_elbow_angle_2d_deg": {
                                "start_value": 10.0,
                                "end_value": 11.0,
                                "change": 1.0,
                            }
                        },
                    },
                },
            },
        ],
    }


def _metric_explorer() -> dict:
    rows = [
        {
            "source_frame_index": frame,
            "movement_end_relative_ms": (frame - 10) * 33.3,
            "value": float(frame),
            "evidence_status": "SUPPORTED",
        }
        for frame in range(11)
    ]
    return {
        "series": {
            "path:compensated_x": rows,
            "path:compensated_y": [
                {**row, "value": float(row["source_frame_index"]) / 2} for row in rows
            ],
            "path:projected_heading_deg": [
                {**row, "value": float(row["source_frame_index"] * 8)} for row in rows
            ],
            "path:normalized_projected_speed_per_s": rows,
            "hka_projected_bilateral_difference_deg": [
                {**row, "value": 2.0 if row["source_frame_index"] < 8 else 24.0}
                for row in rows
            ],
            "left_elbow_angle_2d_deg": [
                {**row, "value": 10.0 if row["source_frame_index"] < 9 else 30.0}
                for row in rows
            ],
        }
    }


def _pose(frame_status_by_source: dict[int, str] | None = None) -> pd.DataFrame:
    frame_status_by_source = frame_status_by_source or {}
    landmarks = [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "right_knee",
        "right_ankle",
    ]
    rows = []
    for frame in range(11):
        for index, name in enumerate(landmarks):
            rows.append(
                {
                    "source_frame_index": frame,
                    "landmark_name": name,
                    "smoothed_x": float(frame + index),
                    "smoothed_y": float(frame + index + 1),
                    "clean_x": float(frame + index),
                    "clean_y": float(frame + index + 1),
                    "rejected": False,
                    "frame_status": frame_status_by_source.get(frame, "VALID_TARGET"),
                    "landmark_status": "OBSERVED_VALID",
                    "interpolated": False,
                    "smoothed": True,
                }
            )
    return pd.DataFrame(rows)
