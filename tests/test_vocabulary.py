from __future__ import annotations

import json

import pandas as pd

from acl_motion.semantics.vocabulary import (
    MovementVocabularyConfig,
    build_controlled_movement_vocabulary,
    build_observable_movement_description_payload,
)


def test_supported_hka_change_produces_lower_limb_description() -> None:
    payload = _payload(_dynamic_df({"injured_hka_angle_2d_deg": [100, 106, 112, 118, 124]}))

    ids = _description_ids(payload)

    assert "LOWER_LIMB_CONFIGURATION_CHANGED" in ids
    assert "INJURED_HKA_INCREASED" in ids
    description = _description(payload, "LOWER_LIMB_CONFIGURATION_CHANGED")
    assert description["supporting_features"]
    assert description["source_frames"]
    assert description["supporting_values"]["injured_hka_angle_2d_deg"]["change"] == 24.0


def test_unsupported_hka_does_not_produce_descriptor() -> None:
    df = _dynamic_df({"injured_hka_angle_2d_deg": [100, 106, 112, 118, 124]})
    df.loc[df["feature_name"].eq("injured_hka_angle_2d_deg"), "feature_status"] = "LOW_CONFIDENCE"

    payload = _payload(df)

    assert "LOWER_LIMB_CONFIGURATION_CHANGED" not in _description_ids(payload)


def test_bilateral_crossing_and_difference_increase_are_neutral() -> None:
    payload = _payload(
        _dynamic_df(
                {
                "hka_projected_bilateral_difference_deg": [-1, -0.5, 0.5, 5, 8],
                "hka_projected_bilateral_absolute_difference_deg": [1, 0.5, 0.5, 5, 8],
            }
        )
    )

    text = json.dumps(payload["descriptions"]).lower()

    assert "BILATERAL_RELATIONSHIP_CROSSING" in _description_ids(payload)
    assert "BILATERAL_DIFFERENCE_INCREASED" in _description_ids(payload)
    assert "abnormal" not in text
    assert "risk" not in text


def test_missing_contralateral_bilateral_limb_prevents_bilateral_descriptor() -> None:
    payload = _payload(
        _dynamic_df(
            {
                "hka_projected_bilateral_difference_deg": [-5, -2, 2, 5, 8],
                "hka_projected_bilateral_absolute_difference_deg": [5, None, None, None, None],
            }
        )
    )

    ids = _description_ids(payload)

    assert "BILATERAL_RELATIONSHIP_CROSSING" not in ids
    assert "BILATERAL_DIFFERENCE_INCREASED" not in ids


def test_trunk_orientation_shift_produces_descriptor() -> None:
    payload = _payload(
        _dynamic_df({"projected_trunk_axis_angle_deg": [10, 16, 21, 27, 33]})
    )

    assert "TRUNK_ORIENTATION_SHIFT" in _description_ids(payload)


def test_path_qa_required_withholds_path_descriptors() -> None:
    payload = _payload(_dynamic_df({"injured_hka_angle_2d_deg": [100, 106, 112, 118, 124]}))

    withheld_ids = {item["descriptor_id"] for item in payload["withheld_descriptions"]}

    assert "PROJECTED_DIRECTION_CHANGE" in withheld_ids
    assert "PROJECTED_SLOWDOWN" in withheld_ids
    assert all(item["evidence_status"] == "WITHHELD" for item in payload["withheld_descriptions"])


def test_supported_interval_is_distinct_from_annotated_movement_end() -> None:
    frame_quality = _frame_quality()
    frame_quality.loc[frame_quality["source_frame_index"].isin([3, 4]), "frame_status"] = [
        "TARGET_IDENTITY_UNCERTAIN",
        "INVALID_TRACK_SEGMENT",
    ]

    payload = _payload(
        _dynamic_df({"injured_hka_angle_2d_deg": [100, 106, 112, 118, 124]}),
        frame_quality=frame_quality,
        config=MovementVocabularyConfig(
            minimum_interval_frames=2,
            minimum_supported_samples=2,
        ),
    )

    coverage = payload["clip_evidence_coverage"]

    assert payload["supported_intervals"][0]["end_frame"] == 2
    assert coverage["clip_end_frame"] == 4
    assert coverage["annotated_movement_end_frame"] == 4
    assert coverage["last_supported_source_frame"] == 2
    assert coverage["has_frames_after_supported_interval"] is True
    assert coverage["supported_interval_reaches_annotated_movement_end"] is False
    assert coverage["post_supported_status_counts"] == {
        "TARGET_IDENTITY_UNCERTAIN": 1,
        "INVALID_TRACK_SEGMENT": 1,
    }


def test_one_frame_spike_does_not_trigger_rapid_language() -> None:
    df = _dynamic_df({"injured_hka_angle_2d_deg": [100, 106, 112, 118, 124]})
    mask = df["feature_name"].eq("injured_hka_angle_2d_deg")
    df.loc[mask, "robust_dynamic_rate"] = [0, 0, 500, 0, 0]

    payload = _payload(df, config=MovementVocabularyConfig(rapid_rate_deg_per_s=100))

    assert "RAPID_LOWER_LIMB_CONFIGURATION_CHANGE" not in _description_ids(payload)


def test_sustained_supported_change_can_trigger_rapid_language() -> None:
    df = _dynamic_df({"injured_hka_angle_2d_deg": [100, 106, 112, 118, 124]})
    mask = df["feature_name"].eq("injured_hka_angle_2d_deg")
    df.loc[mask, "robust_dynamic_rate"] = [0, 120, 130, 140, 0]

    payload = _payload(
        df,
        config=MovementVocabularyConfig(
            minimum_interval_frames=2,
            minimum_supported_samples=2,
            rapid_rate_deg_per_s=100,
            minimum_dynamic_supported_fraction=0.40,
            rapid_min_consecutive_samples=3,
        ),
    )

    assert "RAPID_LOWER_LIMB_CONFIGURATION_CHANGE" in _description_ids(payload)


def test_weak_dynamic_evidence_cannot_trigger_rapid_language() -> None:
    df = _dynamic_df({"injured_hka_angle_2d_deg": [100, 106, 112, 118, 124]})
    mask = df["feature_name"].eq("injured_hka_angle_2d_deg")
    df.loc[mask, "robust_dynamic_rate"] = [120, 130, 140, 150, 160]
    df.loc[mask, "dynamic_status"] = ["SUPPORTED", "MISSING_FEATURE", "MISSING_FEATURE", "MISSING_FEATURE", "SUPPORTED"]

    payload = _payload(df, config=MovementVocabularyConfig(rapid_rate_deg_per_s=100))

    assert "RAPID_LOWER_LIMB_CONFIGURATION_CHANGE" not in _description_ids(payload)


def test_invalid_temporal_gap_prevents_rate_bridge() -> None:
    df = _dynamic_df({"injured_hka_angle_2d_deg": [100, 106, 112, 118, 124]})
    frame_quality = _frame_quality()
    frame_quality.loc[frame_quality["source_frame_index"].eq(2), "frame_status"] = "TARGET_IDENTITY_UNCERTAIN"
    frame_quality["valid_segment_id"] = [1, 1, None, 2, 2]
    mask = df["feature_name"].eq("injured_hka_angle_2d_deg")
    df.loc[mask, "robust_dynamic_rate"] = [120, 130, 140, 150, 160]

    payload = _payload(
        df,
        frame_quality=frame_quality,
        config=MovementVocabularyConfig(
            minimum_interval_frames=2,
            minimum_supported_samples=2,
            rapid_rate_deg_per_s=100,
            rapid_min_consecutive_samples=3,
        ),
    )

    assert "RAPID_LOWER_LIMB_CONFIGURATION_CHANGE" not in _description_ids(payload)


def test_story_salience_caps_families_and_omits_unavailable() -> None:
    payload = _payload(
        _dynamic_df(
            {
                "injured_hka_angle_2d_deg": [100, 106, 112, 118, 124],
                "hka_projected_bilateral_difference_deg": [-5, -2, 2, 5, 8],
                "hka_projected_bilateral_absolute_difference_deg": [5, 2, 2, 5, 8],
                "right_upper_arm_orientation_2d_deg": [10, 30, 50, 70, 90],
                "right_elbow_angle_2d_deg": [80, 90, 100, 110, 120],
                "left_elbow_angle_2d_deg": [90, 100, 110, 120, 130],
            }
        )
    )

    story = payload["default_story_descriptions"]
    families = [item["family"] for item in story]

    assert 2 <= len(story) <= 4
    assert len(families) == len(set(families))
    assert all(item["evidence_status"] != "UNAVAILABLE" for item in story)
    assert "BILATERAL LOWER-LIMB RELATIONSHIP" in families


def test_vocabulary_registry_forbids_clinical_labels() -> None:
    text = json.dumps([item.to_dict() for item in build_controlled_movement_vocabulary()]).lower()

    for forbidden in ("dangerous movement", "injury-causing movement", "valgus collapse"):
        assert forbidden in text
    assert "archetype" not in text


def _payload(
    dynamic_df: pd.DataFrame,
    *,
    frame_quality: pd.DataFrame | None = None,
    config: MovementVocabularyConfig | None = None,
) -> dict:
    return build_observable_movement_description_payload(
        case_id="case",
        source_id="source",
        dynamic_df=dynamic_df,
        frame_quality=frame_quality if frame_quality is not None else _frame_quality(),
        path_summary={"overall_status": "QA_REQUIRED", "reason": "test path withheld"},
        movement_window={
            "movement_start_frame": 0,
            "movement_end_frame": 4,
            "movement_duration_ms": 400.0,
        },
        phase_status="INSUFFICIENT_EVIDENCE_FOR_PHASE_SEGMENTATION",
        config=config
        or MovementVocabularyConfig(
            minimum_interval_frames=2,
            minimum_supported_samples=2,
        ),
    )


def _description_ids(payload: dict) -> set[str]:
    return {item["descriptor_id"] for item in payload["descriptions"]}


def _description(payload: dict, descriptor_id: str) -> dict:
    return next(item for item in payload["descriptions"] if item["descriptor_id"] == descriptor_id)


def _frame_quality() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_frame_index": [0, 1, 2, 3, 4],
            "timestamp_ms": [0.0, 100.0, 200.0, 300.0, 400.0],
            "frame_status": ["VALID_TARGET"] * 5,
            "valid_segment_id": [1] * 5,
        }
    )


def _dynamic_df(values_by_feature: dict[str, list[float | None]]) -> pd.DataFrame:
    rows = []
    frames = [0, 1, 2, 3, 4]
    for feature_name, values in values_by_feature.items():
        for frame, value in zip(frames, values, strict=True):
            supported = value is not None
            rows.append(
                {
                    "case_id": "case",
                    "source_id": "source",
                    "feature_name": feature_name,
                    "source_frame_index": frame,
                    "timestamp_ms": frame * 100.0,
                    "movement_elapsed_ms": frame * 100.0,
                    "movement_end_relative_ms": (frame - 4) * 100.0,
                    "feature_value": value,
                    "unit": "deg",
                    "feature_status": "SUPPORTED" if supported else "INSUFFICIENT_LANDMARKS",
                    "dynamic_status": "SUPPORTED" if supported else "MISSING_FEATURE",
                    "robust_dynamic_rate": 0.0 if supported else None,
                }
            )
    return pd.DataFrame(rows)
