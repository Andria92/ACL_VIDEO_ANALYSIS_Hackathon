from __future__ import annotations

import json

import pytest

from acl_motion.annotations.event_interval_review import (
    EVENT_INTERVAL_REVIEW_QUESTION,
    load_event_interval_review,
    save_event_interval_review,
)
from acl_motion.annotations.registry import default_annotation_cases


def _write_supported_phases(tmp_path, slug: str, *, end_frame: int = 20) -> None:
    path = tmp_path / "phases" / "human" / f"{slug}_movement_phases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "SUPPORTED_PARTIAL_WINDOW",
                "phases": [
                    {
                        "phase_id": "phase_1",
                        "start_frame": 4,
                        "end_frame": end_frame,
                    }
                ],
                "metadata": {
                    "analysis_scope": {
                        "type": "PARTIAL_MOVEMENT_WINDOW",
                        "start_frame": 4,
                        "end_frame": end_frame,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_event_interval_review_is_yes_or_no_and_persists_per_view(tmp_path) -> None:
    case = default_annotation_cases()[0]
    _write_supported_phases(tmp_path, case.slug)

    default_review = load_event_interval_review(case, data_root=tmp_path)
    assert default_review["question"] == EVENT_INTERVAL_REVIEW_QUESTION
    assert default_review["decision"] == "yes"
    assert default_review["review_status"] == "DEFAULT_YES"
    assert default_review["eligible_for_injury_event_comparison"] is True

    saved = save_event_interval_review(
        case,
        decision="yes",
        reviewer_id="researcher_01",
        data_root=tmp_path,
    )
    assert saved["decision"] == "yes"
    assert saved["visible_event_in_supported_phase_interval"] is True
    assert saved["eligible_for_injury_event_comparison"] is True
    assert saved["decision_source"] == "human_operator"

    loaded = load_event_interval_review(case, data_root=tmp_path)
    assert loaded["decision"] == "yes"
    assert loaded["review_status"] == "REVIEWED"


def test_event_interval_review_rejects_any_third_answer(tmp_path) -> None:
    case = default_annotation_cases()[0]
    _write_supported_phases(tmp_path, case.slug)

    with pytest.raises(ValueError, match="must be yes or no"):
        save_event_interval_review(case, decision="unclear", data_root=tmp_path)


def test_event_interval_review_requires_new_answer_after_regeneration(tmp_path) -> None:
    case = default_annotation_cases()[0]
    _write_supported_phases(tmp_path, case.slug, end_frame=20)
    save_event_interval_review(case, decision="no", data_root=tmp_path)

    _write_supported_phases(tmp_path, case.slug, end_frame=24)
    loaded = load_event_interval_review(case, data_root=tmp_path)

    assert loaded["decision"] is None
    assert loaded["previous_decision"] == "no"
    assert loaded["review_status"] == "REVIEW_REQUIRED_AFTER_REGENERATION"
    assert loaded["eligible_for_injury_event_comparison"] is False
