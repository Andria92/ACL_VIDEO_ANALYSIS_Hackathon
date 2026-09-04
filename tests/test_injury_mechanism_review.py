from __future__ import annotations

import json
from dataclasses import replace

import pytest

from acl_motion.annotations.injury_mechanism_review import (
    INJURY_MECHANISM_REVIEW_QUESTION,
    load_injury_mechanism_review,
    save_injury_mechanism_review,
)
from acl_motion.annotations.registry import default_annotation_cases


def test_existing_research_label_is_visible_but_still_requires_confirmation(
    tmp_path,
) -> None:
    case = default_annotation_cases()[0]
    path = tmp_path / "annotations" / "human" / "injury_report_sources.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cases": {
                    case.case_id: {
                        "classification": "indirect_contact",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    review = load_injury_mechanism_review(case, data_root=tmp_path)

    assert review["question"] == INJURY_MECHANISM_REVIEW_QUESTION
    assert review["decision"] == "indirect_contact"
    assert review["decision_label"] == "Indirect contact"
    assert review["review_status"] == "EXISTING_RESEARCH_LABEL"
    assert review["review_required"] is True


def test_injury_mechanism_review_persists_once_for_every_view_of_a_case(tmp_path) -> None:
    case = default_annotation_cases()[0]
    second_view = replace(
        case,
        slug=f"{case.slug}_second_view",
        source_id=f"{case.source_id}_second_view",
        view_id=f"{case.source_id}_second_view",
    )

    saved = save_injury_mechanism_review(
        case,
        decision="contact",
        reviewer_id="reviewer_01",
        data_root=tmp_path,
    )
    loaded = load_injury_mechanism_review(second_view, data_root=tmp_path)

    assert saved["decision"] == "direct_contact"
    assert saved["decision_label"] == "Contact"
    assert loaded["decision"] == "direct_contact"
    assert loaded["review_status"] == "REVIEWED"
    assert loaded["shared_across_case_views"] is True


def test_injury_mechanism_review_accepts_only_the_three_displayed_choices(
    tmp_path,
) -> None:
    case = default_annotation_cases()[0]

    with pytest.raises(ValueError, match="contact, non-contact, or indirect contact"):
        save_injury_mechanism_review(
            case,
            decision="unclear",
            data_root=tmp_path,
        )
