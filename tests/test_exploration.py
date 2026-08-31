from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from acl_motion.analytics.exploration import (
    FEATURE_PLAIN_LANGUAGE,
    assess_group_test_eligibility,
    load_cached_exploration_summary_payload,
    load_exploration_payload,
    load_exploration_summary_payload,
)
from acl_motion.annotations.models import AnnotationCase
from acl_motion.annotations.research_metadata import (
    case_details,
    load_research_metadata,
    save_case_details,
)
from acl_motion.cases.models import InjurySide
from acl_motion.ui.exploration import render_exploration_page


def test_explorer_keeps_charts_readable_and_technical_ids_progressive() -> None:
    html = render_exploration_page()

    assert 'id="distributionCanvas" class="tall-chart"' in html
    assert 'id="relationshipLegend"' in html
    assert "context.fillText(record.player_name" in html
    assert "context.fillText(shortCaseLabel(record.player_name)" not in html
    assert "function evidenceReasonLabel(value)" in html
    assert "The athlete could not be detected reliably in this view." in html
    assert '<tbody id="technicalIdentifierRows">' in html
    assert "Source evidence:" in html


def test_every_current_exploration_measurement_has_plain_language_help() -> None:
    assert len(FEATURE_PLAIN_LANGUAGE) == 36
    assert all(len(description.split()) >= 8 for description in FEATURE_PLAIN_LANGUAGE.values())
    assert "three-dimensional" in FEATURE_PLAIN_LANGUAGE["projected_hip_line_angle_deg"]


def test_current_case_registry_uses_canonical_player_names() -> None:
    root = Path(__file__).resolve().parents[1]
    imported = json.loads(
        (root / "data/annotations/human/imported_video_cases_human.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = json.loads(
        (root / "data/annotations/human/case_research_metadata_human.json").read_text(
            encoding="utf-8"
        )
    )

    imported_names = {case["case_id"]: case["player_name"] for case in imported["cases"]}
    metadata_names = {
        case_id: details["player_name"]
        for case_id, details in metadata["cases"].items()
    }
    assert imported_names["imported_holy_mcnamara_2023_11_19_acl_candidate"] == "Holly McNamara"
    assert imported_names["imported_caroline_wier_2023_09_26_acl_candidate"] == "Caroline Weir"
    assert metadata_names["imported_holy_mcnamara_2023_11_19_acl_candidate"] == "Holly McNamara"
    assert metadata_names["imported_caroline_wier_2023_09_26_acl_candidate"] == "Caroline Weir"


def test_exploration_counts_events_not_camera_views(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summaries"
    summary_dir.mkdir()
    cases = (
        _case("case_a_view_1", "case_a", "source_a_1", "Player A", primary=True),
        _case("case_a_view_2", "case_a", "source_a_2", "Player A", primary=False),
        _case("case_b_view_1", "case_b", "source_b_1", "Player B", primary=True),
    )
    _write_summary(
        summary_dir / "a1_case_feature_summary.parquet",
        case_id="case_a",
        source_id="source_a_1",
        geometry_completeness=0.40,
        geometry_eligible=False,
        mean=120.0,
    )
    _write_summary(
        summary_dir / "a2_case_feature_summary.parquet",
        case_id="case_a",
        source_id="source_a_2",
        geometry_completeness=0.90,
        geometry_eligible=True,
        mean=140.0,
    )
    _write_summary(
        summary_dir / "b1_case_feature_summary.parquet",
        case_id="case_b",
        source_id="source_b_1",
        geometry_completeness=0.80,
        geometry_eligible=True,
        mean=130.0,
    )

    payload = load_exploration_payload(
        cases,
        summary_dir=summary_dir,
        signature_dir=tmp_path / "signatures",
    )
    compact = load_exploration_summary_payload(
        cases,
        summary_dir=summary_dir,
        signature_dir=tmp_path / "signatures",
    )
    cached = load_cached_exploration_summary_payload(
        cases,
        cache_path=tmp_path / "home-summary-cache.json",
        summary_dir=summary_dir,
        signature_dir=tmp_path / "signatures",
    )

    assert payload["analysis_unit"] == "registered_injury_case"
    assert set(compact) == {"summary", "similarity"}
    assert compact["summary"] == payload["summary"]
    assert compact["similarity"] == payload["similarity"]
    assert cached == compact
    assert (tmp_path / "home-summary-cache.json").exists()
    assert payload["summary"]["analysed_case_count"] == 2
    assert payload["summary"]["analysed_view_count"] == 3
    assert len(payload["records"]) == 2
    player_a = next(row for row in payload["records"] if row["case_id"] == "case_a")
    assert player_a["source_id"] == "source_a_2"
    assert player_a["statistical_unit_id"] == "case_a"
    assert player_a["mean"] == 140.0
    assert player_a["view_count"] == 2
    assert player_a["primary_view"] is False
    assert player_a["case_slug"] == "case_a_view_2"
    assert payload["features"][0]["label"] == "Injured-side HKA angle (2D)"
    assert payload["provenance"]["projected_angles_averaged_across_views"] is False
    assert payload["events"][0]["dynamic_eligible_feature_count"] == 1
    assert payload["features"][0]["dynamic_supported_case_count"] == 2


def test_exploration_preserves_human_contact_metadata(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summaries"
    summary_dir.mkdir()
    metadata_path = tmp_path / "research.json"
    metadata_path.write_text(
        json.dumps(
            {
                "cases": {
                    "case_a": {
                        "contact_mechanism": "indirect_contact",
                        "contact_mechanism_source": "human_reviewer",
                        "injury_date": "2026-08-01",
                        "date_of_birth": "2000-08-02",
                        "league": "Women's Super League",
                        "competition": "Example League",
                        "team": "Example FC",
                        "position_group": "midfielder",
                        "match_minute": "67",
                        "preferred_foot": "right",
                        "preferred_foot_source": "EA SPORTS FC 26",
                        "preferred_foot_source_url": "https://example.test/ea-fc-player",
                        "preferred_foot_knee_injured": False,
                        "height_cm": 170,
                        "weight_kg": 63,
                        "height_verification_status": "sourced",
                        "weight_verification_status": "sourced",
                        "biometric_source": "Example player database",
                        "biometric_source_url": "https://example.test/player-profile",
                        "biometric_note": "Example sourced profile values.",
                        "ea_fc_audit_status": "verified",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    case = _case("case_a_view_1", "case_a", "source_a_1", "Player A", primary=True)
    _write_summary(
        summary_dir / "a1_case_feature_summary.parquet",
        case_id="case_a",
        source_id="source_a_1",
        geometry_completeness=0.80,
        geometry_eligible=True,
        mean=135.0,
    )

    payload = load_exploration_payload(
        (case,),
        summary_dir=summary_dir,
        research_metadata_path=metadata_path,
        signature_dir=tmp_path / "signatures",
    )

    assert payload["events"][0]["contact_mechanism"] == "indirect_contact"
    assert payload["events"][0]["injury_date"] == "2026-08-01"
    assert payload["events"][0]["league"] == "Women's Super League"
    assert payload["events"][0]["competition"] == "Example League"
    assert payload["events"][0]["team"] == "Example FC"
    assert payload["events"][0]["position_group"] == "midfielder"
    assert payload["events"][0]["match_minute"] == "67"
    assert payload["events"][0]["age_at_injury"] == 25
    assert payload["events"][0]["age_group"] == "21–25"
    assert payload["events"][0]["preferred_foot"] == "right"
    assert payload["events"][0]["preferred_foot_knee_injured"] is False
    assert payload["events"][0]["height_cm"] == 170
    assert payload["events"][0]["weight_kg"] == 63
    assert payload["events"][0]["biometric_source"] == "Example player database"
    assert payload["records"][0]["preferred_foot_knee_injured"] is False
    assert payload["records"][0]["height_cm"] == 170
    assert payload["records"][0]["weight_kg"] == 63
    assert payload["records"][0]["team"] == "Example FC"
    assert payload["summary"]["known_contact_mechanism_count"] == 1
    assert payload["readiness"]["contact_group_comparison"]["eligible"] is False


def test_current_player_profile_metadata_keeps_missing_biometrics_nullable() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads(
        (root / "data/annotations/human/case_research_metadata_human.json").read_text(
            encoding="utf-8"
        )
    )
    by_player = {
        details["player_name"]: details for details in metadata["cases"].values()
    }

    assert len(by_player) == 36
    assert sum(row["height_cm"] is not None for row in by_player.values()) == 35
    assert sum(row["weight_kg"] is not None for row in by_player.values()) == 32
    assert by_player["Kirsten van de Westeringh"]["height_cm"] is None
    assert by_player["Kirsten van de Westeringh"]["weight_kg"] is None
    assert by_player["Holly McNamara"]["weight_kg"] is None
    assert by_player["Holly McNamara"]["weight_verification_status"] == "source_conflict"
    assert by_player["Charlotte Newsham"]["height_cm"] == 170
    assert by_player["Kayla Duran"]["height_cm"] == 178


def test_injury_report_evidence_overrides_mechanism_and_exposes_provenance(
    tmp_path: Path,
) -> None:
    summary_dir = tmp_path / "summaries"
    summary_dir.mkdir()
    reports_path = tmp_path / "injury_reports.json"
    reports_path.write_text(
        json.dumps(
            {
                "reviewed_at": "2026-08-30",
                "methodology": {"title": "ACL taxonomy", "url": "https://example.test"},
                "cases": {
                    "case_a": {
                        "canonical_player_name": "Correct Player",
                        "previous_classification": "direct_contact",
                        "classification": "unclear",
                        "confidence": "low",
                        "verification_status": "unverified",
                        "evidence_basis": "contact_location_unresolved",
                        "change_status": "changed",
                        "rationale": "A challenge was reported without a contact location.",
                        "investigation_status": "needs_further_investigation",
                        "investigation_note": "Find an angle that shows the injured knee.",
                        "sources": [
                            {
                                "title": "Official report",
                                "publisher": "Example FC",
                                "published_date": "2026-08-01",
                                "url": "https://example.test/report",
                                "evidence": "The source reports a challenge.",
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    case = _case("case_a_view_1", "case_a", "source_a_1", "Player A", primary=True)
    _write_summary(
        summary_dir / "a1_case_feature_summary.parquet",
        case_id="case_a",
        source_id="source_a_1",
        geometry_completeness=0.8,
        geometry_eligible=True,
        mean=135.0,
    )

    payload = load_exploration_payload(
        (case,),
        summary_dir=summary_dir,
        injury_reports_path=reports_path,
        signature_dir=tmp_path / "signatures",
    )
    event = payload["events"][0]

    assert event["player_name"] == "Correct Player"
    assert event["contact_mechanism"] == "unclear"
    assert event["previous_contact_mechanism"] == "direct_contact"
    assert event["mechanism_confidence"] == "low"
    assert event["mechanism_investigation_status"] == "needs_further_investigation"
    assert event["mechanism_investigation_note"] == "Find an angle that shows the injured knee."
    assert event["mechanism_sources"][0]["publisher"] == "Example FC"
    assert payload["summary"]["known_contact_mechanism_count"] == 0
    assert payload["summary"]["unclear_contact_mechanism_count"] == 1
    assert payload["mechanism_review"]["reviewed_at"] == "2026-08-30"
    assert payload["mechanism_methodology"]["title"] == "ACL taxonomy"


def test_phase_support_marks_reference_pool_eligibility(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summaries"
    semantics_dir = tmp_path / "semantics"
    summary_dir.mkdir()
    semantics_dir.mkdir()
    cases = (
        _case("case_a_view_1", "case_a", "source_a_1", "Player A", primary=True),
        _case("case_b_view_1", "case_b", "source_b_1", "Player B", primary=True),
    )
    for case_id, source_id in (("case_a", "source_a_1"), ("case_b", "source_b_1")):
        _write_summary(
            summary_dir / f"{case_id}_case_feature_summary.parquet",
            case_id=case_id,
            source_id=source_id,
            geometry_completeness=0.9,
            geometry_eligible=True,
            mean=130.0,
        )
    for source_id, status in (
        ("source_a_1", "SUPPORTED"),
        ("source_b_1", "INSUFFICIENT_EVIDENCE_FOR_PHASE_SEGMENTATION"),
    ):
        (semantics_dir / f"{source_id}_observable_movement_descriptions.json").write_text(
            json.dumps({"metadata": {"source_id": source_id, "phase_status": status}}),
            encoding="utf-8",
        )

    payload = load_exploration_payload(
        cases,
        summary_dir=summary_dir,
        signature_dir=tmp_path / "signatures",
        semantics_dir=semantics_dir,
    )
    events = {event["case_id"]: event for event in payload["events"]}

    assert events["case_a"]["reference_pool_eligible"] is True
    assert events["case_b"]["reference_pool_eligible"] is False
    assert events["case_b"]["reference_pool_reason"] == (
        "Not eligible as a reference because no completed phase-supported, event-covered "
        "view is available. The case may still be compared as a query when enough "
        "whole-movement measurements are supported."
    )
    assert {row["case_id"] for row in payload["similarity_records"]} == {
        "case_a",
        "case_b",
    }
    assert {row["case_id"] for row in payload["similarity_view_records"]} == {
        "case_a",
        "case_b",
    }
    assert payload["summary"]["phase_supported_case_count"] == 1


def test_case_details_preserve_existing_research_metadata_and_provenance(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "research.json"
    metadata_path.write_text(
        json.dumps(
            {
                "metadata_version": "case_research_metadata_v1",
                "cases": {
                    "case_a": {
                        "statistical_unit_id": "event_a",
                        "contact_mechanism": "direct_contact",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    saved = save_case_details(
        metadata_path,
        "case_a",
        {
            "player_name": "Player A",
            "injury_date": "2026-08-01",
            "league": "Women's Super League",
            "competition": "Example League",
            "team": "Example FC",
            "position_group": "midfielder",
            "match_minute": "45+2",
            "date_of_birth": "1995-01-02",
        },
        annotator_id="researcher_01",
    )

    record = load_research_metadata(metadata_path)["case_a"]
    assert saved["match_minute"] == "45+2"
    assert saved["league"] == "Women's Super League"
    assert record["statistical_unit_id"] == "event_a"
    assert record["contact_mechanism"] == "direct_contact"
    assert record["metadata_source"] == "human_operator_annotation_ui:researcher_01"
    assert record["field_provenance"]["team"] == record["metadata_source"]
    assert case_details(metadata_path, "case_a")["position_group"] == "midfielder"


def test_group_test_gate_requires_independent_cases_per_group() -> None:
    blocked = assess_group_test_eligibility({"non_contact": 12, "direct_contact": 2})
    available_two = assess_group_test_eligibility({"non_contact": 8, "direct_contact": 5})
    available_three = assess_group_test_eligibility(
        {"non_contact": 7, "indirect_contact": 6, "direct_contact": 5}
    )

    assert blocked["status"] == "INSUFFICIENT_INDEPENDENT_CASES"
    assert blocked["eligible"] is False
    assert available_two["recommended_tests"][0] == "Welch t-test"
    assert available_three["recommended_tests"][0] == "Welch ANOVA"


def test_exploration_with_no_summaries_is_explicit(tmp_path: Path) -> None:
    payload = load_exploration_payload((), summary_dir=tmp_path, signature_dir=tmp_path)

    assert payload["summary"]["analysed_case_count"] == 0
    assert payload["readiness"]["descriptive_eda"]["status"] == "NO_ANALYSED_CASES"
    assert payload["readiness"]["confirmatory_inference"]["eligible"] is False


def test_exploration_distinguishes_unavailable_from_limited_values(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summaries"
    summary_dir.mkdir()
    cases = (
        _case("case_a_view_1", "case_a", "source_a_1", "Player A", primary=True),
        _case("case_b_view_1", "case_b", "source_b_1", "Player B", primary=True),
        _case("case_c_view_1", "case_c", "source_c_1", "Player C", primary=True),
    )
    _write_summary(
        summary_dir / "a_case_feature_summary.parquet",
        case_id="case_a",
        source_id="source_a_1",
        geometry_completeness=0.8,
        geometry_eligible=True,
        mean=0.0,
    )
    _write_summary(
        summary_dir / "b_case_feature_summary.parquet",
        case_id="case_b",
        source_id="source_b_1",
        geometry_completeness=0.4,
        geometry_eligible=False,
        mean=125.0,
    )
    _write_summary(
        summary_dir / "c_case_feature_summary.parquet",
        case_id="case_c",
        source_id="source_c_1",
        geometry_completeness=0.0,
        geometry_eligible=False,
        mean=None,
    )

    payload = load_exploration_payload(
        cases,
        summary_dir=summary_dir,
        signature_dir=tmp_path / "signatures",
    )
    feature = payload["features"][0]

    assert feature["supported_case_count"] == 1
    assert feature["available_case_count"] == 2
    assert feature["unsupported_case_count"] == 1
    assert feature["unavailable_case_count"] == 1
    assert next(row for row in payload["records"] if row["case_id"] == "case_a")[
        "mean"
    ] == 0.0
    assert next(row for row in payload["records"] if row["case_id"] == "case_c")[
        "mean"
    ] is None


def test_similarity_readiness_requires_comparable_case_measurements(
    tmp_path: Path,
) -> None:
    signature_dir = tmp_path / "signatures"
    signature_dir.mkdir()
    for name, case_id in (("a", "case_a"), ("b", "case_b")):
        pd.DataFrame({"case_id": [case_id], "feature_name": ["descriptor"]}).to_csv(
            signature_dir / f"{name}_case_movement_signature_long.csv",
            index=False,
        )

    payload = load_exploration_payload(
        (),
        summary_dir=tmp_path / "summaries",
        signature_dir=signature_dir,
    )

    assert payload["similarity"]["available"] is False
    assert payload["similarity"]["status"] == "INSUFFICIENT_COMPARABLE_CASES"
    assert payload["similarity"]["signature_case_count"] == 2
    assert payload["similarity"]["comparable_case_count"] == 0
    assert payload["similarity"]["pairwise_output_count"] == 0


def _case(
    slug: str,
    case_id: str,
    source_id: str,
    player_name: str,
    *,
    primary: bool,
) -> AnnotationCase:
    return AnnotationCase(
        slug=slug,
        case_id=case_id,
        source_id=source_id,
        view_id=source_id,
        view_label=f"View {source_id[-1]}",
        primary_view=primary,
        perspective="oblique",
        injured_side=InjurySide.LEFT,
        player_name=player_name,
        video_path=Path(f"{slug}.mp4"),
    )


def _write_summary(
    path: Path,
    *,
    case_id: str,
    source_id: str,
    geometry_completeness: float,
    geometry_eligible: bool,
    mean: float | None,
) -> None:
    pd.DataFrame(
        [
            {
                "case_id": case_id,
                "source_id": source_id,
                "view_id": source_id,
                "feature_name": "injured_hka_angle_2d_deg",
                "body_region": "lower_limb",
                "feature_family": "lower_limb_geometry",
                "geometry_completeness": geometry_completeness,
                "dynamic_completeness": geometry_completeness - 0.1,
                "mean": mean,
                "range": 20.0,
                "pre_late_change": -5.0,
                "analytics_eligibility": "ANALYTICS_READY",
                "geometry_analytics_eligible": geometry_eligible,
                "dynamic_analytics_eligible": geometry_eligible,
                "analytics_eligible": geometry_eligible,
                "quality_category": "SUPPORTED" if geometry_eligible else "LIMITED",
                "primary_rejection_reason": "",
            }
        ]
    ).to_parquet(path, index=False)
