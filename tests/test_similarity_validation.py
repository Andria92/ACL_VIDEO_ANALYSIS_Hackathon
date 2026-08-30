from acl_motion.validation.expert_judgements import (
    ExpertPairwiseJudgement,
    PairwiseChoice,
    append_expert_judgement,
    build_blinded_assignments,
    evaluate_expert_judgements,
    load_expert_judgements,
    next_blinded_assignment,
)
from acl_motion.validation.similarity import build_internal_similarity_validation_report


def test_internal_similarity_audit_uses_query_excluded_scaling_and_jackknife():
    report = build_internal_similarity_validation_report(_records(), _events())
    query = next(item for item in report["case_audits"] if item["query_case_id"] == "case_a")
    primary = query["lenses"]["overall_movement_difference"]

    assert report["status"] == "INTERNAL_AUDIT_ONLY"
    assert report["summary"]["reference_pool_case_count"] == 5
    assert query["scaling_status"] == "QUERY_EXCLUDED_EXPLORATORY_ESTIMATE"
    assert query["scaler_reference_case_count"] == 4
    assert primary["baseline_top_case_id"] == "case_b"
    assert primary["jackknife_valid_checks"] >= 2
    assert 0.0 <= primary["jackknife_top_retention_frequency"] <= 1.0


def test_blinded_expert_assignment_storage_and_concordance(tmp_path):
    sources = [
        {"case_id": case_id, "slug": f"slug_{case_id}"}
        for case_id in ("case_a", "case_b", "case_c", "case_d", "case_e")
    ]
    assignments = build_blinded_assignments(
        sources,
        {"case_a", "case_b", "case_c", "case_d", "case_e"},
        assessor_id="expert_01",
    )
    assignment = assignments[0]
    closer_case = min(
        (assignment["option_a_case_id"], assignment["option_b_case_id"]),
        key=lambda case_id: abs(_OFFSETS[case_id] - _OFFSETS[assignment["query_case_id"]]),
    )
    choice = (
        PairwiseChoice.OPTION_A
        if assignment["option_a_case_id"] == closer_case
        else PairwiseChoice.OPTION_B
    )
    judgement = ExpertPairwiseJudgement.create(
        assignment=assignment,
        assessor_id="expert_01",
        choice=choice,
    )
    path = append_expert_judgement(tmp_path / "judgements.jsonl", judgement)
    loaded = load_expert_judgements(path)

    assert len(loaded) == 1
    assert next_blinded_assignment(
        assignments,
        loaded,
        assessor_id="expert_01",
    )["assignment_id"] != assignment["assignment_id"]
    assert "player_name" not in assignment
    report = evaluate_expert_judgements(_records(), _events(), loaded)
    assert report["status"] == "CURRENT_CASE_CONCORDANCE"
    assert report["held_out_players"] is False
    assert report["lenses"]["overall_movement_difference"]["evaluated"] == 1


_OFFSETS = {
    "case_a": 0.0,
    "case_b": 0.2,
    "case_c": 4.0,
    "case_d": 8.0,
    "case_e": 12.0,
}


def _events():
    return [
        {
            "case_id": case_id,
            "player_name": f"Player {case_id[-1].upper()}",
            "reference_pool_eligible": True,
            "phase_supported_view_count": 1,
        }
        for case_id in _OFFSETS
    ]


def _records():
    features = (
        ("injured_hka_angle_2d_deg", "lower_limb", 10.0),
        ("contralateral_hka_angle_2d_deg", "lower_limb", 20.0),
        ("projected_trunk_axis_angle_deg", "trunk", 30.0),
        ("trunk_axis", "trunk", 40.0),
    )
    rows = []
    for case_id, offset in _OFFSETS.items():
        for feature_name, family, base in features:
            rows.append(
                {
                    "case_id": case_id,
                    "statistical_unit_id": case_id,
                    "feature_name": feature_name,
                    "body_region": family,
                    "feature_family": family,
                    "perspective": "oblique",
                    "mean": base + offset,
                    "range": base / 2.0 + offset,
                    "range_semantics": (
                        "shortest_directed_arc"
                        if feature_name == "projected_trunk_axis_angle_deg"
                        else "linear"
                    ),
                    "angular_statistics_version": "angular_statistics_v1_shortest_arc",
                    "comparison_statistics_version": (
                        "comparison_statistics_v1_supported_intervals"
                    ),
                    "comparison_support_scope": "observable_supported_intervals",
                    "geometry_analytics_eligible": True,
                    "dynamic_analytics_eligible": False,
                    "geometry_completeness": 0.9,
                    "dynamic_completeness": 0.0,
                }
            )
    return rows
