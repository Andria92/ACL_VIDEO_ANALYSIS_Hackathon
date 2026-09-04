from __future__ import annotations

from acl_motion.analytics.similarity import (
    SIMILARITY_LENSES,
    build_similarity_payload,
    similarity_readiness,
)


def test_similarity_engine_ranks_nearest_supported_case() -> None:
    payload = build_similarity_payload(
        _records(),
        _events(),
        selected_case_id="case_a",
    )

    assert payload["available"] is True
    assert payload["summary"]["comparable_case_count"] == 4
    assert payload["summary"]["comparable_pair_count"] == 6
    assert payload["scaling"]["status"] == "QUERY_EXCLUDED_EXPLORATORY_ESTIMATE"
    assert payload["scaling"]["query_excluded"] is True
    assert payload["scaling"]["reference_case_count"] == 3
    assert [lens["id"] for lens in payload["lenses"]] == [
        lens["id"] for lens in SIMILARITY_LENSES
    ]
    overall = payload["rankings"]["overall_movement_difference"]
    assert overall[0]["case"]["player_name"] == "Player B"
    assert overall[0]["similarity_index"] > overall[1]["similarity_index"]
    assert 0.0 <= overall[0]["similarity_index"] <= 1.0
    assert "score" not in overall[0]["evidence_support"]
    assert overall[0]["stability"]["valid_iterations"] == 200
    assert overall[0]["shared_descriptor_count"] == 9
    assert overall[0]["closest_measurements"]
    assert overall[0]["largest_differences"]


def test_similarity_network_reuses_only_available_pairwise_results() -> None:
    payload = build_similarity_payload(
        _records(),
        _events(),
        selected_case_id="case_a",
        resampling_iterations=0,
    )

    network = payload["network"]
    assert len(network["nodes"]) == 4
    assert network["edge_count"] == 6
    assert len(network["edges"]) == 6
    assert all(edge["indices"] for edge in network["edges"])
    assert all(edge["shared_descriptor_count"] >= 6 for edge in network["edges"])
    assert all(
        "overall_movement_difference" in edge["evidence_support"]
        for edge in network["edges"]
    )
    assert "does not mean" in network["missing_edge_note"]


def test_similarity_measurement_group_filter_is_explicit_and_evidence_gated() -> None:
    records = _records()
    for case_id, offset in {
        "case_a": 0.0,
        "case_b": 0.2,
        "case_c": 8.0,
        "case_d": 12.0,
    }.items():
        records.append(
            {
                "case_id": case_id,
                "statistical_unit_id": case_id,
                "feature_name": "left_elbow_angle_2d_deg",
                "body_region": "upper_body",
                "feature_family": "upper_body",
                "perspective": "oblique",
                "mean": 70.0 + offset,
                "range": 15.0 + offset,
                "range_semantics": "linear",
                "comparison_statistics_version": (
                    "comparison_statistics_v1_supported_intervals"
                ),
                "comparison_support_scope": "observable_supported_intervals",
                "pre_late_change": None,
                "geometry_analytics_eligible": True,
                "dynamic_analytics_eligible": False,
                "geometry_completeness": 0.9,
                "dynamic_completeness": 0.0,
            }
        )

    unfiltered = build_similarity_payload(records, _events(), resampling_iterations=0)
    filtered = build_similarity_payload(
        records,
        _events(),
        selected_case_id="case_a",
        measurement_groups=("lower_limb", "trunk"),
        resampling_iterations=0,
    )
    one_group = build_similarity_payload(
        records,
        _events(),
        selected_case_id="case_a",
        measurement_groups=("lower_limb",),
        resampling_iterations=0,
    )

    assert unfiltered["summary"]["eligible_descriptor_count"] > filtered["summary"][
        "eligible_descriptor_count"
    ]
    assert filtered["measurement_groups"]["selected"] == ["lower_limb", "trunk"]
    assert filtered["measurement_groups"]["scope"] == "FILTERED_MEASUREMENT_GROUPS"
    assert filtered["rankings"]["overall_movement_difference"]
    assert one_group["available"] is False
    assert one_group["network"]["edges"] == []


def test_similarity_engine_excludes_missing_measurements_instead_of_using_zero() -> None:
    records = _records()
    records = [
        record
        for record in records
        if not (record["case_id"] == "case_b" and record["feature_name"] == "trunk_axis")
    ]

    payload = build_similarity_payload(records, _events(), selected_case_id="case_a")
    match = payload["rankings"]["overall_movement_difference"][0]

    assert match["case"]["case_id"] == "case_b"
    assert match["shared_descriptor_count"] == 7


def test_similarity_excludes_stale_wrap_sensitive_ranges_until_rebuilt() -> None:
    records = _records()
    for record in records:
        if record["feature_name"] == "projected_trunk_axis_angle_deg":
            record.pop("range_semantics")

    payload = build_similarity_payload(records, _events(), selected_case_id="case_a")
    match = payload["rankings"]["overall_movement_difference"][0]

    assert match["shared_descriptor_count"] == 8
    assert all(
        item["descriptor_id"] != "projected_trunk_axis_angle_deg::range"
        for item in match["closest_measurements"] + match["largest_differences"]
    )


def test_soft_cosine_is_explicitly_provisional() -> None:
    payload = build_similarity_payload(_records(), _events(), selected_case_id="case_a")
    match = payload["rankings"]["relationship_aware_pattern"][0]

    assert match["evidence_support"]["status"] == "PROVISIONAL"
    assert "soft-cosine relationship map" in match["evidence_support"]["explanation"]
    assert "score" not in match["evidence_support"]


def test_similarity_readiness_reports_computed_pairings() -> None:
    readiness = similarity_readiness(_records(), _events())

    assert readiness["available"] is True
    assert readiness["status"] == "AVAILABLE"
    assert readiness["comparable_case_count"] == 4
    assert readiness["pairwise_output_count"] == 6


def test_query_only_case_can_be_compared_but_cannot_be_a_reference() -> None:
    query_records = []
    for record in _records():
        if record["case_id"] != "case_a":
            continue
        item = dict(record)
        item["case_id"] = "case_query"
        item["statistical_unit_id"] = "case_query"
        item["mean"] += 0.5
        item["range"] += 0.5
        query_records.append(item)
    events = _events() + [
        {
            "case_id": "case_query",
            "player_name": "Query Player",
            "reference_pool_eligible": False,
            "reference_pool_reason": "Phases were not supported.",
        }
    ]
    view_records = []
    for record in _records() + query_records:
        item = dict(record)
        case_id = str(item["case_id"])
        item["source_id"] = f"{case_id}_view_01"
        item["view_label"] = f"{case_id} analysed view"
        item["phase_status"] = (
            "INSUFFICIENT_EVIDENCE_FOR_PHASE_SEGMENTATION"
            if case_id == "case_query"
            else "SUPPORTED"
        )
        item["event_comparison_eligible"] = case_id != "case_query"
        view_records.append(item)

    query_payload = build_similarity_payload(
        _records() + query_records,
        events,
        view_records=view_records,
        selected_case_id="case_query",
        resampling_iterations=0,
    )
    reference_payload = build_similarity_payload(
        _records() + query_records,
        events,
        view_records=view_records,
        selected_case_id="case_a",
        resampling_iterations=0,
    )

    assert query_payload["selected_case"]["reference_pool_eligible"] is False
    assert query_payload["summary"]["reference_pool_case_count"] == 4
    assert query_payload["summary"]["query_only_case_count"] == 1
    assert query_payload["rankings"]["overall_movement_difference"]
    assert all(
        match["case"]["reference_pool_eligible"]
        for match in query_payload["rankings"]["overall_movement_difference"]
    )
    assert all(
        match["selected_view_pair"]["selected_case"]["reference_view_eligible"]
        is False
        for match in query_payload["rankings"]["overall_movement_difference"]
    )
    assert all(
        match["selected_view_pair"]["candidate_case"]["reference_view_eligible"]
        is True
        for match in query_payload["rankings"]["overall_movement_difference"]
    )
    assert "case_query" not in {
        match["case"]["case_id"]
        for match in reference_payload["rankings"]["overall_movement_difference"]
    }


def test_analysed_case_without_supported_descriptors_remains_visible_query_only() -> None:
    events = _events() + [
        {
            "case_id": "case_occluded",
            "player_name": "Occluded Player",
            "reference_pool_eligible": False,
            "reference_pool_reason": "The incomplete view is excluded from the reference pool.",
            "analysed_view_count": 1,
        }
    ]

    payload = build_similarity_payload(
        _records(),
        events,
        selected_case_id="case_occluded",
        resampling_iterations=0,
    )

    selected = payload["selected_case"]
    assert selected["case_id"] == "case_occluded"
    assert selected["reference_pool_eligible"] is False
    assert selected["comparable_descriptor_count"] == 0
    assert selected["query_comparison_ready"] is False
    assert payload["rankings"]["overall_movement_difference"] == []
    assert payload["summary"]["query_only_case_count"] == 1


def test_hidden_unidentified_case_cannot_change_scaling_or_ranking() -> None:
    baseline = build_similarity_payload(_records(), _events(), selected_case_id="case_a")
    hidden_records = []
    for record in _records():
        if record["case_id"] != "case_c":
            continue
        item = dict(record)
        item["case_id"] = "hidden_case"
        item["statistical_unit_id"] = "hidden_case"
        item["mean"] += 1000.0
        item["range"] += 1000.0
        hidden_records.append(item)
    with_hidden = build_similarity_payload(
        _records() + hidden_records,
        _events()
        + [
            {
                "case_id": "hidden_case",
                "player_name": "Imported unidentified clip",
                "reference_pool_eligible": True,
            }
        ],
        selected_case_id="case_a",
    )

    assert with_hidden["summary"]["reference_pool_case_count"] == 4
    assert with_hidden["rankings"] == baseline["rankings"]


def test_phase_dependent_descriptors_require_supported_phase_evidence() -> None:
    records = []
    for record in _records():
        item = dict(record)
        item["pre_late_change"] = item["mean"] / 10.0
        item["dynamic_analytics_eligible"] = True
        item["dynamic_completeness"] = 0.9
        item["phase_status"] = (
            "SUPPORTED"
            if item["case_id"] == "case_a"
            else "INSUFFICIENT_EVIDENCE_FOR_PHASE_SEGMENTATION"
        )
        records.append(item)

    payload = build_similarity_payload(records, _events(), selected_case_id="case_a")

    assert payload["summary"]["eligible_descriptor_count"] == 9


def test_multiple_views_use_one_best_pair_without_averaging() -> None:
    view_records = []
    view_offsets = {
        "case_a": (("source_a_1", "A view 1", 0.0), ("source_a_2", "A view 2", 6.0)),
        "case_b": (("source_b_1", "B view 1", 4.0), ("source_b_2", "B view 2", 0.1)),
        "case_c": (("source_c_1", "C view 1", 8.0),),
        "case_d": (("source_d_1", "D view 1", 12.0),),
    }
    for record in _records():
        case_id = record["case_id"]
        base_offset = {"case_a": 0.0, "case_b": 0.2, "case_c": 8.0, "case_d": 12.0}[case_id]
        for source_id, view_label, offset in view_offsets[case_id]:
            item = dict(record)
            item["source_id"] = source_id
            item["view_label"] = view_label
            item["mean"] = item["mean"] - base_offset + offset
            item["range"] = item["range"] - base_offset + offset
            view_records.append(item)

    payload = build_similarity_payload(
        _records(),
        _events(),
        view_records=view_records,
        selected_case_id="case_a",
        resampling_iterations=0,
    )
    match = payload["rankings"]["overall_movement_difference"][0]

    assert match["case"]["case_id"] == "case_b"
    assert match["selected_view_pair"]["selected_case"]["source_id"] == "source_a_1"
    assert match["selected_view_pair"]["candidate_case"]["source_id"] == "source_b_2"
    assert match["selected_view_pair"]["eligible_view_pair_count"] == 4
    assert match["closest_measurements"][0]["selected_value"] is not None
    assert match["closest_measurements"][0]["candidate_value"] is not None


def test_reference_candidates_use_only_phase_supported_event_covered_views() -> None:
    view_records = []
    view_offsets = {
        "case_a": (("source_a_1", 0.0), ("source_a_2", 6.0)),
        "case_b": (("source_b_1", 4.0), ("source_b_2", 0.1)),
        "case_c": (("source_c_1", 8.0),),
        "case_d": (("source_d_1", 12.0),),
    }
    for record in _records():
        case_id = record["case_id"]
        base_offset = {"case_a": 0.0, "case_b": 0.2, "case_c": 8.0, "case_d": 12.0}[
            case_id
        ]
        for source_id, offset in view_offsets[case_id]:
            item = dict(record)
            item["source_id"] = source_id
            item["view_label"] = source_id
            item["mean"] = item["mean"] - base_offset + offset
            item["range"] = item["range"] - base_offset + offset
            item["phase_status"] = (
                "INSUFFICIENT_EVIDENCE_FOR_PHASE_SEGMENTATION"
                if source_id == "source_b_2"
                else "SUPPORTED"
            )
            item["event_comparison_eligible"] = source_id != "source_b_2"
            view_records.append(item)

    payload = build_similarity_payload(
        _records(),
        _events(),
        view_records=view_records,
        selected_case_id="case_a",
        resampling_iterations=0,
    )
    match = next(
        item
        for item in payload["rankings"]["overall_movement_difference"]
        if item["case"]["case_id"] == "case_b"
    )

    assert match["selected_view_pair"]["candidate_case"]["source_id"] == "source_b_1"
    assert match["selected_view_pair"]["candidate_case"]["reference_view_eligible"] is True
    assert match["selected_view_pair"]["eligible_view_pair_count"] == 2


def _events() -> list[dict]:
    return [
        {
            "case_id": "case_a",
            "player_name": "Player A",
            "team": "Team One",
            "reference_pool_eligible": True,
            "phase_supported_view_count": 1,
        },
        {
            "case_id": "case_b",
            "player_name": "Player B",
            "team": "Team Two",
            "reference_pool_eligible": True,
            "phase_supported_view_count": 1,
        },
        {
            "case_id": "case_c",
            "player_name": "Player C",
            "team": "Team Three",
            "reference_pool_eligible": True,
            "phase_supported_view_count": 1,
        },
        {
            "case_id": "case_d",
            "player_name": "Player D",
            "team": "Team Four",
            "reference_pool_eligible": True,
            "phase_supported_view_count": 1,
        },
    ]


def _records() -> list[dict]:
    case_offsets = {"case_a": 0.0, "case_b": 0.2, "case_c": 8.0, "case_d": 12.0}
    features = (
        ("injured_hka_angle_2d_deg", "lower_limb", 10.0),
        ("contralateral_hka_angle_2d_deg", "lower_limb", 20.0),
        ("projected_trunk_axis_angle_deg", "trunk", 30.0),
        ("trunk_axis", "trunk", 40.0),
        ("pelvis_offset", "trunk", 50.0),
    )
    records = []
    for case_id, offset in case_offsets.items():
        for feature_name, body_region, base in features:
            records.append(
                {
                    "case_id": case_id,
                    "statistical_unit_id": case_id,
                    "feature_name": feature_name,
                    "body_region": body_region,
                    "feature_family": body_region,
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
                    "pre_late_change": None,
                    "geometry_analytics_eligible": True,
                    "dynamic_analytics_eligible": False,
                    "geometry_completeness": 0.9,
                    "dynamic_completeness": 0.0,
                }
            )
    return records
