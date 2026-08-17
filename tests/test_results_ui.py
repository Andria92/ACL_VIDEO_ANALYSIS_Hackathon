from __future__ import annotations

from acl_motion.annotations.models import AnnotationCase
from acl_motion.annotations.registry import default_annotation_cases, views_for_case
from acl_motion.ui.annotation import smoke_test
from acl_motion.ui.results import (
    build_case_synthesis_payload,
    build_human_analysis_regeneration_commands,
    coverage_label,
    explain_status,
    load_human_results_payload,
    load_result_evidence_payload,
    render_results_page,
    result_frame_for_time,
)


def test_human_results_payload_uses_human_namespace() -> None:
    payload = load_human_results_payload(default_annotation_cases()[0])

    assert payload["case"]["slug"] == "christen_press"
    assert payload["case"]["view_id"] == "christen_press_view_01"
    assert payload["view"]["primary_view"] is True
    assert payload["case_views"]["view_count"] == 1
    assert payload["case_synthesis"]["available"] is False
    assert payload["case_synthesis"]["rules"]["averages_projected_angles_across_views"] is False
    assert payload["target_annotation"]["label"] == "Human verified"
    assert payload["source_files"]["movement_profile"].endswith(
        "data/profiles/human/christen_press_movement_profile.json"
    )
    assert all("/human/" in path for path in payload["source_files"].values())
    assert "semantic_observations" in payload["source_files"]
    assert "observable_descriptions" in payload["source_files"]
    assert "movement_phases" in payload["source_files"]
    assert "observable_movement_descriptions" in payload
    assert payload["observable_movement_descriptions"]["supported_intervals"][0]["start_frame"] == 0
    assert payload["observable_movement_descriptions"]["supported_intervals"][0]["end_frame"] == 96
    assert len(payload["observable_movement_descriptions"]["default_story_descriptions"]) <= 4
    assert payload["path_quality_summary"]["overall_status"] in {"SUPPORTED", "UNAVAILABLE", "QA_REQUIRED"}
    assert payload["movement_story"]["status"] in {
        "SUPPORTED",
        "INSUFFICIENT_EVIDENCE_FOR_PHASE_SEGMENTATION",
    }
    assert payload["movement_story"]["phase_count"] >= 0
    assert payload["movement_visual_story"]["story_version"] == "m5_9_movement_first_visual_story_v1"
    if payload["movement_story"]["phase_count"]:
        assert payload["movement_visual_story"]["phases"][0]["snapshot_frames"]
        assert payload["movement_visual_story"]["phases"][0]["observations"]
    else:
        assert payload["movement_visual_story"]["phases"] == []
        assert payload["movement_visual_story"]["whole_movement"]["phase_sequence"] == []


def test_results_movement_window_bounds_and_exact_frame_mapping() -> None:
    payload = load_human_results_payload(default_annotation_cases()[0])
    frames = payload["frames"]

    assert frames[0]["source_frame_index"] == payload["movement_window"]["movement_start_frame"]
    assert frames[-1]["source_frame_index"] == payload["movement_window"]["movement_end_frame"]
    assert result_frame_for_time(payload, 0.0) == 155


def test_result_graph_points_preserve_traceability() -> None:
    payload = load_human_results_payload(default_annotation_cases()[0])
    point = payload["trajectories"]["left_hka_angle_2d_deg"][0]

    assert point["source_frame_index"] == 0
    assert point["movement_end_relative_ms"] == -5166.666666666667
    assert "feature_status" in point
    assert "dynamic_status" in point


def test_evidence_detail_uses_exact_feature_frame() -> None:
    detail = load_result_evidence_payload(
        default_annotation_cases()[0],
        feature_name="left_hka_angle_2d_deg",
        source_frame_index=0,
    )

    assert detail["source_frame_index"] == 0
    assert detail["feature_name"] == "left_hka_angle_2d_deg"
    assert "left_hip" in detail["landmarks_used"]
    assert detail["frame_qc"]["source_frame_index"] == 0


def test_status_to_user_language_mapping() -> None:
    assert explain_status("LOW_DYNAMIC_CONFIDENCE").startswith("The local trajectory")
    assert explain_status("TARGET_IDENTITY_UNCERTAIN").startswith("The target athlete")
    assert coverage_label(1.0) == "HIGH"
    assert coverage_label(0.76) == "GOOD"
    assert coverage_label(0.46) == "LIMITED"


def test_missing_or_limited_feature_rendering_is_explicit() -> None:
    payload = load_human_results_payload(default_annotation_cases()[0])
    card = payload["feature_cards"]["left_hka_angle_2d_deg"]

    assert card["at_movement_end"] is None
    assert card["at_movement_end_status"] == "INVALID_TARGET_FRAME"
    assert card["dynamic_evidence"] in {"SUPPORTED", "LIMITED", "UNAVAILABLE"}


def test_results_page_avoids_front_facing_risk_language() -> None:
    html = render_results_page()

    assert "featureCategorySelect" in html
    assert "featureSelect" in html
    assert "featureGraph" in html
    assert "videoFrame" in html
    assert "backFiveButton" in html
    assert "backOneButton" in html
    assert "playPauseButton" in html
    assert "forwardOneButton" in html
    assert "forwardFiveButton" in html
    assert "trimAnalysisButton" in html
    assert "End analysis here + regenerate" in html
    assert "More statistics" in html
    assert "Evidence" in html
    assert "Technical details" in html
    assert "LOWER LIMB" in html
    assert "BILATERAL" in html
    assert "Left/right projected HKA" in html
    assert "No measurements available" in html
    assert "availableCategories()" in html
    assert "Timing means the source-frame position" in html
    assert "left-minus-right projected HKA difference" in html
    assert "TRUNK & PELVIS" in html
    assert "UPPER BODY" in html
    assert "MOVEMENT PATH" in html
    assert "TIMING" in html
    assert "Movement path graph unavailable in this panel" in html
    assert "left_knee_ankle_distance_normalized" not in html
    assert "right_knee_ankle_distance_normalized" not in html
    removed_clutter = [
        "Explore this movement",
        "Research measurements",
        "HKA Angle",
        "Side / comparison",
        "Delta deg from phase start",
        "target segmentation mask",
        "Mark visible target regions",
        "Target region",
        "Non-target region",
        "Clear frame regions",
        "Descriptive Statistics",
        "Evidence details",
        "Movement Story",
        "Show why",
        "Supported evidence interval",
        "Clip evidence coverage",
        "fall/movement-end frames are therefore unresolved evidence",
        "Visible injury/fall interval awaiting review",
        "No movement description is claimed for this interval yet",
        "Earlier supported measurements, not the injury/fall interval",
        "Not claimed yet",
        "Why these claims are held back",
        "Descriptions withheld",
        "Movement Phase",
        "phaseTimeline",
        "selectedPhaseStory",
        "SelectionMode",
        "Play Phase",
        "Replay Phase",
        "pauseButton",
        "Video overlays & target-region marks",
        "Selected-frame evidence",
        "Open Research Measurement",
        "Inspect frames",
        "Inspect feature",
        "canonicalFeatureSelect",
        "featureSideControl",
        "featureViewControl",
        "Research Details",
        "metricCategorySelect",
        "metricSelect",
        "advancedAngleModeSelect",
        "researchScopeSelect",
        "researchSelectionSentence",
        "Internal Category",
        "Internal Metric",
        "Advanced Transformation",
    ]
    for text in removed_clutter:
        assert text not in html
    assert "featureCategorySelect" in html
    assert "metricList" not in html
    assert "Category Details" not in html
    assert "Explore Metrics" not in html
    assert "Explore measurement" not in html
    assert "ACL risk" not in html
    assert "high risk" not in html.lower()
    assert "danger score" not in html
    assert "critical plant" not in html


def test_annotation_ui_smoke_includes_results_entry_point() -> None:
    result = smoke_test()

    assert result["html_has_view_analysis"] is True
    assert result["results_html_has_single_feature_ui"] is True


def test_results_payload_preserves_phase_story_traceability() -> None:
    payload = load_human_results_payload(default_annotation_cases()[0])
    frame_map = payload["phase_frame_map"]

    if payload["movement_story"]["phase_count"]:
        phase = payload["movement_story"]["phases"][0]
        assert phase["start_frame"] == frame_map[0]["source_frame_index"]
        assert "change_score_summary" in phase
        assert "category_summaries" in phase
    else:
        assert frame_map[0]["phase_id"] is None
        assert frame_map[0]["phase_title"] == "Phase segmentation unavailable"
    assert "transitions" in payload["movement_story"]
    assert "change_score" in frame_map[1]


def test_results_payload_has_scope_aware_visual_story() -> None:
    payload = load_human_results_payload(default_annotation_cases()[0])
    visual_story = payload["movement_visual_story"]

    assert visual_story["whole_movement"]["scope_label"].startswith("Viewing: Whole Movement")
    if not visual_story["phases"]:
        assert payload["movement_story"]["status"] == "INSUFFICIENT_EVIDENCE_FOR_PHASE_SEGMENTATION"
        return
    phase = visual_story["phases"][0]
    assert phase["scope_label"].startswith("Viewing: Phase 1")
    assert 2 <= len(phase["observations"]) <= 4
    assert all("score_components" in item for item in phase["observations"])
    assert "other_observations" in phase
    labels = {item["label"] for item in phase["snapshot_frames"]}
    assert "Phase start" in labels
    assert "Phase end" in labels
    assert "Mid-phase" not in labels
    assert len(phase["snapshot_frames"]) >= 3
    assert all("support" in item for item in phase["observations"])


def test_results_page_default_details_are_collapsed() -> None:
    html = render_results_page()

    assert '<details id="moreStatistics">' in html
    assert "<summary>More statistics</summary>" in html
    assert '<details id="evidenceDetails">' in html
    assert "<summary>Evidence</summary>" in html
    assert '<details id="technicalDetails">' in html
    assert "<summary>Technical details</summary>" in html
    assert 'id="researchMeasurements"' not in html
    assert 'id="exploreMovement"' not in html


def test_simplified_results_controls_have_handlers() -> None:
    html = render_results_page()

    expected_handlers = [
        "$('backFiveButton').onclick = () => stepFrame(-5);",
        "$('backOneButton').onclick = () => stepFrame(-1);",
        "$('forwardOneButton').onclick = () => stepFrame(1);",
        "$('forwardFiveButton').onclick = () => stepFrame(5);",
        "$('playPauseButton').onclick = togglePlayback;",
        "$('trimAnalysisButton').onclick = trimAnalysisWindowAtCurrentFrame;",
        "$('featureCategorySelect').onchange",
        "$('featureSelect').onchange",
        "$('featureGraph').addEventListener('click', graphClickToFrame);",
        "button.onclick = () => setFrame(Number(button.dataset.filmFrame));",
    ]

    for handler in expected_handlers:
        assert handler in html
    assert "drawLine(ctx, item.series" in html
    assert "point.frame !== previousFrame + 1" in html


def test_analysis_regeneration_plan_uses_selected_boundary_and_human_outputs() -> None:
    case = AnnotationCase(
        slug="case_trim",
        case_id="case_trim_acl",
        source_id="case_trim_view_01",
        player_name="Player Trim",
        video_path="/tmp/source.mp4",
    )

    commands = build_human_analysis_regeneration_commands(
        case,
        movement_start_frame=15,
        movement_end_frame=78,
        data_root="/tmp/acl_data",
        python_executable="/tmp/python",
    )
    flattened = [" ".join(command) for command in commands]

    assert any("scripts/extract_pose.py" in command and "--end-frame 78" in command for command in flattened)
    assert all("/human/" in command for command in flattened)
    assert any("scripts/compute_movement_phases.py" in command for command in flattened)
    assert any("scripts/render_qc_overlay.py" in command and "--end-frame 78" in command for command in flattened)


def test_hidden_unavailable_visual_does_not_reserve_chart_space() -> None:
    html = render_results_page()

    assert ".unavailable[hidden]" in html
    assert "display: none !important;" in html
    assert "$('unavailableVisual').style.display = 'none';" in html
    assert "$('unavailableVisual').style.display = 'grid';" in html


def test_multiview_case_grouping_uses_one_acl_event() -> None:
    primary = AnnotationCase(
        slug="case_a_broadcast",
        case_id="case_a_acl",
        source_id="case_a_broadcast_source",
        view_id="broadcast_01",
        view_label="Broadcast",
        primary_view=True,
        player_name="Player A",
        video_path="/tmp/broadcast.mp4",
    )
    replay = AnnotationCase(
        slug="case_a_replay",
        case_id="case_a_acl",
        source_id="case_a_replay_source",
        view_id="replay_02",
        view_label="Replay",
        primary_view=False,
        player_name="Player A",
        video_path="/tmp/replay.mp4",
    )

    views = views_for_case(primary, (replay, primary))

    assert [view.slug for view in views] == ["case_a_broadcast", "case_a_replay"]
    assert {view.case_id for view in views} == {"case_a_acl"}


def test_multiview_synthesis_selects_view_without_averaging_angles(tmp_path) -> None:
    primary = AnnotationCase(
        slug="case_a_broadcast",
        case_id="case_a_acl",
        source_id="case_a_broadcast_source",
        view_id="broadcast_01",
        view_label="Broadcast",
        primary_view=True,
        perspective="sagittal-like",
        player_name="Player A",
        video_path="/tmp/broadcast.mp4",
    )
    replay = AnnotationCase(
        slug="case_a_replay",
        case_id="case_a_acl",
        source_id="case_a_replay_source",
        view_id="replay_02",
        view_label="Replay",
        primary_view=False,
        perspective="frontal-like",
        player_name="Player A",
        video_path="/tmp/replay.mp4",
    )
    _write_case_summary(
        tmp_path,
        "case_a_broadcast",
        [
            {
                "feature_name": "injured_hka_angle_2d_deg",
                "body_region": "lower_limb",
                "quality_category": "SUPPORTED",
                "analytics_eligibility": "ANALYTICS_READY",
                "analytics_eligible": True,
                "geometry_completeness": 0.90,
                "dynamic_completeness": 0.80,
                "mean": 137.0,
                "minimum": 110.0,
                "maximum": 160.0,
                "range": 50.0,
                "pre_late_change": -20.0,
                "primary_rejection_reason": "",
            },
            {
                "feature_name": "projected_trunk_axis_angle_deg",
                "body_region": "trunk_pelvis",
                "quality_category": "SUPPORTED",
                "analytics_eligibility": "ANALYTICS_READY",
                "analytics_eligible": True,
                "geometry_completeness": 0.70,
                "dynamic_completeness": 0.60,
                "mean": 10.0,
                "minimum": -5.0,
                "maximum": 25.0,
                "range": 30.0,
                "pre_late_change": 12.0,
                "primary_rejection_reason": "",
            },
        ],
    )
    _write_case_summary(
        tmp_path,
        "case_a_replay",
        [
            {
                "feature_name": "injured_hka_angle_2d_deg",
                "body_region": "lower_limb",
                "quality_category": "SUPPORTED",
                "analytics_eligibility": "ANALYTICS_READY",
                "analytics_eligible": True,
                "geometry_completeness": 0.60,
                "dynamic_completeness": 0.40,
                "mean": 148.0,
                "minimum": 120.0,
                "maximum": 170.0,
                "range": 50.0,
                "pre_late_change": -12.0,
                "primary_rejection_reason": "",
            },
            {
                "feature_name": "projected_trunk_axis_angle_deg",
                "body_region": "trunk_pelvis",
                "quality_category": "SUPPORTED",
                "analytics_eligibility": "ANALYTICS_READY",
                "analytics_eligible": True,
                "geometry_completeness": 0.95,
                "dynamic_completeness": 0.90,
                "mean": 30.0,
                "minimum": 10.0,
                "maximum": 45.0,
                "range": 35.0,
                "pre_late_change": -15.0,
                "primary_rejection_reason": "",
            },
        ],
    )

    synthesis = build_case_synthesis_payload(
        primary,
        (primary, replay),
        data_root=tmp_path,
    )

    hka = next(
        item
        for item in synthesis["feature_sources"]
        if item["feature_name"] == "injured_hka_angle_2d_deg"
    )
    trunk = next(
        item
        for item in synthesis["feature_sources"]
        if item["feature_name"] == "projected_trunk_axis_angle_deg"
    )

    assert synthesis["available"] is True
    assert synthesis["rules"]["averages_projected_angles_across_views"] is False
    assert synthesis["rules"]["pools_descriptive_statistics_across_views"] is False
    assert hka["automatic_preferred_view"]["view_slug"] == "case_a_broadcast"
    assert hka["automatic_preferred_view"]["mean"] == 137.0
    assert hka["cross_view_status"] == "CROSS_VIEW_CORROBORATED"
    assert trunk["automatic_preferred_view"]["view_slug"] == "case_a_replay"
    assert trunk["cross_view_status"] == "CROSS_VIEW_DISAGREEMENT"


def test_results_payload_includes_metric_explorer() -> None:
    payload = load_human_results_payload(default_annotation_cases()[0])
    explorer = payload["metric_explorer"]

    assert explorer["selection_modes"] == [
        "WHOLE_MOVEMENT",
        "PHASE",
        "FIVE_FRAME_WINDOW",
        "SINGLE_FRAME",
    ]
    assert len(explorer["metrics"]) >= 70
    assert len(explorer["metrics"]) == sum(
        len(items) for items in explorer["categories"].values()
    )
    assert all(
        spec["preferred_visualisation"] or spec["no_visualisation_reason"]
        for spec in explorer["metrics"].values()
    )
    assert "hka_projected_bilateral_absolute_difference_deg" in explorer["phase_statistics"]


def _write_case_summary(tmp_path, slug: str, rows: list[dict]) -> None:
    path = tmp_path / "analytics" / "human" / f"{slug}_case_feature_summary.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_parquet(path)
