from __future__ import annotations

import os

from acl_motion.annotations.models import AnnotationCase, MovementWindowAnnotation
from acl_motion.annotations.registry import default_annotation_cases, views_for_case
from acl_motion.cases.models import InjurySide
from acl_motion.ui.annotation import render_annotation_page, smoke_test
from acl_motion.ui.results import (
    DEFAULT_ANALYSIS_COMMAND_TIMEOUT_SECONDS,
    POSE_EXTRACTION_TIMEOUT_SECONDS,
    _analysis_command_timeout_seconds,
    _analysis_subprocess_env,
    _phase_withholding_explanation,
    _run_regeneration_command,
    _supported_numeric,
    build_case_synthesis_payload,
    build_human_analysis_regeneration_commands,
    coverage_label,
    explain_status,
    load_human_results_payload,
    load_pose_review_frame_payload,
    load_pose_review_timeline_payload,
    load_result_evidence_payload,
    pose_review_analysis_status,
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
    assert "ACL ligament load or strain" in payload["measurement_boundaries"]["not_estimated"]
    assert payload["case_provenance"]["injured_side"]["inferred_from_movement"] is False
    assert payload["case_provenance"]["injury_confirmation"]["inferred_from_video"] is False
    assert payload["source_files"]["movement_profile"].endswith(
        "profiles/human/christen_press_movement_profile.json"
    )
    assert all("/human/" in path for path in payload["source_files"].values())
    assert all(not path.startswith("/") for path in payload["source_files"].values())
    assert "semantic_observations" in payload["source_files"]
    assert "observable_descriptions" in payload["source_files"]
    assert "movement_phases" in payload["source_files"]
    assert "observable_movement_descriptions" in payload
    assert payload["observable_movement_descriptions"]["supported_intervals"][0]["start_frame"] == 0
    assert (
        payload["observable_movement_descriptions"]["supported_intervals"][0]["end_frame"]
        <= payload["movement_window"]["movement_end_frame"]
    )
    assert len(payload["observable_movement_descriptions"]["default_story_descriptions"]) <= 4
    assert payload["path_quality_summary"]["overall_status"] in {"SUPPORTED", "UNAVAILABLE", "QA_REQUIRED"}
    assert payload["movement_story"]["status"] in {
        "SUPPORTED",
        "SUPPORTED_PARTIAL_WINDOW",
        "SUPPORTED_EVIDENCE_INTERVAL",
        "INSUFFICIENT_EVIDENCE_FOR_PHASE_SEGMENTATION",
    }
    assert payload["movement_story"]["phase_count"] >= 0
    assert payload["event_interval_review"]["question"] == (
        "Does the supported phase interval include the visible event you intended to study?"
    )
    assert payload["event_interval_review"]["decision"] in {None, "yes", "no"}
    coverage = payload["geometry_coverage_evidence"]
    assert coverage["definition_version"] == "feature_aware_geometry_coverage_v1"
    assert coverage["movement_window_total_frames"] > 0
    assert any(
        item["name"] == "Movement-window geometry"
        for item in payload["evidence_dimensions"]
    )
    assert any(
        item["name"] == "Reviewed-frame geometry yield"
        for item in payload["evidence_dimensions"]
    )
    assert any(
        item["name"] == "Target-present geometry yield"
        for item in payload["evidence_dimensions"]
    )
    assert payload["movement_visual_story"]["story_version"] == "m5_9_movement_first_visual_story_v1"
    if payload["movement_story"]["phase_count"]:
        assert payload["movement_visual_story"]["phases"][0]["snapshot_frames"]
        assert payload["movement_visual_story"]["phases"][0]["observations"]
    else:
        assert payload["movement_visual_story"]["phases"] == []
        assert payload["movement_visual_story"]["whole_movement"]["phase_sequence"] == []


def test_results_payload_lists_only_completed_analyses() -> None:
    cases = tuple(default_annotation_cases())
    payload = load_human_results_payload(cases[0], analysis_cases=cases)
    analyses = payload["available_analyses"]
    slugs = {item["slug"] for item in analyses}

    assert "christen_press" in slugs
    assert "case_01" in slugs
    assert "leah_williamson_broadcast_wide" not in slugs
    assert next(item for item in analyses if item["slug"] == "christen_press")[
        "current"
    ] is True
    assert all(item["player_name"] and item["view_label"] for item in analyses)


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
    assert "reviewed_frame_yield" in card
    assert "target_present_yield" in card
    assert "longest_continuous_supported" in card


def test_unsupported_numeric_is_suppressed_in_user_facing_payloads() -> None:
    assert _supported_numeric(0.0, "UNAVAILABLE") is None
    assert _supported_numeric(123.4, "LOW_CONFIDENCE") is None
    assert _supported_numeric(12.3, "SUPPORTED") == 12.3


def test_results_page_avoids_front_facing_risk_language() -> None:
    html = render_results_page()

    assert ".results-header-actions {" in html
    assert "flex-wrap: nowrap;" in html
    assert "margin-bottom: 6px;" in html
    assert ".overview-intro {" in html
    assert "font-size: clamp(12.5px, 0.74vw, 14px);" in html
    assert "white-space: nowrap;" in html
    assert html.index('id="overviewSupportStatus"') < html.index('id="overviewIntro"')
    assert "featureCategorySelect" in html
    assert "featureSelect" in html
    assert "featureGraph" in html
    assert 'id="resultsMoreActions"' in html
    assert "More actions" in html
    assert 'id="resultsHeaderTitle"' in html
    assert 'id="analysisCaseSelect"' in html
    assert 'id="openAnalysisButton"' in html
    assert 'id="contextVideoButton"' in html
    assert 'id="contextVideoDialog"' in html
    assert 'id="contextVideoPlayer"' in html
    assert "This video is not used to calculate measurements" in html
    assert "generate the movement summary, or infer contact or causation" in html
    assert "videoFrame" in html
    assert "Movement Story" in html
    assert "Whole Movement Summary" in html
    assert "How phases were divided" in html
    assert "A one-frame spike or a gap in evidence does not create a boundary" in html
    assert "Why this phase starts here" in html
    assert "function phaseStartRationale" in html
    assert "During the opening phase" in html
    assert "After this transition" in html
    assert "Phase summary" in html
    assert "What happened?" not in html
    assert "What defines this phase?" not in html
    assert "supported multidimensional movement change" not in html
    assert "['trunk_pelvis', 'upper_body', 'hip_knee_ankle_chain']" in html
    assert "Main movement changes" in html
    assert "Important frames" in html
    assert "Movement phase story" not in html
    assert "Movement narrative" not in html
    assert 'id="narrativePanel"' not in html
    assert 'id="whyPhasePanel"' not in html
    assert "backFiveButton" in html
    assert "backOneButton" in html
    assert "playPauseButton" in html
    assert "forwardOneButton" in html
    assert "forwardFiveButton" in html
    assert "trimAnalysisButton" in html
    assert "End analysis here + regenerate" in html
    assert '<details class="boundary-control" id="analysisBoundaryControl">' in html
    assert "Researcher action · annotation or regeneration" in html
    assert 'id="editAnnotationButton"' in html
    assert "Edit or extend annotation" in html
    assert "&mode=edit" in html
    assert "analysedViewCount" in html
    assert "not analysed" in html
    assert "analysed view" in html
    assert "View list and alignment note" not in html
    assert "<summary>Alignment note</summary>" in html
    assert "Selected Measurement" in html
    assert 'id="measurementScope"' in html
    assert "Scope: Whole movement · Frames" in html
    assert "Descriptive statistics" in html
    assert "Hip–knee–ankle configuration" in html
    assert "Knee–ankle relationship" in html
    assert "Trunk orientation" in html
    assert "trajectoryInterpretation" in html
    assert "Measurement Support" in html
    assert "supported</span>" in html
    assert "Why are some frames unsupported?" in html
    assert "Target identity was uncertain because the annotated athlete overlapped another player" in html
    assert "Technical status:" in html
    assert "Evidence support" not in html
    assert "Advanced Evidence Details" in html
    assert "Selected-frame QC and raw status" in html
    assert "Case and source provenance" in html
    assert "Cross-case readiness" not in html
    assert "Exact narrative matching" not in html
    assert "LOWER LIMB" in html
    assert "BILATERAL" in html
    assert "Left/right projected HKA" in html
    assert "No measurements available" in html
    assert "availableCategories()" in html
    assert "Timing means the source-frame position" in html
    assert "left-minus-right projected HKA difference" in html
    assert "TRUNK & PELVIS" in html
    assert "UPPER BODY" in html
    assert "TIMING" in html
    assert "'MOVEMENT PATH':" not in html
    assert "drawPhaseBilateralMini" in html
    assert 'data-phase-hka-view="' in html
    assert "Hip–knee–ankle configuration" in html
    assert "drawPhaseOppositeHkaMini" in html
    assert "Injured" in html
    assert "Opposite" in html
    assert "Compare" in html
    assert "Blue dashed = phase start" in html
    assert "Green solid = phase end" in html
    assert "drawPhaseInjuredHkaMini" in html
    assert "drawAngleArc" in html
    assert "Injured projected HKA" in html
    assert "Opposite projected HKA" in html
    assert "function phaseHkaNarrative" in html
    assert "contralateral_hka_angle_2d_deg" in html
    assert "the opposite-side projected hip–knee–ankle configuration" in html
    assert "several measured movement features changed together" in html
    assert "Evidence is moderate because usable measurements were available for only part of this interval" in html
    assert "opened" in html
    assert "more closed through the phase" in html
    assert "drawPhaseTorsoMini" in html
    assert "projected_trunk_axis_angle_deg" in html
    assert "Shoulder line" in html
    assert "drawPhaseUpperBodyMini" in html
    assert "phaseEvidenceDetails.ontoggle" in html
    assert '.phase-mini-visual[data-story-category="upper_body"]' in html
    assert "category === 'upper_body' ? 168 : 132" in html
    assert "fitArmStatesToCanvas" in html
    assert "drawAngleArc(ctx, start[1], start[0], start[2], '#215f9a', true)" in html
    assert 'data-phase-upper-body-view="' in html
    assert 'data-upper-body-mode="' in html
    assert "Right elbow" in html
    assert "Left elbow" in html
    assert "phaseUpperBodyViewModes" in html
    assert "drawPhaseUpperBodySideMini" in html
    assert "drawUpperBodyAngleLabels" not in html
    assert "projected arm orientation and elbow configuration" in html
    assert "Injured limb (" in html
    assert "Opposite limb (" in html
    assert "hka_projected_bilateral_absolute_difference_deg" in html
    assert "Peak gap " in html
    assert "Configuration = shape formed by connected hip, knee, and ankle segments" in html
    assert "Axial reorientation" in html
    assert "Directed: endpoint order matters" in html
    assert "Axis: endpoint order does not matter" in html
    assert "Raw start orientation" in html
    assert "Raw end orientation" in html
    assert "drawWholeMovementPhaseBoundaries" in html
    assert "P' + phase.phase_index + ' begins" in html
    assert "const phaseColor = '#147d73'" in html
    assert "const phaseLabelBackground = '#e6f5f2'" in html
    assert "ctx.fillText('phase begins'" in html
    assert "rgba(255, 208, 80, 0.30)" in html
    assert "salientSnapshotLabel" in html
    assert "Largest supported " in html
    assert "MOST CLOSED" in html
    assert "MOST OPEN" in html
    assert "Minimum HKA · " in html
    assert "Maximum HKA · " in html
    assert "positive image x-axis" in html
    assert "counter-clockwise on screen" in html
    assert "canonical_signed_change" in html
    assert "value: finiteNumberOrNull(point.value)" in html
    assert "function finiteNumberOrNull(value)" in html
    assert "function canonicalAngleDifference" in html
    assert "function canonicalMeasurementRange" in html
    assert "Wrap-aware movement range" in html
    assert "if (change === null)" in html
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
    assert "Observed / measured" in html
    assert "Movement at a glance" in html
    assert "Cross-case context" not in html
    assert 'id="similarCasesPanel"' not in html
    assert "ACL ruptured because" not in html
    assert "same biological cause" not in html
    assert "critical plant" not in html


def test_annotation_ui_smoke_includes_results_entry_point() -> None:
    result = smoke_test()

    assert result["html_has_generate_analysis"] is True
    assert result["html_has_injured_knee"] is True
    assert result["html_has_pose_review"] is True
    assert result["html_has_view_analysis"] is True
    assert result["results_html_has_single_feature_ui"] is True


def test_annotation_ui_uses_one_yolov8n_analysis_path() -> None:
    html = render_annotation_page()

    assert "Skeleton model · YOLOv8n" in html
    assert "established YOLOv8n pose workflow" in html
    assert 'id="generateAnalysis"' in html
    assert 'id="poseProfile"' not in html
    assert 'id="poseComparisonDialog"' not in html
    assert 'id="runModelSensitivity"' not in html


def test_annotation_ui_protects_unsaved_work_and_explains_recovery() -> None:
    html = render_annotation_page()

    assert "annotationDirty: false" in html
    assert 'window.addEventListener("beforeunload"' in html
    assert "This annotation has unsaved changes" in html
    assert 'setSaveFeedback("Unsaved changes.", "unsaved")' in html
    assert "your edits remain on this screen" in html
    assert "annotation is still saved" in html
    assert "Your saved annotation has not been changed" in html


def test_annotation_frame_arrows_survive_workspace_button_and_canvas_focus() -> None:
    html = render_annotation_page()

    assert 'window.addEventListener("keydown", handleWorkspaceKeydown);' in html
    assert 'document.addEventListener("pointerdown", releaseTypingFocusForWorkspacePointer, true);' in html
    assert "function isFrameNavigationEditingTarget(target)" in html
    assert "function releaseTypingFocusForWorkspacePointer(event)" in html
    assert 'target.closest("textarea, select, input")' in html
    assert '"button", "checkbox", "radio", "range", "reset", "submit"' in html
    assert 'event.key !== "ArrowLeft" && event.key !== "ArrowRight"' in html
    assert 'event.preventDefault();\n  loadFrame(app.frame + (event.key === "ArrowLeft" ? -1 : 1));' in html
    assert '["TEXTAREA", "INPUT", "SELECT", "BUTTON"].includes(event.target.tagName)' not in html


def test_results_ui_uses_one_yolov8n_analysis_path() -> None:
    html = render_results_page()

    assert 'id="headerClipSelect"' in html
    assert "'YOLOv8n'" in html
    assert 'id="analysisAvailabilityNotice"' in html
    assert 'id="eventIntervalReview"' in html
    assert 'id="eventIntervalReviewYes"' in html
    assert 'id="eventIntervalReviewNo"' in html
    assert "Does the supported phase interval include the visible event you intended to study?" in html
    assert 'id="eventIntervalReviewUnclear"' not in html
    assert "/api/results/event-interval-review" in html
    assert "This is an evidence limitation, not a missing or failed analysis." in html
    assert 'id="modelSensitivityPanel"' not in html
    assert 'id="headerModelSelect"' not in html
    assert 'id="modelHelpDialog"' not in html
    assert 'id="modelSkeletonDialog"' not in html
    assert "modelQuery" not in html


def test_results_ui_remembers_workspace_and_keeps_video_with_measurements_in_focus_mode() -> None:
    html = render_results_page()

    assert 'id="analysisFocusWorkspace"' in html
    assert 'id="focusModeButton"' in html
    assert "analysis-focus-active" in html
    assert html.index('id="videoReviewPanel"') < html.index('id="featurePanel"')
    assert "acl-analysis-workspace:v" in html
    assert "window.localStorage.setItem" in html
    assert "selectedFeatureId" in html
    assert "currentFrame" in html
    assert "Space: play" in html
    assert "event.shiftKey ? -5 : -1" in html
    assert "No saved annotation or previous analysis was changed" in html

def test_annotation_edit_mode_reopens_saved_context_and_requires_resave() -> None:
    html = render_annotation_page()

    assert 'id="editWorkflowHint"' in html
    assert 'id="injuredSide"' in html
    assert '<option value="left">Left knee</option>' in html
    assert '<option value="right">Right knee</option>' in html
    assert "Editing an existing analysis" in html
    assert 'params.get("frame")' in html
    assert 'params.get("mode") === "edit"' in html
    assert "await loadCase(initialCase.slug, requestedFrame);" in html
    assert 'params.get("focus")' not in html
    assert "markAnnotationDirty();" in html
    assert "Save as ready for validation before regenerating changed annotations." in html


def test_annotation_canvas_preserves_portrait_video_aspect_ratio() -> None:
    html = render_annotation_page()

    assert "width: auto;" in html
    assert "height: auto;" in html
    assert "max-width: 100%;" in html
    assert "max-height: 68vh;" in html


def test_annotation_edit_mode_exposes_previous_pose_and_qc_without_hiding_raw_video() -> None:
    html = render_annotation_page()

    assert 'data-review-mode="video"' in html
    assert 'data-review-mode="roi"' in html
    assert 'id="previousPoseMode"' in html
    assert 'data-review-mode="pose"' in html
    assert "Video only" in html
    assert "Annotation ROI" in html
    assert "Skeleton review + QC" in html
    assert "Accept reviewed skeleton frames" in html
    assert 'id="startAccepted"' in html
    assert 'id="endAccepted"' in html
    assert 'id="removeAccepted"' in html
    assert "raw YOLOv8 skeleton" in html
    assert 'id="analysisUseBadge"' in html
    assert 'id="currentReviewBadge"' in html
    assert 'id="poseAnalysisTimeline"' in html
    assert "not used: insufficient evidence" in html
    assert "Pending change—validate, save, and regenerate" in html
    assert 'app.editMode && app.hasPreviousPoseReview ? "pose" : "roi"' in html
    assert "/api/pose-review/frame" in html
    assert "/api/pose-review?case=" in html
    assert "Previous analysis is now stale" in html
    assert "if (app.hasPreviousPoseReview) app.previousPoseStale = true;" in html


def test_pose_review_playback_keeps_the_viewer_stable() -> None:
    html = render_annotation_page()

    assert "height: 104px;" in html
    assert "-webkit-line-clamp: 2;" in html
    assert "frameImageRequest: 0" in html
    assert "requestedFrame === app.frame" in html
    assert "await Promise.all([imageReady, poseReviewReady]);" in html
    assert "async function advanceReviewPlayback()" in html
    assert "app.reviewTimer = setTimeout(advanceReviewPlayback" in html
    assert "setInterval(" not in html


def test_pose_review_payload_explains_frame_qc_and_staleness(tmp_path) -> None:
    import pandas as pd

    case = AnnotationCase(
        slug="pose_review_case",
        case_id="pose_review_case_acl",
        source_id="pose_review_case_view",
        player_name="Pose Review Player",
        video_path="/tmp/pose_review.mp4",
    )
    quality_path = (
        tmp_path / "quality" / "human" / "pose_review_case_frame_quality.csv"
    )
    pose_path = (
        tmp_path / "processed" / "human" / "pose_review_case_processed_pose.parquet"
    )
    session_path = (
        tmp_path
        / "annotations"
        / "human"
        / "pose_review_case_annotation_session_human.json"
    )
    quality_path.parent.mkdir(parents=True)
    pose_path.parent.mkdir(parents=True)
    session_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "source_frame_index": 12,
                "frame_status": "TARGET_IDENTITY_UNCERTAIN",
                "frame_rejection_reason": "Player overlap made the target pose uncertain.",
                "observed_landmark_count": 2,
                "median_confidence": 0.81,
                "valid_target_frame": False,
                "valid_segment_id": None,
            }
        ]
    ).to_csv(quality_path, index=False)
    pd.DataFrame(
        [
            {
                "source_frame_index": 12,
                "landmark_name": "left_knee",
                "observed": True,
                "interpolated": False,
                "rejected": False,
                "processing_status": "SMOOTHED",
                "smoothed_x": 10.0,
                "smoothed_y": 20.0,
                "clean_x": 10.0,
                "clean_y": 20.0,
            },
            {
                "source_frame_index": 12,
                "landmark_name": "left_ankle",
                "observed": True,
                "interpolated": False,
                "rejected": True,
                "processing_status": "REJECTED_IDENTITY_UNCERTAIN",
                "smoothed_x": None,
                "smoothed_y": None,
                "clean_x": None,
                "clean_y": None,
            },
        ]
    ).to_parquet(pose_path)
    session_path.write_text("{}", encoding="utf-8")
    os.utime(quality_path, (100.0, 100.0))
    os.utime(pose_path, (100.0, 100.0))
    os.utime(session_path, (200.0, 200.0))

    status = pose_review_analysis_status(case, tmp_path)
    payload = load_pose_review_frame_payload(
        case,
        source_frame_index=12,
        data_root=tmp_path,
    )
    timeline = load_pose_review_timeline_payload(case, data_root=tmp_path)

    assert status["available"] is True
    assert status["stale"] is True
    assert payload["status_label"] == "Target continuity uncertain"
    assert payload["status_tone"] == "uncertain"
    assert payload["observed_landmark_count"] == 2
    assert payload["usable_landmark_count"] == 1
    assert payload["rejected_landmark_count"] == 1
    assert payload["median_confidence"] == 0.81
    assert payload["frame_rejection_reason"].startswith("Player overlap")
    assert payload["automatic_frame_status"] == "TARGET_IDENTITY_UNCERTAIN"
    assert payload["manual_review_decision"] == "NOT_REVIEWED"
    assert payload["manual_override_applied"] is False
    assert payload["raw_pose_available"] is False
    assert payload["used_in_analysis"] is False
    assert payload["analysis_use_label"] == "Not used in previous analysis"
    assert "could not confidently separate" in payload["analysis_use_reason"]
    assert payload["skeleton_display_note"].endswith("not included in measurements.")
    assert timeline["intervals"] == [
        {
            "start_frame": 12,
            "end_frame": 12,
            "frame_count": 1,
            "state": "INSUFFICIENT_EVIDENCE",
            "label": "Not used: insufficient evidence",
            "tone": "uncertain",
        }
    ]


def test_annotation_ui_uses_compact_progressive_disclosure() -> None:
    html = render_annotation_page()

    assert 'class="panel annotation-sidebar"' in html
    assert 'id="annotationProgress"' in html
    assert 'class="case-status-strip"' in html
    assert 'class="playback-toolbar" aria-label="Playback and frame navigation"' in html
    assert 'aria-label="Back 5 frames"' in html
    assert 'aria-label="Forward 5 frames"' in html
    assert 'id="roiStep" class="workflow-section workflow-step" open' in html
    assert html.count('class="workflow-section workflow-step" open') == 1
    assert 'id="injuryStep" class="workflow-section workflow-step">' in html
    assert 'id="movementStep" class="workflow-section workflow-step">' in html
    assert 'id="reviewStep" class="workflow-section workflow-step">' in html
    assert "Correct athlete tracking" in html
    assert "Case information" in html
    assert "Validate and generate" in html
    assert 'class="danger-subtle">Delete current keyframe</button>' in html
    assert 'id="unavailableDetails" class="workflow-disclosure priority-disclosure"' in html
    assert "Exclude unreliable frames" in html
    assert "Use when the athlete cannot be identified defensibly" in html
    assert "These frames remain unmeasured." in html
    assert "Start excluded interval" in html
    assert "Correction details" in html
    assert "Confidence and rationale" in html
    assert 'class="sidebar-action-dock" aria-label="Annotation save actions"' in html
    assert "Validate &amp; save" in html
    assert 'id="advancedQa" class="advanced-qa"' in html
    assert "Advanced / QA" in html
    assert "Activity and validation" in html
    assert "Session notes" in html
    assert "Development comparison" in html
    assert 'id="analysisProgress" class="analysis-progress" role="status" aria-live="polite" hidden' in html
    assert 'button.textContent = "Generating...";' not in html
    assert "syncWorkflowSteps(true);" in html
    assert '`${essentials}/3 analysis essentials`' in html
    assert "bindWorkflowAccordion" in html
    assert "if (other !== step) other.open = false;" in html
    assert '$(stepId).open = stepId === id;' in html
    assert 'summary.setAttribute("aria-current", "step")' in html
    assert 'Select the documented injured knee before generating analysis.' in html
    assert "refreshWorkflowState" in html
    assert ":focus-visible" in html
    assert "prefers-reduced-motion: reduce" in html
    assert "The interval is inclusive" not in html


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
    assert "<summary>Descriptive statistics</summary>" in html
    assert '<details class="panel workspace-disclosure" id="featurePanel" open>' in html
    assert '<details class="panel workspace-disclosure" id="phaseStoryPanel">' in html
    assert '<details class="panel workspace-disclosure" id="operatorAnalyticsPanel">' in html
    assert html.index('id="videoReviewPanel"') < html.index('id="featurePanel"')
    assert html.index('id="featurePanel"') < html.index('id="phaseStoryPanel"')
    assert html.index('id="featureGraph"') < html.index('class="measurement-change-block"')
    assert '<details id="advancedEvidenceDetails">' in html
    assert "<summary>Advanced Evidence Details</summary>" in html
    assert '<details id="evidenceDetails">' not in html
    assert '<details id="technicalDetails">' not in html
    assert 'id="researchMeasurements"' not in html
    assert 'id="exploreMovement"' not in html


def test_results_page_prioritises_story_measurements_and_accessible_states() -> None:
    html = render_results_page()

    assert 'id="resultsLoading" role="status" aria-live="polite"' in html
    assert 'id="resultsContent" hidden' in html
    assert 'id="resultOverview"' in html
    assert "Human-guided 2D movement analysis" in html
    assert "Movement at a glance" in html
    assert "function movementAtAGlance(story, phases)" in html
    assert "$('overviewStory').textContent = movementAtAGlance(story, phases);" in html
    assert "Whole Movement Summary" in html
    assert html.index('id="movementStoryAction"') < html.index('id="overviewMeasurements"')
    assert 'id="overviewStory"' in html
    assert 'id="overviewMeasurements"' in html
    assert "Key observable measurements" in html
    assert "Watch the selected movement" in html
    assert 'id="fullscreenReviewButton"' in html
    assert "function initialiseFullscreenReview()" in html
    assert 'aria-label="Back 5 frames"' in html
    assert 'aria-label="Forward 5 frames"' in html
    assert 'role="img" aria-label="Selected projected movement measurement across source frames"' in html
    assert 'id="similarCasesPanel"' not in html
    assert "Cross-case context" not in html
    assert "Compare same-profile similarity rankings" not in html
    assert "Measurement support &amp; Responsible AI" in html
    assert "Unsupported values remain unavailable; they are not displayed as zero." in html
    assert "prefers-reduced-motion: reduce" in html
    assert ":focus-visible" in html


def test_results_page_keeps_video_visible_beside_every_analysis_section() -> None:
    html = render_results_page()

    assert 'class="persistent-video-column"' in html
    assert "This synchronized video remains visible while you inspect every analysis section." in html
    assert 'id="researchResultsColumn"' in html
    assert 'id="resultsSectionNav" aria-label="Analysis sections"' in html
    assert 'data-section-target="resultOverview"' in html
    assert 'data-section-target="featurePanel"' in html
    assert 'data-section-target="phaseStoryPanel"' in html
    assert 'data-section-target="evidenceSection"' in html
    assert "function initialiseSectionNavigation()" in html
    assert "section.scrollIntoView({behavior: 'smooth', block: 'start'});" in html
    assert "position: sticky;" in html
    assert "grid-template-columns: minmax(380px, 0.9fr) minmax(0, 1.55fr);" in html
    assert html.index('id="videoReviewPanel"') < html.index('id="researchResultsColumn"')


def test_results_page_explains_all_supported_intervals_and_circular_statistics() -> None:
    html = render_results_page()

    assert "const supportedRanges = coverage.supported_source_ranges || [];" in html
    assert "Usable measurement interval" in html
    assert "supportedRanges.map((range)" in html
    assert "Movement-window frames" in html
    assert "function circularMean(values, angleType)" in html
    assert "function circularStandardDeviation(values, angleType)" in html
    assert "Axial circular mean" in html
    assert "Circular SD" in html
    assert "Wrap-aware movement range" in html


def test_results_page_keeps_same_case_clip_navigation_in_context() -> None:
    html = render_results_page()

    assert 'id="annotateNextClipButton"' in html
    assert "Annotate next clip" in html
    assert 'id="caseClipsButton"' in html
    assert "All case clips" in html
    assert "function caseViewUrl(view)" in html
    assert "view.results_available" in html
    assert "view.annotation_saved" in html
    assert "?case=" in html


def test_results_page_explains_when_phase_segmentation_is_withheld() -> None:
    html = render_results_page()

    assert 'id="phaseStoryLegend"' in html
    assert "PHASES WITHHELD" in html
    assert "Phase segmentation unavailable" in html
    assert "phaseEvidenceShortfall(story)" in html
    assert "Why phases were withheld" in html
    assert "Why phases were not generated" in html
    assert "No AI or generative model was used" in html
    assert "What the software can and cannot conclude" in html
    assert "Inspect supported measurements" in html
    assert "Frame QC records what evidence failed" not in html
    assert "explanation.cause_note" in html
    assert "if (!phases.length) {\n    panel.hidden = true;" not in html
    assert "Best reviewed-frame yield" in html
    assert "Best target-present yield" in html
    assert "Reviewed accepted-frame yield" in html
    assert "Movement-window coverage" in html
    assert "PARTIAL_MOVEMENT_WINDOW" in html
    assert "frames outside this supported block remain unsegmented" in html


def test_results_page_distinguishes_one_interval_from_a_phase_story() -> None:
    html = render_results_page()

    assert "function isSupportedEvidenceInterval" in html
    assert "function phaseDecisionJustification" in html
    assert "SUPPORTED EVIDENCE INTERVAL" in html
    assert "Supported Evidence Interval · measurement detail" in html
    assert "No supported transition detected" in html
    assert "Why phases were not generated" in html
    assert "Why the AI did not generate phases" not in html
    assert "Includes annotated Movement End:" in html
    assert "One supported interval is available; no before/after phase comparison is claimed." in html


def test_results_graph_preserves_measurements_from_short_valid_sequences() -> None:
    html = render_results_page()

    assert "function measuredSeries(metric)" in html
    assert "function limitedMeasuredSeries(metric)" in html
    assert "series: measuredSeries(metric)" in html
    assert "trendSeries: supportedSeries(metric)" in html
    assert "drawLimitedLine(ctx, item.limitedSeries" in html
    assert "measured · short / trend-limited sequence" in html
    assert "excluded from higher-level trend summaries" in html
    assert "const rows = measuredSeries(metric);" in html


def test_phase_withholding_explanation_separates_pose_geometry_and_phase_rules() -> None:
    explanation = _phase_withholding_explanation(
        {
            "status": "INSUFFICIENT_EVIDENCE_FOR_PHASE_SEGMENTATION",
            "phases": [],
            "eligible_descriptors": [],
            "excluded_descriptors": [
                {
                    "kind": "geometry",
                    "feature_name": "left_hka_angle_2d_deg",
                    "geometry_completeness": 74 / 107,
                }
            ],
            "metadata": {
                "configuration": {
                    "minimum_geometry_completeness": 0.70,
                    "minimum_eligible_descriptors": 4,
                }
            },
        },
        {
            "pose_frame_coverage": 94 / 107,
            "frame_status_counts": {
                "VALID_TARGET": 74,
                "TARGET_NOT_FOUND": 13,
                "INVALID_TRACK_SEGMENT": 13,
                "TARGET_IDENTITY_UNCERTAIN": 7,
            },
        },
        MovementWindowAnnotation(
            movement_start_frame=25,
            movement_end_frame=131,
            movement_start_timestamp_ms=0.0,
            movement_end_timestamp_ms=1766.7,
        ),
        {
            "best_movement_window_feature": {
                "feature_name": "left_hka_angle_2d_deg",
                "movement_window_coverage": 74 / 107,
            },
            "best_reviewed_feature": {
                "feature_name": "left_hka_angle_2d_deg",
                "reviewed_supported_frames": 22,
                "reviewed_total_frames": 23,
                "reviewed_frame_yield": 22 / 23,
            },
            "best_target_present_feature": {
                "feature_name": "left_hka_angle_2d_deg",
                "target_present_supported_frames": 72,
                "target_present_total_frames": 74,
                "target_present_yield": 72 / 74,
            },
            "best_continuous_feature": {
                "feature_name": "left_hka_angle_2d_deg",
                "longest_continuous_supported": {
                    "start_frame": 40,
                    "end_frame": 61,
                    "frame_count": 22,
                    "duration_ms": 733.3,
                },
            },
        },
    )

    assert explanation["withheld"] is True
    assert explanation["total_frames"] == 107
    assert explanation["pose"]["frames"] == 94
    assert explanation["defensible_target"]["frames"] == 74
    assert explanation["best_geometry"]["frames"] == 74
    assert explanation["reviewed_geometry"]["frames"] == 22
    assert explanation["reviewed_geometry"]["total_frames"] == 23
    assert explanation["target_present_geometry"]["frames"] == 72
    assert explanation["continuous_geometry"]["start_frame"] == 40
    assert explanation["continuous_geometry"]["end_frame"] == 61
    assert explanation["phase_rule"] == {
        "minimum_geometry_coverage": 0.70,
        "eligible_descriptors": 0,
        "minimum_eligible_descriptors": 4,
    }
    assert {item["status"] for item in explanation["frame_reasons"]} == {
        "TARGET_NOT_FOUND",
        "INVALID_TRACK_SEGMENT",
        "TARGET_IDENTITY_UNCERTAIN",
    }
    assert "cannot by itself prove" in explanation["cause_note"]
    assert "not silently filled" in explanation["availability_note"]


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
    assert "drawLine(ctx, item.trendSeries" in html
    assert "drawLimitedLine(ctx, item.limitedSeries" in html
    assert "point.frame !== previousFrame + 1" in html


def test_analysis_regeneration_plan_uses_selected_boundary_and_human_outputs() -> None:
    case = AnnotationCase(
        slug="case_trim",
        case_id="case_trim_acl",
        source_id="case_trim_view_01",
        player_name="Player Trim",
        video_path="/tmp/source.mp4",
        injured_side=InjurySide.RIGHT,
        injury_laterality_source="human_operator_test",
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
    assert any(
        "scripts/extract_pose.py" in command
        and "--annotation-session /tmp/acl_data/annotations/human/case_trim_annotation_session_human.json"
        in command
        for command in flattened
    )
    assert any(
        "scripts/extract_pose.py" in command and "--roi-pad 0.0" in command
        for command in flattened
    )
    assert any(
        "scripts/extract_pose.py" in command
        and "--model-path /tmp/acl_data/models/yolov8n-pose.pt" in command
        and "--yolo-selection-strategy largest" in command
        and "--yolo-image-size 640" in command
        and "--yolo-detection-confidence 0.25" in command
        and "--yolo-landmark-confidence 0.25" in command
        for command in flattened
    )
    assert all("--pose-profile-id" not in command for command in flattened)
    assert all("/human/" in command for command in flattened)
    assert any("scripts/compute_movement_phases.py" in command for command in flattened)
    assert any(
        "scripts/compute_geometry_features.py" in command and "--injured-side right" in command
        for command in flattened
    )
    assert any("scripts/render_qc_overlay.py" in command and "--end-frame 78" in command for command in flattened)


def test_analysis_regeneration_plan_rejects_negative_roi_context_padding() -> None:
    case = AnnotationCase(
        slug="case_padding",
        case_id="case_padding_acl",
        source_id="case_padding_view",
        player_name="Player Padding",
        video_path="/tmp/source.mp4",
    )

    try:
        build_human_analysis_regeneration_commands(
            case,
            movement_start_frame=0,
            movement_end_frame=10,
            roi_context_padding_fraction=-0.1,
        )
    except ValueError as exc:
        assert str(exc) == "ROI context padding fraction cannot be negative."
    else:
        raise AssertionError("Negative ROI context padding should be rejected.")


def test_pose_extraction_gets_a_cold_start_timeout_without_weakening_other_stages() -> None:
    assert _analysis_command_timeout_seconds(
        ["/tmp/python", "scripts/extract_pose.py"]
    ) == POSE_EXTRACTION_TIMEOUT_SECONDS
    assert _analysis_command_timeout_seconds(
        ["/tmp/python", "scripts/process_pose_quality.py"]
    ) == DEFAULT_ANALYSIS_COMMAND_TIMEOUT_SECONDS


def test_analysis_subprocess_uses_one_macos_openmp_runtime(
    monkeypatch, tmp_path
) -> None:
    from acl_motion.ui import results

    environment_root = tmp_path / "analysis_env"
    python_executable = environment_root / "bin" / "python3.11"
    torch_library_dir = (
        environment_root
        / "lib"
        / "python3.11"
        / "site-packages"
        / "torch"
        / "lib"
    )
    python_executable.parent.mkdir(parents=True)
    python_executable.touch()
    torch_library_dir.mkdir(parents=True)
    (torch_library_dir / "libomp.dylib").touch()
    monkeypatch.setattr(results.sys, "platform", "darwin")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/existing/runtime")

    env = _analysis_subprocess_env(python_executable)

    assert env["DYLD_LIBRARY_PATH"].split(os.pathsep) == [
        str(torch_library_dir),
        "/existing/runtime",
    ]
    assert "KMP_DUPLICATE_LIB_OK" not in env


def test_analysis_timeout_becomes_a_clean_value_error(monkeypatch) -> None:
    from acl_motion.ui import results

    observed = {}

    def time_out(command, **kwargs):
        observed["timeout"] = kwargs["timeout"]
        raise results.subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(results.subprocess, "run", time_out)

    try:
        _run_regeneration_command(["/tmp/python", "scripts/extract_pose.py"])
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("A subprocess timeout should become a ValueError response.")

    assert observed["timeout"] == POSE_EXTRACTION_TIMEOUT_SECONDS
    assert "YOLO/PyTorch startup and pose extraction" in message
    assert "No completed results were published" in message


def test_analysis_regeneration_uses_explicit_human_laterality() -> None:
    case = AnnotationCase(
        slug="case_laterality",
        case_id="case_laterality_acl",
        source_id="case_laterality_view",
        player_name="Player Laterality",
        video_path="/tmp/source.mp4",
        injured_side=InjurySide.RIGHT,
    )

    commands = build_human_analysis_regeneration_commands(
        case,
        movement_start_frame=0,
        movement_end_frame=10,
        data_root="/tmp/acl_data",
        injured_side=InjurySide.LEFT,
    )
    geometry_command = next(
        command for command in commands if "scripts/compute_geometry_features.py" in command
    )

    assert geometry_command[geometry_command.index("--injured-side") + 1] == "left"


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
