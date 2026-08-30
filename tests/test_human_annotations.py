from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from acl_motion.analytics.guard import (
    CROSS_CASE_ANALYTIC_NAMES,
    CrossCaseAnalyticsUnavailable,
    require_cross_case_analytics_ready,
)
from acl_motion.annotations.case_intake import injury_case_options, register_analysis_clip
from acl_motion.annotations.models import (
    AnnotationProvenance,
    EventConfidence,
    HumanAnnotationSession,
    MovementWindowAnnotation,
    OperatorFlag,
    RoiKeyframeAnnotation,
    TargetAcceptedIntervalAnnotation,
    TargetUnavailableIntervalAnnotation,
)
from acl_motion.annotations.movement_window import (
    add_movement_timing_columns,
    filter_to_movement_window,
    infer_movement_start_frame,
    migrate_session_to_movement_window,
)
from acl_motion.annotations.propagation import propagated_bbox
from acl_motion.annotations.registry import case_by_slug, imported_annotation_cases
from acl_motion.annotations.research_metadata import load_research_metadata
from acl_motion.annotations.storage import (
    assert_human_annotation_path,
    load_human_annotation_session,
    load_movement_window_json,
    load_pipeline_event_annotation,
    load_pipeline_roi_timeline,
    load_roi_keyframes_csv,
    load_target_unavailable_intervals_csv,
    save_human_annotation_session,
)
from acl_motion.annotations.validation import (
    bbox_iou,
    compare_event_anchors,
    compare_independent_annotation_sessions,
    validate_annotation_session,
)
from acl_motion.annotations.view_alignment import (
    ViewAlignmentAnchor,
    load_view_alignment,
    save_view_alignment,
    view_alignment_path,
)
from acl_motion.cases.annotations import AnchorType, EventAnnotation
from acl_motion.cases.models import InjurySide
from acl_motion.ui.annotation import (
    AnnotationUiState,
    _delete_case_entry,
    _imported_case_for_video,
    _load_imported_cases,
    _parse_video_upload,
    _save_imported_cases,
    render_annotation_page,
    smoke_test,
)
from acl_motion.video.roi import BBox


def test_roi_serialization_and_pipeline_loading(tmp_path) -> None:
    session = _session()
    paths = save_human_annotation_session(session, tmp_path / "human", "case")

    loaded_keyframes = load_roi_keyframes_csv(paths.roi_csv)
    pipeline_timeline = load_pipeline_roi_timeline(paths.roi_csv)
    loaded_event = load_pipeline_event_annotation(paths.event_json)

    assert loaded_keyframes[0].flags == (OperatorFlag.PLAYER_OVERLAP,)
    assert pipeline_timeline.bbox_for_frame(5).x == 50.0
    assert loaded_event.event_anchor_frame == 3
    assert loaded_event.annotation_method == "human_ui_annotation"


def test_session_json_preserves_provenance(tmp_path) -> None:
    session = _session()
    paths = save_human_annotation_session(session, tmp_path / "human", "case")

    loaded = load_human_annotation_session(paths.session_json)

    assert loaded.provenance.annotator_type.value == "HUMAN"
    assert loaded.provenance.annotator_id == "researcher_01"
    assert loaded.provenance.view_id == "source"
    assert loaded.manual_roi_keyframe_count == 2
    assert loaded.injured_side is InjurySide.LEFT
    assert loaded.injury_laterality_source == "human_operator_test"


def test_target_unavailable_intervals_preserve_absence_and_provenance(tmp_path) -> None:
    session = _session().with_changes(
        target_unavailable_intervals=(
            TargetUnavailableIntervalAnnotation(
                start_frame=4,
                end_frame=6,
                reason=OperatorFlag.PLAYER_OVERLAP,
                note="Jordan is fully occluded by another player.",
            ),
        )
    )

    paths = save_human_annotation_session(session, tmp_path / "human", "case")
    loaded = load_human_annotation_session(paths.session_json)
    csv_intervals = load_target_unavailable_intervals_csv(paths.target_unavailable_csv)

    assert loaded.manual_target_unavailable_frame_count == 3
    assert loaded.target_unavailable_interval_at(5) is not None
    assert loaded.target_unavailable_interval_at(3) is None
    assert csv_intervals[0].reason == OperatorFlag.PLAYER_OVERLAP
    assert csv_intervals[0].note.startswith("Jordan")


def test_target_accepted_intervals_preserve_human_pose_review(tmp_path) -> None:
    session = _session().with_changes(
        target_accepted_intervals=(
            TargetAcceptedIntervalAnnotation(
                start_frame=4,
                end_frame=8,
                note="Raw skeleton follows the documented athlete.",
            ),
        )
    )

    paths = save_human_annotation_session(session, tmp_path / "human", "case")
    loaded = load_human_annotation_session(paths.session_json)

    assert loaded.manual_target_accepted_frame_count == 5
    assert loaded.target_accepted_interval_at(6) is not None
    assert loaded.target_accepted_interval_at(3) is None
    assert loaded.target_accepted_intervals[0].note.startswith("Raw skeleton")


def test_target_accepted_interval_cannot_overlap_excluded_interval() -> None:
    with pytest.raises(ValueError, match="cannot overlap target-unavailable"):
        _session().with_changes(
            target_unavailable_intervals=(
                TargetUnavailableIntervalAnnotation(start_frame=4, end_frame=6),
            ),
            target_accepted_intervals=(
                TargetAcceptedIntervalAnnotation(start_frame=6, end_frame=8),
            ),
        )


def test_target_unavailable_interval_rejects_roi_keyframe_inside_gap() -> None:
    with pytest.raises(ValueError, match="cannot exist inside"):
        HumanAnnotationSession(
            provenance=_provenance(),
            roi_keyframes=(
                RoiKeyframeAnnotation(frame_index=5, bbox=BBox(0, 0, 10, 10)),
            ),
            target_unavailable_intervals=(
                TargetUnavailableIntervalAnnotation(start_frame=4, end_frame=6),
            ),
        )


def test_invalid_box_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        BBox(0, 0, 0, 10)


def test_keyframe_replacement_and_deletion() -> None:
    session = _session()
    replaced = session.replace_keyframe(
        RoiKeyframeAnnotation(frame_index=0, bbox=BBox(10, 0, 20, 20))
    )
    deleted = replaced.delete_keyframe(10)

    assert replaced.roi_keyframes[0].bbox.x == 10
    assert [keyframe.frame_index for keyframe in deleted.roi_keyframes] == [0]


def test_annotation_validation_warns_for_missing_event() -> None:
    session = HumanAnnotationSession(
        provenance=_provenance(),
        roi_keyframes=(RoiKeyframeAnnotation(frame_index=1, bbox=BBox(0, 0, 10, 10)),),
    )
    result = validate_annotation_session(session, frame_count=20)

    assert result.ok
    assert "No Movement End has been marked yet." in result.warnings
    assert result.summary["movement_start"] == 1


def test_finalized_annotation_preserves_unknown_injured_knee() -> None:
    session = HumanAnnotationSession(
        provenance=_provenance(),
        roi_keyframes=(RoiKeyframeAnnotation(frame_index=1, bbox=BBox(0, 0, 10, 10)),),
        finalized=True,
    )

    result = validate_annotation_session(session, frame_count=20)

    assert result.ok
    assert result.summary["injured_side"] == "unknown"
    assert any("will remain unknown" in warning for warning in result.warnings)


def test_movement_start_infers_from_first_manual_roi_keyframe() -> None:
    session = _session()

    assert infer_movement_start_frame(session) == 0


def test_human_migration_preserves_roi_and_movement_window() -> None:
    session = _session()

    migrated = migrate_session_to_movement_window(session, fps=30.0)

    assert migrated.manual_roi_keyframe_count == session.manual_roi_keyframe_count
    assert migrated.movement_window is not None
    assert migrated.movement_window.movement_start_frame == 0
    assert migrated.movement_window.movement_end_frame == 3
    assert migrated.movement_window.duration_ms == pytest.approx(100.0)
    assert migrated.event_annotation is not None
    assert migrated.event_annotation.annotation_method == "human_movement_window_compatibility"


def test_explicit_movement_end_persists(tmp_path) -> None:
    session = _session().with_changes(
        movement_window=MovementWindowAnnotation(
            movement_start_frame=0,
            movement_start_timestamp_ms=0.0,
            movement_end_frame=10,
            movement_end_timestamp_ms=333.3333333333,
            confidence=EventConfidence.MODERATE,
            rationale="visible sequence ended",
        )
    )

    paths = save_human_annotation_session(session, tmp_path / "human", "case")
    loaded_session = load_human_annotation_session(paths.session_json)
    loaded_window = load_movement_window_json(paths.movement_window_json)

    assert loaded_session.movement_window is not None
    assert loaded_session.movement_window.movement_end_frame == 10
    assert loaded_window.movement_end_frame == 10
    assert loaded_window.rationale == "visible sequence ended"


def test_missing_movement_end_refuses_development_fallback() -> None:
    session = HumanAnnotationSession(
        provenance=_provenance(),
        roi_keyframes=(RoiKeyframeAnnotation(frame_index=0, bbox=BBox(0, 0, 10, 10)),),
    )

    with pytest.raises(ValueError, match="development anchors cannot be substituted"):
        migrate_session_to_movement_window(session, fps=30.0)

    development_like = session.with_changes(
        event_annotation=EventAnnotation(
            case_id="case",
            source_id="source",
            event_anchor_frame=10,
            event_anchor_type=AnchorType.CRITICAL_PLANT,
            annotation_method="development_manual",
        )
    )

    with pytest.raises(ValueError, match="current HUMAN annotation session"):
        migrate_session_to_movement_window(development_like, fps=30.0)


def test_movement_window_filters_primary_profile_frames() -> None:
    frame_df = pd.DataFrame(
        {
            "source_frame_index": [0, 1, 2, 3, 4],
            "timestamp_ms": [0.0, 33.0, 66.0, 99.0, 132.0],
        }
    )
    movement_window = MovementWindowAnnotation(
        movement_start_frame=1,
        movement_start_timestamp_ms=33.0,
        movement_end_frame=3,
        movement_end_timestamp_ms=99.0,
    )

    filtered = filter_to_movement_window(frame_df, movement_window)

    assert filtered["source_frame_index"].tolist() == [1, 2, 3]


def test_movement_window_temporal_transforms() -> None:
    frame_df = pd.DataFrame(
        {
            "source_frame_index": [0, 5, 10],
            "timestamp_ms": [0.0, 500.0, 1000.0],
        }
    )
    movement_window = MovementWindowAnnotation(
        movement_start_frame=0,
        movement_start_timestamp_ms=0.0,
        movement_end_frame=10,
        movement_end_timestamp_ms=1000.0,
    )

    timed = add_movement_timing_columns(frame_df, movement_window)

    assert timed["movement_elapsed_ms"].tolist() == [0.0, 500.0, 1000.0]
    assert timed["movement_end_relative_ms"].tolist() == [-1000.0, -500.0, 0.0]
    assert timed["movement_phase_pct"].tolist() == [0.0, 50.0, 100.0]


def test_cross_case_analytics_guard_requires_more_than_one_human_case() -> None:
    for analytic_name in CROSS_CASE_ANALYTIC_NAMES:
        with pytest.raises(CrossCaseAnalyticsUnavailable):
            require_cross_case_analytics_ready(
                human_validated_case_count=1,
                analytic_name=analytic_name,
            )

    assert (
        require_cross_case_analytics_ready(
            human_validated_case_count=2,
            analytic_name="similarity",
        )
        is None
    )


def test_manual_view_alignment_preserves_per_view_source_frames(tmp_path) -> None:
    empty = load_view_alignment(tmp_path / "human", "case/a acl")
    anchor = ViewAlignmentAnchor(
        anchor_id="foot_plant_01",
        label="Critical foot plant",
        case_id="case/a acl",
        view_frames={"broadcast_01": 142, "replay_02": 87},
        notes="same observable movement event",
    )

    updated = empty.upsert(anchor)
    path = save_view_alignment(updated, tmp_path / "human")
    loaded = load_view_alignment(tmp_path / "human", "case/a acl")

    assert path == view_alignment_path(tmp_path / "human", "case/a acl")
    assert loaded.anchors[0].view_frames == {"broadcast_01": 142, "replay_02": 87}
    assert loaded.to_dict()["note"].startswith("Alignment anchors relate local source frames")


def test_roi_propagation_linear_and_endpoint_hold() -> None:
    keyframes = (
        RoiKeyframeAnnotation(frame_index=0, bbox=BBox(0, 0, 10, 10)),
        RoiKeyframeAnnotation(frame_index=10, bbox=BBox(100, 20, 20, 20)),
    )

    assert propagated_bbox(keyframes, 5).x == 50.0
    assert propagated_bbox(keyframes, 5).y == 10.0
    assert propagated_bbox(keyframes, -1).x == 0.0
    assert propagated_bbox(keyframes, 11).x == 100.0


def test_single_keyframe_propagation_holds_constant() -> None:
    keyframes = (RoiKeyframeAnnotation(frame_index=5, bbox=BBox(25, 10, 50, 60)),)

    assert propagated_bbox(keyframes, 0).x == 25.0
    assert propagated_bbox(keyframes, 10).height == 60.0


def test_iou_cases() -> None:
    assert bbox_iou(BBox(0, 0, 10, 10), BBox(0, 0, 10, 10)) == 1.0
    assert bbox_iou(BBox(0, 0, 10, 10), BBox(20, 20, 10, 10)) == 0.0
    assert bbox_iou(BBox(0, 0, 10, 10), BBox(5, 0, 10, 10)) == pytest.approx(1 / 3)


def test_event_anchor_comparison_uses_fps() -> None:
    human = EventAnnotation(
        case_id="case",
        source_id="source",
        event_anchor_frame=113,
        event_anchor_type=AnchorType.CRITICAL_PLANT,
    )
    development = EventAnnotation(
        case_id="case",
        source_id="source",
        event_anchor_frame=111,
        event_anchor_type=AnchorType.CRITICAL_PLANT,
    )

    comparison = compare_event_anchors(human, development, fps=30.0)

    assert comparison.frame_difference == 2
    assert comparison.time_difference_ms == pytest.approx(66.6667)


def test_independent_annotation_agreement_preserves_disagreements() -> None:
    first = _session().with_changes(
        target_unavailable_intervals=(
            TargetUnavailableIntervalAnnotation(start_frame=4, end_frame=5),
        ),
        movement_window=MovementWindowAnnotation(
            movement_start_frame=0,
            movement_end_frame=10,
            movement_start_timestamp_ms=0.0,
            movement_end_timestamp_ms=333.3,
        ),
    )
    second = HumanAnnotationSession(
        provenance=AnnotationProvenance.create(
            case_id="case",
            source_id="source",
            video_path="/tmp/video.mp4",
            annotator_id="researcher_02",
        ),
        roi_keyframes=(
            RoiKeyframeAnnotation(frame_index=0, bbox=BBox(2, 0, 20, 20)),
            RoiKeyframeAnnotation(frame_index=10, bbox=BBox(102, 0, 20, 20)),
        ),
        target_unavailable_intervals=(
            TargetUnavailableIntervalAnnotation(start_frame=5, end_frame=6),
        ),
        movement_window=MovementWindowAnnotation(
            movement_start_frame=1,
            movement_end_frame=12,
            movement_start_timestamp_ms=33.3,
            movement_end_timestamp_ms=400.0,
        ),
        injured_side=InjurySide.LEFT,
    )

    agreement = compare_independent_annotation_sessions(
        first,
        second,
        frame_count=20,
        fps=30.0,
    )

    assert agreement.independent_sessions is True
    assert agreement.same_case_and_source is True
    assert agreement.roi_agreement.mean_iou == pytest.approx(18 / 22)
    assert agreement.target_availability_exact_agreement == pytest.approx(0.9)
    assert agreement.unavailable_frame_jaccard == pytest.approx(1 / 3)
    assert agreement.movement_start_difference_frames == -1
    assert agreement.movement_end_difference_frames == -2
    assert agreement.injured_side_agreement is True


def test_development_annotation_paths_are_not_human_save_targets(tmp_path) -> None:
    with pytest.raises(ValueError, match="development"):
        assert_human_annotation_path("data/annotations/christen_press_roi_keyframes.csv")
    with pytest.raises(ValueError, match="_human"):
        assert_human_annotation_path(tmp_path / "roi.csv")


def test_ui_smoke_test_does_not_write_human_annotations(tmp_path) -> None:
    result = smoke_test(output_dir=tmp_path / "human")

    assert result["case_count"] == 7
    assert "case_01" in result["cases"]
    assert "leah_williamson_broadcast_wide" in result["cases"]
    assert "leah_williamson_replay_close_oblique" in result["cases"]
    assert "leah_williamson_replay_frontal_oblique" in result["cases"]
    assert "leah_williamson_replay_close_sagittal" in result["cases"]
    assert result["html_has_canvas"] is True
    assert result["html_has_save"] is True
    assert result["html_has_import_video"] is True
    assert result["html_has_open_video"] is True
    assert result["html_has_generate_analysis"] is True
    assert result["html_has_case_details"] is True
    assert result["html_has_target_unavailable_intervals"] is True
    assert result["writes_files"] is False


def test_annotation_routes_new_video_views_through_player_first_cutter() -> None:
    html = render_annotation_page()

    assert "Add / cut video view" in html
    assert "openVideoCutterForCase" in html
    assert '`/video-cutter?case=${encodeURIComponent(caseId)}&return=${encodeURIComponent(returnPath)}`' in html
    assert 'id="localVideoInput"' not in html


def test_local_video_import_parser_and_registry_roundtrip(tmp_path) -> None:
    boundary = "----codex-test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="player_name"\r\n\r\n'
        "Custom Player\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="video"; filename="Leah Sample.mp4"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
    ).encode() + b"not-a-real-video-for-parser-only\r\n" + f"--{boundary}--\r\n".encode()

    upload = _parse_video_upload(f"multipart/form-data; boundary={boundary}", body)

    assert upload["filename"] == "Leah Sample.mp4"
    assert upload["player_name"] == "Custom Player"
    assert upload["data"] == b"not-a-real-video-for-parser-only"

    case = _imported_case_for_video(
        video_path=tmp_path / "Leah Sample.mp4",
        player_name=upload["player_name"],
        cases=[],
    )
    state = AnnotationUiState(
        cases=[case],
        output_dir=tmp_path / "human",
        video_root=tmp_path,
    )

    _save_imported_cases(state)
    loaded = _load_imported_cases(tmp_path / "human", tmp_path)
    registry_loaded = imported_annotation_cases(
        tmp_path,
        imported_cases_path=tmp_path / "human" / "imported_video_cases_human.json",
    )
    resolved = case_by_slug("imported_custom_player", registry_loaded)

    assert loaded[0].slug == "imported_custom_player"
    assert loaded[0].player_name == "Custom Player"
    assert loaded[0].view_label == "Imported local video"
    assert resolved.video_path == tmp_path / "Leah Sample.mp4"


def test_imported_registry_accepts_legacy_project_relative_video_paths(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    video_root = Path("data/videos/analysis_clips")
    video_path = video_root / "legacy-case.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"video")
    output_dir = Path("data/annotations/human")
    output_dir.mkdir(parents=True)
    registry_path = output_dir / "imported_video_cases_human.json"
    registry_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "slug": "imported_legacy_case",
                        "case_id": "legacy_case_acl",
                        "source_id": "legacy_case_view_01",
                        "video_path": str(video_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    ui_cases = _load_imported_cases(output_dir, video_root)
    registry_cases = imported_annotation_cases(
        video_root,
        imported_cases_path=registry_path,
    )

    assert ui_cases[0].video_path == video_path
    assert registry_cases[0].video_path == video_path


def test_imported_registry_omits_case_superseded_by_canonical_view(tmp_path) -> None:
    registry_path = tmp_path / "imported_video_cases_human.json"
    registry_path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "slug": "imported_duplicate",
                        "case_id": "imported_case",
                        "source_id": "imported_source",
                        "video_path": "duplicate.mp4",
                        "superseded_by": "canonical_view",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cases = imported_annotation_cases(tmp_path, imported_cases_path=registry_path)

    assert cases == ()


def test_delete_view_preserves_siblings_and_promotes_a_primary_view(tmp_path) -> None:
    primary = _imported_case_for_video(
        video_path=tmp_path / "primary.mp4",
        player_name="Example Player",
        cases=[],
    )
    replay = replace(
        primary,
        slug=f"{primary.slug}_replay",
        source_id=f"{primary.source_id}_replay",
        view_id=f"{primary.source_id}_replay",
        view_label="Replay view",
        primary_view=False,
    )
    state = AnnotationUiState(
        cases=[primary, replay],
        output_dir=tmp_path / "human",
        video_root=tmp_path,
        trash_dir=tmp_path / "trash",
    )
    _save_imported_cases(state)

    response = _delete_case_entry(
        {"scope": "view", "case_id": primary.case_id, "slug": primary.slug},
        state,
    )

    assert response["remaining_view_count"] == 1
    assert response["source_files_preserved"] is False
    assert [case.slug for case in state.cases] == [replay.slug]
    assert state.cases[0].primary_view is True
    assert _load_imported_cases(state.output_dir, tmp_path)[0].slug == replay.slug


def test_delete_case_persists_a_library_tombstone_without_deleting_files(tmp_path) -> None:
    video_path = tmp_path / "case.mp4"
    video_path.write_bytes(b"video remains")
    case = _imported_case_for_video(
        video_path=video_path,
        player_name="Example Player",
        cases=[],
    )
    state = AnnotationUiState(
        cases=[case],
        output_dir=tmp_path / "human",
        video_root=tmp_path,
        trash_dir=tmp_path / "trash",
    )

    response = _delete_case_entry(
        {"scope": "case", "case_id": case.case_id},
        state,
    )

    assert response["deleted"] is True
    assert state.cases == []
    assert not video_path.exists()
    trash_bundle = Path(response["trash_bundle"])
    assert (trash_bundle / "source_videos" / "case.mp4").read_bytes() == b"video remains"


def test_case_intake_groups_multiple_views_under_one_dated_injury_case(tmp_path) -> None:
    registry_path = tmp_path / "human" / "imported_video_cases_human.json"
    metadata_path = tmp_path / "human" / "case_research_metadata_human.json"
    primary_clip = tmp_path / "primary.mp4"
    replay_clip = tmp_path / "replay.mp4"
    primary_clip.write_bytes(b"primary")
    replay_clip.write_bytes(b"replay")

    primary, primary_details = register_analysis_clip(
        {
            "assignment_mode": "new",
            "player_name": "Example Player",
            "injury_date": "2026-08-24",
            "team": "Example FC",
            "opponent": "Rivals FC",
            "competition": "Example League",
            "position_group": "defender",
            "match_minute": "67",
            "view_label": "Live wide",
            "perspective": "high-wide",
        },
        video_path=primary_clip,
        cases=(),
        imported_cases_path=registry_path,
        research_metadata_path=metadata_path,
    )
    replay, replay_details = register_analysis_clip(
        {
            "assignment_mode": "existing",
            "case_id": primary.case_id,
            "view_label": "Close slow-motion replay",
            "perspective": "oblique",
            "slow_motion": True,
        },
        video_path=replay_clip,
        cases=(primary,),
        imported_cases_path=registry_path,
        research_metadata_path=metadata_path,
    )

    records = json.loads(registry_path.read_text(encoding="utf-8"))["cases"]
    metadata = load_research_metadata(metadata_path)[primary.case_id]
    options = injury_case_options((primary, replay), research_metadata_path=metadata_path)

    assert primary.case_id == replay.case_id
    assert primary.primary_view is True
    assert replay.primary_view is False
    assert replay.source_id.endswith("view_02")
    assert replay.slow_motion is True
    assert primary_details["injury_date"] == "2026-08-24"
    assert replay_details["opponent"] == "Rivals FC"
    assert metadata["team"] == "Example FC"
    assert metadata["opponent"] == "Rivals FC"
    assert len(records) == 2
    assert options[0]["view_count"] == 2
    assert options[0]["injury_date"] == "2026-08-24"
    assert options[0]["match_minute"] == "67"


def test_new_injury_case_requires_a_date(tmp_path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")

    with pytest.raises(ValueError, match="date is required"):
        register_analysis_clip(
            {"assignment_mode": "new", "player_name": "Example Player"},
            video_path=clip,
            cases=(),
            imported_cases_path=tmp_path / "registry.json",
            research_metadata_path=tmp_path / "metadata.json",
        )


def test_case_intake_refuses_duplicate_player_and_injury_date(tmp_path) -> None:
    registry_path = tmp_path / "human" / "imported_video_cases_human.json"
    metadata_path = tmp_path / "human" / "case_research_metadata_human.json"
    first_clip = tmp_path / "first.mp4"
    duplicate_clip = tmp_path / "duplicate.mp4"
    first_clip.write_bytes(b"first")
    duplicate_clip.write_bytes(b"duplicate")
    first, _ = register_analysis_clip(
        {
            "assignment_mode": "new",
            "player_name": "Example Player",
            "injury_date": "2026-08-24",
        },
        video_path=first_clip,
        cases=(),
        imported_cases_path=registry_path,
        research_metadata_path=metadata_path,
    )

    with pytest.raises(ValueError, match="already exists for this player and date"):
        register_analysis_clip(
            {
                "assignment_mode": "new",
                "player_name": "Example Player",
                "injury_date": "2026-08-24",
            },
            video_path=duplicate_clip,
            cases=(first,),
            imported_cases_path=registry_path,
            research_metadata_path=metadata_path,
        )


def _session() -> HumanAnnotationSession:
    return HumanAnnotationSession(
        provenance=_provenance(),
        roi_keyframes=(
            RoiKeyframeAnnotation(
                frame_index=0,
                bbox=BBox(0, 0, 20, 20),
                flags=(OperatorFlag.PLAYER_OVERLAP,),
                note="first",
            ),
            RoiKeyframeAnnotation(frame_index=10, bbox=BBox(100, 0, 20, 20)),
        ),
        event_annotation=EventAnnotation(
            case_id="case",
            source_id="source",
            view_id="source",
            event_anchor_frame=3,
            event_anchor_type=AnchorType.CRITICAL_PLANT,
            annotation_confidence=0.6,
            annotation_method="human_ui_annotation",
            annotator="researcher_01",
        ),
        event_confidence_label=EventConfidence.MODERATE,
        injured_side=InjurySide.LEFT,
        injury_laterality_source="human_operator_test",
    )


def _provenance() -> AnnotationProvenance:
    return AnnotationProvenance.create(
        case_id="case",
        source_id="source",
        video_path="/tmp/video.mp4",
        annotator_id="researcher_01",
    )
