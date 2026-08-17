from __future__ import annotations

import pandas as pd
import pytest

from acl_motion.analytics.guard import (
    CROSS_CASE_ANALYTIC_NAMES,
    CrossCaseAnalyticsUnavailable,
    require_cross_case_analytics_ready,
)
from acl_motion.annotations.models import (
    AnnotationProvenance,
    EventConfidence,
    HumanAnnotationSession,
    MovementWindowAnnotation,
    OperatorFlag,
    RoiKeyframeAnnotation,
)
from acl_motion.annotations.movement_window import (
    add_movement_timing_columns,
    filter_to_movement_window,
    infer_movement_start_frame,
    migrate_session_to_movement_window,
)
from acl_motion.annotations.propagation import propagated_bbox
from acl_motion.annotations.registry import case_by_slug, imported_annotation_cases
from acl_motion.annotations.storage import (
    assert_human_annotation_path,
    load_human_annotation_session,
    load_movement_window_json,
    load_pipeline_event_annotation,
    load_pipeline_roi_timeline,
    load_roi_keyframes_csv,
    save_human_annotation_session,
)
from acl_motion.annotations.validation import (
    bbox_iou,
    compare_event_anchors,
    validate_annotation_session,
)
from acl_motion.annotations.view_alignment import (
    ViewAlignmentAnchor,
    load_view_alignment,
    save_view_alignment,
    view_alignment_path,
)
from acl_motion.cases.annotations import AnchorType, EventAnnotation
from acl_motion.ui.annotation import (
    AnnotationUiState,
    _imported_case_for_video,
    _load_imported_cases,
    _parse_video_upload,
    _save_imported_cases,
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
    assert result["writes_files"] is False


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
    )


def _provenance() -> AnnotationProvenance:
    return AnnotationProvenance.create(
        case_id="case",
        source_id="source",
        video_path="/tmp/video.mp4",
        annotator_id="researcher_01",
    )
