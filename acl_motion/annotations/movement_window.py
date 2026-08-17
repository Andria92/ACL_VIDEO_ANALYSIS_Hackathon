"""Human Movement Window migration and timing helpers."""

from __future__ import annotations

from acl_motion.annotations.models import (
    EventConfidence,
    HumanAnnotationSession,
    MovementWindowAnnotation,
)
from acl_motion.cases.annotations import AnchorType, EventAnnotation


def infer_movement_start_frame(session: HumanAnnotationSession) -> int:
    """Infer Movement Start from the first manual human ROI keyframe."""

    if not session.roi_keyframes:
        raise ValueError("Cannot infer Movement Start without a human ROI keyframe.")
    return int(session.roi_keyframes[0].frame_index)


def migrate_session_to_movement_window(
    session: HumanAnnotationSession,
    *,
    fps: float,
) -> HumanAnnotationSession:
    """Upgrade a legacy human session to the Movement Window schema.

    The previous human UI stored the operator's visible-sequence-end click in the
    compatibility EventAnnotation. This migration treats that HUMAN marker as
    Movement End. Development anchors are never used here.
    """

    if session.movement_window is not None:
        return session
    start_frame = infer_movement_start_frame(session)
    end_frame = _human_end_marker(session)
    movement_window = MovementWindowAnnotation(
        movement_start_frame=start_frame,
        movement_start_timestamp_ms=_timestamp_ms(start_frame, fps),
        movement_end_frame=end_frame,
        movement_end_timestamp_ms=_timestamp_ms(end_frame, fps),
        confidence=session.event_confidence_label,
        rationale=session.event_annotation.notes if session.event_annotation is not None else "",
        source="migrated_from_human_visible_sequence_end",
    )
    compatibility_event = movement_window_to_event_annotation(
        session,
        movement_window,
    )
    return session.with_changes(
        movement_window=movement_window,
        event_annotation=compatibility_event,
    )


def movement_window_to_event_annotation(
    session: HumanAnnotationSession,
    movement_window: MovementWindowAnnotation,
) -> EventAnnotation:
    """Build a legacy-compatible EventAnnotation with Movement End as zero."""

    return EventAnnotation(
        case_id=session.provenance.case_id,
        source_id=session.provenance.source_id,
        view_id=session.provenance.source_id,
        analysis_start_frame=movement_window.movement_start_frame,
        analysis_end_frame=movement_window.movement_end_frame,
        estimated_event_end_frame=movement_window.movement_end_frame,
        event_anchor_frame=movement_window.movement_end_frame,
        event_anchor_type=AnchorType.ESTIMATED_EVENT_END,
        annotation_confidence=_confidence_value(movement_window.confidence),
        annotation_method="human_movement_window_compatibility",
        annotator=session.provenance.annotator_id,
        notes=(
            "Compatibility timing anchor for computation only: 0 ms is Movement End, "
            "not an ACL rupture, injury frame, critical plant, or biomechanical event."
        ),
    )


def add_movement_timing_columns(
    df,
    movement_window: MovementWindowAnnotation,
):
    """Return a copy with start-relative, end-relative, and phase timing columns."""

    output = df.copy()
    start_ms = movement_window.movement_start_timestamp_ms
    end_ms = movement_window.movement_end_timestamp_ms
    duration_ms = movement_window.duration_ms
    output["movement_elapsed_ms"] = output["timestamp_ms"].astype(float) - start_ms
    output["movement_end_relative_ms"] = output["timestamp_ms"].astype(float) - end_ms
    output["movement_phase_pct"] = (
        output["movement_elapsed_ms"] / duration_ms * 100.0 if duration_ms > 0 else 0.0
    )
    output["movement_start_frame"] = movement_window.movement_start_frame
    output["movement_end_frame"] = movement_window.movement_end_frame
    return output


def filter_to_movement_window(df, movement_window: MovementWindowAnnotation):
    """Return rows inside the inclusive human Movement Window."""

    return df[
        df["source_frame_index"].astype(int).between(
            movement_window.movement_start_frame,
            movement_window.movement_end_frame,
            inclusive="both",
        )
    ].copy()


def _human_end_marker(session: HumanAnnotationSession) -> int:
    if session.event_annotation is None:
        raise ValueError("Human Movement End is missing; development anchors cannot be substituted.")
    if session.event_annotation.annotation_method != "human_ui_annotation":
        raise ValueError("Movement End must come from the current HUMAN annotation session.")
    marker = session.event_annotation.t0_frame()
    if marker is None:
        raise ValueError("Human Movement End is missing; development anchors cannot be substituted.")
    return int(marker)


def _timestamp_ms(frame_index: int, fps: float) -> float:
    return frame_index / fps * 1000.0 if fps else 0.0


def _confidence_value(confidence: EventConfidence | None) -> float | None:
    if confidence is None:
        return None
    return {"high": 0.9, "moderate": 0.6, "low": 0.3}[confidence.value]
