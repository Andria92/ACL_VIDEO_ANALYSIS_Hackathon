"""Storage helpers for human annotation sessions."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from acl_motion.annotations.models import (
    AnnotationProvenance,
    HumanAnnotationSession,
    MovementWindowAnnotation,
    RoiKeyframeAnnotation,
    TargetAcceptedIntervalAnnotation,
    TargetUnavailableIntervalAnnotation,
)
from acl_motion.cases.annotations import load_event_annotation, write_event_annotation
from acl_motion.cases.models import InjurySide
from acl_motion.video.roi import BBox, RoiTimeline


@dataclass(frozen=True, slots=True)
class HumanAnnotationPaths:
    """Canonical file paths for a saved human annotation session."""

    session_json: Path
    roi_csv: Path
    target_unavailable_csv: Path
    movement_window_json: Path
    event_json: Path


def human_annotation_paths(output_dir: str | Path, slug: str) -> HumanAnnotationPaths:
    """Return canonical human annotation file paths for a case slug."""

    root = Path(output_dir)
    return HumanAnnotationPaths(
        session_json=root / f"{slug}_annotation_session_human.json",
        roi_csv=root / f"{slug}_target_roi_human.csv",
        target_unavailable_csv=root / f"{slug}_target_unavailable_intervals_human.csv",
        movement_window_json=root / f"{slug}_movement_window_human.json",
        event_json=root / f"{slug}_event_annotation_human.json",
    )


def save_human_annotation_session(
    session: HumanAnnotationSession,
    output_dir: str | Path,
    slug: str,
) -> HumanAnnotationPaths:
    """Save a human session plus pipeline-compatible ROI/event files."""

    paths = human_annotation_paths(output_dir, slug)
    for path in (
        paths.session_json,
        paths.roi_csv,
        paths.target_unavailable_csv,
        paths.movement_window_json,
        paths.event_json,
    ):
        assert_human_annotation_path(path)
    paths.session_json.parent.mkdir(parents=True, exist_ok=True)
    write_session_json(session, paths.session_json)
    if session.roi_keyframes:
        write_roi_keyframes_csv(session, paths.roi_csv)
    write_target_unavailable_intervals_csv(session, paths.target_unavailable_csv)
    if session.movement_window is not None:
        write_movement_window_json(session, paths.movement_window_json)
    if session.event_annotation is not None:
        write_event_annotation(session.event_annotation, paths.event_json)
    return paths


def write_session_json(session: HumanAnnotationSession, path: str | Path) -> Path:
    """Write the full resumable annotation session JSON."""

    output = Path(path)
    assert_human_annotation_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")
    return output


def load_human_annotation_session(path: str | Path) -> HumanAnnotationSession:
    """Load a resumable human annotation session JSON."""

    return HumanAnnotationSession.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def write_movement_window_json(session: HumanAnnotationSession, path: str | Path) -> Path:
    """Write the canonical human Movement Window JSON."""

    if session.movement_window is None:
        raise ValueError("Cannot write movement window: session has no movement_window.")
    output = Path(path)
    assert_human_annotation_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": session.provenance.to_dict(),
        "case_id": session.provenance.case_id,
        "source_id": session.provenance.source_id,
        "view_id": session.provenance.view_id or session.provenance.source_id,
        "manual_roi_keyframe_count": session.manual_roi_keyframe_count,
        "manual_target_unavailable_frame_count": session.manual_target_unavailable_frame_count,
        "target_unavailable_intervals": [
            interval.to_dict() for interval in session.target_unavailable_intervals
        ],
        "manual_target_accepted_frame_count": session.manual_target_accepted_frame_count,
        "target_accepted_intervals": [
            interval.to_dict() for interval in session.target_accepted_intervals
        ],
        "movement_window": session.movement_window.to_dict(),
        "injured_side": session.injured_side.value,
        "injury_laterality_source": session.injury_laterality_source,
        "notes": session.notes,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def load_movement_window_json(path: str | Path) -> MovementWindowAnnotation:
    """Load the canonical human Movement Window JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return MovementWindowAnnotation.from_dict(payload["movement_window"])


def write_roi_keyframes_csv(session: HumanAnnotationSession, path: str | Path) -> Path:
    """Write pipeline-compatible human ROI keyframes."""

    output = Path(path)
    assert_human_annotation_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "frame_index",
            "x",
            "y",
            "width",
            "height",
            "view_id",
            "annotation_source",
            "annotation_session_id",
            "annotator_id",
            "flags",
            "note",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for keyframe in session.roi_keyframes:
            writer.writerow(
                {
                    "frame_index": keyframe.frame_index,
                    "x": keyframe.bbox.x,
                    "y": keyframe.bbox.y,
                    "width": keyframe.bbox.width,
                    "height": keyframe.bbox.height,
                    "view_id": session.provenance.view_id or session.provenance.source_id,
                    "annotation_source": "human_ui",
                    "annotation_session_id": session.provenance.annotation_session_id,
                    "annotator_id": session.provenance.annotator_id,
                    "flags": "|".join(flag.value for flag in keyframe.flags),
                    "note": keyframe.note,
                }
            )
    return output


def load_roi_keyframes_csv(path: str | Path) -> tuple[RoiKeyframeAnnotation, ...]:
    """Load human ROI keyframes from CSV, preserving optional flags and notes."""

    keyframes: list[RoiKeyframeAnnotation] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_index", "x", "y", "width", "height"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"ROI keyframe CSV missing columns: {sorted(missing)}")
        for row in reader:
            flags = tuple(flag for flag in row.get("flags", "").split("|") if flag)
            keyframes.append(
                RoiKeyframeAnnotation(
                    frame_index=int(row["frame_index"]),
                    bbox=BBox(
                        x=float(row["x"]),
                        y=float(row["y"]),
                        width=float(row["width"]),
                        height=float(row["height"]),
                    ),
                    flags=flags,
                    note=row.get("note", ""),
                )
            )
    return tuple(sorted(keyframes, key=lambda item: item.frame_index))


def write_target_unavailable_intervals_csv(
    session: HumanAnnotationSession,
    path: str | Path,
) -> Path:
    """Write explicit human target-unavailable intervals with provenance."""

    output = Path(path)
    assert_human_annotation_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "start_frame",
            "end_frame",
            "frame_count",
            "reason",
            "note",
            "view_id",
            "annotation_source",
            "annotation_session_id",
            "annotator_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for interval in session.target_unavailable_intervals:
            writer.writerow(
                {
                    "start_frame": interval.start_frame,
                    "end_frame": interval.end_frame,
                    "frame_count": interval.frame_count,
                    "reason": interval.reason.value,
                    "note": interval.note,
                    "view_id": session.provenance.view_id or session.provenance.source_id,
                    "annotation_source": "human_ui",
                    "annotation_session_id": session.provenance.annotation_session_id,
                    "annotator_id": session.provenance.annotator_id,
                }
            )
    return output


def load_target_unavailable_intervals_csv(
    path: str | Path,
) -> tuple[TargetUnavailableIntervalAnnotation, ...]:
    """Load human target-unavailable intervals from CSV."""

    intervals: list[TargetUnavailableIntervalAnnotation] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"start_frame", "end_frame", "reason"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Target-unavailable interval CSV missing columns: {sorted(missing)}")
        for row in reader:
            intervals.append(
                TargetUnavailableIntervalAnnotation(
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    reason=row["reason"],
                    note=row.get("note", ""),
                )
            )
    return tuple(sorted(intervals, key=lambda item: (item.start_frame, item.end_frame)))


def load_pipeline_roi_timeline(path: str | Path) -> RoiTimeline:
    """Load a saved human ROI file using the existing pipeline reader."""

    return RoiTimeline.from_csv(path)


def load_pipeline_event_annotation(path: str | Path):
    """Load a saved human event JSON using the existing pipeline reader."""

    return load_event_annotation(path)


def new_human_session(
    *,
    case_id: str,
    source_id: str,
    video_path: str | Path,
    annotator_id: str,
    view_id: str | None = None,
    roi_keyframes: tuple[RoiKeyframeAnnotation, ...] = (),
    target_unavailable_intervals: tuple[TargetUnavailableIntervalAnnotation, ...] = (),
    target_accepted_intervals: tuple[TargetAcceptedIntervalAnnotation, ...] = (),
    movement_window: MovementWindowAnnotation | None = None,
    event_annotation=None,
    event_confidence_label=None,
    injured_side: InjurySide | str = InjurySide.UNKNOWN,
    injury_laterality_source: str = "",
    notes: str = "",
    finalized: bool = False,
    existing_provenance: AnnotationProvenance | None = None,
) -> HumanAnnotationSession:
    """Create a new or updated human annotation session."""

    provenance = existing_provenance or AnnotationProvenance.create(
        case_id=case_id,
        source_id=source_id,
        video_path=video_path,
        annotator_id=annotator_id,
        view_id=view_id or source_id,
    )
    if existing_provenance is not None:
        provenance = AnnotationProvenance(
            annotation_session_id=existing_provenance.annotation_session_id,
            annotator_type=existing_provenance.annotator_type,
            annotator_id=annotator_id,
            created_at=existing_provenance.created_at,
            updated_at=existing_provenance.touch().updated_at,
            case_id=existing_provenance.case_id,
            source_id=existing_provenance.source_id,
            video_path=existing_provenance.video_path,
            view_id=existing_provenance.view_id or existing_provenance.source_id,
            ui_tool_version=existing_provenance.ui_tool_version,
        )
    return HumanAnnotationSession(
        provenance=provenance,
        roi_keyframes=roi_keyframes,
        target_unavailable_intervals=target_unavailable_intervals,
        target_accepted_intervals=target_accepted_intervals,
        movement_window=movement_window,
        event_annotation=event_annotation,
        event_confidence_label=event_confidence_label,
        injured_side=InjurySide(injured_side),
        injury_laterality_source=injury_laterality_source,
        notes=notes,
        finalized=finalized,
    )


def assert_human_annotation_path(path: str | Path) -> None:
    """Refuse paths that look like development annotation outputs."""

    candidate = Path(path)
    parts = set(candidate.parts)
    if "annotations" in parts and "human" not in parts:
        raise ValueError(f"Human annotations must not be saved over development files: {candidate}")
    if "_human" not in candidate.stem:
        raise ValueError(f"Human annotation filename must include '_human': {candidate}")
