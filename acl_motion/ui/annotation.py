"""Dependency-light local annotation UI for M5.5."""

from __future__ import annotations

import json
import mimetypes
import re
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as email_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO
from urllib.parse import parse_qs, urlparse

from acl_motion.annotations.models import (
    ANNOTATION_UI_VERSION,
    AnnotationCase,
    EventConfidence,
    MovementWindowAnnotation,
    RoiKeyframeAnnotation,
)
from acl_motion.annotations.movement_window import movement_window_to_event_annotation
from acl_motion.annotations.registry import default_annotation_cases, views_for_case
from acl_motion.annotations.storage import (
    human_annotation_paths,
    load_human_annotation_session,
    new_human_session,
    save_human_annotation_session,
)
from acl_motion.annotations.validation import (
    compare_roi_timelines,
    validate_annotation_session,
)
from acl_motion.annotations.view_alignment import (
    ViewAlignmentAnchor,
    load_view_alignment,
    save_view_alignment,
)
from acl_motion.ui.results import (
    clear_result_mask_prompts,
    human_results_available,
    load_human_results_payload,
    load_result_evidence_payload,
    read_result_frame_jpeg,
    render_results_page,
    save_result_mask_prompt,
    trim_human_analysis_window_and_regenerate,
    undo_result_mask_prompt,
)
from acl_motion.video.io import VideoMetadata, read_video_metadata
from acl_motion.video.roi import BBox, RoiTimeline

IMPORTED_CASES_FILENAME = "imported_video_cases_human.json"
IMPORTED_CASE_SLUG_PREFIX = "imported_"
SUPPORTED_IMPORT_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".webm", ".mkv"}


@dataclass(slots=True)
class AnnotationUiState:
    """Runtime state shared by annotation UI requests."""

    cases: list[AnnotationCase]
    output_dir: Path
    video_root: Path


def run_annotation_ui(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    output_dir: str | Path = "data/annotations/human",
    video_root: str | Path = "/Users/andriagryffinpro/Desktop/injury_videos",
) -> None:
    """Run the local annotation UI until interrupted."""

    output_path = Path(output_dir)
    root = Path(video_root)
    state = AnnotationUiState(
        cases=[*default_annotation_cases(root), *_load_imported_cases(output_path, root)],
        output_dir=output_path,
        video_root=root,
    )
    server = build_server(host=host, port=port, state=state)
    print(f"ACL Movement Explorer annotation UI: http://{host}:{port}")
    print(f"Human annotations will save under: {state.output_dir}")
    server.serve_forever()


def build_server(
    *,
    host: str,
    port: int,
    state: AnnotationUiState,
) -> ThreadingHTTPServer:
    """Build a configured HTTP server for tests or local launch."""

    handler = make_handler(state)
    return ThreadingHTTPServer((host, port), handler)


def make_handler(state: AnnotationUiState):
    """Create a request handler class bound to annotation UI state."""

    class AnnotationHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_html(render_annotation_page())
                elif parsed.path == "/results":
                    self._send_html(render_results_page())
                elif parsed.path == "/api/cases":
                    self._send_json({"cases": [_case_payload(case) for case in state.cases]})
                elif parsed.path == "/api/session":
                    self._send_json(_session_response(_case_from_query(parsed.query), state))
                elif parsed.path == "/api/frame":
                    query = parse_qs(parsed.query)
                    case = _case_by_slug(_one(query, "case"), state.cases)
                    frame_index = int(_one(query, "frame", "0"))
                    image = read_frame_jpeg(case.video_path, frame_index)
                    self._send_bytes(image, "image/jpeg")
                elif parsed.path == "/api/video":
                    case = _case_from_query(parsed.query)
                    self._send_file(case.video_path)
                elif parsed.path == "/api/results":
                    case = _case_from_optional_query(parsed.query, state)
                    self._send_json(
                        load_human_results_payload(
                            case,
                            case_views=views_for_case(case, state.cases),
                        )
                    )
                elif parsed.path == "/api/results/evidence":
                    query = parse_qs(parsed.query)
                    self._send_json(
                        load_result_evidence_payload(
                            _case_from_optional_query(parsed.query, state),
                            feature_name=_one(query, "feature"),
                            source_frame_index=int(_one(query, "frame")),
                        )
                    )
                elif parsed.path == "/api/results/frame":
                    query = parse_qs(parsed.query)
                    image = read_result_frame_jpeg(
                        _case_from_optional_query(parsed.query, state),
                        source_frame_index=int(_one(query, "frame")),
                        show_roi=_one(query, "roi", "1") == "1",
                        show_pose=_one(query, "pose", "1") == "1",
                        show_mask=_one(query, "mask", "0") == "1",
                    )
                    self._send_bytes(image, "image/jpeg")
                elif parsed.path == "/api/compare":
                    self._send_json(_comparison_response(_case_from_query(parsed.query), state))
                elif parsed.path == "/api/view-alignment":
                    case = _case_from_query(parsed.query)
                    self._send_json(load_view_alignment(state.output_dir, case.case_id).to_dict())
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")
            except (json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/results/mask-prompt":
                    payload = self._read_json()
                    response = save_result_mask_prompt(
                        _case_by_slug(str(payload["case"]), state.cases),
                        source_frame_index=int(payload["frame"]),
                        x_px=float(payload["x"]),
                        y_px=float(payload["y"]),
                        label=str(payload["label"]),
                    )
                    self._send_json(response)
                    return
                if parsed.path == "/api/results/mask-prompts/undo":
                    payload = self._read_json()
                    response = undo_result_mask_prompt(
                        _case_by_slug(str(payload["case"]), state.cases),
                        source_frame_index=int(payload["frame"]),
                    )
                    self._send_json(response)
                    return
                if parsed.path == "/api/results/mask-prompts/clear":
                    payload = self._read_json()
                    response = clear_result_mask_prompts(
                        _case_by_slug(str(payload["case"]), state.cases),
                        source_frame_index=int(payload["frame"]),
                    )
                    self._send_json(response)
                    return
                if parsed.path == "/api/results/trim-analysis-window":
                    payload = self._read_json()
                    response = trim_human_analysis_window_and_regenerate(
                        _case_by_slug(str(payload["case"]), state.cases),
                        movement_end_frame=int(payload["frame"]),
                        rationale=str(
                            payload.get(
                                "rationale",
                                "Post-injury frames excluded by human operator.",
                            )
                        ),
                        annotator_id=str(payload.get("annotator_id", "researcher_01")),
                    )
                    self._send_json(response)
                    return
                if parsed.path == "/api/view-alignment":
                    payload = self._read_json()
                    case = _case_by_slug(str(payload["case"]), state.cases)
                    existing = load_view_alignment(state.output_dir, case.case_id)
                    anchor = ViewAlignmentAnchor(
                        anchor_id=str(payload["anchor_id"]),
                        label=str(payload["label"]),
                        case_id=case.case_id,
                        view_frames={
                            str(view): int(frame)
                            for view, frame in dict(payload["view_frames"]).items()
                        },
                        notes=str(payload.get("notes", "")),
                        created_by=str(payload.get("created_by", "researcher_01")),
                    )
                    updated = existing.upsert(anchor)
                    path = save_view_alignment(updated, state.output_dir)
                    self._send_json({"saved": True, "path": str(path), **updated.to_dict()})
                    return
                if parsed.path == "/api/import-video":
                    response = _import_video_response(self.headers, self.rfile, state)
                    self._send_json(response, HTTPStatus.CREATED)
                    return
                if parsed.path != "/api/save":
                    self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")
                    return
                payload = self._read_json()
                response = _save_response(payload, state)
                status = HTTPStatus.OK if response["validation"]["ok"] else HTTPStatus.BAD_REQUEST
                self._send_json(response, status)
            except (json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args) -> None:
            return

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw or "{}")

        def _send_html(self, html: str) -> None:
            self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")

        def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_bytes(self, data: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_file(self, path: str | Path) -> None:
            file_path = Path(path)
            if not file_path.exists():
                raise ValueError(f"Could not open video: {file_path}")
            content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            data = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Content-Disposition", f'inline; filename="{file_path.name}"')
            self.end_headers()
            self.wfile.write(data)

    def _case_from_query(query_string: str) -> AnnotationCase:
        query = parse_qs(query_string)
        return _case_by_slug(_one(query, "case"), state.cases)

    def _case_from_optional_query(query_string: str, state: AnnotationUiState) -> AnnotationCase:
        query = parse_qs(query_string)
        return _case_by_slug(_one(query, "case", "christen_press"), state.cases)

    return AnnotationHandler


def smoke_test(
    *,
    output_dir: str | Path = "data/annotations/human",
    video_root: str | Path = "/Users/andriagryffinpro/Desktop/injury_videos",
) -> dict:
    """Run a non-writing UI smoke test."""

    cases = default_annotation_cases(video_root)
    payloads = [_case_payload(case) for case in cases]
    html = render_annotation_page()
    return {
        "ui_version": ANNOTATION_UI_VERSION,
        "case_count": len(payloads),
        "cases": [item["slug"] for item in payloads],
        "html_has_canvas": "<canvas" in html,
        "html_has_save": "Save annotation" in html,
        "html_has_import_video": "Open video from computer" in html,
        "html_has_open_video": "Open selected video" in html,
        "html_has_view_analysis": "View Analysis" in html,
        "results_html_has_single_feature_ui": (
            "featureCategorySelect" in render_results_page()
            and "featureSelect" in render_results_page()
            and "featureGraph" in render_results_page()
            and "Movement Story" in render_results_page()
            and "Selected Measurement" in render_results_page()
        ),
        "output_dir": str(output_dir),
        "writes_files": False,
    }


def read_frame_jpeg(video_path: str | Path, frame_index: int) -> bytes:
    """Read one video frame and return a JPEG byte payload."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_index = min(max(frame_index, 0), max(frame_count - 1, 0))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Could not read frame {frame_index} from {video_path}")
        encoded_ok, buffer = cv2.imencode(".jpg", frame)
        if not encoded_ok:
            raise ValueError("Could not encode video frame as JPEG.")
        return buffer.tobytes()
    finally:
        capture.release()


def _import_video_response(headers, stream: BinaryIO, state: AnnotationUiState) -> dict:
    content_type = headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type:
        raise ValueError("Video import expects multipart/form-data.")
    length = int(headers.get("Content-Length", "0"))
    if length <= 0:
        raise ValueError("No video upload was received.")

    body = stream.read(length)
    upload = _parse_video_upload(content_type, body)
    file_path = _unique_import_path(state.video_root, upload["filename"])
    file_path.write_bytes(upload["data"])

    try:
        read_video_metadata(file_path)
    except ValueError:
        file_path.unlink(missing_ok=True)
        raise ValueError("The selected file could not be read as a video.") from None

    case = _imported_case_for_video(
        video_path=file_path,
        player_name=upload["player_name"],
        cases=state.cases,
    )
    state.cases.append(case)
    _save_imported_cases(state)
    return {"imported": True, "case": _case_payload(case)}


def _parse_video_upload(content_type: str, body: bytes) -> dict:
    message = BytesParser(policy=email_policy).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    if not message.is_multipart():
        raise ValueError("Video import did not contain multipart form data.")

    filename = ""
    file_content_type = ""
    player_name = ""
    video_data: bytes | None = None
    for part in message.iter_parts():
        field_name = part.get_param("name", header="content-disposition")
        if field_name == "video":
            filename = Path(part.get_filename() or "imported_video.mp4").name
            file_content_type = part.get_content_type()
            video_data = part.get_payload(decode=True) or b""
        elif field_name == "player_name":
            player_name = _decode_form_field(part.get_payload(decode=True) or b"")

    if not video_data:
        raise ValueError("No video file was attached.")
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_IMPORT_EXTENSIONS and not file_content_type.startswith("video/"):
        raise ValueError(
            "Unsupported video file type. Choose an mp4, mov, m4v, avi, webm, or mkv file."
        )
    return {
        "filename": filename,
        "player_name": player_name.strip() or _label_from_filename(filename),
        "data": video_data,
    }


def _decode_form_field(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def _unique_import_path(video_root: Path, filename: str) -> Path:
    video_root.mkdir(parents=True, exist_ok=True)
    original = Path(filename).name
    extension = Path(original).suffix.lower() or ".mp4"
    stem = _safe_slug(Path(original).stem) or "imported_video"
    candidate = video_root / f"{stem}{extension}"
    suffix = 2
    while candidate.exists():
        candidate = video_root / f"{stem}_{suffix}{extension}"
        suffix += 1
    return candidate


def _imported_case_for_video(
    *,
    video_path: Path,
    player_name: str,
    cases: list[AnnotationCase],
) -> AnnotationCase:
    label = player_name.strip() or _label_from_filename(video_path.name)
    slug_base = f"{IMPORTED_CASE_SLUG_PREFIX}{_safe_slug(label)}"
    slug = _unique_slug(slug_base, cases)
    source_id = f"{slug}_view_01"
    return AnnotationCase(
        slug=slug,
        case_id=f"{slug}_acl_candidate",
        source_id=source_id,
        view_id=source_id,
        view_label="Imported local video",
        primary_view=True,
        perspective="unknown",
        occlusion_level="unknown",
        view_quality="human_annotation_pending",
        player_name=label,
        video_path=video_path,
        notes=(
            "Imported through the human annotation UI. View category and injury "
            "laterality should be supplied by the operator before analysis."
        ),
    )


def _load_imported_cases(output_dir: Path, video_root: Path) -> tuple[AnnotationCase, ...]:
    path = _imported_cases_path(output_dir)
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    cases = []
    for record in payload.get("cases", ()):
        try:
            cases.append(_imported_case_from_record(record, video_root))
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(cases)


def _save_imported_cases(state: AnnotationUiState) -> Path:
    path = _imported_cases_path(state.output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        _imported_case_record(case)
        for case in state.cases
        if case.slug.startswith(IMPORTED_CASE_SLUG_PREFIX)
    ]
    path.write_text(json.dumps({"cases": records}, indent=2), encoding="utf-8")
    return path


def _imported_cases_path(output_dir: Path) -> Path:
    return output_dir / IMPORTED_CASES_FILENAME


def _imported_case_record(case: AnnotationCase) -> dict:
    return {
        "slug": case.slug,
        "case_id": case.case_id,
        "source_id": case.source_id,
        "view_id": case.view_id or case.source_id,
        "view_label": case.view_label,
        "primary_view": case.primary_view,
        "perspective": case.perspective,
        "occlusion_level": case.occlusion_level,
        "view_quality": case.view_quality,
        "slow_motion": case.slow_motion,
        "cropped_or_zoomed": case.cropped_or_zoomed,
        "real_time_scale": case.real_time_scale,
        "player_name": case.player_name,
        "video_path": str(case.video_path),
        "notes": case.notes,
    }


def _imported_case_from_record(record: dict, video_root: Path) -> AnnotationCase:
    video_path = Path(str(record["video_path"]))
    if not video_path.is_absolute():
        video_path = video_root / video_path
    return AnnotationCase(
        slug=str(record["slug"]),
        case_id=str(record["case_id"]),
        source_id=str(record["source_id"]),
        view_id=str(record.get("view_id") or record["source_id"]),
        view_label=str(record.get("view_label", "Imported local video")),
        primary_view=bool(record.get("primary_view", True)),
        perspective=str(record.get("perspective", "unknown")),
        occlusion_level=str(record.get("occlusion_level", "unknown")),
        view_quality=str(record.get("view_quality", "human_annotation_pending")),
        slow_motion=bool(record.get("slow_motion", False)),
        cropped_or_zoomed=bool(record.get("cropped_or_zoomed", False)),
        real_time_scale=record.get("real_time_scale"),
        player_name=str(record.get("player_name", _label_from_filename(video_path.name))),
        video_path=video_path,
        notes=str(record.get("notes", "")),
    )


def _unique_slug(slug_base: str, cases: list[AnnotationCase]) -> str:
    base = slug_base or f"{IMPORTED_CASE_SLUG_PREFIX}video"
    existing = {case.slug for case in cases}
    candidate = base
    suffix = 2
    while candidate in existing:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:80]


def _label_from_filename(filename: str) -> str:
    label = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return label or "Imported video"


def _case_payload(case: AnnotationCase) -> dict:
    payload = case.to_dict()
    try:
        metadata = read_video_metadata(case.video_path)
        payload["metadata"] = _metadata_payload(metadata)
        payload["video_available"] = True
    except ValueError as exc:
        payload["metadata"] = None
        payload["video_available"] = False
        payload["video_error"] = str(exc)
    return payload


def _metadata_payload(metadata: VideoMetadata) -> dict:
    return {
        "fps": metadata.fps,
        "width": metadata.width,
        "height": metadata.height,
        "frame_count": metadata.frame_count,
        "duration_seconds": metadata.duration_seconds,
    }


def _session_response(case: AnnotationCase, state: AnnotationUiState) -> dict:
    paths = human_annotation_paths(state.output_dir, case.slug)
    if paths.session_json.exists():
        session = load_human_annotation_session(paths.session_json)
        session_payload = session.to_dict()
        resume = True
    else:
        session_payload = {
            "provenance": None,
            "manual_roi_keyframe_count": 0,
            "roi_keyframes": [],
            "movement_window": None,
            "event_annotation": None,
            "event_confidence_label": None,
            "operator_flags": [],
            "notes": "",
            "finalized": False,
        }
        resume = False
    return {
        "case": _case_payload(case),
        "session": session_payload,
        "resume_available": resume,
        "human_results_available": human_results_available(case),
        "human_paths": {
            "session_json": str(paths.session_json),
            "roi_csv": str(paths.roi_csv),
            "movement_window_json": str(paths.movement_window_json),
            "event_json": str(paths.event_json),
        },
    }


def _save_response(payload: dict, state: AnnotationUiState) -> dict:
    case = _case_by_slug(str(payload["case_slug"]), state.cases)
    paths = human_annotation_paths(state.output_dir, case.slug)
    existing = load_human_annotation_session(paths.session_json) if paths.session_json.exists() else None
    annotator_id = str(payload.get("annotator_id", "")).strip() or "researcher_01"
    keyframes = tuple(_keyframe_from_payload(item) for item in payload.get("roi_keyframes", ()))
    movement_window, confidence_label = _movement_window_from_payload(payload, case, keyframes)
    provisional_session = new_human_session(
        case_id=case.case_id,
        source_id=case.source_id,
        video_path=case.video_path,
        annotator_id=annotator_id,
        view_id=case.view_id or case.source_id,
        roi_keyframes=keyframes,
        movement_window=movement_window,
        event_confidence_label=confidence_label,
        notes=str(payload.get("notes", "")),
        finalized=bool(payload.get("finalized", False)),
        existing_provenance=existing.provenance if existing else None,
    )
    event_annotation = (
        movement_window_to_event_annotation(provisional_session, movement_window)
        if movement_window is not None
        else None
    )
    session = provisional_session.with_changes(event_annotation=event_annotation)
    frame_count = None
    try:
        frame_count = read_video_metadata(case.video_path).frame_count
    except ValueError:
        pass
    validation = validate_annotation_session(session, frame_count=frame_count)
    saved_paths = save_human_annotation_session(session, state.output_dir, case.slug)
    return {
        "saved": True,
        "session": session.to_dict(),
        "validation": validation.to_dict(),
        "paths": {
            "session_json": str(saved_paths.session_json),
            "roi_csv": str(saved_paths.roi_csv),
            "movement_window_json": str(saved_paths.movement_window_json),
            "event_json": str(saved_paths.event_json),
        },
    }


def _comparison_response(case: AnnotationCase, state: AnnotationUiState) -> dict:
    paths = human_annotation_paths(state.output_dir, case.slug)
    if not paths.session_json.exists() or not paths.roi_csv.exists():
        return {"available": False, "reason": "No saved human ROI annotation exists yet."}
    if case.development_roi_path is None or not case.development_roi_path.exists():
        return {"available": False, "reason": "No development ROI annotation is registered."}
    metadata = read_video_metadata(case.video_path)
    human_session = load_human_annotation_session(paths.session_json)
    development_timeline = RoiTimeline.from_csv(case.development_roi_path)
    frames = tuple(range(metadata.frame_count))
    roi_summary = compare_roi_timelines(
        human_session.roi_keyframes,
        development_timeline.keyframes,
        frames,
    )
    return {
        "available": True,
        "roi_agreement": roi_summary.to_dict(),
        "movement_window": (
            human_session.movement_window.to_dict()
            if human_session.movement_window is not None
            else None
        ),
        "note": (
            "Agreement describes annotation-box similarity only, not biomechanical validity. "
            "Human Movement End is not compared to development critical-plant/event anchors."
        ),
    }


def _keyframe_from_payload(data: dict) -> RoiKeyframeAnnotation:
    bbox = data["bbox"]
    return RoiKeyframeAnnotation(
        frame_index=int(data["frame_index"]),
        bbox=BBox(
            x=float(bbox["x"]),
            y=float(bbox["y"]),
            width=float(bbox["width"]),
            height=float(bbox["height"]),
        ),
        flags=tuple(data.get("flags", ())),
        note=str(data.get("note", "")),
    )


def _movement_window_from_payload(
    payload: dict,
    case: AnnotationCase,
    keyframes: tuple[RoiKeyframeAnnotation, ...],
) -> tuple[MovementWindowAnnotation | None, EventConfidence | None]:
    movement_window = payload.get("movement_window") or {}
    if not movement_window or movement_window.get("movement_end_frame") in (None, ""):
        return None, None
    confidence_label = (
        EventConfidence(str(movement_window["confidence"]).lower())
        if movement_window.get("confidence")
        else None
    )
    if not keyframes:
        raise ValueError("Movement Start requires at least one ROI keyframe.")
    metadata = read_video_metadata(case.video_path)
    start_frame = int(keyframes[0].frame_index)
    end_frame = _optional_int(movement_window.get("movement_end_frame"))
    if end_frame is None:
        return None, confidence_label
    return (
        MovementWindowAnnotation(
            movement_start_frame=start_frame,
            movement_start_timestamp_ms=start_frame / metadata.fps * 1000 if metadata.fps else 0.0,
            movement_end_frame=end_frame,
            movement_end_timestamp_ms=end_frame / metadata.fps * 1000 if metadata.fps else 0.0,
            confidence=confidence_label,
            rationale=str(movement_window.get("rationale", "")),
            source="human_ui_movement_window",
        ),
        confidence_label,
    )


def _case_by_slug(slug: str, cases: list[AnnotationCase] | tuple[AnnotationCase, ...]) -> AnnotationCase:
    for case in cases:
        if case.slug == slug:
            return case
    raise KeyError(f"Unknown case: {slug}")


def _one(query: dict[str, list[str]], key: str, default: str | None = None) -> str:
    values = query.get(key)
    if not values:
        if default is None:
            raise KeyError(f"Missing query parameter: {key}")
        return default
    return values[0]


def _optional_int(value) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def render_annotation_page() -> str:
    """Return the self-contained annotation UI page."""

    return r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ACL Movement Explorer - Human Annotation</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f8;
      --ink: #1f2933;
      --muted: #5d6673;
      --line: #d6dbe1;
      --panel: #ffffff;
      --green: #148a54;
      --amber: #c47b00;
      --blue: #1d68c4;
      --red: #b42335;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
    }
    h1 { font-size: 18px; margin: 0; }
    .app {
      display: grid;
      grid-template-columns: minmax(520px, 1fr) 360px;
      gap: 14px;
      padding: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .viewer {
      min-height: 560px;
    }
    canvas {
      display: block;
      width: 100%;
      max-height: 68vh;
      background: #111827;
      border-radius: 6px;
      cursor: crosshair;
    }
    .controls, .row {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .row { margin-bottom: 10px; }
    label { font-size: 13px; color: var(--muted); }
    select, input, textarea, button {
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 9px;
      background: #fff;
    }
    button {
      cursor: pointer;
      font-weight: 600;
    }
    button.primary {
      background: var(--blue);
      color: white;
      border-color: var(--blue);
    }
    button.good {
      background: var(--green);
      color: white;
      border-color: var(--green);
    }
    button.warn {
      background: #fff8e6;
      color: #784900;
      border-color: #f0c96b;
    }
    button.danger {
      background: #fff1f2;
      color: var(--red);
      border-color: #f3a9b2;
    }
    input[type="range"] { width: 100%; }
    textarea { width: 100%; min-height: 62px; resize: vertical; }
    .meta {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      font-size: 13px;
      margin: 10px 0;
    }
    .meta div {
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px;
    }
    .timeline {
      position: relative;
      height: 22px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #eef2f6;
      margin: 8px 0 12px;
    }
    .marker {
      position: absolute;
      top: 3px;
      width: 14px;
      height: 14px;
      margin-left: -7px;
      border-radius: 50%;
      background: var(--green);
      border: 2px solid white;
      box-shadow: 0 0 0 1px var(--green);
    }
    .event-marker {
      position: absolute;
      top: 0;
      width: 2px;
      height: 20px;
      background: var(--red);
    }
    .hint {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.35;
    }
    .status {
      white-space: pre-wrap;
      font-size: 13px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      min-height: 42px;
    }
    details { margin-top: 10px; }
    summary { cursor: pointer; font-weight: 700; }
    .flag-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 6px;
      font-size: 13px;
    }
    .legend {
      display: flex;
      gap: 14px;
      font-size: 13px;
      margin-top: 8px;
    }
    .swatch {
      display: inline-block;
      width: 18px;
      height: 10px;
      border: 2px solid currentColor;
      margin-right: 5px;
    }
  </style>
</head>
<body>
  <header>
    <h1>Human Annotation - ACL Movement Explorer</h1>
    <div class="controls">
      <label>Case <select id="caseSelect"></select></label>
      <button id="importVideo" type="button">Open video from computer</button>
      <input id="localVideoInput" type="file" accept="video/*" hidden />
      <button id="openCurrentVideo" type="button">Open selected video</button>
      <label>Annotator <input id="annotatorId" value="researcher_01" /></label>
    </div>
  </header>
  <main class="app">
    <section class="panel viewer">
      <canvas id="frameCanvas"></canvas>
      <div class="legend">
        <span><span class="swatch" style="color: var(--green)"></span>manual keyframe</span>
        <span><span class="swatch" style="color: var(--amber); border-style: dashed"></span>propagated ROI</span>
      </div>
      <div class="meta">
        <div>Player<br><strong id="playerName">-</strong></div>
        <div>Source frame<br><strong id="frameLabel">-</strong></div>
        <div>Timestamp<br><strong id="timeLabel">-</strong></div>
        <div>Video<br><strong id="videoLabel">-</strong></div>
      </div>
      <input id="scrub" type="range" min="0" max="0" value="0" />
      <div id="timeline" class="timeline"></div>
      <div class="controls">
        <button id="back5">-5</button>
        <button id="prev">Previous</button>
        <button id="next">Next</button>
        <button id="fwd5">+5</button>
        <label>Jump <input id="jumpFrame" type="number" min="0" style="width: 92px" /></label>
        <button id="jump">Go</button>
        <button id="review" class="warn">Review play</button>
      </div>
    </section>
    <aside class="panel">
      <p class="hint">
        Draw a box around the documented injured athlete. Keep the full visible body
        inside the box where possible. Add a correction when the propagated box no
        longer follows the athlete or clips visible limbs. Mark Movement End where
        the visible movement sequence has effectively finished.
      </p>
      <h3>Select athlete</h3>
      <div class="controls">
        <button id="saveKeyframe" class="good">Add / replace correction</button>
        <button id="copyPrevious">Copy previous ROI</button>
        <button id="deleteKeyframe" class="danger">Delete current</button>
        <button id="undo">Undo</button>
      </div>
      <details>
        <summary>Optional keyframe flags</summary>
        <div class="flag-grid" id="flagGrid"></div>
        <label>Keyframe note <textarea id="keyframeNote"></textarea></label>
      </details>
      <h3>Movement window</h3>
      <p class="hint">
        Movement Start is automatically set from your first target ROI keyframe.
        This is the first clearly analysable frame. You do not need to identify
        when the ACL injury occurred; the system analyses the movement inside this window.
      </p>
      <div class="row">
        <button id="setMovementEnd" class="primary">Mark Movement End</button>
      </div>
      <div class="row">
        <label>Confidence
          <select id="confidence">
            <option value="moderate">moderate</option>
            <option value="high">high</option>
            <option value="low">low</option>
          </select>
        </label>
      </div>
      <label>Movement End rationale <textarea id="movementRationale"></textarea></label>
      <pre id="windowSummary" class="status"></pre>
      <h3>Review and save</h3>
      <label>Session notes <textarea id="sessionNotes"></textarea></label>
      <div class="controls">
        <button id="save" class="primary">Save annotation</button>
        <button id="finalSave" class="good">Save as ready for validation</button>
        <a id="viewAnalysis" class="button primary" href="/results?case=christen_press">View Analysis</a>
      </div>
      <div id="status" class="status"></div>
      <details>
        <summary>Comparison after human save</summary>
        <p class="hint">Development boxes are not shown during annotation. This comparison is available only after your human annotation exists.</p>
        <button id="compare">Compare with development annotation</button>
        <pre id="compareStatus" class="status"></pre>
      </details>
    </aside>
  </main>
<script>
const flags = ["PLAYER_OVERLAP", "TARGET_PARTIALLY_OCCLUDED", "TARGET_NOT_VISIBLE", "CAMERA_CUT", "ROI_DIFFICULT", "OTHER"];
let app = {
  cases: [],
  currentCase: null,
  meta: {fps: 0, frame_count: 0, width: 0, height: 0},
  frame: 0,
  image: new Image(),
  keyframes: [],
  draftBox: null,
  drawing: null,
  movementWindow: {},
  sessionNotes: "",
  history: [],
  reviewTimer: null
};

const canvas = document.getElementById("frameCanvas");
const ctx = canvas.getContext("2d");

function $(id) { return document.getElementById(id); }

async function api(path, options) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) return response;
  const data = await response.json();
  if (!response.ok && data.error) throw new Error(data.error);
  return data;
}

function escapeHtml(value) {
  const replacements = {"&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"};
  return String(value).replace(/[&<>"']/g, character => replacements[character]);
}

function caseOptionLabel(caseItem) {
  const label = caseItem.view_label && caseItem.view_label !== "Primary view"
    ? `${caseItem.player_name} - ${caseItem.view_label}`
    : caseItem.player_name;
  return label;
}

function renderCaseSelect(selectedSlug) {
  $("caseSelect").innerHTML = app.cases
    .map(c => `<option value="${escapeHtml(c.slug)}">${escapeHtml(caseOptionLabel(c))}</option>`)
    .join("");
  if (selectedSlug) $("caseSelect").value = selectedSlug;
}

async function init() {
  flags.forEach(flag => {
    const label = document.createElement("label");
    label.innerHTML = `<input type="checkbox" value="${flag}" /> ${flag.replaceAll("_", " ").toLowerCase()}`;
    $("flagGrid").appendChild(label);
  });
  const data = await api("/api/cases");
  app.cases = data.cases;
  renderCaseSelect();
  $("caseSelect").addEventListener("change", () => loadCase($("caseSelect").value));
  bindControls();
  const requestedCase = new URLSearchParams(window.location.search).get("case");
  const initialCase = app.cases.find(c => c.slug === requestedCase) || app.cases[0];
  await loadCase(initialCase.slug);
}

async function loadCase(slug) {
  const data = await api(`/api/session?case=${encodeURIComponent(slug)}`);
  app.currentCase = data.case;
  $("caseSelect").value = app.currentCase.slug;
  $("viewAnalysis").href = `/results?case=${encodeURIComponent(app.currentCase.slug)}`;
  $("viewAnalysis").style.display = data.human_results_available ? "inline-block" : "none";
  app.meta = data.case.metadata || {fps: 0, frame_count: 1, width: 0, height: 0};
  app.keyframes = (data.session.roi_keyframes || []).map(normalizeKeyframe);
  app.movementWindow = data.session.movement_window || {};
  if (!app.movementWindow.movement_end_frame && data.session.event_annotation?.event_anchor_frame !== undefined) {
    app.movementWindow.movement_end_frame = data.session.event_annotation.event_anchor_frame;
    app.movementWindow.confidence = data.session.event_confidence_label || "moderate";
    app.movementWindow.rationale = data.session.event_annotation.notes || "";
  }
  app.sessionNotes = data.session.notes || "";
  $("sessionNotes").value = app.sessionNotes;
  $("annotatorId").value = data.session.provenance?.annotator_id || $("annotatorId").value || "researcher_01";
  $("playerName").textContent = data.case.player_name;
  $("videoLabel").textContent = `${app.meta.width || "-"}x${app.meta.height || "-"} @ ${app.meta.fps || "-"} fps`;
  $("scrub").max = Math.max((app.meta.frame_count || 1) - 1, 0);
  app.frame = 0;
  if (app.movementWindow.movement_start_frame !== null && app.movementWindow.movement_start_frame !== undefined) {
    app.frame = app.movementWindow.movement_start_frame;
  } else if (app.keyframes.length) {
    app.frame = app.keyframes[0].frame_index;
  } else if (app.movementWindow.movement_end_frame !== null && app.movementWindow.movement_end_frame !== undefined) {
    app.frame = app.movementWindow.movement_end_frame;
  }
  updateMovementWindowControls();
  await loadFrame(app.frame);
  renderTimeline();
  setStatus(data.resume_available ? "Resumed saved human annotation." : "New independent human annotation session.");
}

function bindControls() {
  $("scrub").addEventListener("input", e => loadFrame(Number(e.target.value)));
  $("prev").onclick = () => loadFrame(app.frame - 1);
  $("next").onclick = () => loadFrame(app.frame + 1);
  $("back5").onclick = () => loadFrame(app.frame - 5);
  $("fwd5").onclick = () => loadFrame(app.frame + 5);
  $("jump").onclick = () => loadFrame(Number($("jumpFrame").value || 0));
  $("saveKeyframe").onclick = saveKeyframe;
  $("copyPrevious").onclick = copyPrevious;
  $("deleteKeyframe").onclick = deleteKeyframe;
  $("undo").onclick = undo;
  $("setMovementEnd").onclick = setMovementEnd;
  $("save").onclick = () => saveSession(false);
  $("finalSave").onclick = () => saveSession(true);
  $("compare").onclick = compareAnnotations;
  $("review").onclick = toggleReview;
  $("importVideo").onclick = importLocalVideo;
  $("localVideoInput").addEventListener("change", handleLocalVideoSelected);
  $("openCurrentVideo").onclick = openCurrentVideo;
  window.addEventListener("keydown", event => {
    if (event.target.tagName === "TEXTAREA" || event.target.tagName === "INPUT") return;
    if (event.key === "ArrowLeft") loadFrame(app.frame - 1);
    if (event.key === "ArrowRight") loadFrame(app.frame + 1);
    if (event.key === "s") saveSession(false);
  });
  canvas.addEventListener("mousedown", startDraw);
  canvas.addEventListener("mousemove", moveDraw);
  canvas.addEventListener("mouseup", finishDraw);
}

function importLocalVideo() {
  $("localVideoInput").value = "";
  $("localVideoInput").click();
}

async function handleLocalVideoSelected(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const suggestedLabel = file.name.replace(/\.[^.]+$/, "").replaceAll("_", " ").trim();
  const playerName = window.prompt(
    "Player / case label for this video",
    suggestedLabel || "Imported video"
  );
  if (playerName === null) {
    setStatus("Video import cancelled.");
    return;
  }
  const formData = new FormData();
  formData.append("video", file);
  formData.append("player_name", playerName.trim() || suggestedLabel || "Imported video");
  setStatus(`Importing ${file.name}...`);
  const data = await api("/api/import-video", {
    method: "POST",
    body: formData
  });
  const existingIndex = app.cases.findIndex(c => c.slug === data.case.slug);
  if (existingIndex >= 0) app.cases[existingIndex] = data.case;
  else app.cases.push(data.case);
  renderCaseSelect(data.case.slug);
  await loadCase(data.case.slug);
  setStatus(`Opened ${data.case.player_name} for annotation.\nImported copy: ${data.case.video_path}`);
}

function openCurrentVideo() {
  if (!app.currentCase?.slug) {
    setStatus("Choose a case before opening a video.");
    return;
  }
  window.open(`/api/video?case=${encodeURIComponent(app.currentCase.slug)}`, "_blank", "noopener");
}

async function loadFrame(frame) {
  const maxFrame = Math.max((app.meta.frame_count || 1) - 1, 0);
  app.frame = Math.min(Math.max(Math.round(frame), 0), maxFrame);
  $("scrub").value = app.frame;
  $("jumpFrame").value = app.frame;
  $("frameLabel").textContent = `${app.frame} / ${maxFrame}`;
  const timeMs = app.meta.fps ? (app.frame / app.meta.fps * 1000) : 0;
  $("timeLabel").textContent = `${timeMs.toFixed(1)} ms`;
  app.image.onload = () => {
    canvas.width = app.image.naturalWidth;
    canvas.height = app.image.naturalHeight;
    draw();
  };
  app.image.src = `/api/frame?case=${encodeURIComponent(app.currentCase.slug)}&frame=${app.frame}&t=${Date.now()}`;
  app.draftBox = null;
  updateKeyframeNote();
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(app.image, 0, 0);
  const manual = keyframeAt(app.frame);
  const propagated = propagatedBox(app.frame);
  if (propagated && !manual) drawBox(propagated, "#c47b00", "Propagated ROI", true);
  if (manual) drawBox(manual.bbox, "#148a54", "Manual keyframe", false);
  if (app.draftBox) drawBox(app.draftBox, "#1d68c4", "Draft ROI", false);
  if (app.movementWindow.movement_end_frame === app.frame) {
    ctx.fillStyle = "#b42335";
    ctx.fillText("Movement End", 12, 24);
  }
}

function drawBox(box, color, label, dashed) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 4;
  ctx.setLineDash(dashed ? [12, 8] : []);
  ctx.strokeRect(box.x, box.y, box.width, box.height);
  ctx.setLineDash([]);
  ctx.fillStyle = color;
  ctx.font = "20px sans-serif";
  ctx.fillText(label, box.x + 6, Math.max(box.y - 8, 22));
  ctx.restore();
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * canvas.width / rect.width,
    y: (event.clientY - rect.top) * canvas.height / rect.height
  };
}

function startDraw(event) {
  const point = canvasPoint(event);
  app.drawing = {start: point, current: point};
}

function moveDraw(event) {
  if (!app.drawing) return;
  app.drawing.current = canvasPoint(event);
  app.draftBox = boxFromPoints(app.drawing.start, app.drawing.current);
  draw();
}

function finishDraw(event) {
  if (!app.drawing) return;
  app.drawing.current = canvasPoint(event);
  app.draftBox = boxFromPoints(app.drawing.start, app.drawing.current);
  app.drawing = null;
  draw();
}

function boxFromPoints(a, b) {
  const x = Math.max(0, Math.min(a.x, b.x));
  const y = Math.max(0, Math.min(a.y, b.y));
  const width = Math.abs(a.x - b.x);
  const height = Math.abs(a.y - b.y);
  return {x, y, width, height};
}

function saveKeyframe() {
  const box = app.draftBox || keyframeAt(app.frame)?.bbox || propagatedBox(app.frame);
  if (!box || box.width < 2 || box.height < 2) {
    setStatus("Draw a target box before saving this keyframe.");
    return;
  }
  pushHistory();
  const keyframe = {
    frame_index: app.frame,
    bbox: box,
    flags: selectedFlags(),
    note: $("keyframeNote").value || ""
  };
  app.keyframes = app.keyframes.filter(k => k.frame_index !== app.frame).concat([keyframe]);
  app.keyframes.sort((a, b) => a.frame_index - b.frame_index);
  app.draftBox = null;
  renderTimeline();
  draw();
  setStatus(`Saved ROI keyframe at source frame ${app.frame}.`);
}

function copyPrevious() {
  const previous = [...app.keyframes].reverse().find(k => k.frame_index <= app.frame);
  if (!previous) {
    setStatus("No previous ROI keyframe to copy.");
    return;
  }
  app.draftBox = {...previous.bbox};
  draw();
}

function deleteKeyframe() {
  if (!keyframeAt(app.frame)) {
    setStatus("There is no manual keyframe at this frame.");
    return;
  }
  pushHistory();
  app.keyframes = app.keyframes.filter(k => k.frame_index !== app.frame);
  renderTimeline();
  draw();
  setStatus(`Deleted ROI keyframe at source frame ${app.frame}.`);
}

function undo() {
  const previous = app.history.pop();
  if (!previous) {
    setStatus("Nothing to undo.");
    return;
  }
  app.keyframes = previous;
  renderTimeline();
  draw();
}

function pushHistory() {
  app.history.push(JSON.parse(JSON.stringify(app.keyframes)));
  if (app.history.length > 25) app.history.shift();
}

function setMovementEnd() {
  app.movementWindow.movement_end_frame = app.frame;
  app.movementWindow.confidence = $("confidence").value;
  app.movementWindow.rationale = $("movementRationale").value || "";
  updateMovementWindowControls();
  renderTimeline();
  draw();
  setStatus(`Marked Movement End at source frame ${app.frame}.`);
}

async function saveSession(finalized) {
  app.movementWindow.confidence = $("confidence").value;
  app.movementWindow.rationale = $("movementRationale").value || "";
  const payload = {
    case_slug: app.currentCase.slug,
    annotator_id: $("annotatorId").value || "researcher_01",
    roi_keyframes: app.keyframes,
    movement_window: app.movementWindow,
    notes: $("sessionNotes").value || "",
    finalized
  };
  const response = await api("/api/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const lines = [
    finalized ? "Saved as ready for validation." : "Saved partial annotation.",
    `ROI keyframes: ${response.validation.summary.target_roi_keyframes}`,
    `Movement Start: ${response.validation.summary.movement_start ?? "-"}`,
    `Movement End: ${response.validation.summary.movement_end ?? "-"}`,
    `Movement duration: ${response.validation.summary.movement_duration_ms ?? "-"} ms`,
    `Confidence: ${response.validation.summary.confidence ?? "-"}`,
    ...response.validation.warnings.map(w => `Warning: ${w}`),
    ...response.validation.errors.map(e => `Error: ${e}`),
    `ROI file: ${response.paths.roi_csv}`,
    `Movement Window file: ${response.paths.movement_window_json}`,
    `Event file: ${response.paths.event_json}`,
    `Session file: ${response.paths.session_json}`
  ];
  setStatus(lines.join("\n"));
}

async function compareAnnotations() {
  const data = await api(`/api/compare?case=${encodeURIComponent(app.currentCase.slug)}`);
  $("compareStatus").textContent = JSON.stringify(data, null, 2);
}

function toggleReview() {
  if (app.reviewTimer) {
    clearInterval(app.reviewTimer);
    app.reviewTimer = null;
    $("review").textContent = "Review play";
    return;
  }
  $("review").textContent = "Stop review";
  app.reviewTimer = setInterval(() => {
    const next = app.frame + 1;
    if (next >= (app.meta.frame_count || 1)) toggleReview();
    else loadFrame(next);
  }, 180);
}

function keyframeAt(frame) {
  return app.keyframes.find(k => k.frame_index === frame);
}

function propagatedBox(frame) {
  if (!app.keyframes.length) return null;
  const sorted = [...app.keyframes].sort((a, b) => a.frame_index - b.frame_index);
  if (frame <= sorted[0].frame_index) return sorted[0].bbox;
  if (frame >= sorted[sorted.length - 1].frame_index) return sorted[sorted.length - 1].bbox;
  for (let i = 1; i < sorted.length; i++) {
    const prev = sorted[i - 1];
    const next = sorted[i];
    if (frame <= next.frame_index) {
      const t = (frame - prev.frame_index) / (next.frame_index - prev.frame_index);
      return {
        x: prev.bbox.x + (next.bbox.x - prev.bbox.x) * t,
        y: prev.bbox.y + (next.bbox.y - prev.bbox.y) * t,
        width: prev.bbox.width + (next.bbox.width - prev.bbox.width) * t,
        height: prev.bbox.height + (next.bbox.height - prev.bbox.height) * t
      };
    }
  }
  return sorted[sorted.length - 1].bbox;
}

function selectedFlags() {
  return [...document.querySelectorAll("#flagGrid input:checked")].map(input => input.value);
}

function updateKeyframeNote() {
  const manual = keyframeAt(app.frame);
  $("keyframeNote").value = manual?.note || "";
  document.querySelectorAll("#flagGrid input").forEach(input => {
    input.checked = manual?.flags?.includes(input.value) || false;
  });
}

function updateMovementWindowControls() {
  if (app.movementWindow.confidence) $("confidence").value = app.movementWindow.confidence;
  $("movementRationale").value = app.movementWindow.rationale || "";
  renderWindowSummary();
}

function renderWindowSummary() {
  const start = movementStartFrame();
  const end = app.movementWindow.movement_end_frame;
  const startMs = start !== null && app.meta.fps ? (start / app.meta.fps * 1000) : null;
  const endMs = end !== null && end !== undefined && app.meta.fps ? (end / app.meta.fps * 1000) : null;
  const duration = startMs !== null && endMs !== null ? endMs - startMs : null;
  $("windowSummary").textContent = [
    `Movement Start: ${start ?? "-"}`,
    `Movement End: ${end ?? "-"}`,
    `Duration: ${duration !== null ? duration.toFixed(1) + " ms" : "-"}`,
    "Movement End is not an ACL rupture or injury frame."
  ].join("\n");
}

function renderTimeline() {
  const maxFrame = Math.max((app.meta.frame_count || 1) - 1, 1);
  $("timeline").innerHTML = "";
  const start = movementStartFrame();
  const end = app.movementWindow.movement_end_frame;
  if (start !== null && end !== null && end !== undefined) {
    const band = document.createElement("div");
    band.style.position = "absolute";
    band.style.top = "5px";
    band.style.height = "10px";
    band.style.borderRadius = "8px";
    band.style.background = "rgba(29,104,196,0.25)";
    band.style.left = `${Math.min(start, end) / maxFrame * 100}%`;
    band.style.width = `${Math.abs(end - start) / maxFrame * 100}%`;
    band.title = "Analysis window";
    $("timeline").appendChild(band);
  }
  app.keyframes.forEach(keyframe => {
    const marker = document.createElement("div");
    marker.className = "marker";
    marker.title = `ROI keyframe ${keyframe.frame_index}`;
    marker.style.left = `${keyframe.frame_index / maxFrame * 100}%`;
    marker.onclick = () => loadFrame(keyframe.frame_index);
    $("timeline").appendChild(marker);
  });
  if (app.movementWindow.movement_end_frame !== null && app.movementWindow.movement_end_frame !== undefined) {
    const marker = document.createElement("div");
    marker.className = "event-marker";
    marker.title = `Movement End ${app.movementWindow.movement_end_frame}`;
    marker.style.left = `${app.movementWindow.movement_end_frame / maxFrame * 100}%`;
    marker.onclick = () => loadFrame(app.movementWindow.movement_end_frame);
    $("timeline").appendChild(marker);
  }
}

function movementStartFrame() {
  if (!app.keyframes.length) return null;
  return [...app.keyframes].sort((a, b) => a.frame_index - b.frame_index)[0].frame_index;
}

function normalizeKeyframe(keyframe) {
  return {
    frame_index: Number(keyframe.frame_index),
    bbox: keyframe.bbox,
    flags: keyframe.flags || [],
    note: keyframe.note || ""
  };
}

function setStatus(message) {
  $("status").textContent = message;
}

init().catch(error => setStatus(error.message));
</script>
</body>
</html>
"""
