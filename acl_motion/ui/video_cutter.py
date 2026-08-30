"""Local browser UI for reviewing and cutting video clips."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from acl_motion.annotations.case_intake import injury_case_options, register_analysis_clip
from acl_motion.annotations.registry import analysis_annotation_cases
from acl_motion.annotations.research_metadata import RESEARCH_METADATA_FILENAME
from acl_motion.ui.app_shell import app_shell_css, app_site_header
from acl_motion.video.context import (
    CONTEXT_CLIP_ROLE,
    CONTEXT_CLIPS_FILENAME,
    context_clip_registry_path,
    new_context_video_clip,
    save_context_video_clip,
)
from acl_motion.video.io import VideoMetadata, read_video_metadata

VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
DEFAULT_MAIN_MENU_URL = "http://127.0.0.1:8765/"
DEFAULT_VIDEO_ROOTS = (
    "data/videos",
    "data/videos/analysis_clips",
)


@dataclass(slots=True)
class VideoCutterState:
    """Runtime state shared by video cutter UI requests."""

    video_roots: tuple[Path, ...]
    output_dir: Path
    main_menu_url: str = DEFAULT_MAIN_MENU_URL
    context_clips_path: Path = field(
        default_factory=lambda: context_clip_registry_path().resolve()
    )
    manual_videos: set[Path] = field(default_factory=set)


def create_video_cutter_state(
    *,
    video_roots: tuple[str | Path, ...] = DEFAULT_VIDEO_ROOTS,
    output_dir: str | Path = "data/videos/analysis_clips",
    main_menu_url: str = DEFAULT_MAIN_MENU_URL,
    annotation_output_dir: str | Path = "data/annotations/human",
) -> VideoCutterState:
    """Build cutter state for either the standalone or integrated UI."""

    return VideoCutterState(
        video_roots=tuple(_resolve_path(root) for root in video_roots),
        output_dir=_resolve_path(output_dir),
        main_menu_url=main_menu_url,
        context_clips_path=_resolve_path(
            Path(annotation_output_dir) / CONTEXT_CLIPS_FILENAME
        ),
    )


def run_video_cutter_ui(
    *,
    host: str = "127.0.0.1",
    port: int = 8770,
    video_roots: tuple[str | Path, ...] = DEFAULT_VIDEO_ROOTS,
    output_dir: str | Path = "data/videos/analysis_clips",
    main_menu_url: str = DEFAULT_MAIN_MENU_URL,
) -> None:
    """Run the local video cutter UI until interrupted."""

    state = create_video_cutter_state(
        video_roots=video_roots,
        output_dir=output_dir,
        main_menu_url=main_menu_url,
    )
    server = build_server(host=host, port=port, state=state)
    print(f"ACL Movement Analytics Lab video cutter: http://{host}:{port}")
    print("Video roots:")
    for root in state.video_roots:
        print(f"  - {root}")
    print(f"Cut clips will save under: {state.output_dir}")
    server.serve_forever()


def build_server(
    *,
    host: str,
    port: int,
    state: VideoCutterState,
) -> ThreadingHTTPServer:
    """Build a configured HTTP server for tests or local launch."""

    handler = make_handler(state)
    return ThreadingHTTPServer((host, port), handler)


def make_handler(state: VideoCutterState):
    """Create a request handler class bound to video cutter state."""

    class VideoCutterHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_html(
                        render_video_cutter_page(main_menu_url=state.main_menu_url)
                    )
                elif parsed.path == "/api/videos":
                    self._send_json(_videos_response(state))
                elif parsed.path == "/api/context-cases":
                    self._send_json(_context_cases_response(state))
                elif parsed.path == "/api/metadata":
                    video_path = _video_path_from_query(parsed.query, state)
                    self._send_json(_video_payload(video_path))
                elif parsed.path == "/api/video":
                    video_path = _video_path_from_query(parsed.query, state)
                    self._send_file(video_path)
                elif parsed.path == "/api/download":
                    output_path = _output_path_from_query(parsed.query, state)
                    self._send_file(output_path, attachment=True)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")
            except (BrokenPipeError, ConnectionResetError):
                return
            except (KeyError, OSError, ValueError) as exc:
                try:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                except (BrokenPipeError, ConnectionResetError):
                    return

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_html(
                        render_video_cutter_page(main_menu_url=state.main_menu_url),
                        send_body=False,
                    )
                elif parsed.path == "/api/videos":
                    self._send_json(_videos_response(state), send_body=False)
                elif parsed.path == "/api/context-cases":
                    self._send_json(_context_cases_response(state), send_body=False)
                elif parsed.path == "/api/metadata":
                    video_path = _video_path_from_query(parsed.query, state)
                    self._send_json(_video_payload(video_path), send_body=False)
                elif parsed.path == "/api/video":
                    video_path = _video_path_from_query(parsed.query, state)
                    self._send_file(video_path, send_body=False)
                elif parsed.path == "/api/download":
                    output_path = _output_path_from_query(parsed.query, state)
                    self._send_file(output_path, attachment=True, send_body=False)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")
            except (BrokenPipeError, ConnectionResetError):
                return
            except (KeyError, OSError, ValueError) as exc:
                try:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST, send_body=False)
                except (BrokenPipeError, ConnectionResetError):
                    return

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/open-path":
                    payload = self._read_json()
                    self._send_json(open_video_path_response(payload, state))
                    return
                if parsed.path == "/api/assign-analysis-clip":
                    payload = self._read_json()
                    response, _ = assign_analysis_clip(payload, state)
                    self._send_json(response, HTTPStatus.CREATED)
                    return
                if parsed.path != "/api/cut":
                    self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")
                    return
                payload = self._read_json()
                self._send_json(cut_video_response(payload, state, api_base="/api"))
            except (BrokenPipeError, ConnectionResetError):
                return
            except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
                try:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                except (BrokenPipeError, ConnectionResetError):
                    return

        def log_message(self, format: str, *args) -> None:
            return

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw or "{}")

        def _send_html(self, html: str, *, send_body: bool = True) -> None:
            self._send_bytes(
                html.encode("utf-8"), "text/html; charset=utf-8", send_body=send_body
            )

        def _send_json(
            self,
            payload: dict,
            status: HTTPStatus = HTTPStatus.OK,
            *,
            send_body: bool = True,
        ) -> None:
            data = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if send_body:
                self.wfile.write(data)

        def _send_file(
            self,
            path: Path,
            *,
            attachment: bool = False,
            send_body: bool = True,
        ) -> None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            file_size = path.stat().st_size
            range_header = self.headers.get("Range")
            if range_header:
                try:
                    start, end = _parse_range(range_header, file_size)
                except (TypeError, ValueError):
                    self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", "0")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    return
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            else:
                start, end = 0, max(file_size - 1, 0)
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(max(end - start + 1, 0)))
            self.send_header("Cache-Control", "no-store")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            if not send_body:
                return
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

        def _send_bytes(self, data: bytes, content_type: str, *, send_body: bool = True) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if send_body:
                self.wfile.write(data)

    return VideoCutterHandler


def video_cutter_videos_response(state: VideoCutterState) -> dict:
    """Return the video listing used by an integrated cutter route."""

    return _videos_response(state)


def video_cutter_context_cases_response(state: VideoCutterState) -> dict:
    """Return context-case options used by an integrated cutter route."""

    return _context_cases_response(state)


def video_cutter_video_path(query_string: str, state: VideoCutterState) -> Path:
    """Resolve a validated source video from an integrated request."""

    return _video_path_from_query(query_string, state)


def video_cutter_output_path(query_string: str, state: VideoCutterState) -> Path:
    """Resolve a validated cut output from an integrated request."""

    return _output_path_from_query(query_string, state)


def open_video_path_response(payload: dict, state: VideoCutterState) -> dict:
    """Open a local video path and add it to the current cutter session."""

    video_path = _validate_video_file(_resolve_path(str(payload["path"])))
    state.manual_videos.add(video_path)
    return {"video": _video_payload(video_path)}


def cut_video_response(
    payload: dict,
    state: VideoCutterState,
    *,
    api_base: str = "/api",
) -> dict:
    """Cut a video and tailor browser URLs to the route hosting the cutter."""

    video_path = _decode_video_id(str(payload["video_id"]), state)
    result = cut_video_segment(
        video_path=video_path,
        output_dir=state.output_dir,
        start_seconds=float(payload["start_seconds"]),
        end_seconds=float(payload["end_seconds"]),
        output_name=str(payload.get("output_name", "")),
        mode=str(payload.get("mode", "accurate")),
    )
    result["source_video_path"] = str(video_path)
    if str(payload.get("clip_role", "ANALYSIS_CLIP")) == CONTEXT_CLIP_ROLE:
        case = _context_case_by_id(str(payload.get("case_id", "")), state)
        context_clip = new_context_video_clip(
            case_id=case.case_id,
            video_path=result["path"],
            source_video_path=video_path,
            start_seconds=float(result["start_seconds"]),
            end_seconds=float(result["end_seconds"]),
            created_by=str(payload.get("created_by", "researcher_01")),
        )
        save_context_video_clip(context_clip, state.context_clips_path)
        result["context_clip"] = {
            **context_clip.to_dict(),
            "player_name": case.player_name,
        }
    result["download_url"] = (
        f"{api_base.rstrip('/')}/download?id={result['video_id']}"
    )
    return result


def _context_cases_response(state: VideoCutterState) -> dict:
    return {
        "cases": list(
            injury_case_options(
                _registered_cases(state),
                research_metadata_path=_research_path(state),
            )
        )
    }


def _context_case_options(state: VideoCutterState):
    by_case_id = {}
    for case in _registered_cases(state):
        by_case_id.setdefault(case.case_id, case)
    return tuple(
        sorted(
            by_case_id.values(),
            key=lambda case: (case.player_name.casefold(), case.case_id),
        )
    )


def _registered_cases(state: VideoCutterState):
    return analysis_annotation_cases(imported_cases_path=_imported_cases_path(state))


def _context_case_by_id(case_id: str, state: VideoCutterState):
    if not case_id:
        raise ValueError("Choose the injury case for this real-time context clip.")
    for case in _context_case_options(state):
        if case.case_id == case_id:
            return case
    raise ValueError("The selected injury case is not registered.")


def assign_analysis_clip(
    payload: dict,
    state: VideoCutterState,
    *,
    cases=None,
) -> tuple[dict, object]:
    """Attach one cut analysis clip to a new or existing injury event."""

    video_path = _decode_video_id(str(payload["video_id"]), state)
    registered_cases = tuple(cases) if cases is not None else _registered_cases(state)
    case, details = register_analysis_clip(
        payload,
        video_path=video_path,
        cases=registered_cases,
        imported_cases_path=_imported_cases_path(state),
        research_metadata_path=_research_path(state),
    )
    case_payload = case.to_dict()
    case_payload["video_path"] = case.video_path.name
    return (
        {
            "assigned": True,
            "case": case_payload,
            "case_details": details,
            "annotation_status": "needs_annotation",
        },
        case,
    )


def _imported_cases_path(state: VideoCutterState) -> Path:
    return state.context_clips_path.with_name("imported_video_cases_human.json")


def _research_path(state: VideoCutterState) -> Path:
    return state.context_clips_path.with_name(RESEARCH_METADATA_FILENAME)


def cut_video_segment(
    *,
    video_path: str | Path,
    output_dir: str | Path,
    start_seconds: float,
    end_seconds: float,
    output_name: str = "",
    mode: str = "accurate",
) -> dict:
    """Cut a segment from a video and return a browser payload for the new clip."""

    source_path = _validate_video_file(_resolve_path(video_path))
    metadata = read_video_metadata(source_path)
    start, end = _validate_cut_bounds(start_seconds, end_seconds, metadata.duration_seconds)
    target_dir = _resolve_path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = _unique_output_path(
        target_dir / _safe_output_filename(output_name, source_path, start, end)
    )
    requested_mode = mode.lower().strip()
    if requested_mode not in {"accurate", "copy", "opencv"}:
        raise ValueError("Cut mode must be accurate, copy, or opencv.")

    method = "opencv"
    if requested_mode == "opencv":
        _trim_with_opencv(source_path, output_path, start, end)
    else:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is not None:
            method = "ffmpeg_accurate" if requested_mode == "accurate" else "ffmpeg_copy"
            try:
                _trim_with_ffmpeg(
                    ffmpeg_path=ffmpeg_path,
                    source_path=source_path,
                    output_path=output_path,
                    start_seconds=start,
                    end_seconds=end,
                    accurate=requested_mode == "accurate",
                )
            except subprocess.CalledProcessError:
                if output_path.exists():
                    output_path.unlink()
                method = "opencv"
                _trim_with_opencv(source_path, output_path, start, end)
        else:
            _trim_with_opencv(source_path, output_path, start, end)

    output_metadata = read_video_metadata(output_path)
    output_id = _encode_path(output_path)
    return {
        "saved": True,
        "method": method,
        "start_seconds": start,
        "end_seconds": end,
        "path": str(output_path),
        "file_name": output_path.name,
        "video_id": output_id,
        "download_url": f"/api/download?id={output_id}",
        "duration_seconds": output_metadata.duration_seconds,
        "frame_count": output_metadata.frame_count,
        "fps": output_metadata.fps,
        "width": output_metadata.width,
        "height": output_metadata.height,
    }


def smoke_test(
    *,
    output_dir: str | Path = "data/videos/analysis_clips",
    video_roots: tuple[str | Path, ...] = DEFAULT_VIDEO_ROOTS,
) -> dict:
    """Run a non-writing UI smoke test."""

    state = VideoCutterState(
        video_roots=tuple(_resolve_path(root) for root in video_roots),
        output_dir=_resolve_path(output_dir),
    )
    html = render_video_cutter_page()
    return {
        "video_root_count": len(state.video_roots),
        "video_count": len(_discover_videos(state.video_roots)),
        "html_has_video_player": '<video id="player"' in html,
        "html_has_mark_in": 'id="setInButton"' in html,
        "html_has_mark_out": 'id="setOutButton"' in html,
        "html_has_five_frame_controls": (
            'id="backFiveFrameButton"' in html and 'id="forwardFiveFrameButton"' in html
        ),
        "html_has_reload_player": 'id="reloadPlayerButton"' in html,
        "html_has_player_error_recovery": "handlePlayerError" in html,
        "html_has_cut": 'id="cutButton"' in html,
        "output_dir": str(state.output_dir),
        "writes_files": False,
    }


def _videos_response(state: VideoCutterState) -> dict:
    videos = list(_discover_videos(state.video_roots))
    videos.extend(sorted(state.manual_videos, key=lambda path: str(path).lower()))
    unique_videos = tuple(dict.fromkeys(videos))
    registered_by_path = {}
    try:
        for case in _registered_cases(state):
            registered_by_path.setdefault(case.video_path.resolve(), case)
    except (OSError, ValueError):
        # Video discovery remains available while an optional case registry is
        # being created or repaired.
        registered_by_path = {}

    inventory = []
    for path in unique_videos:
        payload = _video_inventory_payload(path)
        case = registered_by_path.get(path.resolve())
        if case is not None:
            payload.update(
                {
                    "registered_view": True,
                    "case_id": case.case_id,
                    "player_name": case.player_name,
                    "view_label": case.view_label,
                }
            )
        inventory.append(payload)
    return {
        # Keep discovery cheap. Opening every video with OpenCV here made the
        # dropdown wait for the slowest or damaged file in either video root.
        # Metadata is fetched only after the researcher selects a video.
        "videos": inventory,
        "skipped": [],
        "roots": [str(root) for root in state.video_roots],
        "output_dir": str(state.output_dir),
        "ffmpeg_available": shutil.which("ffmpeg") is not None,
    }


def _discover_videos(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    videos: list[Path] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for path in root.rglob("*"):
            if _is_video_file(path) and path.is_file():
                videos.append(path.resolve())
    return tuple(sorted(dict.fromkeys(videos), key=lambda path: str(path).lower()))


def _video_payload(path: Path) -> dict:
    metadata = read_video_metadata(path)
    return {
        **_video_inventory_payload(path),
        "metadata": _metadata_payload(metadata),
    }


def _video_inventory_payload(path: Path) -> dict:
    """Return list-view data without opening the video container."""

    return {
        "id": _encode_path(path),
        "name": path.name,
        "path": str(path),
    }


def _metadata_payload(metadata: VideoMetadata) -> dict:
    return {
        "fps": metadata.fps,
        "width": metadata.width,
        "height": metadata.height,
        "frame_count": metadata.frame_count,
        "duration_seconds": metadata.duration_seconds,
    }


def _video_path_from_query(query_string: str, state: VideoCutterState) -> Path:
    query = parse_qs(query_string)
    return _decode_video_id(_one(query, "id"), state)


def _output_path_from_query(query_string: str, state: VideoCutterState) -> Path:
    query = parse_qs(query_string)
    output_path = _decode_path(_one(query, "id"))
    if not _path_inside(output_path, state.output_dir):
        raise ValueError("Requested output is outside the configured cut directory.")
    return _validate_video_file(output_path)


def _decode_video_id(video_id: str, state: VideoCutterState) -> Path:
    video_path = _decode_path(video_id)
    allowed_roots = (*state.video_roots, state.output_dir)
    if video_path not in state.manual_videos and not any(
        _path_inside(video_path, root) for root in allowed_roots
    ):
        raise ValueError("Video is outside the configured video roots.")
    return _validate_video_file(video_path)


def _encode_path(path: Path) -> str:
    encoded = base64.urlsafe_b64encode(str(path.resolve()).encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def _decode_path(encoded_path: str) -> Path:
    padding = "=" * (-len(encoded_path) % 4)
    decoded = base64.urlsafe_b64decode((encoded_path + padding).encode("ascii")).decode("utf-8")
    return _resolve_path(decoded)


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _validate_video_file(path: Path) -> Path:
    if not path.exists():
        raise ValueError(f"Video does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Video path is not a file: {path}")
    if not _is_video_file(path):
        raise ValueError(f"Unsupported video extension: {path.suffix}")
    return path


def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _path_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _one(query: dict[str, list[str]], key: str) -> str:
    values = query.get(key)
    if not values:
        raise KeyError(f"Missing query parameter: {key}")
    return values[0]


def _parse_range(range_header: str, file_size: int) -> tuple[int, int]:
    units, _, range_spec = range_header.partition("=")
    if units != "bytes" or "-" not in range_spec:
        raise ValueError("Unsupported Range header.")
    start_text, end_text = range_spec.split("-", 1)
    if start_text == "":
        suffix_size = min(int(end_text), file_size)
        return max(file_size - suffix_size, 0), max(file_size - 1, 0)
    start = int(start_text)
    end = int(end_text) if end_text else file_size - 1
    if start < 0 or end < start or start >= file_size:
        raise ValueError("Requested byte range is outside the file.")
    return start, min(end, file_size - 1)


def _validate_cut_bounds(
    start_seconds: float,
    end_seconds: float,
    duration_seconds: float,
) -> tuple[float, float]:
    if start_seconds < 0:
        raise ValueError("Start time must be 0 or greater.")
    if end_seconds <= start_seconds:
        raise ValueError("End time must be after start time.")
    if duration_seconds > 0 and start_seconds >= duration_seconds:
        raise ValueError("Start time is beyond the video duration.")
    end = min(end_seconds, duration_seconds) if duration_seconds > 0 else end_seconds
    if end <= start_seconds:
        raise ValueError("Cut duration is empty.")
    return start_seconds, end


def _safe_output_filename(
    output_name: str,
    source_path: Path,
    start_seconds: float,
    end_seconds: float,
) -> str:
    cleaned_name = output_name.strip()
    if not cleaned_name:
        cleaned_name = (
            f"{source_path.stem}_{_time_token(start_seconds)}_{_time_token(end_seconds)}"
        )
    cleaned_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", cleaned_name).strip(" .")
    if not cleaned_name:
        cleaned_name = "clip"
    suffix = Path(cleaned_name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        cleaned_name = f"{cleaned_name}.mp4"
    return cleaned_name


def _time_token(seconds: float) -> str:
    millis = round(seconds * 1000)
    total_seconds, ms = divmod(millis, 1000)
    minutes, sec = divmod(total_seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}h{minute:02d}m{sec:02d}s{ms:03d}"
    return f"{minute:02d}m{sec:02d}s{ms:03d}"


def _unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"Could not create a unique output path for {path.name}.")


def _trim_with_ffmpeg(
    *,
    ffmpeg_path: str,
    source_path: Path,
    output_path: Path,
    start_seconds: float,
    end_seconds: float,
    accurate: bool,
) -> None:
    duration = end_seconds - start_seconds
    if accurate:
        command = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_seconds:.6f}",
            "-i",
            str(source_path),
            "-t",
            f"{duration:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    else:
        command = [
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_seconds:.6f}",
            "-i",
            str(source_path),
            "-t",
            f"{duration:.6f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
    subprocess.run(command, check=True)


def _trim_with_opencv(
    source_path: Path,
    output_path: Path,
    start_seconds: float,
    end_seconds: float,
) -> None:
    import cv2

    capture = cv2.VideoCapture(str(source_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {source_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video metadata is incomplete; cannot write cut clip.")
        start_frame = max(0, round(start_seconds * fps))
        end_frame = max(start_frame + 1, round(end_seconds * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        try:
            if not writer.isOpened():
                raise ValueError(f"Could not create output video: {output_path}")
            frame_index = start_frame
            while frame_index < end_frame:
                ok, frame = capture.read()
                if not ok:
                    break
                writer.write(frame)
                frame_index += 1
        finally:
            writer.release()
    finally:
        capture.release()


def render_video_cutter_page(
    *,
    main_menu_url: str = DEFAULT_MAIN_MENU_URL,
    api_base: str = "/api",
) -> str:
    """Return the self-contained video cutter UI page."""

    return r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ACL Movement Analytics Lab - Video Cutter</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f3f5f7;
      --panel: #ffffff;
      --ink: #17212f;
      --muted: #647084;
      --line: #d4dbe4;
      --blue: #1d5fa7;
      --green: #177245;
      --red: #b42335;
      --amber: #a86700;
      --shadow: 0 12px 30px rgba(23, 33, 47, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      min-height: 58px;
      padding: 12px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }
    h1 {
      font-size: 18px;
      margin: 0;
      font-weight: 760;
    }
    .shell {
      display: grid;
      grid-template-columns: minmax(560px, 1fr) 360px;
      gap: 14px;
      padding: 14px;
      max-width: 1480px;
      margin: 0 auto;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .viewer {
      padding: 12px;
      min-width: 0;
    }
    .case-setup {
      grid-column: 1 / -1;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .case-setup-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
    }
    .case-setup h2 { margin: 0 0 4px; font-size: 18px; }
    .case-setup p { margin: 0; color: var(--muted); font-size: 13px; line-height: 1.45; }
    .case-setup-form { display: flex; flex-direction: column; gap: 12px; }
    .case-setup.is-active { padding: 10px 12px; }
    .case-step {
      color: var(--blue);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
    }
    .case-active {
      align-items: center;
      border: 1px solid #a8cfba;
      border-radius: 6px;
      background: #f0faf4;
      display: flex;
      gap: 14px;
      justify-content: space-between;
      padding: 10px 12px;
      font-size: 13px;
      line-height: 1.4;
    }
    .case-active-copy { min-width: 0; }
    .case-active-label {
      color: #23714c;
      display: block;
      font-size: 10px;
      font-weight: 800;
      letter-spacing: .07em;
      text-transform: uppercase;
    }
    .case-active-title {
      color: #135f3b;
      display: inline;
      font-size: 16px;
      font-weight: 780;
    }
    .case-active-meta { color: var(--ink); font-size: 12px; font-weight: 650; overflow-wrap: anywhere; }
    .case-active-note {
      display: block;
      margin-top: 2px;
      color: #135f3b;
      font-size: 11px;
    }
    .case-active-actions { display: flex; flex: 0 0 auto; flex-wrap: wrap; gap: 7px; }
    .side {
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .toolbar, .row, .button-row {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .toolbar { justify-content: space-between; margin-bottom: 10px; }
    .toolbar > label { min-width: 0; }
    #videoSelect { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    #videoSearchInput { min-width: 190px; }
    .source-verification {
      flex-direction: row;
      align-items: flex-start;
      gap: 9px;
      padding: 10px 12px;
      border: 1px solid #d7c47a;
      border-radius: 6px;
      background: #fffaf0;
      color: var(--ink);
      line-height: 1.4;
    }
    .source-verification input { min-height: auto; margin-top: 2px; }
    .grow { flex: 1 1 240px; min-width: 170px; }
    label {
      display: flex;
      flex-direction: column;
      gap: 5px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 650;
    }
    [hidden] { display: none !important; }
    .control-block {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .field-label {
      font-size: 12px;
      color: var(--muted);
      font-weight: 650;
    }
    input, select, button {
      min-height: 44px;
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 7px 9px;
    }
    select, input { width: 100%; }
    button {
      width: auto;
      cursor: pointer;
      font-weight: 720;
      white-space: nowrap;
    }
    a.button {
      min-height: 44px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 7px 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font-weight: 720;
      text-decoration: none;
      white-space: nowrap;
    }
    a.button.primary {
      background: var(--blue);
      border-color: var(--blue);
      color: #fff;
    }
    button.primary {
      background: var(--blue);
      color: #fff;
      border-color: var(--blue);
    }
    button.good {
      background: var(--green);
      color: #fff;
      border-color: var(--green);
    }
    button.warn {
      background: #fff8ea;
      color: #6f4300;
      border-color: #e8bf63;
    }
    button.danger {
      background: #fff1f2;
      color: var(--red);
      border-color: #efb1bb;
    }
    button:disabled {
      opacity: 0.48;
      cursor: not-allowed;
    }
    video {
      width: 100%;
      max-height: calc(100vh - 215px);
      min-height: 340px;
      background: #101826;
      border-radius: 6px;
      display: block;
    }
    .timeline {
      position: relative;
      height: 30px;
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: linear-gradient(90deg, #e8edf3, #f7f9fb);
      overflow: hidden;
    }
    .selection {
      position: absolute;
      inset-block: 0;
      left: var(--selection-left, 0%);
      width: var(--selection-width, 0%);
      background: rgba(23, 114, 69, 0.24);
      border-inline: 2px solid var(--green);
    }
    .playhead {
      position: absolute;
      top: 0;
      bottom: 0;
      left: var(--playhead-left, 0%);
      width: 2px;
      background: var(--red);
    }
    .meta-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .meta {
      min-height: 58px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px;
      background: #f8fafc;
      font-size: 12px;
      color: var(--muted);
    }
    .meta strong {
      display: block;
      margin-top: 3px;
      color: var(--ink);
      font-size: 15px;
      overflow-wrap: anywhere;
    }
    .field-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .segment {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 6px;
    }
    .segment button[aria-pressed="true"] {
      background: #e7f0fb;
      border-color: var(--blue);
      color: #124d88;
    }
    .status {
      min-height: 54px;
      padding: 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .context-boundary {
      margin: -4px 0 0;
      padding: 8px;
      border-left: 3px solid var(--blue);
      background: #f1f6fc;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .output {
      display: none;
      gap: 8px;
      flex-direction: column;
    }
    .output.is-visible { display: flex; }
    .output a {
      color: var(--blue);
      font-weight: 720;
      overflow-wrap: anywhere;
    }
    .assignment {
      margin-top: 4px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .assignment h2 { margin: 0; font-size: 16px; }
    .assignment p { margin: 0; color: var(--muted); font-size: 12px; line-height: 1.4; }
    .view-details {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfd;
    }
    .view-details > summary {
      align-items: center;
      cursor: pointer;
      display: flex;
      gap: 8px;
      justify-content: space-between;
      list-style: none;
      padding: 10px;
    }
    .view-details > summary::-webkit-details-marker { display: none; }
    .view-details > summary::after { color: var(--blue); content: "+"; font-size: 18px; font-weight: 800; }
    .view-details[open] > summary::after { content: "−"; }
    .view-details-summary strong { color: var(--ink); display: block; font-size: 13px; }
    .view-details-summary small { color: var(--muted); display: block; font-size: 10px; font-weight: 600; margin-top: 2px; }
    .view-details-body { border-top: 1px solid var(--line); display: grid; gap: 10px; padding: 10px; }
    .cut-action-row { display: grid; gap: 5px; }
    .cut-action-row button { min-height: 48px; width: 100%; }
    .cut-action-row small { color: var(--muted); font-size: 10px; line-height: 1.35; text-align: center; }
    .output-summary {
      background: #f7fafc;
      border: 1px solid var(--line);
      border-radius: 6px;
      display: grid;
      gap: 5px;
      padding: 10px;
    }
    .output-eyebrow { color: var(--green); font-size: 10px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }
    #outputName { font-size: 13px; line-height: 1.35; overflow-wrap: anywhere; }
    .output-secondary { align-items: center; display: flex; flex-wrap: wrap; gap: 12px; }
    .output-technical { color: var(--muted); font-size: 10px; }
    .output-technical summary { cursor: pointer; font-weight: 700; }
    .output-technical code { display: block; margin-top: 6px; overflow-wrap: anywhere; white-space: normal; }
    .assignment.completion-card { border: 1px solid #a8cfba; border-radius: 6px; margin-top: 0; padding: 11px; }
    .completion-feedback { color: var(--muted); font-size: 12px; line-height: 1.4; }
    .side > #status { min-height: 0; }
    .check-row { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
    .check-row label { flex-direction: row; align-items: center; gap: 6px; }
    .check-row input { width: auto; min-height: auto; }
    @media (max-width: 980px) {
      .shell {
        grid-template-columns: 1fr;
      }
      .case-active { align-items: flex-start; flex-direction: column; }
      video {
        min-height: 240px;
        max-height: 58vh;
      }
    }
    @media (max-width: 620px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }
      .shell { padding: 10px; }
      .meta-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .field-grid { grid-template-columns: 1fr; }
      .button-row button { flex: 1 1 120px; }
    }
    __APP_SHELL_CSS__
  </style>
</head>
<body>
  <a class="app-skip-link" href="#mainContent">Skip to video preparation</a>
  __APP_SITE_HEADER__
  <header class="app-tool-header">
    <h1>Video Cutter</h1>
    <div class="row">
      <a class="button" id="cancelCutterLink" href="__MAIN_MENU_URL__" hidden>Cancel and return</a>
      <a class="button" href="__MAIN_MENU_URL__">Main menu</a>
      <button id="refreshButton">Refresh</button>
    </div>
  </header>

  <main id="mainContent" class="shell app-page-main" tabindex="-1">
    <section id="caseSetupPanel" class="panel case-setup">
      <div id="caseSetupForm" class="case-setup-form">
      <div class="case-setup-header">
        <div>
          <span class="case-step">Step 1 · Player and injury case</span>
          <h2>Create or choose the case before opening video</h2>
          <p>A case represents one player’s injury event. Every clip you cut in this session will be attached as another video view of that case.</p>
        </div>
        <a class="button" href="__MAIN_MENU_URL__#review">Edit case library</a>
      </div>
      <label>
        What are you working on?
        <select id="assignmentModeSelect">
          <option value="new">Create a new player injury case</option>
          <option value="existing">Add video views to an existing case</option>
        </select>
      </label>
      <label id="existingAnalysisCaseField" hidden>
        Existing injury case
        <select id="analysisCaseSelect"></select>
      </label>
      <div id="newCaseFields" class="field-grid">
        <label>Player name<input id="casePlayerInput" autocomplete="off" /></label>
        <label>Match / injury date<input id="caseDateInput" type="date" /></label>
        <label>Team<input id="caseTeamInput" autocomplete="off" /></label>
        <label>Opponent<input id="caseOpponentInput" autocomplete="off" /></label>
        <label>League / competition<input id="caseCompetitionInput" autocomplete="off" /></label>
        <label>Position
          <select id="casePositionSelect">
            <option value="unknown">Unknown</option>
            <option value="goalkeeper">Goalkeeper</option>
            <option value="defender">Defender</option>
            <option value="midfielder">Midfielder</option>
            <option value="forward">Forward</option>
          </select>
        </label>
        <label>Documented injured knee
          <select id="caseInjuredSideSelect">
            <option value="unknown">Unknown</option>
            <option value="left">Left</option>
            <option value="right">Right</option>
          </select>
        </label>
        <label>Match minute<input id="caseMatchMinuteInput" placeholder="67 or 45+2" /></label>
      </div>
      <div class="button-row">
        <button id="beginCaseButton" type="button" class="primary">Continue to video</button>
      </div>
      <span id="caseSetupFeedback" class="status">Enter the player and injury-event information first.</span>
      </div>
      <div id="activeCaseSummary" class="case-active" hidden>
        <div class="case-active-copy">
          <span class="case-active-label">Active case</span>
          <strong id="activeCaseTitle" class="case-active-title">Selected player</strong>
          <span id="activeCaseMeta" class="case-active-meta"></span>
          <span id="activeCaseNote" class="case-active-note">All new cuts will be attached to this case.</span>
        </div>
        <div class="case-active-actions">
          <button id="changeCaseButton" type="button">Change or edit case</button>
          <a class="button" href="__MAIN_MENU_URL__#review">Case library</a>
        </div>
      </div>
    </section>
    <section id="videoWorkspace" class="panel viewer" hidden>
      <div class="toolbar">
        <label class="grow">
          Find source video
          <input id="videoSearchInput" type="search" placeholder="Player, view, or filename" disabled />
        </label>
        <label class="grow">
          Source video
          <select id="videoSelect" disabled></select>
        </label>
        <label class="grow">
          Local path
          <input id="pathInput" placeholder="/path/to/video.mp4" disabled />
        </label>
        <button id="openPathButton" class="primary" disabled>Open</button>
      </div>
      <video id="player" controls preload="metadata" playsinline></video>
      <div class="timeline" id="timeline">
        <div class="selection"></div>
        <div class="playhead"></div>
      </div>
      <div class="meta-grid">
        <div class="meta">Current<strong id="currentTimeLabel">0:00.000</strong></div>
        <div class="meta">In<strong id="inTimeLabel">0:00.000</strong></div>
        <div class="meta">Out<strong id="outTimeLabel">0:00.000</strong></div>
        <div class="meta">Selection<strong id="selectionTimeLabel">0:00.000</strong></div>
      </div>
    </section>

    <aside id="cutControlsPanel" class="panel side" hidden>
      <div class="button-row">
        <button id="backFiveButton" aria-label="Back 5 seconds">-5s</button>
        <button id="backFiveFrameButton" aria-label="Back 5 frames">-5f</button>
        <button id="backFrameButton" aria-label="Back 1 frame">-1f</button>
        <button id="forwardFrameButton" aria-label="Forward 1 frame">+1f</button>
        <button id="forwardFiveFrameButton" aria-label="Forward 5 frames">+5f</button>
        <button id="forwardFiveButton" aria-label="Forward 5 seconds">+5s</button>
      </div>
      <div class="button-row">
        <button id="setInButton" class="primary">Set In</button>
        <button id="setOutButton" class="primary">Set Out</button>
        <button id="reviewButton" class="warn">Review Cut</button>
        <button id="reloadPlayerButton">Reload</button>
      </div>
      <div class="field-grid">
        <label>
          In seconds
          <input id="startInput" type="number" min="0" step="0.001" value="0" />
        </label>
        <label>
          Out seconds
          <input id="endInput" type="number" min="0" step="0.001" value="0" />
        </label>
      </div>
      <label>
        Clip purpose
        <select id="clipRoleSelect">
          <option value="ANALYSIS_CLIP">Analysis clip</option>
          <option value="REAL_TIME_CONTEXT">Real-time context clip</option>
        </select>
      </label>
      <label id="contextCaseField" hidden>
        Associated injury case
        <select id="contextCaseSelect"></select>
      </label>
      <p id="contextBoundary" class="context-boundary" hidden>
        Context only. This clip will not be used for measurements or movement summaries.
      </p>
      <details id="viewDetailsPanel" class="view-details">
        <summary>
          <span class="view-details-summary">
            <strong>Optional view and export details</strong>
            <small>View name, camera perspective, replay settings, and technical export options</small>
          </span>
        </summary>
        <div class="view-details-body">
        <p>These details belong to this clip only. They do not create another injury case.</p>
        <div class="field-grid">
          <label>View name<input id="viewLabelInput" placeholder="Live wide, close replay…" /></label>
          <label>Camera perspective
            <select id="perspectiveSelect">
              <option value="unknown">Unknown</option>
              <option value="frontal-like">Frontal-like</option>
              <option value="sagittal-like">Sagittal-like</option>
              <option value="oblique">Oblique</option>
              <option value="high-wide">High / wide</option>
            </select>
          </label>
        </div>
        <div class="check-row">
          <label><input id="slowMotionInput" type="checkbox" /> Slow-motion replay</label>
          <label><input id="croppedInput" type="checkbox" /> Cropped or zoomed</label>
        </div>
        <label>
          Output name
          <input id="nameInput" placeholder="optional_clip_name.mp4" />
        </label>
        <div class="control-block">
          <span class="field-label">Export mode</span>
          <div class="segment">
            <button id="accurateModeButton" type="button" aria-pressed="true">Accurate</button>
            <button id="copyModeButton" type="button" aria-pressed="false">Fast</button>
          </div>
        </div>
        <div class="meta-grid">
          <div class="meta">FPS<strong id="fpsLabel">-</strong></div>
          <div class="meta">Frames<strong id="frameCountLabel">-</strong></div>
          <div class="meta">Size<strong id="sizeLabel">-</strong></div>
          <div class="meta">Duration<strong id="durationLabel">-</strong></div>
        </div>
        </div>
      </details>
      <div class="cut-action-row">
        <button id="cutButton" class="good" disabled>Cut and add view</button>
        <small id="cutActionNote">The new clip will be attached to the active injury case.</small>
      </div>
      <p id="sourceAssignmentWarning" class="context-boundary" hidden></p>
      <label id="sourceVerificationField" class="source-verification" hidden>
        <input id="sourceVerifiedInput" type="checkbox" disabled />
        <span>I verified that this source video shows the active athlete and the correct injury event.</span>
      </label>
      <div id="status" class="status">Warming up the video player…</div>
      <div id="output" class="output">
        <div class="output-summary">
          <span class="output-eyebrow">Cut prepared</span>
          <strong id="outputName"></strong>
          <div class="output-secondary">
            <a id="downloadLink" href="#">Download cut</a>
            <details class="output-technical">
              <summary>Technical output details</summary>
              <code id="outputPath"></code>
            </details>
          </div>
        </div>
        <section id="assignmentPanel" class="assignment completion-card" hidden>
          <h2 id="assignmentTitle">Adding video view to case…</h2>
          <span id="assignmentFeedback" class="completion-feedback">Saving this view to the active case…</span>
          <div class="button-row">
            <button id="assignClipButton" type="button" class="good" hidden>Retry assignment</button>
            <button id="cutAnotherViewButton" type="button">Cut another view</button>
            <a id="annotateAssignedLink" class="button primary" href="#" hidden>Annotate this view</a>
          </div>
        </section>
      </div>
    </aside>
  </main>

<script>
const $ = id => document.getElementById(id);
const app = {
  videos: [],
  selected: null,
  duration: 0,
  fps: 30,
  inTime: 0,
  outTime: 0,
  mode: "accurate",
  reviewing: false,
  switchingVideo: false,
  recoverAttempts: 0,
  lastKnownTime: 0,
  contextCases: [],
  clipRole: "ANALYSIS_CLIP",
  latestCut: null,
  caseReady: false,
  videoReady: false,
  activeAssignmentMode: null,
  activeCaseId: "",
  activeCaseLabel: "",
  activeCasePlayerName: "",
  requestedVideoApplied: false,
};

const player = $("player");
const apiBase = __API_BASE_JSON__;
const annotateBase = __ANNOTATE_BASE_JSON__;
const mainMenuBase = __MAIN_MENU_JSON__;
const requestedParams = new URLSearchParams(window.location.search);
const requestedCaseId = requestedParams.get("case");
const requestedVideoRef = requestedParams.get("video");
const requestedReturnRef = requestedParams.get("return");

function safeReturnUrl(rawValue) {
  if (!rawValue) return "";
  try {
    const target = new URL(rawValue, window.location.origin);
    const mainMenuOrigin = new URL(mainMenuBase, window.location.href).origin;
    if (![window.location.origin, mainMenuOrigin].includes(target.origin)) return "";
    return target.href;
  } catch (_error) {
    return "";
  }
}

const requestedReturnUrl = safeReturnUrl(requestedReturnRef);

function selectedCaseReturnUrl() {
  const caseId = app.activeCaseId || requestedCaseId;
  if (!caseId) return "";
  const destination = new URL(mainMenuBase, window.location.href);
  destination.hash = "review";
  destination.search = "";
  destination.searchParams.set("case", caseId);
  return destination.href;
}

function updateCutterReturnLink() {
  const link = $("cancelCutterLink");
  const destination = requestedReturnUrl || selectedCaseReturnUrl();
  link.hidden = !destination;
  if (!destination) return;
  link.href = destination;
  const pathname = new URL(destination, window.location.href).pathname;
  link.textContent = pathname === "/results"
    ? "Cancel and return to analysis"
    : pathname === "/annotate"
      ? "Cancel and return to annotation"
      : "Back to selected case";
}

function fmt(seconds) {
  const value = Number.isFinite(seconds) ? Math.max(seconds, 0) : 0;
  const millis = Math.floor((value % 1) * 1000);
  const total = Math.floor(value);
  const sec = total % 60;
  const min = Math.floor(total / 60) % 60;
  const hour = Math.floor(total / 3600);
  const prefix = hour ? `${hour}:` : "";
  const minuteText = hour ? String(min).padStart(2, "0") : String(min);
  return `${prefix}${minuteText}:${String(sec).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

function friendlyCutLabel(viewLabel = "Video view") {
  const label = String(viewLabel || "").trim() || "Video view";
  return `${label} · ${fmt(app.inTime)}–${fmt(app.outTime)}`;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

async function loadVideos() {
  setStatus("Warming up the video player…");
  const response = await fetch(`${apiBase}/videos`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not load videos.");
  app.videos = data.videos || [];
  renderVideoOptions();
  if (app.videos.length && app.caseReady) {
    if (app.selected && app.videos.some(video => video.id === app.selected.id)) {
      selectVideo(app.selected.id);
    } else {
      selectInitialVideo();
    }
  } else {
    if (!app.videos.length) {
      $("videoSelect").innerHTML = '<option value="">No videos found</option>';
    }
    $("cutButton").disabled = true;
    setStatus(app.videos.length
      ? "Create or choose the player injury case above before opening video."
      : `No videos found.\nRoots:\n${(data.roots || []).join("\n")}`
    );
  }
}

function selectInitialVideo() {
  if (!app.videos.length) return;
  const requestedVideo = requestedVideoRef
    ? app.videos.find(video => (
        video.id === requestedVideoRef
        || video.name === requestedVideoRef
        || video.path === requestedVideoRef
      ))
    : null;
  if (requestedVideo && !app.requestedVideoApplied) {
    app.requestedVideoApplied = true;
    selectVideo(requestedVideo.id);
    return;
  }
  resetVideoSelection(`Choose and verify the source video for ${app.activeCasePlayerName || "the active player"}.`);
}

async function loadContextCases() {
  const response = await fetch(`${apiBase}/context-cases`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not load injury cases.");
  app.contextCases = data.cases || [];
  renderCaseOptions();
  const requestedCase = app.contextCases.find(item => item.case_id === requestedCaseId);
  if (requestedCase) {
    $("assignmentModeSelect").value = "existing";
    setAssignmentMode("existing");
    $("analysisCaseSelect").value = requestedCase.case_id;
    beginCaseWorkflow();
  } else {
    $("assignmentModeSelect").value = "new";
    setAssignmentMode("new");
  }
}

function caseOptionLabel(item) {
  const matchup = [item.team, item.opponent ? `vs ${item.opponent}` : ""].filter(Boolean).join(" ");
  const details = [item.injury_date, matchup, `${item.view_count || 1} view${item.view_count === 1 ? "" : "s"}`]
    .filter(Boolean);
  return `${item.player_name}${details.length ? ` — ${details.join(" · ")}` : ""}`;
}

function caseCategoryLabel(value) {
  const text = String(value || "").trim();
  if (!text || text.toLowerCase() === "unknown") return "Not recorded";
  return text.replace(/[-_]+/g, " ").replace(/\b\w/g, char => char.toUpperCase());
}

function renderActiveCaseSummary(item, note = "All cuts will be saved as views of this case.") {
  if (!item) return;
  const value = raw => {
    const text = String(raw || "").trim();
    return text || "Not recorded";
  };
  const matchup = [item.team, item.opponent ? `vs ${item.opponent}` : ""]
    .map(part => String(part || "").trim())
    .filter(Boolean)
    .join(" ");
  const details = [
    item.injury_date,
    matchup,
    item.competition,
    item.position_group && item.position_group !== "unknown" ? caseCategoryLabel(item.position_group) : "",
    item.injured_side && item.injured_side !== "unknown" ? `${caseCategoryLabel(item.injured_side)} knee` : "",
    item.match_minute ? `minute ${item.match_minute}` : "",
  ].map(part => String(part || "").trim()).filter(Boolean);
  $("activeCaseTitle").textContent = value(item.player_name);
  $("activeCaseMeta").textContent = details.length ? ` · ${details.join(" · ")}` : "";
  $("activeCaseNote").textContent = note;
  $("activeCaseSummary").hidden = false;
}

function renderCaseOptions() {
  const options = app.contextCases.map(item => (
    `<option value="${escapeHtml(item.case_id)}">${escapeHtml(caseOptionLabel(item))}</option>`
  )).join("");
  $("contextCaseSelect").innerHTML = options;
  $("analysisCaseSelect").innerHTML = options || '<option value="">No injury cases yet</option>';
  if (!app.contextCases.length) {
    $("assignmentModeSelect").value = "new";
    setAssignmentMode("new");
  }
}

function setClipRole(role) {
  if (role === "REAL_TIME_CONTEXT" && app.activeAssignmentMode === "new") {
    $("clipRoleSelect").value = "ANALYSIS_CLIP";
    app.clipRole = "ANALYSIS_CLIP";
    setStatus("Save the first analysis view before adding a context-only clip to a new case.");
    return;
  }
  app.clipRole = role;
  const contextual = role === "REAL_TIME_CONTEXT";
  $("contextCaseField").hidden = !contextual;
  $("contextBoundary").hidden = !contextual;
  $("viewDetailsPanel").hidden = contextual;
  $("cutButton").textContent = contextual ? "Save Context Clip" : "Cut and add view";
  $("cutActionNote").textContent = contextual
    ? "This clip will be stored as context only and will not be measured."
    : "The new clip will be attached to the active injury case.";
}

function setAssignmentMode(mode) {
  const creating = mode === "new";
  $("newCaseFields").hidden = !creating;
  $("existingAnalysisCaseField").hidden = creating;
  $("caseSetupFeedback").textContent = creating
    ? "Enter the player and injury-event information first. The case will be saved with its first cut view."
    : "Choose the existing player injury case that should receive every cut in this session.";
}

function selectedCaseOption() {
  return app.contextCases.find(item => item.case_id === $("analysisCaseSelect").value) || null;
}

function setCaseFieldsDisabled(disabled) {
  [
    "assignmentModeSelect", "analysisCaseSelect", "casePlayerInput", "caseDateInput",
    "caseTeamInput", "caseOpponentInput", "caseCompetitionInput", "casePositionSelect",
    "caseInjuredSideSelect", "caseMatchMinuteInput",
  ].forEach(id => { $(id).disabled = disabled; });
}

function setWorkspaceEnabled(enabled) {
  $("videoWorkspace").hidden = !enabled;
  $("cutControlsPanel").hidden = !enabled;
  $("videoSelect").disabled = !enabled;
  $("videoSearchInput").disabled = !enabled;
  $("pathInput").disabled = !enabled;
  $("openPathButton").disabled = !enabled;
  if (!enabled) $("sourceVerifiedInput").disabled = true;
  if (!enabled) {
    app.videoReady = false;
    player.pause();
  }
  setVideoControlsEnabled(enabled && app.videoReady);
  $("cutButton").disabled = !enabled || !app.selected || !app.videoReady || !$("sourceVerifiedInput").checked;
}

function setVideoControlsEnabled(enabled) {
  [
    "backFiveButton", "backFiveFrameButton", "backFrameButton",
    "forwardFrameButton", "forwardFiveFrameButton", "forwardFiveButton",
    "setInButton", "setOutButton", "reviewButton",
    "startInput", "endInput", "clipRoleSelect", "contextCaseSelect",
    "viewLabelInput", "perspectiveSelect", "slowMotionInput", "croppedInput",
    "nameInput", "accurateModeButton", "copyModeButton",
  ].forEach(id => { $(id).disabled = !enabled; });
  $("reloadPlayerButton").disabled = !app.caseReady || !app.selected;
}

function beginCaseWorkflow() {
  const mode = $("assignmentModeSelect").value;
  let activeCase;
  if (mode === "existing") {
    const selected = selectedCaseOption();
    if (!selected) {
      $("caseSetupFeedback").textContent = "Choose an existing injury case.";
      return;
    }
    app.activeAssignmentMode = "existing";
    app.activeCaseId = selected.case_id;
    app.activeCaseLabel = caseOptionLabel(selected);
    app.activeCasePlayerName = selected.player_name;
    activeCase = selected;
    $("contextCaseSelect").value = selected.case_id;
  } else {
    const playerName = $("casePlayerInput").value.trim();
    const injuryDate = $("caseDateInput").value;
    if (!playerName) {
      $("caseSetupFeedback").textContent = "Enter the player name before opening video.";
      return;
    }
    if (!injuryDate) {
      $("caseSetupFeedback").textContent = "Choose the match / injury date before opening video.";
      return;
    }
    app.activeAssignmentMode = "new";
    app.activeCaseId = "";
    app.activeCaseLabel = `${playerName} — ${injuryDate}`;
    app.activeCasePlayerName = playerName;
    activeCase = {
      player_name: playerName,
      injury_date: injuryDate,
      team: $("caseTeamInput").value,
      opponent: $("caseOpponentInput").value,
      competition: $("caseCompetitionInput").value,
      position_group: $("casePositionSelect").value,
      match_minute: $("caseMatchMinuteInput").value,
      injured_side: $("caseInjuredSideSelect").value,
    };
  }
  app.caseReady = true;
  setClipRole($("clipRoleSelect").value);
  setCaseFieldsDisabled(true);
  $("caseSetupForm").hidden = true;
  $("caseSetupPanel").classList.add("is-active");
  $("beginCaseButton").hidden = true;
  $("changeCaseButton").hidden = false;
  $("caseSetupFeedback").hidden = true;
  renderActiveCaseSummary(activeCase);
  updateCutterReturnLink();
  setWorkspaceEnabled(true);
  renderVideoOptions();
  if (app.videos.length) {
    selectInitialVideo();
  }
  else {
    setStatus("Finding the available source videos…");
    loadVideos().catch(error => setStatus(error.message));
  }
}

function changeCaseWorkflow() {
  if (app.latestCut && app.activeCaseId) {
    const confirmed = window.confirm("Start working on a different injury case? Existing assigned views will remain in the case library.");
    if (!confirmed) return;
  }
  app.caseReady = false;
  app.activeAssignmentMode = null;
  app.activeCaseId = "";
  app.activeCaseLabel = "";
  app.activeCasePlayerName = "";
  app.latestCut = null;
  $("clipRoleSelect").value = "ANALYSIS_CLIP";
  setClipRole("ANALYSIS_CLIP");
  setCaseFieldsDisabled(false);
  $("beginCaseButton").hidden = false;
  $("changeCaseButton").hidden = true;
  $("caseSetupFeedback").hidden = false;
  $("activeCaseSummary").hidden = true;
  $("caseSetupForm").hidden = false;
  $("caseSetupPanel").classList.remove("is-active");
  $("output").classList.remove("is-visible");
  setWorkspaceEnabled(false);
  setAssignmentMode($("assignmentModeSelect").value);
  setStatus("Create or choose the player injury case above before opening video.");
  updateCutterReturnLink();
}

function renderVideoOptions() {
  const query = $("videoSearchInput").value.trim().toLowerCase();
  const visible = app.videos.filter(video => {
    const haystack = [video.name, video.path, video.player_name, video.view_label]
      .filter(Boolean).join(" ").toLowerCase();
    return !query || haystack.includes(query);
  });
  const currentCase = visible.filter(video => (
    video.registered_view && app.activeCaseId && video.case_id === app.activeCaseId
  ));
  const otherRegistered = visible.filter(video => (
    video.registered_view && (!app.activeCaseId || video.case_id !== app.activeCaseId)
  ));
  const local = visible.filter(video => !video.registered_view);
  const option = video => `<option value="${escapeHtml(video.id)}">${escapeHtml(friendlyVideoLabel(video))}</option>`;
  const group = (label, videos) => videos.length
    ? `<optgroup label="${escapeHtml(label)}">${videos.map(option).join("")}</optgroup>`
    : "";
  $("videoSelect").innerHTML = '<option value="">Choose a source video…</option>'
    + group("Views already linked to this case", currentCase)
    + group("Views linked to other cases", otherRegistered)
    + group("Other local source videos", local);
  if (app.selected && visible.some(video => video.id === app.selected.id)) {
    $("videoSelect").value = app.selected.id;
  }
}

function friendlyVideoLabel(video) {
  if (video.registered_view) {
    return `${video.player_name || "Registered case"} · ${video.view_label || "Video view"}`;
  }
  return String(video.name || "Local video")
    .replace(/\.[^.]+$/, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function resetVideoSelection(message = "Choose and verify a source video before setting cut points.") {
  app.selected = null;
  app.videoReady = false;
  app.duration = 0;
  app.inTime = 0;
  app.outTime = 0;
  player.pause();
  player.removeAttribute("src");
  player.load();
  $("videoSelect").value = "";
  $("sourceAssignmentWarning").hidden = true;
  $("sourceAssignmentWarning").textContent = "";
  $("sourceVerificationField").hidden = true;
  $("sourceVerifiedInput").checked = false;
  $("sourceVerifiedInput").disabled = true;
  setVideoControlsEnabled(false);
  $("cutButton").disabled = true;
  updateMeta({});
  updateInputs();
  updateTimeline();
  setStatus(message);
}

async function loadSelectedVideoMetadata(video) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(
      `${apiBase}/metadata?id=${encodeURIComponent(video.id)}`,
      {signal: controller.signal}
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not read video metadata.");
    video.metadata = data.metadata || {};
    if (!app.selected || app.selected.id !== video.id) return;
    const meta = video.metadata;
    app.duration = Number(meta.duration_seconds || app.duration || 0);
    app.fps = Number(meta.fps || app.fps || 30) || 30;
    if (!app.outTime) app.outTime = app.duration;
    updateMeta(video);
    updateInputs();
    updateTimeline();
  } catch (error) {
    if (!app.selected || app.selected.id !== video.id) return;
    setStatus(error.name === "AbortError"
      ? "The video opened, but metadata inspection timed out. You can still review it or choose another file."
      : `The selected video opened with limited metadata. ${error.message}`
    );
  } finally {
    window.clearTimeout(timeout);
  }
}

function selectVideo(id) {
  if (!id) {
    resetVideoSelection();
    return;
  }
  const video = app.videos.find(item => item.id === id);
  if (!video) return;
  app.selected = video;
  app.videoReady = false;
  setVideoControlsEnabled(false);
  app.switchingVideo = true;
  app.recoverAttempts = 0;
  app.lastKnownTime = 0;
  $("videoSelect").value = video.id;
  const meta = video.metadata || {};
  app.duration = Number(meta.duration_seconds || 0);
  app.fps = Number(meta.fps || 30) || 30;
  app.inTime = 0;
  app.outTime = app.duration;
  player.src = videoUrl(video.id);
  player.load();
  updateMeta(video);
  updateInputs();
  updateTimeline();
  $("cutButton").disabled = true;
  $("sourceAssignmentWarning").textContent = `Selected source: ${friendlyVideoLabel(video)}. Verify that the footage shows ${app.activeCasePlayerName || "the active player"} and the correct injury event.`;
  $("sourceAssignmentWarning").hidden = false;
  $("sourceVerificationField").hidden = false;
  $("sourceVerifiedInput").checked = false;
  $("sourceVerifiedInput").disabled = true;
  setStatus("Opening the selected video…");
  loadSelectedVideoMetadata(video);
}

function updateMeta(video) {
  const meta = video.metadata || {};
  $("fpsLabel").textContent = meta.fps ? Number(meta.fps).toFixed(3) : "-";
  $("frameCountLabel").textContent = meta.frame_count || "-";
  $("sizeLabel").textContent = meta.width && meta.height ? `${meta.width} x ${meta.height}` : "-";
  $("durationLabel").textContent = fmt(Number(meta.duration_seconds || 0));
}

function updateInputs() {
  $("startInput").value = app.inTime.toFixed(3);
  $("endInput").value = app.outTime.toFixed(3);
  $("inTimeLabel").textContent = fmt(app.inTime);
  $("outTimeLabel").textContent = fmt(app.outTime);
  $("selectionTimeLabel").textContent = fmt(Math.max(app.outTime - app.inTime, 0));
}

function updateTimeline() {
  const duration = app.duration || player.duration || 0;
  const current = duration ? clamp(player.currentTime / duration * 100, 0, 100) : 0;
  const left = duration ? clamp(app.inTime / duration * 100, 0, 100) : 0;
  const right = duration ? clamp(app.outTime / duration * 100, 0, 100) : 0;
  $("timeline").style.setProperty("--playhead-left", `${current}%`);
  $("timeline").style.setProperty("--selection-left", `${left}%`);
  $("timeline").style.setProperty("--selection-width", `${Math.max(right - left, 0)}%`);
  $("currentTimeLabel").textContent = fmt(player.currentTime || 0);
}

function setStatus(text) {
  $("status").textContent = text;
}

function videoUrl(id, reloadToken = "") {
  const suffix = reloadToken ? `&reload=${encodeURIComponent(reloadToken)}` : "";
  return `${apiBase}/video?id=${encodeURIComponent(id)}${suffix}`;
}

function setMode(mode) {
  app.mode = mode;
  $("accurateModeButton").setAttribute("aria-pressed", String(mode === "accurate"));
  $("copyModeButton").setAttribute("aria-pressed", String(mode === "copy"));
}

function setIn(time) {
  app.inTime = clamp(Number(time), 0, app.duration || Number.MAX_SAFE_INTEGER);
  if (app.outTime <= app.inTime) {
    app.outTime = Math.min(app.inTime + frameStep(), app.duration || app.inTime + frameStep());
  }
  updateInputs();
  updateTimeline();
}

function setOut(time) {
  app.outTime = clamp(Number(time), 0, app.duration || Number.MAX_SAFE_INTEGER);
  if (app.outTime <= app.inTime) {
    app.inTime = Math.max(0, app.outTime - frameStep());
  }
  updateInputs();
  updateTimeline();
}

function frameStep() {
  return 1 / (app.fps || 30);
}

function seekBy(delta) {
  player.currentTime = clamp((player.currentTime || 0) + delta, 0, app.duration || player.duration || 0);
  app.lastKnownTime = player.currentTime || app.lastKnownTime;
}

function reloadCurrentVideo(message = "Reloading video...") {
  if (!app.selected) return;
  const duration = app.duration || player.duration || 0;
  const currentTime = Number.isFinite(player.currentTime)
    ? player.currentTime
    : (app.lastKnownTime || 0);
  const resumeAt = clamp(currentTime, 0, duration || Number.MAX_SAFE_INTEGER);
  app.switchingVideo = true;
  app.videoReady = false;
  app.reviewing = false;
  setVideoControlsEnabled(false);
  $("cutButton").disabled = true;
  setStatus(message);
  player.pause();
  player.addEventListener("loadedmetadata", () => {
    if (resumeAt > 0) {
      const loadedDuration = player.duration || app.duration || resumeAt;
      const lastPlayableTime = Math.max(0, loadedDuration - frameStep());
      player.currentTime = Math.min(resumeAt, lastPlayableTime);
    }
    updateTimeline();
  }, {once: true});
  player.src = videoUrl(app.selected.id, Date.now());
  player.load();
}

async function openPath() {
  const path = $("pathInput").value.trim();
  if (!path) return;
  setStatus("Opening video...");
  const response = await fetch(`${apiBase}/open-path`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({path}),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not open video.");
  const video = data.video;
  const existing = app.videos.find(item => item.id === video.id);
  if (!existing) app.videos.push(video);
  renderVideoOptions();
  selectVideo(video.id);
}

async function cutVideo() {
  if (!app.selected) return;
  if (!app.caseReady) {
    setStatus("Create or choose the player injury case before cutting video.");
    return;
  }
  if (!$("sourceVerifiedInput").checked) {
    setStatus("Verify the selected athlete and injury event before adding this view.");
    return;
  }
  const start = Number($("startInput").value);
  const end = Number($("endInput").value);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    setStatus("Choose an out time after the in time.");
    return;
  }
  if (app.clipRole === "REAL_TIME_CONTEXT" && !app.activeCaseId) {
    setStatus("Save the first analysis view before adding real-time context to this case.");
    return;
  }
  const sourceLabel = friendlyVideoLabel(app.selected);
  const targetLabel = app.activeCasePlayerName || app.activeCaseLabel || "the active player";
  const purpose = app.clipRole === "REAL_TIME_CONTEXT"
    ? `save it as context for ${targetLabel}`
    : `attach it as a measurement view for ${targetLabel}`;
  const confirmed = window.confirm(
    `Cut ${fmt(start)}–${fmt(end)} from “${sourceLabel}” and ${purpose}?\n\nVerify that this footage shows the correct athlete and injury event.`
  );
  if (!confirmed) return;
  app.inTime = start;
  app.outTime = end;
  app.reviewing = false;
  app.lastKnownTime = player.currentTime || app.lastKnownTime;
  player.pause();
  updateInputs();
  updateTimeline();
  $("cutButton").disabled = true;
  setStatus("Cutting video...");
  $("output").classList.remove("is-visible");
  const response = await fetch(`${apiBase}/cut`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      video_id: app.selected.id,
      start_seconds: app.inTime,
      end_seconds: app.outTime,
      output_name: $("nameInput").value,
      mode: app.mode,
      clip_role: app.clipRole,
      case_id: app.activeCaseId,
      created_by: "researcher_01",
    }),
  });
  const data = await response.json();
  $("cutButton").disabled = false;
  if (!response.ok) throw new Error(data.error || "Could not cut video.");
  app.latestCut = data;
  const preparedViewLabel = app.clipRole === "REAL_TIME_CONTEXT"
    ? "Real-time context clip"
    : ($("viewLabelInput").value || "Video view");
  $("outputName").textContent = friendlyCutLabel(preparedViewLabel);
  $("downloadLink").href = data.download_url;
  $("outputPath").textContent = data.path;
  $("output").classList.add("is-visible");
  const context = data.context_clip;
  $("assignmentPanel").hidden = Boolean(context);
  $("assignClipButton").hidden = true;
  $("assignClipButton").disabled = false;
  $("annotateAssignedLink").hidden = true;
  $("assignmentTitle").textContent = "Adding video view to case…";
  $("assignmentFeedback").textContent = "Saving this view to the active case…";
  setStatus(context
    ? `Saved as real-time context for ${context.player_name}. It is now available on Results.`
    : `Cut created with ${data.method}. Attaching it to ${app.activeCaseLabel}…`
  );
  if (!context) await assignAnalysisClip();
}

async function assignAnalysisClip() {
  if (!app.latestCut) return;
  const mode = app.activeAssignmentMode;
  if (mode === "existing" && !app.activeCaseId) {
    $("assignmentFeedback").textContent = "The active injury case is unavailable.";
    return;
  }
  if (mode === "new" && !$("casePlayerInput").value.trim()) {
    $("assignmentFeedback").textContent = "Enter the player name for the new injury case.";
    return;
  }
  if (mode === "new" && !$("caseDateInput").value) {
    $("assignmentFeedback").textContent = "Choose the match / injury date for the new case.";
    return;
  }
  $("assignClipButton").disabled = true;
  $("assignmentFeedback").textContent = "Assigning video view…";
  const response = await fetch(`${apiBase}/assign-analysis-clip`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      video_id: app.latestCut.video_id,
      assignment_mode: mode,
      case_id: app.activeCaseId,
      player_name: $("casePlayerInput").value,
      injury_date: $("caseDateInput").value,
      team: $("caseTeamInput").value,
      opponent: $("caseOpponentInput").value,
      competition: $("caseCompetitionInput").value,
      position_group: $("casePositionSelect").value,
      injured_side: $("caseInjuredSideSelect").value,
      match_minute: $("caseMatchMinuteInput").value,
      view_label: $("viewLabelInput").value,
      perspective: $("perspectiveSelect").value,
      slow_motion: $("slowMotionInput").checked,
      cropped_or_zoomed: $("croppedInput").checked,
      source_video_path: app.latestCut.source_video_path,
      clip_start_seconds: app.latestCut.start_seconds,
      clip_end_seconds: app.latestCut.end_seconds,
      created_by: "researcher_01",
    }),
  });
  const data = await response.json();
  $("assignClipButton").disabled = false;
  if (!response.ok) {
    $("assignClipButton").hidden = false;
    $("assignmentFeedback").textContent = data.error || "Could not assign this video view.";
    throw new Error(data.error || "Could not assign this video view.");
  }
  const item = data.case;
  const details = data.case_details || {};
  const existingOption = app.contextCases.find(option => option.case_id === item.case_id);
  if (existingOption) {
    existingOption.view_count = Number(existingOption.view_count || 0) + 1;
  } else {
    app.contextCases.push({
      case_id: item.case_id,
      player_name: details.player_name || item.player_name,
      injury_date: details.injury_date || "",
      team: details.team || "",
      opponent: details.opponent || "",
      competition: details.competition || "",
      position_group: details.position_group || "unknown",
      match_minute: details.match_minute || "",
      injured_side: item.injured_side || "unknown",
      view_count: 1,
    });
  }
  renderCaseOptions();
  $("analysisCaseSelect").value = item.case_id;
  $("contextCaseSelect").value = item.case_id;
  app.activeAssignmentMode = "existing";
  app.activeCaseId = item.case_id;
  const activeCase = app.contextCases.find(option => option.case_id === item.case_id);
  app.activeCaseLabel = caseOptionLabel(activeCase);
  app.activeCasePlayerName = details.player_name || item.player_name;
  updateCutterReturnLink();
  renderActiveCaseSummary(
    activeCase,
    "All further cuts will be saved as additional views of this case.",
  );
  const playerName = details.player_name || item.player_name;
  $("assignmentTitle").textContent = `View added to ${playerName}`;
  $("outputName").textContent = friendlyCutLabel(item.view_label || "Video view");
  $("assignmentFeedback").textContent = "Ready for annotation, or keep cutting another view for the same case.";
  $("annotateAssignedLink").href = `${annotateBase}?case=${encodeURIComponent(item.slug)}`;
  $("annotateAssignedLink").hidden = false;
  $("assignClipButton").hidden = true;
  $("assignClipButton").disabled = true;
  setStatus("Video view added successfully. Choose the next action below.");
  window.requestAnimationFrame(() => $("output").scrollIntoView({block: "nearest"}));
}

function prepareAnotherView() {
  app.latestCut = null;
  $("output").classList.remove("is-visible");
  $("viewLabelInput").value = "";
  $("perspectiveSelect").value = "unknown";
  $("slowMotionInput").checked = false;
  $("croppedInput").checked = false;
  $("nameInput").value = "";
  $("viewDetailsPanel").open = false;
  $("annotateAssignedLink").hidden = true;
  $("assignmentPanel").hidden = true;
  resetVideoSelection(`Choose and verify the source video for the next view of ${app.activeCasePlayerName || "the active player"}.`);
}

function reviewCut() {
  if (!app.selected) return;
  app.reviewing = true;
  app.lastKnownTime = app.inTime;
  player.currentTime = app.inTime;
  player.play();
}

function handlePlayerError() {
  const error = player.error;
  if (!error) return;
  if (app.switchingVideo && error.code === 1) return;
  app.switchingVideo = false;
  if (error.code === 1 && app.recoverAttempts < 2) {
    app.recoverAttempts += 1;
    window.setTimeout(() => {
      reloadCurrentVideo("Video loading was interrupted. Reloading the player...");
    }, 250);
    return;
  }
  app.videoReady = false;
  setVideoControlsEnabled(false);
  $("cutButton").disabled = true;
  setStatus(mediaErrorMessage(error));
}

function mediaErrorMessage(error) {
  if (error.code === 1) return "Video loading was interrupted. Use Reload to resume this video.";
  if (error.code === 2) return "Video loading hit a network error. Use Reload to try again.";
  if (error.code === 3) return "The browser could not decode this video.";
  if (error.code === 4) return "This browser does not support this video source.";
  return "The video player stopped unexpectedly. Use Reload to try again.";
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[char]));
}

$("refreshButton").addEventListener("click", () => loadVideos().catch(error => setStatus(error.message)));
$("videoSelect").addEventListener("change", event => selectVideo(event.target.value));
$("videoSearchInput").addEventListener("input", () => {
  renderVideoOptions();
  if (app.selected && $("videoSelect").value !== app.selected.id) {
    resetVideoSelection("The selected source is outside the current search. Choose and verify a visible source video.");
  }
});
$("sourceVerifiedInput").addEventListener("change", event => {
  $("cutButton").disabled = !event.target.checked || !app.caseReady || !app.selected || !app.videoReady;
  if (event.target.checked) {
    setStatus("Source verified. Set the In and Out points, review the selection, then add the view.");
  }
});
$("openPathButton").addEventListener("click", () => openPath().catch(error => setStatus(error.message)));
$("setInButton").addEventListener("click", () => setIn(player.currentTime || 0));
$("setOutButton").addEventListener("click", () => setOut(player.currentTime || 0));
$("reviewButton").addEventListener("click", reviewCut);
$("reloadPlayerButton").addEventListener("click", () => {
  app.recoverAttempts = 0;
  reloadCurrentVideo("Reloaded video.");
});
$("cutButton").addEventListener("click", () => cutVideo().catch(error => {
  $("cutButton").disabled = false;
  setStatus(error.message);
}));
$("backFiveButton").addEventListener("click", () => seekBy(-5));
$("forwardFiveButton").addEventListener("click", () => seekBy(5));
$("backFiveFrameButton").addEventListener("click", () => seekBy(-5 * frameStep()));
$("backFrameButton").addEventListener("click", () => seekBy(-frameStep()));
$("forwardFrameButton").addEventListener("click", () => seekBy(frameStep()));
$("forwardFiveFrameButton").addEventListener("click", () => seekBy(5 * frameStep()));
$("accurateModeButton").addEventListener("click", () => setMode("accurate"));
$("copyModeButton").addEventListener("click", () => setMode("copy"));
$("clipRoleSelect").addEventListener("change", event => setClipRole(event.target.value));
$("assignmentModeSelect").addEventListener("change", event => setAssignmentMode(event.target.value));
$("beginCaseButton").addEventListener("click", beginCaseWorkflow);
$("changeCaseButton").addEventListener("click", changeCaseWorkflow);
$("cutAnotherViewButton").addEventListener("click", prepareAnotherView);
$("assignClipButton").addEventListener("click", () => assignAnalysisClip().catch(error => {
  $("assignClipButton").disabled = false;
  $("assignmentFeedback").textContent = error.message;
}));
$("startInput").addEventListener("change", event => setIn(Number(event.target.value)));
$("endInput").addEventListener("change", event => setOut(Number(event.target.value)));
player.addEventListener("loadedmetadata", () => {
  app.duration = Number.isFinite(player.duration) ? player.duration : app.duration;
  app.switchingVideo = false;
  app.recoverAttempts = 0;
  app.videoReady = true;
  if (!app.outTime) app.outTime = app.duration;
  updateInputs();
  updateTimeline();
  setVideoControlsEnabled(true);
  $("sourceVerifiedInput").disabled = false;
  $("cutButton").disabled = !app.caseReady || !app.selected || !$("sourceVerifiedInput").checked;
  setStatus("Video ready. Verify the athlete and injury event, then set the In and Out points.");
});
player.addEventListener("timeupdate", () => {
  app.lastKnownTime = player.currentTime || app.lastKnownTime;
  if (app.reviewing && player.currentTime >= app.outTime) {
    player.pause();
    app.reviewing = false;
    player.currentTime = app.outTime;
  }
  updateTimeline();
});
player.addEventListener("pause", () => {
  app.reviewing = false;
});
player.addEventListener("error", handlePlayerError);

updateCutterReturnLink();
Promise.all([loadVideos(), loadContextCases()]).catch(error => setStatus(error.message));
</script>
</body>
</html>
    """.replace("__MAIN_MENU_URL__", escape(main_menu_url, quote=True)).replace(
        "__API_BASE_JSON__", json.dumps(api_base.rstrip("/"))
    ).replace(
        "__MAIN_MENU_JSON__", json.dumps(main_menu_url)
    ).replace(
        "__ANNOTATE_BASE_JSON__",
        json.dumps(f"{main_menu_url.rstrip('/')}/annotate"),
    ).replace("__APP_SHELL_CSS__", app_shell_css()).replace(
        "__APP_SITE_HEADER__",
        app_site_header("Video Cutter", home_url=main_menu_url),
    )
