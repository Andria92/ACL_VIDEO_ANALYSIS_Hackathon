"""Local browser UI for reviewing and cutting video clips."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from acl_motion.video.io import VideoMetadata, read_video_metadata

VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
DEFAULT_VIDEO_ROOTS = (
    "data/videos",
    "/Users/andriagryffinpro/Desktop/injury_videos",
)


@dataclass(slots=True)
class VideoCutterState:
    """Runtime state shared by video cutter UI requests."""

    video_roots: tuple[Path, ...]
    output_dir: Path
    manual_videos: set[Path] = field(default_factory=set)


def run_video_cutter_ui(
    *,
    host: str = "127.0.0.1",
    port: int = 8770,
    video_roots: tuple[str | Path, ...] = DEFAULT_VIDEO_ROOTS,
    output_dir: str | Path = "data/videos/cuts",
) -> None:
    """Run the local video cutter UI until interrupted."""

    state = VideoCutterState(
        video_roots=tuple(_resolve_path(root) for root in video_roots),
        output_dir=_resolve_path(output_dir),
    )
    server = build_server(host=host, port=port, state=state)
    print(f"ACL Movement Explorer video cutter: http://{host}:{port}")
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
                    self._send_html(render_video_cutter_page())
                elif parsed.path == "/api/videos":
                    self._send_json(_videos_response(state))
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
            except BrokenPipeError:
                return
            except (KeyError, OSError, ValueError) as exc:
                try:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                except BrokenPipeError:
                    return

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_html(render_video_cutter_page(), send_body=False)
                elif parsed.path == "/api/videos":
                    self._send_json(_videos_response(state), send_body=False)
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
            except BrokenPipeError:
                return
            except (KeyError, OSError, ValueError) as exc:
                try:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST, send_body=False)
                except BrokenPipeError:
                    return

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/open-path":
                    payload = self._read_json()
                    video_path = _validate_video_file(_resolve_path(str(payload["path"])))
                    state.manual_videos.add(video_path)
                    self._send_json({"video": _video_payload(video_path)})
                    return
                if parsed.path != "/api/cut":
                    self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")
                    return
                payload = self._read_json()
                video_path = _decode_video_id(str(payload["video_id"]), state)
                result = cut_video_segment(
                    video_path=video_path,
                    output_dir=state.output_dir,
                    start_seconds=float(payload["start_seconds"]),
                    end_seconds=float(payload["end_seconds"]),
                    output_name=str(payload.get("output_name", "")),
                    mode=str(payload.get("mode", "accurate")),
                )
                self._send_json(result)
            except BrokenPipeError:
                return
            except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as exc:
                try:
                    self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                except BrokenPipeError:
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
                start, end = _parse_range(range_header, file_size)
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            else:
                start, end = 0, max(file_size - 1, 0)
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(max(end - start + 1, 0)))
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
                    except BrokenPipeError:
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
    output_dir: str | Path = "data/videos/cuts",
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
    payloads: list[dict] = []
    skipped: list[dict] = []
    for path in unique_videos:
        try:
            payloads.append(_video_payload(path))
        except ValueError as exc:
            skipped.append({"path": str(path), "error": str(exc)})
    return {
        "videos": payloads,
        "skipped": skipped,
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
        "id": _encode_path(path),
        "name": path.name,
        "path": str(path),
        "metadata": _metadata_payload(metadata),
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


def render_video_cutter_page() -> str:
    """Return the self-contained video cutter UI page."""

    return r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ACL Movement Explorer - Video Cutter</title>
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
    .grow { flex: 1 1 240px; min-width: 170px; }
    label {
      display: flex;
      flex-direction: column;
      gap: 5px;
      font-size: 12px;
      color: var(--muted);
      font-weight: 650;
    }
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
      min-height: 36px;
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
    @media (max-width: 980px) {
      .shell {
        grid-template-columns: 1fr;
      }
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
  </style>
</head>
<body>
  <header>
    <h1>Video Cutter</h1>
    <div class="row">
      <button id="refreshButton">Refresh</button>
      <button id="cutButton" class="good" disabled>Cut Video</button>
    </div>
  </header>

  <main class="shell">
    <section class="panel viewer">
      <div class="toolbar">
        <label class="grow">
          Video
          <select id="videoSelect"></select>
        </label>
        <label class="grow">
          Local path
          <input id="pathInput" placeholder="/path/to/video.mp4" />
        </label>
        <button id="openPathButton" class="primary">Open</button>
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

    <aside class="panel side">
      <div class="button-row">
        <button id="backFiveButton">-5s</button>
        <button id="backFiveFrameButton">-5f</button>
        <button id="backFrameButton">-1f</button>
        <button id="forwardFrameButton">+1f</button>
        <button id="forwardFiveFrameButton">+5f</button>
        <button id="forwardFiveButton">+5s</button>
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
      <div id="status" class="status">Loading videos...</div>
      <div id="output" class="output">
        <strong id="outputName"></strong>
        <a id="downloadLink" href="#">Download cut</a>
        <a id="reviewOutputLink" href="#">Open cut in player</a>
        <span id="outputPath" class="status"></span>
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
};

const player = $("player");

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

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

async function loadVideos() {
  setStatus("Loading videos...");
  const response = await fetch("/api/videos");
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Could not load videos.");
  app.videos = data.videos || [];
  renderVideoOptions();
  if (app.videos.length) {
    selectVideo(app.videos[0].id);
  } else {
    $("videoSelect").innerHTML = '<option value="">No videos found</option>';
    $("cutButton").disabled = true;
    setStatus(`No videos found.\nRoots:\n${(data.roots || []).join("\n")}`);
  }
}

function renderVideoOptions() {
  $("videoSelect").innerHTML = app.videos.map(video => {
    return `<option value="${video.id}">${escapeHtml(video.name)}</option>`;
  }).join("");
}

function selectVideo(id) {
  const video = app.videos.find(item => item.id === id);
  if (!video) return;
  app.selected = video;
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
  $("cutButton").disabled = false;
  setStatus(video.path);
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
  return `/api/video?id=${encodeURIComponent(id)}${suffix}`;
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
  const resumeAt = clamp(player.currentTime || app.lastKnownTime || 0, 0, duration || Number.MAX_SAFE_INTEGER);
  app.switchingVideo = true;
  app.reviewing = false;
  setStatus(message);
  player.pause();
  player.src = videoUrl(app.selected.id, Date.now());
  player.load();
  player.addEventListener("loadedmetadata", () => {
    if (resumeAt > 0) {
      player.currentTime = Math.min(resumeAt, player.duration || app.duration || resumeAt);
    }
    updateTimeline();
  }, {once: true});
}

async function openPath() {
  const path = $("pathInput").value.trim();
  if (!path) return;
  setStatus("Opening video...");
  const response = await fetch("/api/open-path", {
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
  const start = Number($("startInput").value);
  const end = Number($("endInput").value);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    setStatus("Choose an out time after the in time.");
    return;
  }
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
  const response = await fetch("/api/cut", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      video_id: app.selected.id,
      start_seconds: app.inTime,
      end_seconds: app.outTime,
      output_name: $("nameInput").value,
      mode: app.mode,
    }),
  });
  const data = await response.json();
  $("cutButton").disabled = false;
  if (!response.ok) throw new Error(data.error || "Could not cut video.");
  $("outputName").textContent = data.file_name;
  $("downloadLink").href = data.download_url;
  $("reviewOutputLink").href = "#";
  $("reviewOutputLink").onclick = event => {
    event.preventDefault();
    const video = {
      id: data.video_id,
      name: data.file_name,
      path: data.path,
      metadata: {
        fps: data.fps,
        width: data.width,
        height: data.height,
        frame_count: data.frame_count,
        duration_seconds: data.duration_seconds,
      },
    };
    const existingIndex = app.videos.findIndex(item => item.id === video.id);
    if (existingIndex >= 0) {
      app.videos[existingIndex] = video;
    } else {
      app.videos.push(video);
    }
    renderVideoOptions();
    selectVideo(video.id);
  };
  $("outputPath").textContent = data.path;
  $("output").classList.add("is-visible");
  setStatus(`Saved with ${data.method}.`);
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
$("startInput").addEventListener("change", event => setIn(Number(event.target.value)));
$("endInput").addEventListener("change", event => setOut(Number(event.target.value)));
player.addEventListener("loadedmetadata", () => {
  app.duration = Number.isFinite(player.duration) ? player.duration : app.duration;
  app.switchingVideo = false;
  app.recoverAttempts = 0;
  if (!app.outTime) app.outTime = app.duration;
  updateInputs();
  updateTimeline();
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

loadVideos().catch(error => setStatus(error.message));
</script>
</body>
</html>
"""
