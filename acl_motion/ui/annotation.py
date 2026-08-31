"""Dependency-light local annotation UI for M5.5."""

from __future__ import annotations

import json
import mimetypes
import queue
import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from email.parser import BytesParser
from email.policy import default as email_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, RLock, Thread
from typing import BinaryIO
from urllib.parse import parse_qs, quote, urlparse

from acl_motion.analytics.exploration import (
    load_cached_exploration_summary_payload,
    load_exploration_payload,
)
from acl_motion.analytics.similarity import (
    SimilarityComputationCancelled,
    build_similarity_payload,
)
from acl_motion.annotations.event_interval_review import save_event_interval_review
from acl_motion.annotations.models import (
    ANNOTATION_UI_VERSION,
    AnnotationCase,
    EventConfidence,
    MovementWindowAnnotation,
    RoiKeyframeAnnotation,
    TargetAcceptedIntervalAnnotation,
    TargetUnavailableIntervalAnnotation,
)
from acl_motion.annotations.movement_window import movement_window_to_event_annotation
from acl_motion.annotations.registry import (
    default_annotation_cases,
    resolve_imported_video_path,
    views_for_case,
)
from acl_motion.annotations.research_metadata import (
    case_details,
    delete_case_details,
    research_metadata_path,
    save_case_details,
)
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
from acl_motion.cases.models import InjurySide
from acl_motion.persistence import (
    CaseArtifactTransaction,
    atomic_write_bytes,
    atomic_write_json,
    path_lock,
)
from acl_motion.runtime import ensure_supported_runtime
from acl_motion.ui.app_shell import app_shell_css, app_site_header
from acl_motion.ui.comparison import render_comparison_page
from acl_motion.ui.exploration import render_exploration_page
from acl_motion.ui.home import render_home_page
from acl_motion.ui.results import (
    clear_result_mask_prompts,
    context_video_path,
    generate_human_analysis_from_annotation,
    human_results_available,
    human_results_generated_at,
    load_human_results_payload,
    load_pose_review_frame_payload,
    load_pose_review_timeline_payload,
    load_result_evidence_payload,
    pose_review_analysis_status,
    read_pose_review_frame_jpeg,
    read_result_frame_jpeg,
    render_results_page,
    save_result_mask_prompt,
    trim_human_analysis_window_and_regenerate,
    undo_result_mask_prompt,
)
from acl_motion.ui.similarity_validation import render_similarity_validation_page
from acl_motion.ui.video_cutter import (
    VideoCutterState,
    assign_analysis_clip,
    create_video_cutter_state,
    cut_video_response,
    open_video_path_response,
    render_video_cutter_page,
    video_cutter_context_cases_response,
    video_cutter_output_path,
    video_cutter_video_path,
    video_cutter_videos_response,
)
from acl_motion.validation.expert_judgements import (
    ExpertPairwiseJudgement,
    PairwiseChoice,
    append_expert_judgement,
    build_blinded_assignments,
    evaluate_expert_judgements,
    load_expert_judgements,
    next_blinded_assignment,
)
from acl_motion.validation.similarity import build_internal_similarity_validation_report
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
    video_cutter_state: VideoCutterState | None = None
    trash_dir: Path | None = None
    analysis_jobs: dict[str, dict] = field(default_factory=dict)
    analysis_jobs_lock: Lock = field(default_factory=Lock, repr=False)
    analysis_queue: queue.Queue = field(default_factory=queue.Queue, repr=False)
    analysis_worker_lock: Lock = field(default_factory=Lock, repr=False)
    analysis_worker_started: bool = False
    catalog_lock: RLock = field(default_factory=RLock, repr=False)
    case_locks: dict[str, RLock] = field(default_factory=dict, repr=False)
    case_locks_lock: Lock = field(default_factory=Lock, repr=False)
    case_revisions: dict[str, int] = field(default_factory=dict)
    comparison_cache: dict[tuple[int, str], dict] = field(default_factory=dict, repr=False)
    comparison_cache_lock: Lock = field(default_factory=Lock, repr=False)
    comparison_generation: int = 0
    comparison_request_lock: Lock = field(default_factory=Lock, repr=False)
    comparison_latest_requests: dict[str, str] = field(default_factory=dict, repr=False)


class StateConflictError(RuntimeError):
    """Raised when a stale or concurrent operation cannot be applied safely."""


@contextmanager
def _guard_case_mutation(
    state: AnnotationUiState,
    case_id: str,
    operation: str,
):
    """Reject overlapping writes instead of blocking an HTTP request indefinitely."""

    operation_lock = _case_lock(state, case_id)
    if not operation_lock.acquire(blocking=False):
        raise StateConflictError(
            f"Another operation is already changing this case. Retry {operation} when it finishes."
        )
    try:
        if _case_has_active_analysis(state, case_id):
            raise StateConflictError(
                f"Analysis is queued or running for this case. Wait for it to finish before {operation}."
            )
        yield
    finally:
        operation_lock.release()


def run_annotation_ui(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    output_dir: str | Path = "data/annotations/human",
    video_root: str | Path = "data/videos/analysis_clips",
) -> None:
    """Run the local annotation UI until interrupted."""

    ensure_supported_runtime()
    output_path = Path(output_dir)
    root = Path(video_root)
    state = AnnotationUiState(
        cases=_ensure_primary_views(
            [
                case
                for case in [
                    *default_annotation_cases(root),
                    *_load_imported_cases(output_path, root),
                ]
                if case.video_path.exists()
            ]
        ),
        output_dir=output_path,
        video_root=root,
    )
    server = build_server(host=host, port=port, state=state)
    print(f"ACL Movement Analytics Lab annotation UI: http://{host}:{port}")
    print(f"Human annotations will save under: {state.output_dir}")
    server.serve_forever()


def build_server(
    *,
    host: str,
    port: int,
    state: AnnotationUiState,
) -> ThreadingHTTPServer:
    """Build a configured HTTP server for tests or local launch."""

    _hydrate_analysis_jobs(state)
    _hydrate_case_display_names(state)
    handler = make_handler(state)
    return ThreadingHTTPServer((host, port), handler)


def _analysis_job_payload(state: AnnotationUiState, case_slug: str) -> dict:
    """Return a safe snapshot of the current analysis transition for one view."""

    with state.analysis_jobs_lock:
        job = state.analysis_jobs.get(case_slug)
        if job is None:
            return {
                "case": case_slug,
                "status": "idle",
                "stage": "not_started",
                "result_url": None,
                "error": None,
            }
        return dict(job)


def _case_has_active_analysis(state: AnnotationUiState, case_id: str) -> bool:
    with state.analysis_jobs_lock:
        return any(
            job.get("case_id") == case_id
            and job.get("status") in {"queued", "running"}
            for job in state.analysis_jobs.values()
        )


def _case_lock(state: AnnotationUiState, case_id: str) -> RLock:
    """Return the stable in-process lock for one injury case and all of its views."""

    with state.case_locks_lock:
        return state.case_locks.setdefault(case_id, RLock())


def _case_revision(state: AnnotationUiState, case_id: str) -> int:
    with state.case_locks_lock:
        return state.case_revisions.setdefault(case_id, 0)


def _bump_case_revision(state: AnnotationUiState, case_id: str) -> int:
    with state.case_locks_lock:
        revision = state.case_revisions.get(case_id, 0) + 1
        state.case_revisions[case_id] = revision
        return revision


def _invalidate_comparison_cache(state: AnnotationUiState) -> None:
    with state.comparison_cache_lock:
        state.comparison_generation += 1
        state.comparison_cache.clear()


def _movement_comparison_response(
    state: AnnotationUiState,
    selected_case_id: str = "",
    *,
    client_id: str = "",
    request_id: str = "",
) -> dict:
    """Build each comparison generation once, then reuse it across requests."""

    if client_id and request_id:
        with state.comparison_request_lock:
            state.comparison_latest_requests[client_id] = request_id

    def cancelled() -> bool:
        if not client_id or not request_id:
            return False
        with state.comparison_request_lock:
            return state.comparison_latest_requests.get(client_id) != request_id

    with state.comparison_cache_lock:
        generation = state.comparison_generation
        cache_key = (generation, selected_case_id)
        cached = state.comparison_cache.get(cache_key)
        if cached is not None:
            return cached
        exploration_key = (generation, "__exploration_payload__")
        exploration = state.comparison_cache.get(exploration_key)
        if exploration is None:
            exploration = load_exploration_payload(
                tuple(state.cases),
                research_metadata_path=research_metadata_path(state.output_dir),
                data_root=_annotation_data_root(state),
            )
            state.comparison_cache[exploration_key] = exploration
        try:
            payload = build_similarity_payload(
                exploration["similarity_records"],
                exploration["events"],
                view_records=exploration["similarity_view_records"],
                selected_case_id=selected_case_id,
                cancelled=cancelled,
            )
        except SimilarityComputationCancelled as exc:
            raise StateConflictError(str(exc)) from exc
        state.comparison_cache[cache_key] = payload
        return payload


def _analysis_jobs_path(state: AnnotationUiState) -> Path:
    return state.output_dir / ".analysis_jobs_human.json"


def _persist_analysis_jobs_locked(state: AnnotationUiState) -> None:
    atomic_write_json(
        _analysis_jobs_path(state),
        {
            "updated_at": datetime.now(UTC).isoformat(),
            "jobs": state.analysis_jobs,
        },
    )


def _hydrate_analysis_jobs(state: AnnotationUiState) -> None:
    """Recover terminal job history and mark abandoned work after a restart."""

    path = _analysis_jobs_path(state)
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    jobs = payload.get("jobs", {})
    if not isinstance(jobs, dict):
        return
    now = datetime.now(UTC).isoformat()
    with state.analysis_jobs_lock:
        state.analysis_jobs = {
            str(slug): dict(job)
            for slug, job in jobs.items()
            if isinstance(job, dict)
        }
        changed = False
        for job in state.analysis_jobs.values():
            if job.get("status") in {"queued", "running"}:
                job.update(
                    status="failed",
                    stage="interrupted_by_restart",
                    error=(
                        "The application restarted before this analysis completed. "
                        "The saved annotation is intact; start analysis again."
                    ),
                    updated_at=now,
                    result_url=None,
                )
                changed = True
        if changed:
            _persist_analysis_jobs_locked(state)


def _update_analysis_job(state: AnnotationUiState, case_slug: str, **changes) -> dict:
    with state.analysis_jobs_lock:
        job = state.analysis_jobs[case_slug]
        job.update(changes)
        job["updated_at"] = datetime.now(UTC).isoformat()
        _persist_analysis_jobs_locked(state)
        return dict(job)


def _run_analysis_job(case: AnnotationCase, state: AnnotationUiState) -> None:
    _update_analysis_job(
        state,
        case.slug,
        status="running",
        stage="warming_runtime_and_running_pipeline",
    )
    try:
        with _case_lock(state, case.case_id):
            result = generate_human_analysis_from_annotation(case)
    except Exception as exc:  # noqa: BLE001 - jobs must always reach a terminal state.
        _update_analysis_job(
            state,
            case.slug,
            status="failed",
            stage="failed",
            error=str(exc),
            result_url=None,
        )
        return
    _bump_case_revision(state, case.case_id)
    _invalidate_comparison_cache(state)
    _update_analysis_job(
        state,
        case.slug,
        status="completed",
        stage="completed",
        error=None,
        result_url=result["result_url"],
    )


def _analysis_worker(state: AnnotationUiState) -> None:
    """Run analysis transitions serially so expensive pose jobs cannot stampede."""

    while True:
        case = state.analysis_queue.get()
        try:
            _run_analysis_job(case, state)
        finally:
            state.analysis_queue.task_done()


def _ensure_analysis_worker(state: AnnotationUiState) -> None:
    with state.analysis_worker_lock:
        if state.analysis_worker_started:
            return
        Thread(
            target=_analysis_worker,
            args=(state,),
            name="analysis-queue",
            daemon=True,
        ).start()
        state.analysis_worker_started = True


def _start_analysis_job(case: AnnotationCase, state: AnnotationUiState) -> dict:
    """Start one guarded analysis transition without blocking the HTTP request."""

    now = datetime.now(UTC).isoformat()
    operation_lock = _case_lock(state, case.case_id)
    if not operation_lock.acquire(blocking=False):
        raise StateConflictError(
            "This case is currently being saved, trimmed, or deleted. Retry analysis when "
            "that operation finishes."
        )
    try:
        with state.analysis_jobs_lock:
            existing = state.analysis_jobs.get(case.slug)
            if existing and existing.get("status") in {"queued", "running"}:
                return {**existing, "already_running": True}
            state.analysis_jobs[case.slug] = {
                "case": case.slug,
                "case_id": case.case_id,
                "status": "queued",
                "stage": "queued",
                "started_at": now,
                "updated_at": now,
                "result_url": None,
                "error": None,
                "already_running": False,
            }
            _persist_analysis_jobs_locked(state)
    finally:
        operation_lock.release()
    _ensure_analysis_worker(state)
    state.analysis_queue.put(case)
    return _analysis_job_payload(state, case.slug)


def make_handler(state: AnnotationUiState):
    """Create a request handler class bound to annotation UI state."""

    cutter_api_base = "/video-cutter/api"
    cutter_state = state.video_cutter_state or create_video_cutter_state(
        video_roots=(Path("data/videos"), state.video_root),
        output_dir=Path("data/videos/analysis_clips"),
        main_menu_url="/",
        annotation_output_dir=state.output_dir,
    )

    class AnnotationHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_html(render_home_page())
                elif parsed.path == "/annotate":
                    self._send_html(render_annotation_page())
                elif parsed.path == "/results":
                    self._send_html(render_results_page())
                elif parsed.path == "/compare":
                    self._send_html(render_comparison_page())
                elif parsed.path == "/validate-similarity":
                    self._send_html(render_similarity_validation_page())
                elif parsed.path == "/explore":
                    self._send_html(render_exploration_page())
                elif parsed.path in {"/video-cutter", "/video-cutter/"}:
                    self._send_html(
                        render_video_cutter_page(
                            main_menu_url="/",
                            api_base=cutter_api_base,
                        )
                    )
                elif parsed.path == f"{cutter_api_base}/videos":
                    self._send_json(video_cutter_videos_response(cutter_state))
                elif parsed.path == f"{cutter_api_base}/context-cases":
                    self._send_json(video_cutter_context_cases_response(cutter_state))
                elif parsed.path == f"{cutter_api_base}/metadata":
                    video_path = video_cutter_video_path(parsed.query, cutter_state)
                    self._send_json(_metadata_payload(read_video_metadata(video_path)))
                elif parsed.path == f"{cutter_api_base}/video":
                    video_path = video_cutter_video_path(parsed.query, cutter_state)
                    self._send_file(video_path)
                elif parsed.path == f"{cutter_api_base}/download":
                    output_path = video_cutter_output_path(parsed.query, cutter_state)
                    self._send_file(output_path, attachment=True)
                elif parsed.path == "/api/cases":
                    metadata_path = research_metadata_path(state.output_dir)
                    include_video_metadata = (
                        _one(
                            parse_qs(parsed.query),
                            "include_video_metadata",
                            "1",
                        )
                        != "0"
                    )
                    self._send_json(
                        {
                            "cases": [
                                _case_payload(
                                    case,
                                    display_details=case_details(
                                        metadata_path,
                                        case.case_id,
                                        fallback_player_name=case.player_name,
                                    ),
                                    annotation_output_dir=state.output_dir,
                                    include_video_metadata=include_video_metadata,
                                )
                                for case in state.cases
                            ]
                        }
                    )
                elif parsed.path == "/api/explore/summary":
                    self._send_json(
                        load_cached_exploration_summary_payload(
                            state.cases,
                            cache_path=(
                                state.output_dir / ".home_exploration_summary_cache.json"
                            ),
                            research_metadata_path=research_metadata_path(state.output_dir),
                            data_root=_annotation_data_root(state),
                        )
                    )
                elif parsed.path == "/api/explore":
                    self._send_json(
                        load_exploration_payload(
                            state.cases,
                            research_metadata_path=research_metadata_path(state.output_dir),
                            data_root=_annotation_data_root(state),
                        )
                    )
                elif parsed.path == "/api/movement-comparison":
                    query = parse_qs(parsed.query)
                    selected_case_id = _one(query, "case", "")
                    self._send_json(
                        _movement_comparison_response(
                            state,
                            selected_case_id,
                            client_id=_one(query, "client_id", ""),
                            request_id=_one(query, "request_id", ""),
                        )
                    )
                elif parsed.path == "/api/similarity-validation/assignment":
                    self._send_json(
                        _similarity_validation_assignment_response(
                            _one(parse_qs(parsed.query), "assessor"),
                            state,
                        )
                    )
                elif parsed.path == "/api/similarity-validation/report":
                    self._send_json(_similarity_validation_report(state))
                elif parsed.path == "/api/similarity-validation/video":
                    query = parse_qs(parsed.query)
                    case = _similarity_validation_video_case(
                        assessor_id=_one(query, "assessor"),
                        assignment_id=_one(query, "assignment"),
                        role=_one(query, "role"),
                        state=state,
                    )
                    self._send_file(case.video_path)
                elif parsed.path == "/api/session":
                    self._send_json(_session_response(_case_from_query(parsed.query), state))
                elif parsed.path == "/api/frame":
                    query = parse_qs(parsed.query)
                    case = _case_by_slug(_one(query, "case"), state.cases)
                    frame_index = int(_one(query, "frame", "0"))
                    image = read_frame_jpeg(case.video_path, frame_index)
                    self._send_bytes(image, "image/jpeg")
                elif parsed.path == "/api/pose-review":
                    query = parse_qs(parsed.query)
                    self._send_json(
                        load_pose_review_frame_payload(
                            _case_from_optional_query(parsed.query, state),
                            source_frame_index=int(_one(query, "frame")),
                        )
                    )
                elif parsed.path == "/api/pose-review/timeline":
                    self._send_json(
                        load_pose_review_timeline_payload(
                            _case_from_optional_query(parsed.query, state)
                        )
                    )
                elif parsed.path == "/api/pose-review/frame":
                    query = parse_qs(parsed.query)
                    image = read_pose_review_frame_jpeg(
                        _case_from_optional_query(parsed.query, state),
                        source_frame_index=int(_one(query, "frame")),
                    )
                    self._send_bytes(image, "image/jpeg")
                elif parsed.path == "/api/video":
                    case = _case_from_query(parsed.query)
                    self._send_file(case.video_path)
                elif parsed.path == "/api/analysis-status":
                    case = _case_from_optional_query(parsed.query, state)
                    self._send_json(_analysis_job_payload(state, case.slug))
                elif parsed.path == "/api/results":
                    case = _case_from_optional_query(parsed.query, state)
                    result_payload = load_human_results_payload(
                        case,
                        data_root=_annotation_data_root(state),
                        case_views=views_for_case(case, state.cases),
                        analysis_cases=tuple(state.cases),
                    )
                    result_payload["case_views"] = _results_navigation_payload(case, state)
                    self._send_json(result_payload)
                elif parsed.path == "/api/results/evidence":
                    query = parse_qs(parsed.query)
                    case = _case_from_optional_query(parsed.query, state)
                    self._send_json(
                        load_result_evidence_payload(
                            case,
                            feature_name=_one(query, "feature"),
                            source_frame_index=int(_one(query, "frame")),
                            data_root=_annotation_data_root(state),
                        )
                    )
                elif parsed.path == "/api/results/frame":
                    query = parse_qs(parsed.query)
                    case = _case_from_optional_query(parsed.query, state)
                    image = read_result_frame_jpeg(
                        case,
                        source_frame_index=int(_one(query, "frame")),
                        show_roi=_one(query, "roi", "1") == "1",
                        show_pose=_one(query, "pose", "1") == "1",
                        show_mask=_one(query, "mask", "0") == "1",
                        data_root=_annotation_data_root(state),
                    )
                    self._send_bytes(image, "image/jpeg")
                elif parsed.path == "/api/results/context-video":
                    query = parse_qs(parsed.query)
                    case = _case_from_optional_query(parsed.query, state)
                    self._send_file(
                        context_video_path(case, clip_id=_one(query, "clip"))
                    )
                elif parsed.path == "/api/compare":
                    self._send_json(_comparison_response(_case_from_query(parsed.query), state))
                elif parsed.path == "/api/view-alignment":
                    case = _case_from_query(parsed.query)
                    self._send_json(load_view_alignment(state.output_dir, case.case_id).to_dict())
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")
            except (BrokenPipeError, ConnectionResetError):
                return
            except StateConflictError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            except (json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_HEAD(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    self._send_html(render_home_page(), send_body=False)
                elif parsed.path == "/annotate":
                    self._send_html(render_annotation_page(), send_body=False)
                elif parsed.path == "/results":
                    self._send_html(render_results_page(), send_body=False)
                elif parsed.path == "/compare":
                    self._send_html(render_comparison_page(), send_body=False)
                elif parsed.path == "/validate-similarity":
                    self._send_html(render_similarity_validation_page(), send_body=False)
                elif parsed.path == "/explore":
                    self._send_html(render_exploration_page(), send_body=False)
                elif parsed.path in {"/video-cutter", "/video-cutter/"}:
                    self._send_html(
                        render_video_cutter_page(
                            main_menu_url="/",
                            api_base=cutter_api_base,
                        ),
                        send_body=False,
                    )
                elif parsed.path == f"{cutter_api_base}/videos":
                    self._send_json(
                        video_cutter_videos_response(cutter_state),
                        send_body=False,
                    )
                elif parsed.path == f"{cutter_api_base}/context-cases":
                    self._send_json(
                        video_cutter_context_cases_response(cutter_state),
                        send_body=False,
                    )
                elif parsed.path == f"{cutter_api_base}/metadata":
                    video_path = video_cutter_video_path(parsed.query, cutter_state)
                    self._send_json(
                        _metadata_payload(read_video_metadata(video_path)),
                        send_body=False,
                    )
                elif parsed.path == f"{cutter_api_base}/video":
                    video_path = video_cutter_video_path(parsed.query, cutter_state)
                    self._send_file(video_path, send_body=False)
                elif parsed.path == f"{cutter_api_base}/download":
                    output_path = video_cutter_output_path(parsed.query, cutter_state)
                    self._send_file(output_path, attachment=True, send_body=False)
                elif parsed.path == "/api/video":
                    case = _case_from_query(parsed.query)
                    self._send_file(case.video_path, send_body=False)
                elif parsed.path == "/api/results/context-video":
                    query = parse_qs(parsed.query)
                    case = _case_from_optional_query(parsed.query, state)
                    self._send_file(
                        context_video_path(case, clip_id=_one(query, "clip")),
                        send_body=False,
                    )
                elif parsed.path == "/api/similarity-validation/video":
                    query = parse_qs(parsed.query)
                    case = _similarity_validation_video_case(
                        assessor_id=_one(query, "assessor"),
                        assignment_id=_one(query, "assignment"),
                        role=_one(query, "role"),
                        state=state,
                    )
                    self._send_file(case.video_path, send_body=False)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Unknown route")
            except (BrokenPipeError, ConnectionResetError):
                return
            except (json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
                self._send_json(
                    {"error": str(exc)},
                    HTTPStatus.BAD_REQUEST,
                    send_body=False,
                )

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == f"{cutter_api_base}/open-path":
                    self._send_json(open_video_path_response(self._read_json(), cutter_state))
                    return
                if parsed.path == f"{cutter_api_base}/assign-analysis-clip":
                    with state.catalog_lock:
                        response, case = assign_analysis_clip(
                            self._read_json(),
                            cutter_state,
                            cases=state.cases,
                        )
                        if not any(existing.slug == case.slug for existing in state.cases):
                            state.cases.append(case)
                        _bump_case_revision(state, case.case_id)
                        _invalidate_comparison_cache(state)
                    self._send_json(response, HTTPStatus.CREATED)
                    return
                if parsed.path == "/api/cases/delete":
                    self._send_json(_delete_case_entry(self._read_json(), state))
                    return
                if parsed.path == "/api/similarity-validation/judgement":
                    self._send_json(
                        _save_similarity_validation_judgement(self._read_json(), state),
                        HTTPStatus.CREATED,
                    )
                    return
                if parsed.path == f"{cutter_api_base}/cut":
                    self._send_json(
                        cut_video_response(
                            self._read_json(),
                            cutter_state,
                            api_base=cutter_api_base,
                        )
                    )
                    return
                if parsed.path == "/api/results/mask-prompt":
                    payload = self._read_json()
                    case = _case_by_slug(str(payload["case"]), state.cases)
                    with _guard_case_mutation(state, case.case_id, "saving mask prompts"):
                        response = save_result_mask_prompt(
                            case,
                            source_frame_index=int(payload["frame"]),
                            x_px=float(payload["x"]),
                            y_px=float(payload["y"]),
                            label=str(payload["label"]),
                            data_root=_annotation_data_root(state),
                        )
                        _bump_case_revision(state, case.case_id)
                    self._send_json(response)
                    return
                if parsed.path == "/api/results/mask-prompts/undo":
                    payload = self._read_json()
                    case = _case_by_slug(str(payload["case"]), state.cases)
                    with _guard_case_mutation(state, case.case_id, "undoing mask prompts"):
                        response = undo_result_mask_prompt(
                            case,
                            source_frame_index=int(payload["frame"]),
                            data_root=_annotation_data_root(state),
                        )
                        _bump_case_revision(state, case.case_id)
                    self._send_json(response)
                    return
                if parsed.path == "/api/results/mask-prompts/clear":
                    payload = self._read_json()
                    case = _case_by_slug(str(payload["case"]), state.cases)
                    with _guard_case_mutation(state, case.case_id, "clearing mask prompts"):
                        response = clear_result_mask_prompts(
                            case,
                            source_frame_index=int(payload["frame"]),
                            data_root=_annotation_data_root(state),
                        )
                        _bump_case_revision(state, case.case_id)
                    self._send_json(response)
                    return
                if parsed.path == "/api/results/trim-analysis-window":
                    payload = self._read_json()
                    case = _case_by_slug(str(payload["case"]), state.cases)
                    operation_lock = _case_lock(state, case.case_id)
                    if not operation_lock.acquire(blocking=False):
                        raise StateConflictError(
                            "Another operation is already changing this case. Retry when it finishes."
                        )
                    try:
                        if _case_has_active_analysis(state, case.case_id):
                            raise StateConflictError(
                                "Analysis is already queued or running for this case. Wait for it "
                                "to finish before changing the analysis boundary."
                            )
                        response = trim_human_analysis_window_and_regenerate(
                            case,
                            movement_end_frame=int(payload["frame"]),
                            rationale=str(
                                payload.get(
                                    "rationale",
                                    "Post-injury frames excluded by human operator.",
                                )
                            ),
                            annotator_id=str(
                                payload.get("annotator_id", "researcher_01")
                            ),
                            data_root=_annotation_data_root(state),
                        )
                        _bump_case_revision(state, case.case_id)
                        _invalidate_comparison_cache(state)
                    finally:
                        operation_lock.release()
                    self._send_json(response)
                    return
                if parsed.path == "/api/results/event-interval-review":
                    payload = self._read_json()
                    case = _case_by_slug(str(payload["case"]), state.cases)
                    with _guard_case_mutation(
                        state,
                        case.case_id,
                        "saving the event-interval review",
                    ):
                        response = save_event_interval_review(
                            case,
                            decision=str(payload.get("decision", "")),
                            reviewer_id=str(payload.get("reviewer_id", "researcher_01")),
                            data_root=_annotation_data_root(state),
                        )
                        _bump_case_revision(state, case.case_id)
                        _invalidate_comparison_cache(state)
                    self._send_json(response)
                    return
                if parsed.path == "/api/generate-analysis":
                    payload = self._read_json()
                    case = _case_by_slug(str(payload["case"]), state.cases)
                    self._send_json(_start_analysis_job(case, state), HTTPStatus.ACCEPTED)
                    return
                if parsed.path == "/api/view-alignment":
                    payload = self._read_json()
                    case = _case_by_slug(str(payload["case"]), state.cases)
                    with _guard_case_mutation(state, case.case_id, "saving view alignment"):
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
                        _bump_case_revision(state, case.case_id)
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
            except (BrokenPipeError, ConnectionResetError):
                return
            except StateConflictError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.CONFLICT)
            except (
                json.JSONDecodeError,
                KeyError,
                OSError,
                RuntimeError,
                ValueError,
                subprocess.CalledProcessError,
            ) as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, format: str, *args) -> None:
            return

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            return json.loads(raw or "{}")

        def _send_html(self, html: str, *, send_body: bool = True) -> None:
            self._send_bytes(
                html.encode("utf-8"),
                "text/html; charset=utf-8",
                send_body=send_body,
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
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if send_body:
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    return

        def _send_bytes(
            self,
            data: bytes,
            content_type: str,
            *,
            send_body: bool = True,
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if send_body:
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    return

        def _send_file(
            self,
            path: str | Path,
            *,
            attachment: bool = False,
            send_body: bool = True,
        ) -> None:
            file_path = Path(path)
            if not file_path.exists():
                raise ValueError(f"Could not open video: {file_path}")
            content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            file_size = file_path.stat().st_size
            range_header = self.headers.get("Range")
            if range_header:
                try:
                    start, end = _parse_byte_range(range_header, file_size)
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
            if attachment:
                ascii_name = re.sub(
                    r"[^A-Za-z0-9._ -]+",
                    "_",
                    file_path.name.encode("ascii", "ignore").decode("ascii"),
                ).strip(" .") or "download"
                encoded_name = quote(file_path.name, safe="")
                self.send_header(
                    "Content-Disposition",
                    (
                        f'attachment; filename="{ascii_name}"; '
                        f"filename*=UTF-8''{encoded_name}"
                    ),
                )
            self.end_headers()
            if not send_body:
                return
            with file_path.open("rb") as handle:
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
    video_root: str | Path = "data/videos/analysis_clips",
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
        "html_has_save": 'id="save"' in html,
        "html_has_import_video": "Add / cut video view" in html,
        "html_has_open_video": "Open selected video" in html,
        "html_has_generate_analysis": "Generate analysis" in html,
        "html_has_injured_knee": 'id="injuredSide"' in html,
        "html_has_case_details": 'id="caseDetailsDisclosure"' in html,
        "html_has_target_unavailable_intervals": "Start excluded interval" in html,
        "html_has_pose_review": 'id="previousPoseMode"' in html,
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
    """Reject the legacy one-file/one-case import path.

    Video views now enter through the player-first cutter so that repeated clips cannot
    silently create independent injury cases.
    """

    raise ValueError(
        "Direct video import from annotation is disabled. Add the video through the "
        "Video Cutter and attach it to a player injury case first."
    )
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
    with path_lock(path):
        records = [
            _imported_case_record(case)
            for case in state.cases
            if case.slug.startswith(IMPORTED_CASE_SLUG_PREFIX)
        ]
        return atomic_write_json(path, {"cases": records})


def _ensure_primary_views(cases: list[AnnotationCase]) -> list[AnnotationCase]:
    for case_id in {case.case_id for case in cases}:
        siblings = [case for case in cases if case.case_id == case_id]
        if siblings and not any(case.primary_view for case in siblings):
            promoted = siblings[0]
            cases[cases.index(promoted)] = replace(promoted, primary_view=True)
    return cases


def _trash_destination(bundle: Path, path: Path, data_root: Path | None) -> Path:
    if data_root is not None:
        try:
            relative = path.relative_to(data_root)
            return bundle / "data" / relative
        except ValueError:
            pass
    return bundle / "source_videos" / path.name


def _move_case_files_to_trash(
    removed_cases: list[AnnotationCase],
    state: AnnotationUiState,
) -> tuple[Path, list[tuple[Path, Path]]]:
    trash_root = state.trash_dir or Path.home() / ".Trash" / "ACL Movement Explorer"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    bundle = trash_root / f"deleted-case-{stamp}"
    data_root = (
        state.output_dir.parent.parent
        if state.output_dir.parent.name == "annotations"
        else None
    )
    remaining_paths = {
        case.video_path.resolve()
        for case in state.cases
        if case not in removed_cases and case.video_path.exists()
    }
    paths: set[Path] = {
        case.video_path
        for case in removed_cases
        if case.video_path.exists() and case.video_path.resolve() not in remaining_paths
    }
    if data_root is not None and data_root.exists():
        slugs = tuple(case.slug for case in removed_cases)
        paths.update(
            path
            for path in data_root.rglob("*")
            if path.is_file()
            and any(path.name.startswith(f"{slug}_") for slug in slugs)
        )

    bundle.mkdir(parents=True, exist_ok=False)
    moved: list[tuple[Path, Path]] = []
    try:
        for path in sorted(paths):
            destination = _trash_destination(bundle, path, data_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                destination = destination.with_name(
                    f"{destination.stem}-{len(moved) + 1}{destination.suffix}"
                )
            shutil.move(str(path), destination)
            moved.append((path, destination))
        atomic_write_json(
            bundle / "deletion_manifest.json",
            {
                "deleted_at": datetime.now(UTC).isoformat(),
                "case_ids": sorted({case.case_id for case in removed_cases}),
                "view_slugs": [case.slug for case in removed_cases],
                "moved_file_count": len(moved),
                "moves": [
                    {"source": str(source), "trash": str(destination)}
                    for source, destination in moved
                ],
            },
        )
    except Exception:
        _restore_case_files_from_trash(moved, bundle)
        raise
    return bundle, moved


def _restore_case_files_from_trash(
    moved: list[tuple[Path, Path]],
    bundle: Path,
) -> None:
    for source, destination in reversed(moved):
        if not destination.exists():
            continue
        source.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(destination), source)
    shutil.rmtree(bundle, ignore_errors=True)


def _remove_case_metadata(output_dir: Path, case_id: str) -> None:
    delete_case_details(research_metadata_path(output_dir), case_id)


def _delete_case_entry(payload: dict, state: AnnotationUiState) -> dict:
    case_id = str(payload.get("case_id", "")).strip()
    if not case_id:
        raise ValueError("A case_id is required.")
    operation_lock = _case_lock(state, case_id)
    if not operation_lock.acquire(blocking=False):
        raise StateConflictError(
            "Another operation is already changing this case. Retry deletion when it finishes."
        )
    try:
        if _case_has_active_analysis(state, case_id):
            raise StateConflictError(
                "Analysis is queued or running for this case. Wait for it to finish before deleting."
            )
        with state.catalog_lock:
            return _delete_case_entry_unlocked(payload, state)
    finally:
        operation_lock.release()


def _delete_case_entry_unlocked(payload: dict, state: AnnotationUiState) -> dict:
    """Remove a registered case/view and send all of its files to system Trash."""

    scope = str(payload.get("scope", "case")).strip().lower()
    case_id = str(payload.get("case_id", "")).strip()
    if scope not in {"case", "view"}:
        raise ValueError("Delete scope must be 'case' or 'view'.")
    if not case_id:
        raise ValueError("A case_id is required.")

    siblings = [case for case in state.cases if case.case_id == case_id]
    if not siblings:
        raise ValueError("The selected injury case is no longer in the library.")

    if scope == "case":
        removed = siblings
    else:
        slug = str(payload.get("slug", "")).strip()
        target = next((case for case in siblings if case.slug == slug), None)
        if target is None:
            raise ValueError("The selected video view is no longer in this case.")
        if len(siblings) == 1:
            raise ValueError("Delete the case to remove its only video view.")
        removed = [target]

    registry_path = _imported_cases_path(state.output_dir)
    metadata_path = research_metadata_path(state.output_dir)
    with path_lock(registry_path), path_lock(metadata_path):
        return _delete_case_catalog_entry(
            scope=scope,
            case_id=case_id,
            removed=removed,
            state=state,
            registry_path=registry_path,
            metadata_path=metadata_path,
        )


def _delete_case_catalog_entry(
    *,
    scope: str,
    case_id: str,
    removed: list[AnnotationCase],
    state: AnnotationUiState,
    registry_path: Path,
    metadata_path: Path,
) -> dict:
    """Apply one deletion while both shared case catalogs are locked."""

    registry_before = registry_path.read_bytes() if registry_path.exists() else None
    metadata_before = metadata_path.read_bytes() if metadata_path.exists() else None
    cases_before = list(state.cases)
    trash_bundle, moved = _move_case_files_to_trash(removed, state)
    removed_slugs = {case.slug for case in removed}
    try:
        state.cases[:] = _ensure_primary_views(
            [case for case in state.cases if case.slug not in removed_slugs]
        )
        _save_imported_cases(state)
        remaining_view_count = sum(case.case_id == case_id for case in state.cases)
        if not remaining_view_count:
            _remove_case_metadata(state.output_dir, case_id)
    except Exception:
        state.cases[:] = cases_before
        if registry_before is None:
            registry_path.unlink(missing_ok=True)
        else:
            atomic_write_bytes(registry_path, registry_before)
        if metadata_before is None:
            metadata_path.unlink(missing_ok=True)
        else:
            atomic_write_bytes(metadata_path, metadata_before)
        _restore_case_files_from_trash(moved, trash_bundle)
        raise
    _bump_case_revision(state, case_id)
    _invalidate_comparison_cache(state)
    with state.analysis_jobs_lock:
        for slug in removed_slugs:
            state.analysis_jobs.pop(slug, None)
        _persist_analysis_jobs_locked(state)
    return {
        "deleted": True,
        "scope": scope,
        "case_id": case_id,
        "removed_view_slugs": sorted(removed_slugs),
        "remaining_view_count": remaining_view_count,
        "source_files_preserved": False,
        "moved_file_count": len(moved),
        "trash_bundle": str(trash_bundle),
    }


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
        "injured_side": InjurySide(case.injured_side).value,
        "injury_laterality_source": case.injury_laterality_source,
        "player_name": case.player_name,
        "video_path": str(case.video_path),
        "notes": case.notes,
    }


def _imported_case_from_record(record: dict, video_root: Path) -> AnnotationCase:
    video_path = resolve_imported_video_path(record["video_path"], video_root)
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
        injured_side=InjurySide(record.get("injured_side", "unknown")),
        injury_laterality_source=str(record.get("injury_laterality_source", "")),
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


def _case_payload(
    case: AnnotationCase,
    *,
    display_details: dict[str, str] | None = None,
    annotation_output_dir: str | Path | None = None,
    include_video_metadata: bool = True,
) -> dict:
    payload = case.to_dict()
    if display_details is not None:
        payload["case_details"] = dict(display_details)
        payload["player_name"] = display_details.get("player_name") or case.player_name
        for field in (
            "injury_date",
            "team",
            "opponent",
            "competition",
            "position_group",
            "match_minute",
        ):
            payload[field] = display_details.get(field, "")
    # Keep registered local paths server-side. The browser only needs a safe label.
    payload["video_path"] = case.video_path.name
    payload["video_source_label"] = case.video_path.name
    payload["results_available"] = human_results_available(case)
    payload["analysis_generated_at"] = human_results_generated_at(case)
    payload["annotation_saved"] = (
        human_annotation_paths(annotation_output_dir, case.slug).session_json.exists()
        if annotation_output_dir is not None
        else False
    )
    if include_video_metadata:
        try:
            metadata = read_video_metadata(case.video_path)
            payload["metadata"] = _metadata_payload(metadata)
            payload["video_available"] = True
        except ValueError:
            payload["metadata"] = None
            payload["video_available"] = False
            payload["video_error"] = "The registered source video is unavailable."
    else:
        payload["metadata"] = None
        payload["video_available"] = case.video_path.exists()
    return payload


def _metadata_payload(metadata: VideoMetadata) -> dict:
    return {
        "fps": metadata.fps,
        "width": metadata.width,
        "height": metadata.height,
        "frame_count": metadata.frame_count,
        "duration_seconds": metadata.duration_seconds,
    }


def _annotation_data_root(state: AnnotationUiState) -> Path:
    """Resolve the data root from the configured human-annotation directory."""

    output = state.output_dir
    if output.name == "human" and output.parent.name == "annotations":
        return output.parent.parent
    return Path("data")


def _results_navigation_payload(
    case: AnnotationCase,
    state: AnnotationUiState,
) -> dict:
    """Return clip navigation for the shared case."""

    root = _annotation_data_root(state)
    navigation_views = []
    for view in views_for_case(case, state.cases):
        annotation_saved = human_annotation_paths(
            state.output_dir,
            view.slug,
        ).session_json.exists()
        navigation_views.append(
            {
                "slug": view.slug,
                "case_id": view.case_id,
                "source_id": view.source_id,
                "view_id": view.view_id or view.source_id,
                "view_label": view.view_label,
                "perspective": view.perspective,
                "primary_view": view.primary_view,
                "annotation_saved": annotation_saved,
                "results_available": human_results_available(view, data_root=root),
            }
        )
    return {
        "case_id": case.case_id,
        "current_view_slug": case.slug,
        "view_count": len(navigation_views),
        "views": navigation_views,
    }


def _session_response(case: AnnotationCase, state: AnnotationUiState) -> dict:
    with _case_lock(state, case.case_id):
        response = _session_response_unlocked(case, state)
        response["revision"] = _case_revision(state, case.case_id)
        return response


def _session_response_unlocked(case: AnnotationCase, state: AnnotationUiState) -> dict:
    paths = human_annotation_paths(state.output_dir, case.slug)
    if paths.session_json.exists():
        session = load_human_annotation_session(paths.session_json)
        if session.injured_side is not InjurySide.UNKNOWN:
            case = _set_case_laterality(
                case,
                state,
                session.injured_side,
                session.injury_laterality_source,
                persist=False,
            )
        session_payload = session.to_dict()
        resume = True
    else:
        session_payload = {
            "provenance": None,
            "manual_roi_keyframe_count": 0,
            "roi_keyframes": [],
            "manual_target_unavailable_frame_count": 0,
            "target_unavailable_intervals": [],
            "manual_target_accepted_frame_count": 0,
            "target_accepted_intervals": [],
            "movement_window": None,
            "event_annotation": None,
            "event_confidence_label": None,
            "injured_side": InjurySide(case.injured_side).value,
            "injury_laterality_source": case.injury_laterality_source,
            "operator_flags": [],
            "notes": "",
            "finalized": False,
        }
        resume = False
    saved_case_details = case_details(
        research_metadata_path(state.output_dir),
        case.case_id,
        fallback_player_name=case.player_name,
    )
    return {
        "case": _case_payload(case, display_details=saved_case_details),
        "case_details": saved_case_details,
        "session": session_payload,
        "resume_available": resume,
        "human_results_available": human_results_available(case),
        "pose_review": pose_review_analysis_status(case),
        "human_paths": {
            "session_json": paths.session_json.name,
            "roi_csv": paths.roi_csv.name,
            "target_unavailable_csv": paths.target_unavailable_csv.name,
            "movement_window_json": paths.movement_window_json.name,
            "event_json": paths.event_json.name,
        },
    }


def _save_response(payload: dict, state: AnnotationUiState) -> dict:
    case = _case_by_slug(str(payload["case_slug"]), state.cases)
    operation_lock = _case_lock(state, case.case_id)
    if not operation_lock.acquire(blocking=False):
        raise StateConflictError(
            "Another operation is already changing this case. Retry saving when it finishes."
        )
    try:
        if _case_has_active_analysis(state, case.case_id):
            raise StateConflictError(
                "Analysis is queued or running for this case. Wait for it to finish before saving."
            )
        expected = payload.get("revision")
        current = _case_revision(state, case.case_id)
        if expected is not None and int(expected) != current:
            raise StateConflictError(
                "This case changed in another tab. Reload it before saving so newer work "
                "is not overwritten."
            )
        registry_path = _imported_cases_path(state.output_dir)
        metadata_path = research_metadata_path(state.output_dir)
        with path_lock(registry_path), path_lock(metadata_path):
            registry_before = registry_path.read_bytes() if registry_path.exists() else None
            metadata_before = metadata_path.read_bytes() if metadata_path.exists() else None
            cases_before = list(state.cases)
            try:
                with CaseArtifactTransaction(state.output_dir, case.slug):
                    response = _save_response_unlocked(payload, state)
            except Exception:
                state.cases[:] = cases_before
                _restore_optional_file(registry_path, registry_before)
                _restore_optional_file(metadata_path, metadata_before)
                raise
        response["revision"] = _bump_case_revision(state, case.case_id)
        _invalidate_comparison_cache(state)
        return response
    finally:
        operation_lock.release()


def _restore_optional_file(path: Path, previous: bytes | None) -> None:
    """Restore a small catalog file after a failed multi-file operation."""

    if previous is None:
        path.unlink(missing_ok=True)
    else:
        atomic_write_bytes(path, previous)


def _save_response_unlocked(payload: dict, state: AnnotationUiState) -> dict:
    case = _case_by_slug(str(payload["case_slug"]), state.cases)
    paths = human_annotation_paths(state.output_dir, case.slug)
    existing = load_human_annotation_session(paths.session_json) if paths.session_json.exists() else None
    annotator_id = str(payload.get("annotator_id", "")).strip() or "researcher_01"
    try:
        injured_side = InjurySide(str(payload.get("injured_side", "unknown")).lower())
    except ValueError as exc:
        raise ValueError("Injured knee must be left, right, or unknown.") from exc
    injury_laterality_source = (
        f"human_operator_annotation_ui:{annotator_id}"
        if injured_side is not InjurySide.UNKNOWN
        else ""
    )
    keyframes = tuple(_keyframe_from_payload(item) for item in payload.get("roi_keyframes", ()))
    target_unavailable_intervals = tuple(
        _target_unavailable_interval_from_payload(item)
        for item in payload.get("target_unavailable_intervals", ())
    )
    target_accepted_intervals = tuple(
        _target_accepted_interval_from_payload(item)
        for item in payload.get("target_accepted_intervals", ())
    )
    movement_window, confidence_label = _movement_window_from_payload(payload, case, keyframes)
    provisional_session = new_human_session(
        case_id=case.case_id,
        source_id=case.source_id,
        video_path=case.video_path,
        annotator_id=annotator_id,
        view_id=case.view_id or case.source_id,
        roi_keyframes=keyframes,
        target_unavailable_intervals=target_unavailable_intervals,
        target_accepted_intervals=target_accepted_intervals,
        movement_window=movement_window,
        event_confidence_label=confidence_label,
        injured_side=injured_side,
        injury_laterality_source=injury_laterality_source,
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
    metadata_path = research_metadata_path(state.output_dir)
    if "case_details" in payload:
        saved_case_details = save_case_details(
            metadata_path,
            case.case_id,
            payload.get("case_details", {}),
            annotator_id=annotator_id,
            fallback_player_name=case.player_name,
        )
    else:
        saved_case_details = case_details(
            metadata_path,
            case.case_id,
            fallback_player_name=case.player_name,
        )
    case = _set_case_player_name(case, state, saved_case_details["player_name"])
    case = _set_case_laterality(
        case,
        state,
        injured_side,
        injury_laterality_source,
        persist=True,
    )
    return {
        "saved": True,
        "case": _case_payload(case, display_details=saved_case_details),
        "case_details": saved_case_details,
        "session": session.to_dict(),
        "validation": validation.to_dict(),
        "human_results_available": human_results_available(case),
        "paths": {
            "session_json": saved_paths.session_json.name,
            "roi_csv": saved_paths.roi_csv.name,
            "target_unavailable_csv": saved_paths.target_unavailable_csv.name,
            "movement_window_json": saved_paths.movement_window_json.name,
            "event_json": saved_paths.event_json.name,
            "case_research_metadata_json": metadata_path.name,
        },
    }


def _hydrate_case_display_names(state: AnnotationUiState) -> None:
    """Apply saved case-level player names to every runtime video view."""

    metadata_path = research_metadata_path(state.output_dir)
    for index, case in enumerate(state.cases):
        details = case_details(
            metadata_path,
            case.case_id,
            fallback_player_name=case.player_name,
        )
        player_name = details["player_name"] or case.player_name
        if player_name != case.player_name:
            state.cases[index] = replace(case, player_name=player_name)


def _set_case_player_name(
    case: AnnotationCase,
    state: AnnotationUiState,
    player_name: str,
) -> AnnotationCase:
    """Update the shared runtime identity for every view of one injury case."""

    normalized = player_name.strip() or case.player_name
    selected = replace(case, player_name=normalized)
    for index, candidate in enumerate(state.cases):
        if candidate.case_id == case.case_id:
            updated = replace(candidate, player_name=normalized)
            state.cases[index] = updated
            if candidate.slug == case.slug:
                selected = updated
    return selected


def _set_case_laterality(
    case: AnnotationCase,
    state: AnnotationUiState,
    injured_side: InjurySide,
    source: str,
    *,
    persist: bool,
) -> AnnotationCase:
    """Update runtime case laterality and persist imported-case metadata when requested."""

    updated = replace(
        case,
        injured_side=InjurySide(injured_side),
        injury_laterality_source=source,
    )
    for index, candidate in enumerate(state.cases):
        if candidate.slug == case.slug:
            state.cases[index] = updated
            break
    if persist and case.slug.startswith(IMPORTED_CASE_SLUG_PREFIX):
        _save_imported_cases(state)
    return updated


def _comparison_response(case: AnnotationCase, state: AnnotationUiState) -> dict:
    paths = human_annotation_paths(state.output_dir, case.slug)
    if not paths.session_json.exists() or not paths.roi_csv.exists():
        return {"available": False, "reason": "No saved human ROI annotation exists yet."}
    if case.development_roi_path is None or not case.development_roi_path.exists():
        return {"available": False, "reason": "No development ROI annotation is registered."}
    metadata = read_video_metadata(case.video_path)
    human_session = load_human_annotation_session(paths.session_json)
    development_timeline = RoiTimeline.from_csv(case.development_roi_path)
    frames = tuple(
        frame_index
        for frame_index in range(metadata.frame_count)
        if human_session.target_unavailable_interval_at(frame_index) is None
    )
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


def _similarity_validation_context(
    state: AnnotationUiState,
) -> tuple[dict, list[dict], set[str]]:
    exploration = load_exploration_payload(
        state.cases,
        research_metadata_path=research_metadata_path(state.output_dir),
        data_root=_annotation_data_root(state),
    )
    similarity_index = build_similarity_payload(
        exploration["similarity_records"],
        exploration["events"],
        view_records=exploration["similarity_view_records"],
        resampling_iterations=0,
    )
    public_ids = {str(case["case_id"]) for case in similarity_index["cases"]}
    reference_ids = {
        str(case["case_id"])
        for case in similarity_index["cases"]
        if bool(case["reference_pool_eligible"])
    }
    sources = []
    for event in exploration["events"]:
        statistical_unit_id = str(event["case_id"])
        if statistical_unit_id not in public_ids:
            continue
        registered_ids = {
            str(value)
            for value in event.get("registered_case_ids", [statistical_unit_id])
        }
        candidate_views = [
            case
            for case in state.cases
            if case.case_id in registered_ids and case.video_path.exists()
        ]
        if not candidate_views:
            continue
        selected_view = min(
            candidate_views,
            key=lambda case: (not case.primary_view, case.slug),
        )
        sources.append(
            {
                "case_id": statistical_unit_id,
                "slug": selected_view.slug,
            }
        )
    return exploration, sources, reference_ids


def _similarity_validation_assignment_response(
    assessor_id: str,
    state: AnnotationUiState,
) -> dict:
    assessor = assessor_id.strip()
    exploration, sources, reference_ids = _similarity_validation_context(state)
    assignments = build_blinded_assignments(
        sources,
        reference_ids,
        assessor_id=assessor,
    )
    judgements = load_expert_judgements(_expert_judgement_path(state))
    selected = next_blinded_assignment(
        assignments,
        judgements,
        assessor_id=assessor,
    )
    assignment_ids = {str(item["assignment_id"]) for item in assignments}
    completed_count = len(
        {
            item.assignment_id
            for item in judgements
            if item.assessor_id == assessor and item.assignment_id in assignment_ids
        }
    )
    public_assignment = None
    if selected is not None:
        encoded_assessor = quote(assessor, safe="")
        encoded_assignment = quote(str(selected["assignment_id"]), safe="")
        base_url = (
            "/api/similarity-validation/video?"
            f"assessor={encoded_assessor}&assignment={encoded_assignment}"
        )
        public_assignment = {
            "assignment_id": selected["assignment_id"],
            "query_video_url": f"{base_url}&role=query",
            "option_a_video_url": f"{base_url}&role=option_a",
            "option_b_video_url": f"{base_url}&role=option_b",
        }
    return {
        "protocol": "expert_pairwise_similarity_v1",
        "assignment": public_assignment,
        "assignment_count": len(assignments),
        "completed_count": completed_count,
        "reference_pool_case_count": len(reference_ids),
        "query_case_count": len(sources),
        "blinding": (
            "Algorithm scores, rankings, and recorded case metadata are hidden. Visual identity "
            "may still be apparent from the source footage."
        ),
        "data_status": exploration["similarity"]["status"],
    }


def _save_similarity_validation_judgement(
    payload: dict,
    state: AnnotationUiState,
) -> dict:
    assessor = str(payload["assessor_id"]).strip()
    _, sources, reference_ids = _similarity_validation_context(state)
    assignments = build_blinded_assignments(
        sources,
        reference_ids,
        assessor_id=assessor,
    )
    assignment = next(
        (
            item
            for item in assignments
            if item["assignment_id"] == str(payload["assignment_id"])
        ),
        None,
    )
    if assignment is None:
        raise ValueError("The blinded assignment is unavailable or no longer eligible.")
    judgement = ExpertPairwiseJudgement.create(
        assignment=assignment,
        assessor_id=assessor,
        choice=PairwiseChoice(str(payload["choice"])),
        notes=str(payload.get("notes", "")),
    )
    path = append_expert_judgement(_expert_judgement_path(state), judgement)
    return {
        "saved": True,
        "judgement_id": judgement.judgement_id,
        "path": str(path),
    }


def _similarity_validation_video_case(
    *,
    assessor_id: str,
    assignment_id: str,
    role: str,
    state: AnnotationUiState,
) -> AnnotationCase:
    _, sources, reference_ids = _similarity_validation_context(state)
    assignments = build_blinded_assignments(
        sources,
        reference_ids,
        assessor_id=assessor_id,
    )
    assignment = next(
        (item for item in assignments if item["assignment_id"] == assignment_id),
        None,
    )
    if assignment is None:
        raise ValueError("Unknown blinded validation assignment.")
    role_key = {
        "query": "query_slug",
        "option_a": "option_a_slug",
        "option_b": "option_b_slug",
    }.get(role)
    if role_key is None:
        raise ValueError("Validation video role must be query, option_a, or option_b.")
    return _case_by_slug(str(assignment[role_key]), state.cases)


def _similarity_validation_report(state: AnnotationUiState) -> dict:
    exploration, _, _ = _similarity_validation_context(state)
    judgements = load_expert_judgements(_expert_judgement_path(state))
    return {
        "internal_audit": build_internal_similarity_validation_report(
            exploration["records"], exploration["events"]
        ),
        "expert_concordance": evaluate_expert_judgements(
            exploration["records"], exploration["events"], judgements
        ),
        "scope_note": (
            "Internal sensitivity and current-case expert concordance are not substitutes for "
            "external laboratory validation or genuinely new held-out players."
        ),
    }


def _expert_judgement_path(state: AnnotationUiState) -> Path:
    annotation_root = Path(state.output_dir)
    data_root = (
        annotation_root.parent.parent
        if annotation_root.name == "human" and annotation_root.parent.name == "annotations"
        else annotation_root
    )
    return data_root / "validation" / "human" / "expert_pairwise_judgements.jsonl"


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


def _target_unavailable_interval_from_payload(
    data: dict,
) -> TargetUnavailableIntervalAnnotation:
    return TargetUnavailableIntervalAnnotation.from_dict(data)


def _target_accepted_interval_from_payload(
    data: dict,
) -> TargetAcceptedIntervalAnnotation:
    return TargetAcceptedIntervalAnnotation.from_dict(data)


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


def _parse_byte_range(range_header: str, file_size: int) -> tuple[int, int]:
    """Parse a single HTTP byte range for browser video seeking."""

    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match or file_size <= 0:
        raise ValueError("Unsupported video byte range.")
    start_text, end_text = match.groups()
    if not start_text:
        suffix_length = int(end_text or "0")
        if suffix_length <= 0:
            raise ValueError("Unsupported video byte range.")
        start = max(file_size - suffix_length, 0)
        end = file_size - 1
    else:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    if start < 0 or start >= file_size or end < start:
        raise ValueError("Requested video byte range is outside the file.")
    return start, min(end, file_size - 1)


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
  <title>ACL Movement Analytics Lab - Human Annotation</title>
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
      --unavailable: #8f2f3f;
    }
    * { box-sizing: border-box; }
    :focus-visible {
      outline: 3px solid #f0a929;
      outline-offset: 3px;
    }
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
      grid-template-columns: minmax(520px, 1fr) minmax(350px, 384px);
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
      width: auto;
      height: auto;
      max-width: 100%;
      max-height: 68vh;
      margin-inline: auto;
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
      min-height: 44px;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    a.button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--ink);
      font-weight: 600;
      padding: 7px 9px;
      text-decoration: none;
      min-height: 44px;
    }
    a.button.primary {
      background: var(--blue);
      border-color: var(--blue);
      color: #fff;
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
    .case-status-strip {
      align-items: center;
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      gap: 4px 8px;
      font-size: 13px;
      line-height: 1.4;
      margin: 9px 0;
      min-height: 34px;
      padding: 5px 2px;
    }
    .case-status-strip strong { color: var(--ink); }
    .case-status-strip span:not(:last-child)::after {
      color: #9aa4af;
      content: "·";
      margin-left: 8px;
    }
    .timeline {
      position: relative;
      height: 22px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #eef2f6;
      margin: 8px 0 12px;
    }
    .pose-timeline-wrap[hidden] { display: none; }
    .pose-timeline-wrap { margin: 7px 0 2px; }
    .pose-timeline-legend {
      color: var(--muted);
      display: flex;
      flex-wrap: wrap;
      font-size: 11px;
      gap: 5px 12px;
      margin-bottom: 4px;
    }
    .pose-timeline-legend i {
      border-radius: 3px;
      display: inline-block;
      height: 8px;
      margin-right: 4px;
      width: 14px;
    }
    .pose-analysis-timeline {
      background: #eef2f6;
      border: 1px solid var(--line);
      border-radius: 7px;
      height: 16px;
      overflow: hidden;
      position: relative;
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
    .annotation-sidebar {
      align-self: start;
      max-height: calc(100vh - 92px);
      overflow-y: auto;
      padding: 0;
      position: sticky;
      scrollbar-gutter: stable;
      top: 14px;
    }
    .sidebar-title {
      align-items: center;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      padding: 13px 14px;
      position: sticky;
      top: 0;
      z-index: 4;
    }
    .sidebar-title h2 {
      font-size: 17px;
      margin: 0;
    }
    .progress-badge, .summary-badge {
      background: #eaf2fc;
      border-radius: 999px;
      color: #174f91;
      font-size: 12px;
      font-weight: 700;
      padding: 4px 8px;
      white-space: nowrap;
    }
    .summary-badge.attention {
      background: #f6dfe4;
      color: var(--unavailable);
    }
    .workflow-section {
      border-bottom: 1px solid var(--line);
      margin: 0;
      padding: 0;
    }
    .workflow-section:last-child { border-bottom: 0; }
    .workflow-step > summary {
      align-items: flex-start;
      cursor: pointer;
      display: flex;
      gap: 9px;
      list-style: none;
      min-height: 64px;
      padding: 13px 14px;
    }
    .workflow-step > summary::-webkit-details-marker { display: none; }
    .workflow-step > summary::after {
      color: var(--muted);
      content: "+";
      flex: 0 0 auto;
      font-size: 18px;
      line-height: 1;
      margin-left: auto;
      padding-top: 2px;
    }
    .workflow-step[open] > summary::after { content: "−"; }
    .workflow-step[open] > summary {
      background: linear-gradient(90deg, #f3f8fe, #fbfdff);
      box-shadow: inset 3px 0 0 var(--blue);
    }
    .workflow-step[data-state="complete"] .step-number {
      background: #e2f4e9;
      color: #11643f;
      font-size: 0;
    }
    .workflow-step[data-state="complete"] .step-number::after {
      content: "✓";
      font-size: 13px;
    }
    .workflow-step-body {
      padding: 2px 14px 16px 47px;
    }
    .workflow-step-copy { min-width: 0; }
    .section-heading {
      align-items: flex-start;
      display: flex;
      gap: 9px;
      margin-bottom: 9px;
    }
    .step-number {
      align-items: center;
      background: #eaf2fc;
      border-radius: 50%;
      color: #174f91;
      display: inline-flex;
      flex: 0 0 24px;
      font-size: 12px;
      font-weight: 800;
      height: 24px;
      justify-content: center;
    }
    .section-heading h3 {
      font-size: 15px;
      margin: 1px 0 2px;
    }
    .section-state {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      margin: 0;
    }
    .field-stack {
      display: grid;
      gap: 5px;
    }
    .field-stack select, .field-stack input { width: 100%; }
    .field-note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }
    .primary-action {
      width: 100%;
    }
    .action-context {
      color: var(--muted);
      display: block;
      font-size: 11px;
      font-weight: 750;
      letter-spacing: .025em;
      margin: 0 0 6px;
      text-transform: uppercase;
    }
    .compact-actions {
      display: grid;
      gap: 7px;
      grid-template-columns: 1fr 1fr;
      margin-top: 7px;
    }
    .compact-actions > * { min-width: 0; }
    .compact-actions .wide { grid-column: 1 / -1; }
    .destructive-row {
      border-top: 1px solid #edf0f3;
      display: flex;
      justify-content: flex-end;
      margin-top: 8px;
      padding-top: 8px;
    }
    button.danger-subtle {
      background: transparent;
      border-color: transparent;
      color: var(--red);
      font-size: 13px;
      min-height: 36px;
      padding: 5px 7px;
    }
    button.danger-subtle:hover { background: #fff1f2; border-color: #f3a9b2; }
    .case-details-grid {
      display: grid;
      gap: 8px;
      grid-template-columns: 1fr 1fr;
      margin-top: 9px;
    }
    .case-details-grid .full { grid-column: 1 / -1; }
    .workflow-disclosure {
      margin: 0;
    }
    .workflow-disclosure > summary {
      align-items: center;
      display: flex;
      font-size: 14px;
      gap: 8px;
      justify-content: space-between;
      list-style: none;
    }
    .workflow-disclosure > summary::-webkit-details-marker { display: none; }
    .workflow-disclosure > summary::before {
      color: var(--muted);
      content: "+";
      font-size: 18px;
      font-weight: 500;
      line-height: 1;
    }
    .workflow-disclosure[open] > summary::before { content: "-"; }
    .workflow-disclosure > summary .summary-label { flex: 1; }
    .priority-disclosure {
      background: #fffaf0;
      border: 1px solid #ead08a;
      border-radius: 7px;
      margin-top: 10px;
      padding: 9px 10px;
    }
    .priority-disclosure > summary { color: #784900; min-height: 28px; }
    .priority-disclosure .summary-label strong { display: block; font-size: 13px; }
    .priority-disclosure .summary-label small {
      color: #8d641b;
      display: block;
      font-size: 11px;
      font-weight: 600;
      line-height: 1.35;
      margin-top: 2px;
    }
    .disclosure-body {
      border-top: 1px solid #e7ebef;
      margin-top: 10px;
      padding-top: 10px;
    }
    .compact-copy {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      margin: 0 0 9px;
    }
    .window-summary {
      align-items: baseline;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 6px;
      display: flex;
      gap: 8px;
      justify-content: space-between;
      margin-top: 8px;
      padding: 8px 9px;
    }
    .window-summary strong { font-size: 13px; }
    .window-summary span {
      color: var(--muted);
      font-size: 12px;
    }
    .sidebar-action-dock {
      background: rgba(255, 255, 255, 0.97);
      border-top: 1px solid var(--line);
      bottom: 0;
      display: grid;
      gap: 8px;
      padding: 11px 14px 13px;
      position: sticky;
      z-index: 4;
    }
    .dock-feedback {
      align-items: center;
      display: flex;
      gap: 7px;
      margin: 0;
      min-height: 18px;
    }
    .dock-feedback::before {
      background: #7c8794;
      border-radius: 999px;
      content: "";
      flex: 0 0 8px;
      height: 8px;
      width: 8px;
    }
    .dock-feedback.unsaved { color: #784900; font-weight: 700; }
    .dock-feedback.unsaved::before { background: var(--amber); }
    .dock-feedback.saving { color: var(--blue); font-weight: 700; }
    .dock-feedback.saving::before { background: var(--blue); }
    .dock-feedback.saved { color: #0c6a43; font-weight: 700; }
    .dock-feedback.saved::before { background: var(--green); }
    .dock-actions { display: grid; gap: 7px; grid-template-columns: .8fr 1.2fr; }
    .dock-actions button { min-width: 0; }
    .analysis-actions {
      display: grid;
      gap: 7px;
      margin-top: 2px;
    }
    .analysis-actions .button, .analysis-actions button { width: 100%; }
    .pose-profile-card {
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 7px;
      display: grid;
      gap: 7px;
      padding: 9px;
    }
    .pose-profile-card p { margin: 0; }
    .analysis-progress {
      align-items: center;
      color: var(--muted);
      display: flex;
      font-size: 12px;
      gap: 8px;
      line-height: 1.4;
      margin-top: 8px;
    }
    .analysis-progress[hidden] { display: none; }
    .progress-dot {
      animation: pulse 1.4s ease-in-out infinite;
      background: var(--blue);
      border-radius: 50%;
      height: 8px;
      width: 8px;
    }
    @keyframes pulse { 50% { opacity: 0.35; } }
    .advanced-qa {
      background: #f8fafc;
      border: 1px solid #e2e7ec;
      border-radius: 6px;
      margin-top: 12px;
      padding: 9px 10px;
    }
    .advanced-qa > summary {
      align-items: center;
      display: flex;
      justify-content: space-between;
      list-style: none;
      min-height: 28px;
    }
    .advanced-qa > summary::-webkit-details-marker { display: none; }
    .advanced-qa > summary::after { color: var(--muted); content: "+"; font-size: 18px; }
    .advanced-qa[open] > summary::after { content: "−"; }
    .advanced-qa-body { border-top: 1px solid #e2e7ec; margin-top: 8px; padding-top: 2px; }
    .quiet-status {
      background: transparent;
      border: 0;
      color: var(--muted);
      min-height: 0;
      padding: 8px 0 0;
    }
    .advanced-stack details {
      border-top: 1px solid #e7ebef;
      margin: 0;
      padding: 9px 0;
    }
    .advanced-stack details:first-child { border-top: 0; }
    .advanced-stack summary {
      color: var(--ink);
      font-size: 13px;
    }
    .edit-hint {
      background: #fff8e6;
      border: 1px solid #f0c96b;
      border-radius: 6px;
      color: #784900;
      font-size: 13px;
      line-height: 1.4;
      margin: 10px 0;
      padding: 9px;
    }
    .review-toolbar {
      align-items: center;
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      margin-bottom: 9px;
    }
    .playback-toolbar {
      align-items: center;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 7px;
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      padding: 7px;
    }
    .playback-toolbar .frame-buttons {
      display: inline-flex;
      gap: 3px;
    }
    .playback-toolbar .frame-buttons button {
      border-radius: 4px;
      min-width: 48px;
    }
    .jump-control {
      align-items: center;
      display: inline-flex;
      gap: 5px;
    }
    .jump-control input { width: 82px; }
    .review-control { margin-left: auto; }
    .review-modes {
      background: #eef2f6;
      border: 1px solid var(--line);
      border-radius: 6px;
      display: inline-flex;
      gap: 2px;
      padding: 2px;
    }
    .review-modes button {
      background: transparent;
      border: 0;
      color: var(--muted);
      padding: 6px 9px;
    }
    .review-modes button.active {
      background: #fff;
      box-shadow: 0 1px 2px rgba(31, 41, 51, 0.12);
      color: var(--ink);
    }
    .pose-review-summary {
      align-items: center;
      color: var(--muted);
      display: grid;
      flex: 1 1 340px;
      font-size: 12px;
      gap: 4px 9px;
      grid-template-columns: max-content max-content max-content minmax(0, 1fr);
      grid-template-rows: 24px 18px 32px 18px;
      height: 104px;
      align-content: start;
      min-width: 0;
      overflow: hidden;
    }
    .pose-review-summary[hidden], .pose-review-stale[hidden] { display: none; }
    .qc-badge {
      border-radius: 999px;
      font-size: 11px;
      font-weight: 800;
      padding: 3px 7px;
      white-space: nowrap;
    }
    .qc-badge.supported { background: #e2f4e9; color: #11643f; }
    .qc-badge.limited, .qc-badge.uncertain { background: #fff0c7; color: #784900; }
    .qc-badge.rejected, .qc-badge.missing { background: #f6dfe4; color: var(--unavailable); }
    .qc-badge.neutral { background: #e8edf2; color: #485563; }
    #poseReviewFrame {
      grid-column: 1 / -1;
      grid-row: 2;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .pose-review-reason {
      color: #485563;
      display: -webkit-box;
      grid-column: 1 / -1;
      grid-row: 3;
      line-height: 16px;
      min-height: 32px;
      overflow: hidden;
      -webkit-box-orient: vertical;
      -webkit-line-clamp: 2;
    }
    .pose-review-use-note {
      color: var(--muted);
      font-size: 12px;
      grid-column: 1 / -1;
      grid-row: 4;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .pose-review-stale {
      background: #fff8e6;
      border-left: 3px solid var(--amber);
      color: #784900;
      flex: 1 0 100%;
      font-size: 12px;
      line-height: 1.35;
      padding: 6px 8px;
    }
    canvas.raw-review { cursor: default; }
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
    @media (max-width: 980px) {
      header { align-items: flex-start; flex-direction: column; }
      .app { grid-template-columns: 1fr; }
      .annotation-sidebar {
        max-height: none;
        position: static;
      }
    }
    @media (max-width: 620px) {
      .app { padding: 8px; }
      header .controls {
        display: grid;
        grid-template-columns: 1fr 1fr;
        width: 100%;
      }
      header .controls > label:first-child {
        display: grid;
        gap: 4px;
        grid-column: 1 / -1;
      }
      header .controls > label:last-child {
        align-items: center;
        display: grid;
        gap: 6px;
        grid-column: 1 / -1;
        grid-template-columns: auto minmax(0, 1fr);
      }
      #caseSelect, #annotatorId {
        min-width: 0;
        width: 100%;
      }
      #importVideo, #openCurrentVideo {
        min-width: 0;
        white-space: normal;
        width: 100%;
      }
      .compact-actions, .dock-actions { grid-template-columns: 1fr; }
      .compact-actions .wide { grid-column: auto; }
      .review-modes { display: grid; grid-template-columns: 1fr; width: 100%; }
      .review-modes button { width: 100%; }
      .workflow-step-body { padding-left: 14px; }
      .playback-toolbar, .playback-toolbar .frame-buttons, .jump-control { width: 100%; }
      .playback-toolbar .frame-buttons button { flex: 1 1 0; min-width: 0; }
      .jump-control input { flex: 1 1 auto; min-width: 0; }
      .review-control { margin-left: 0; width: 100%; }
    }
    @media (prefers-reduced-motion: reduce) {
      .progress-dot { animation: none; }
    }
    __APP_SHELL_CSS__
  </style>
</head>
<body>
  <a class="app-skip-link" href="#mainContent">Skip to annotation workspace</a>
  __APP_SITE_HEADER__
  <header class="app-tool-header">
    <h1>Human Annotation - ACL Movement Analytics Lab</h1>
    <div class="controls">
      <a class="button" href="/">Main menu</a>
      <a class="button" id="caseClipsLink" href="/">All case clips</a>
      <label>Case <select id="caseSelect"></select></label>
      <button id="importVideo" type="button">Add / cut video view</button>
      <button id="openCurrentVideo" type="button">Open selected video</button>
      <label>Annotator <input id="annotatorId" value="researcher_01" /></label>
    </div>
  </header>
  <main id="mainContent" class="app app-page-main" tabindex="-1">
    <section class="panel viewer">
      <div class="review-toolbar">
        <div class="review-modes" role="group" aria-label="Frame review display">
          <button type="button" data-review-mode="video">Video only</button>
          <button type="button" data-review-mode="roi">Annotation ROI</button>
          <button id="previousPoseMode" type="button" data-review-mode="pose">Skeleton review + QC</button>
        </div>
        <div id="poseReviewSummary" class="pose-review-summary" hidden>
          <span id="analysisUseBadge" class="qc-badge neutral">Previous analysis status</span>
          <span id="poseReviewBadge" class="qc-badge neutral">Frame QC</span>
          <span id="currentReviewBadge" class="qc-badge neutral">No manual decision</span>
          <span id="poseReviewFrame"></span>
          <span id="poseReviewReason" class="pose-review-reason"></span>
          <span id="poseReviewUseNote" class="pose-review-use-note"></span>
        </div>
        <div id="poseReviewStale" class="pose-review-stale" hidden>
          Previous analysis is now stale. Save and regenerate to update the skeleton and QC.
        </div>
      </div>
      <canvas id="frameCanvas"></canvas>
      <div id="annotationLegend" class="legend">
        <span><span class="swatch" style="color: var(--green)"></span>manual keyframe</span>
        <span><span class="swatch" style="color: var(--amber); border-style: dashed"></span>propagated ROI</span>
        <span><span class="swatch" style="color: var(--unavailable); background: #f6dfe4"></span>target unavailable</span>
      </div>
      <div class="case-status-strip" aria-label="Current annotation frame">
        <span>Case: <strong id="playerName">-</strong></span>
        <span>Frame <strong id="frameLabel">-</strong></span>
        <span><strong id="timeLabel">-</strong></span>
        <span><strong id="videoLabel">-</strong></span>
      </div>
      <input id="scrub" type="range" min="0" max="0" value="0" />
      <div id="poseTimelineWrap" class="pose-timeline-wrap" hidden>
        <div class="pose-timeline-legend" aria-label="Previous analysis skeleton status legend">
          <span><i style="background:#148a54"></i>used</span>
          <span><i style="background:#c47b00"></i>not used: insufficient evidence</span>
          <span><i style="background:#8f2f3f"></i>human excluded</span>
          <span><i style="background:#7c8794"></i>no usable skeleton</span>
        </div>
        <div id="poseAnalysisTimeline" class="pose-analysis-timeline" aria-label="Skeletons used in the previous analysis"></div>
      </div>
      <div id="timeline" class="timeline"></div>
      <div class="playback-toolbar" aria-label="Playback and frame navigation">
        <div class="frame-buttons">
          <button id="back5" aria-label="Back 5 frames">-5</button>
          <button id="prev">Previous</button>
          <button id="next">Next</button>
          <button id="fwd5" aria-label="Forward 5 frames">+5</button>
        </div>
        <div class="jump-control">
          <label for="jumpFrame">Jump to frame</label>
          <input id="jumpFrame" type="number" min="0" />
          <button id="jump">Go</button>
        </div>
        <button id="review" class="warn review-control">Review play</button>
      </div>
    </section>
    <aside class="panel annotation-sidebar">
      <div class="sidebar-title">
        <h2>Annotation workflow</h2>
        <span id="annotationProgress" class="progress-badge">0/3 essentials</span>
      </div>
      <div id="editWorkflowHint" class="edit-hint" hidden>
        <strong>Editing an existing analysis.</strong> Extend or correct the saved evidence,
        then validate and regenerate.
      </div>

      <details id="roiStep" class="workflow-section workflow-step" open>
        <summary>
          <span class="step-number">1</span>
          <div class="workflow-step-copy">
            <h3>Correct athlete tracking</h3>
            <p id="roiSummary" class="section-state">No ROI corrections yet</p>
          </div>
        </summary>
        <div class="workflow-step-body">
        <span class="action-context">When the athlete is identifiable</span>
        <button id="saveKeyframe" class="good primary-action">Add / replace ROI correction</button>
        <div class="compact-actions">
          <button id="copyPrevious">Copy previous ROI</button>
          <button id="undo">Undo</button>
        </div>
        <div class="destructive-row">
          <button id="deleteKeyframe" class="danger-subtle">Delete current keyframe</button>
        </div>
        <details id="acceptedDetails" class="workflow-disclosure priority-disclosure">
          <summary>
            <span class="summary-label">
              <strong>Accept reviewed skeleton frames</strong>
              <small>Confirm that the raw YOLOv8 skeleton belongs to this athlete</small>
            </span>
            <span id="acceptedBadge" class="summary-badge">None</span>
          </summary>
          <div class="disclosure-body">
            <p class="compact-copy">
              Open Skeleton review + QC, inspect the raw skeleton, then accept an individual
              frame or a continuous range. Acceptance confirms identity only; missing or
              incomplete joints can still remain unmeasured.
            </p>
            <label class="field-stack">Review note <textarea id="acceptedNote"></textarea></label>
            <div class="compact-actions">
              <button id="startAccepted" class="good wide">Start accepted range</button>
              <button id="endAccepted" class="good" disabled>Accept through here</button>
              <button id="removeAccepted">Remove acceptance at frame</button>
            </div>
            <pre id="acceptedSummary" class="status quiet-status"></pre>
          </div>
        </details>
        <details id="unavailableDetails" class="workflow-disclosure priority-disclosure">
          <summary>
            <span class="summary-label">
              <strong>Exclude unreliable frames</strong>
              <small>Use when the athlete cannot be identified defensibly</small>
            </span>
            <span id="unavailableBadge" class="summary-badge">None</span>
          </summary>
          <div class="disclosure-body">
            <p class="compact-copy">
              Mark only frames where the documented athlete cannot be identified defensibly.
              These frames remain unmeasured.
            </p>
            <label class="field-stack">Reason
              <select id="unavailableReason">
                <option value="CANNOT_JUDGE">cannot judge safely</option>
                <option value="TARGET_NOT_VISIBLE">target not visible</option>
                <option value="PLAYER_OVERLAP">player overlap</option>
                <option value="TARGET_PARTIALLY_OCCLUDED">severe partial occlusion</option>
                <option value="CAMERA_CUT">camera cut</option>
                <option value="OTHER">other</option>
              </select>
            </label>
            <label class="field-stack">Interval note <textarea id="unavailableNote"></textarea></label>
            <div class="compact-actions">
              <button id="startUnavailable" class="warn wide">Start excluded interval</button>
              <button id="endUnavailable" class="danger" disabled>End here</button>
              <button id="removeUnavailable">Remove at frame</button>
            </div>
            <pre id="unavailableSummary" class="status quiet-status"></pre>
          </div>
        </details>
        <details class="workflow-disclosure">
          <summary><span class="summary-label">Correction details</span></summary>
          <div class="disclosure-body">
            <div class="flag-grid" id="flagGrid"></div>
            <label class="field-stack">Keyframe note <textarea id="keyframeNote"></textarea></label>
          </div>
        </details>
        </div>
      </details>

      <details id="injuryStep" class="workflow-section workflow-step">
        <summary>
          <span class="step-number">2</span>
          <div class="workflow-step-copy">
            <h3>Case information</h3>
            <p id="injurySummary" class="section-state">Record the documented injured knee, or leave unknown</p>
          </div>
        </summary>
        <div class="workflow-step-body">
        <label class="field-stack">Documented injured knee
          <select id="injuredSide">
            <option value="unknown">Unknown / not documented</option>
            <option value="left">Left knee</option>
            <option value="right">Right knee</option>
          </select>
          <span class="field-note">Human-supplied case information; never inferred from pose.</span>
        </label>
        <details class="workflow-disclosure" id="caseDetailsDisclosure">
          <summary>
            <span class="summary-label">Case details (optional)</span>
            <span id="caseDetailsBadge" class="summary-badge">Not recorded</span>
          </summary>
          <div class="disclosure-body case-details-grid">
            <label class="field-stack full">Player name
              <input id="casePlayerName" type="text" autocomplete="off" />
            </label>
            <label class="field-stack">Injury date
              <input id="injuryDate" type="date" />
            </label>
            <label class="field-stack">Match minute
              <input id="matchMinute" type="text" inputmode="numeric" placeholder="67 or 45+2" />
            </label>
            <label class="field-stack full">Team
              <input id="caseTeam" type="text" autocomplete="off" placeholder="Unknown" />
            </label>
            <label class="field-stack full">Opponent
              <input id="caseOpponent" type="text" autocomplete="off" placeholder="Unknown" />
            </label>
            <label class="field-stack full">League / competition
              <input id="caseCompetition" type="text" autocomplete="off" placeholder="Unknown" />
            </label>
            <label class="field-stack">Position
              <select id="positionGroup">
                <option value="unknown">Unknown</option>
                <option value="goalkeeper">Goalkeeper</option>
                <option value="defender">Defender</option>
                <option value="midfielder">Midfielder</option>
                <option value="forward">Forward</option>
              </select>
            </label>
            <label class="field-stack">Date of birth
              <input id="dateOfBirth" type="date" />
            </label>
            <span class="field-note full">Recorded as operator-supplied metadata with provenance. Leave any field blank when unknown.</span>
          </div>
        </details>
        </div>
      </details>

      <details id="movementStep" class="workflow-section workflow-step">
        <summary>
          <span class="step-number">3</span>
          <div class="workflow-step-copy">
            <h3>Movement window</h3>
            <p id="movementStepSummary" class="section-state">Start comes from the first ROI keyframe</p>
          </div>
        </summary>
        <div class="workflow-step-body">
        <p class="compact-copy">Movement Start comes from the first ROI keyframe.</p>
        <button id="setMovementEnd" class="primary primary-action">Mark current frame as Movement End</button>
        <div id="windowSummary" class="window-summary"></div>
        <details class="workflow-disclosure">
          <summary><span class="summary-label">Confidence and rationale</span></summary>
          <div class="disclosure-body">
            <label class="field-stack">Confidence
              <select id="confidence">
                <option value="moderate">moderate</option>
                <option value="high">high</option>
                <option value="low">low</option>
              </select>
            </label>
            <label class="field-stack">Movement End rationale <textarea id="movementRationale"></textarea></label>
            <span class="field-note">Movement End marks the observable analysis boundary, not an inferred rupture frame.</span>
          </div>
        </details>
        </div>
      </details>

      <details id="reviewStep" class="workflow-section workflow-step">
        <summary>
          <span class="step-number">4</span>
          <div class="workflow-step-copy">
            <h3>Validate and generate</h3>
            <p id="reviewStepSummary" class="section-state">Validate the annotation before analysis</p>
          </div>
        </summary>
        <div class="workflow-step-body">
        <p class="compact-copy">Once the annotation essentials are complete, save for validation and generate or regenerate the analysis.</p>
        <div class="analysis-actions">
          <div class="pose-profile-card">
            <strong>Skeleton model · YOLOv8n</strong>
            <p class="field-note">The analysis uses the established YOLOv8n pose workflow for every case.</p>
          </div>
          <button id="generateAnalysis" class="primary" type="button">Generate analysis</button>
          <span class="field-note">Available after the annotation is validated and the injured knee is recorded.</span>
          <a id="viewAnalysis" class="button" href="/results?case=christen_press">View Analysis</a>
        </div>
        <div id="analysisProgress" class="analysis-progress" role="status" aria-live="polite" hidden>
          <span class="progress-dot" aria-hidden="true"></span>
          <span id="analysisProgressText">Reviewing the replay and building the movement analysis…</span>
        </div>
        <details id="advancedQa" class="advanced-qa">
          <summary>Advanced / QA</summary>
          <div class="advanced-qa-body advanced-stack">
            <details id="activityDetails">
              <summary>Activity and validation</summary>
              <div id="status" class="status quiet-status" role="status" aria-live="polite"></div>
            </details>
            <details>
              <summary>Session notes</summary>
              <label class="field-stack">Operator notes <textarea id="sessionNotes"></textarea></label>
            </details>
            <details>
              <summary>Development comparison</summary>
              <p class="compact-copy">Available after the human annotation has been saved.</p>
              <button id="compare">Compare annotations</button>
              <pre id="compareStatus" class="status quiet-status"></pre>
            </details>
          </div>
        </details>
        </div>
      </details>
      <div class="sidebar-action-dock" aria-label="Annotation save actions">
        <p id="saveFeedback" class="field-note dock-feedback saved" role="status" aria-live="polite">All changes saved.</p>
        <div class="dock-actions">
          <button id="save">Save draft</button>
          <button id="finalSave" class="good">Validate &amp; save</button>
        </div>
      </div>
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
  unavailableIntervals: [],
  pendingUnavailableStart: null,
  acceptedIntervals: [],
  pendingAcceptedStart: null,
  draftBox: null,
  drawing: null,
  movementWindow: {},
  sessionNotes: "",
  history: [],
  reviewTimer: null,
  reviewMode: "roi",
  hasPreviousPoseReview: false,
  previousPoseStale: false,
  poseReviewRequest: 0,
  poseReviewIntervals: [],
  frameImageRequest: 0,
  annotationDirty: false,
  saveInProgress: false,
  revision: 0,
  caseLoadVersion: 0,
  caseLoadAbortController: null,
  handlingPopState: false
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

function syncAnnotationUrl(slug, mode = "replace") {
  const url = new URL(window.location.href);
  url.searchParams.set("case", slug);
  url.searchParams.delete("frame");
  const method = mode === "push" ? "pushState" : "replaceState";
  window.history[method]({case: slug}, "", url);
}

function stopReviewPlayback() {
  if (app.reviewTimer) clearTimeout(app.reviewTimer);
  app.reviewTimer = null;
  const reviewButton = $("review");
  if (reviewButton) reviewButton.textContent = "Review play";
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
  $("caseSelect").addEventListener("change", handleCaseChange);
  bindControls();
  const params = new URLSearchParams(window.location.search);
  const requestedCase = params.get("case");
  const requestedFrameParam = params.get("frame");
  const requestedFrameValue = Number(requestedFrameParam);
  const requestedFrame = requestedFrameParam !== null && Number.isInteger(requestedFrameValue)
    ? requestedFrameValue
    : null;
  app.editMode = params.get("mode") === "edit";
  const initialCase = app.cases.find(c => c.slug === requestedCase) || app.cases[0];
  await loadCase(initialCase.slug, requestedFrame);
}

async function loadCase(slug, requestedFrame = null, {historyMode = "replace"} = {}) {
  const requestVersion = ++app.caseLoadVersion;
  if (app.caseLoadAbortController) app.caseLoadAbortController.abort();
  const controller = new AbortController();
  app.caseLoadAbortController = controller;
  stopReviewPlayback();
  app.history = [];
  app.pendingUnavailableStart = null;
  app.pendingAcceptedStart = null;
  app.frameImageRequest += 1;
  app.poseReviewRequest += 1;
  const data = await api(
    `/api/session?case=${encodeURIComponent(slug)}`,
    {signal: controller.signal}
  );
  if (requestVersion !== app.caseLoadVersion) return false;
  app.currentCase = data.case;
  app.revision = Number(data.revision || 0);
  $("caseClipsLink").href = `/?case=${encodeURIComponent(app.currentCase.case_id)}`;
  $("caseSelect").value = app.currentCase.slug;
  $("viewAnalysis").href = `/results?case=${encodeURIComponent(app.currentCase.slug)}`;
  app.meta = data.case.metadata || {fps: 0, frame_count: 1, width: 0, height: 0};
  app.keyframes = (data.session.roi_keyframes || []).map(normalizeKeyframe);
  app.unavailableIntervals = (data.session.target_unavailable_intervals || [])
    .map(normalizeUnavailableInterval);
  app.pendingUnavailableStart = null;
  app.acceptedIntervals = (data.session.target_accepted_intervals || [])
    .map(normalizeAcceptedInterval);
  app.pendingAcceptedStart = null;
  app.movementWindow = data.session.movement_window || {};
  if (!app.movementWindow.movement_end_frame && data.session.event_annotation?.event_anchor_frame !== undefined) {
    app.movementWindow.movement_end_frame = data.session.event_annotation.event_anchor_frame;
    app.movementWindow.confidence = data.session.event_confidence_label || "moderate";
    app.movementWindow.rationale = data.session.event_annotation.notes || "";
  }
  app.sessionNotes = data.session.notes || "";
  $("sessionNotes").value = app.sessionNotes;
  $("annotatorId").value = data.session.provenance?.annotator_id || $("annotatorId").value || "researcher_01";
  const savedInjuredSide = data.session.injured_side || "unknown";
  $("injuredSide").value = savedInjuredSide !== "unknown"
    ? savedInjuredSide
    : (data.case.injured_side || "unknown");
  const caseDetails = data.case_details || {};
  $("casePlayerName").value = caseDetails.player_name || data.case.player_name || "";
  $("injuryDate").value = caseDetails.injury_date || "";
  $("matchMinute").value = caseDetails.match_minute || "";
  $("caseTeam").value = caseDetails.team || "";
  $("caseOpponent").value = caseDetails.opponent || "";
  $("caseCompetition").value = caseDetails.competition || "";
  $("positionGroup").value = caseDetails.position_group || "unknown";
  $("dateOfBirth").value = caseDetails.date_of_birth || "";
  updateCaseDetailsBadge();
  updateAnalysisActions(data.human_results_available, Boolean(data.session.finalized));
  app.hasPreviousPoseReview = Boolean(data.pose_review?.available);
  app.previousPoseStale = Boolean(data.pose_review?.stale);
  app.poseReviewIntervals = [];
  if (app.hasPreviousPoseReview) {
    try {
      const timelineData = await api(
        `/api/pose-review/timeline?case=${encodeURIComponent(app.currentCase.slug)}`,
        {signal: controller.signal}
      );
      if (requestVersion !== app.caseLoadVersion) return false;
      app.poseReviewIntervals = timelineData.intervals || [];
    } catch (error) {
      if (error.name === "AbortError") return false;
      app.poseReviewIntervals = [];
    }
  }
  if (requestVersion !== app.caseLoadVersion) return false;
  setReviewMode(
    app.editMode && app.hasPreviousPoseReview ? "pose" : "roi",
    false
  );
  $("playerName").textContent = data.case.player_name;
  $("videoLabel").textContent = `${app.meta.width || "-"}×${app.meta.height || "-"} · ${app.meta.fps || "-"} fps`;
  $("scrub").max = Math.max((app.meta.frame_count || 1) - 1, 0);
  app.frame = 0;
  if (requestedFrame !== null) {
    app.frame = requestedFrame;
  } else if (app.movementWindow.movement_start_frame !== null && app.movementWindow.movement_start_frame !== undefined) {
    app.frame = app.movementWindow.movement_start_frame;
  } else if (app.keyframes.length) {
    app.frame = app.keyframes[0].frame_index;
  } else if (app.movementWindow.movement_end_frame !== null && app.movementWindow.movement_end_frame !== undefined) {
    app.frame = app.movementWindow.movement_end_frame;
  }
  updateMovementWindowControls();
  updateUnavailableControls();
  updateAcceptedControls();
  $("editWorkflowHint").hidden = !app.editMode;
  await loadFrame(app.frame);
  if (requestVersion !== app.caseLoadVersion) return false;
  renderTimeline();
  syncWorkflowSteps(true);
  app.annotationDirty = false;
  setSaveFeedback("All changes saved.", "saved");
  const status = app.editMode
    ? "Editing the saved annotation at its current Movement End. Move forward to extend it, add any needed ROI corrections, then save as ready for validation and regenerate analysis."
    : data.resume_available
      ? "Resumed saved human annotation."
      : "New independent human annotation session.";
  setStatus(status);
  syncAnnotationUrl(app.currentCase.slug, historyMode);
  return true;
}

function bindControls() {
  bindWorkflowAccordion();
  document.querySelectorAll("[data-review-mode]").forEach(button => {
    button.onclick = () => setReviewMode(button.dataset.reviewMode);
  });
  $("scrub").addEventListener("input", e => loadFrame(Number(e.target.value)));
  $("prev").onclick = () => loadFrame(app.frame - 1);
  $("next").onclick = () => loadFrame(app.frame + 1);
  $("back5").onclick = () => loadFrame(app.frame - 5);
  $("fwd5").onclick = () => loadFrame(app.frame + 5);
  $("jump").onclick = () => loadFrame(Number($("jumpFrame").value || 0));
  $("saveKeyframe").onclick = saveKeyframe;
  $("copyPrevious").onclick = copyPrevious;
  $("deleteKeyframe").onclick = deleteKeyframe;
  $("startUnavailable").onclick = startUnavailableInterval;
  $("endUnavailable").onclick = endUnavailableInterval;
  $("removeUnavailable").onclick = removeUnavailableInterval;
  $("startAccepted").onclick = startAcceptedInterval;
  $("endAccepted").onclick = endAcceptedInterval;
  $("removeAccepted").onclick = removeAcceptedInterval;
  $("undo").onclick = undo;
  $("setMovementEnd").onclick = setMovementEnd;
  $("save").onclick = () => saveSession(false);
  $("finalSave").onclick = () => saveSession(true);
  $("generateAnalysis").onclick = generateAnalysis;
  $("compare").onclick = compareAnnotations;
  $("review").onclick = toggleReview;
  $("importVideo").onclick = openVideoCutterForCase;
  $("openCurrentVideo").onclick = openCurrentVideo;
  $("injuredSide").addEventListener("change", () => {
    markAnnotationDirty();
    updateSidebarSummary();
  });
  ["casePlayerName", "injuryDate", "matchMinute", "caseTeam", "caseOpponent", "caseCompetition", "positionGroup", "dateOfBirth"]
    .forEach(id => {
      const eventName = $(id).tagName === "SELECT" ? "change" : "input";
      $(id).addEventListener(eventName, () => {
        updateCaseDetailsBadge();
        markAnnotationDirty();
      });
    });
  ["annotatorId", "movementRationale", "sessionNotes"].forEach(id => {
    $(id).addEventListener("input", markAnnotationDirty);
  });
  $("confidence").addEventListener("change", markAnnotationDirty);
  window.addEventListener("keydown", event => {
    if (["TEXTAREA", "INPUT", "SELECT", "BUTTON"].includes(event.target.tagName)) return;
    if (event.key === "ArrowLeft") loadFrame(app.frame - 1);
    if (event.key === "ArrowRight") loadFrame(app.frame + 1);
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      saveSession(false);
    }
  });
  window.addEventListener("beforeunload", event => {
    if (!app.annotationDirty || app.saveInProgress) return;
    event.preventDefault();
    event.returnValue = "";
  });
  window.addEventListener("popstate", async () => {
    const requested = new URLSearchParams(window.location.search).get("case");
    const target = app.cases.find(item => item.slug === requested);
    if (!target) {
      if (app.currentCase) syncAnnotationUrl(app.currentCase.slug, "replace");
      return;
    }
    if (target.slug === app.currentCase?.slug) return;
    try {
      await loadCase(target.slug, null, {historyMode: "replace"});
    } catch (error) {
      if (error.name !== "AbortError") setStatus(`Could not restore that clip. ${error.message}`);
    }
  });
  canvas.addEventListener("mousedown", startDraw);
  canvas.addEventListener("mousemove", moveDraw);
  canvas.addEventListener("mouseup", finishDraw);
}

async function handleCaseChange() {
  const selectedSlug = $("caseSelect").value;
  const currentSlug = app.currentCase?.slug || "";
  if (app.annotationDirty && !window.confirm(
    "This annotation has unsaved changes. Leave them behind and open another clip?"
  )) {
    $("caseSelect").value = currentSlug;
    return;
  }
  try {
    await loadCase(selectedSlug, null, {historyMode: "push"});
  } catch (error) {
    if (error.name === "AbortError") return;
    $("caseSelect").value = currentSlug;
    setStatus(`Could not open the selected clip. ${error.message}`);
  }
}

function bindWorkflowAccordion() {
  const steps = [...document.querySelectorAll(".annotation-sidebar > .workflow-step")];
  steps.forEach(step => {
    step.addEventListener("toggle", () => {
      if (!step.open) return;
      steps.forEach(other => {
        if (other !== step) other.open = false;
      });
    });
  });
}

function openWorkflowStep(id) {
  ["roiStep", "injuryStep", "movementStep", "reviewStep"].forEach(stepId => {
    $(stepId).open = stepId === id;
  });
}

function setReviewMode(mode, reload = true) {
  const requestedMode = ["video", "roi", "pose"].includes(mode) ? mode : "roi";
  if (requestedMode === "pose" && !app.hasPreviousPoseReview) {
    setStatus("No previous processed pose is available for this case yet.");
    return;
  }
  app.reviewMode = requestedMode;
  document.querySelectorAll("[data-review-mode]").forEach(button => {
    const active = button.dataset.reviewMode === app.reviewMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });
  $("previousPoseMode").disabled = !app.hasPreviousPoseReview;
  $("poseReviewSummary").hidden = app.reviewMode !== "pose";
  $("poseTimelineWrap").hidden = app.reviewMode !== "pose";
  $("annotationLegend").hidden = app.reviewMode === "video";
  canvas.classList.toggle("raw-review", app.reviewMode === "video");
  updatePoseReviewStaleNotice();
  renderPoseAnalysisTimeline();
  if (reload && app.currentCase) loadFrame(app.frame);
}

function updatePoseReviewStaleNotice() {
  $("poseReviewStale").hidden = !(
    app.reviewMode === "pose" && app.previousPoseStale
  );
}

async function loadPoseReview(frame) {
  const requestId = ++app.poseReviewRequest;
  $("analysisUseBadge").className = "qc-badge neutral";
  $("analysisUseBadge").textContent = "Loading previous analysis status";
  $("poseReviewBadge").className = "qc-badge neutral";
  $("poseReviewBadge").textContent = "Loading frame QC";
  $("currentReviewBadge").className = "qc-badge neutral";
  $("currentReviewBadge").textContent = "Checking current decision";
  $("poseReviewFrame").textContent = `Frame ${frame}`;
  $("poseReviewReason").textContent = "";
  $("poseReviewUseNote").textContent = "";
  try {
    const data = await api(
      `/api/pose-review?case=${encodeURIComponent(app.currentCase.slug)}&frame=${frame}`
    );
    if (requestId !== app.poseReviewRequest || frame !== app.frame || app.reviewMode !== "pose") {
      return;
    }
    app.previousPoseStale = app.previousPoseStale || Boolean(data.stale);
    $("analysisUseBadge").className = `qc-badge ${data.analysis_use_tone || "neutral"}`;
    $("analysisUseBadge").textContent = data.analysis_use_label || "Previous analysis status unavailable";
    $("poseReviewBadge").className = `qc-badge ${data.status_tone || "neutral"}`;
    $("poseReviewBadge").textContent = `QC: ${data.status_label || "unavailable"}`;
    const previousDecision = data.manual_review_decision || "NOT_REVIEWED";
    const decisionPresentation = currentReviewDecisionPresentation(frame, previousDecision);
    $("currentReviewBadge").className = `qc-badge ${decisionPresentation.tone}`;
    $("currentReviewBadge").textContent = decisionPresentation.label;
    const details = [
      `Frame ${data.source_frame_index}`,
      `${data.observed_landmark_count} observed`,
      `${data.usable_landmark_count} usable`
    ];
    if (data.interpolated_landmark_count) {
      details.push(`${data.interpolated_landmark_count} interpolated`);
    }
    if (data.median_confidence !== null && data.median_confidence !== undefined) {
      details.push(`${Math.round(data.median_confidence * 100)}% median pose confidence`);
    }
    if (data.raw_pose_available) details.push("raw skeleton shown");
    if (data.automatic_frame_status && data.automatic_frame_status !== data.frame_status) {
      details.push(`automatic check: ${data.automatic_frame_status.replaceAll("_", " ").toLowerCase()}`);
    }
    $("poseReviewFrame").textContent = details.join(" · ");
    $("poseReviewReason").textContent = data.analysis_use_reason || data.frame_rejection_reason || "";
    $("poseReviewUseNote").textContent = [
      data.skeleton_display_note || "",
      decisionPresentation.pending
        ? "Pending change—validate, save, and regenerate before it affects the analysis."
        : ""
    ].filter(Boolean).join(" ");
    updatePoseReviewStaleNotice();
  } catch (error) {
    if (requestId !== app.poseReviewRequest) return;
    $("analysisUseBadge").className = "qc-badge neutral";
    $("analysisUseBadge").textContent = "Previous analysis status unavailable";
    $("poseReviewBadge").className = "qc-badge neutral";
    $("poseReviewBadge").textContent = "Previous QC unavailable";
    $("currentReviewBadge").className = "qc-badge neutral";
    $("currentReviewBadge").textContent = "Decision unavailable";
    $("poseReviewReason").textContent = error.message;
    $("poseReviewUseNote").textContent = "";
  }
}

function currentManualReviewDecision(frame) {
  if (unavailableIntervalAt(frame)) return "EXCLUDED";
  if (acceptedIntervalAt(frame)) return "ACCEPTED";
  return "NOT_REVIEWED";
}

function currentReviewDecisionPresentation(frame, previousDecision) {
  const currentDecision = currentManualReviewDecision(frame);
  const pending = currentDecision !== previousDecision;
  if (pending) {
    const labels = {
      ACCEPTED: "Pending: human accepted",
      EXCLUDED: "Pending: human excluded",
      NOT_REVIEWED: "Pending: return to automatic QC"
    };
    return {label: labels[currentDecision], tone: "uncertain", pending: true};
  }
  if (currentDecision === "ACCEPTED") {
    return {label: "Human accepted · applied", tone: "supported", pending: false};
  }
  if (currentDecision === "EXCLUDED") {
    return {label: "Human excluded · applied", tone: "rejected", pending: false};
  }
  return {label: "No manual decision", tone: "neutral", pending: false};
}

function refreshCurrentPoseReviewStatus() {
  if (app.reviewMode === "pose" && app.hasPreviousPoseReview) {
    loadPoseReview(app.frame);
  }
}

function openVideoCutterForCase() {
  const caseId = app.currentCase?.case_id;
  const returnPath = app.currentCase?.slug
    ? `/annotate?case=${encodeURIComponent(app.currentCase.slug)}`
    : "/";
  window.location.href = caseId
    ? `/video-cutter?case=${encodeURIComponent(caseId)}&return=${encodeURIComponent(returnPath)}`
    : "/video-cutter";
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
  const requestedFrame = app.frame;
  const imageRequest = ++app.frameImageRequest;
  $("scrub").value = app.frame;
  $("jumpFrame").value = app.frame;
  $("frameLabel").textContent = `${app.frame} / ${maxFrame}`;
  const timeMs = app.meta.fps ? (app.frame / app.meta.fps * 1000) : 0;
  $("timeLabel").textContent = `${(timeMs / 1000).toFixed(2)} s`;
  const previousPose = app.reviewMode === "pose" && app.hasPreviousPoseReview;
  const frameImage = new Image();
  const imageReady = new Promise(resolve => {
    frameImage.onload = () => {
      if (imageRequest === app.frameImageRequest && requestedFrame === app.frame) {
        app.image = frameImage;
        canvas.width = frameImage.naturalWidth;
        canvas.height = frameImage.naturalHeight;
        draw();
      }
      resolve();
    };
    frameImage.onerror = () => resolve();
  });
  frameImage.src = previousPose
    ? `/api/pose-review/frame?case=${encodeURIComponent(app.currentCase.slug)}&frame=${requestedFrame}&t=${Date.now()}`
    : `/api/frame?case=${encodeURIComponent(app.currentCase.slug)}&frame=${requestedFrame}&t=${Date.now()}`;
  const poseReviewReady = previousPose
    ? loadPoseReview(requestedFrame)
    : Promise.resolve();
  if (!previousPose) {
    app.poseReviewRequest += 1;
  }
  app.draftBox = null;
  updateKeyframeNote();
  updateUnavailableControls();
  updateAcceptedControls();
  renderPoseAnalysisTimeline();
  await Promise.all([imageReady, poseReviewReady]);
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(app.image, 0, 0);
  if (app.reviewMode === "video") return;
  const manual = keyframeAt(app.frame);
  const propagated = propagatedBox(app.frame);
  const unavailable = unavailableIntervalAt(app.frame);
  if (propagated && !manual) drawBox(propagated, "#c47b00", "Propagated ROI", true);
  if (manual) drawBox(manual.bbox, "#148a54", "Manual keyframe", false);
  if (app.draftBox) drawBox(app.draftBox, "#1d68c4", "Draft ROI", false);
  if (unavailable) drawUnavailableLabel(unavailable);
  else {
    const accepted = acceptedIntervalAt(app.frame);
    if (accepted) drawAcceptedLabel(accepted);
  }
  if (app.movementWindow.movement_end_frame === app.frame) {
    ctx.fillStyle = "#b42335";
    ctx.fillText("Movement End", 12, 24);
  }
}

function drawUnavailableLabel(interval) {
  const label = `Current review: excluded | frames ${interval.start_frame}-${interval.end_frame}`;
  ctx.save();
  ctx.font = "bold 20px sans-serif";
  const width = Math.min(ctx.measureText(label).width + 24, canvas.width - 24);
  ctx.fillStyle = "rgba(143,47,63,0.92)";
  ctx.fillRect(12, 12, width, 38);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, 22, 38, width - 20);
  ctx.restore();
}

function drawAcceptedLabel(interval) {
  const label = `Current review: human accepted | frames ${interval.start_frame}-${interval.end_frame}`;
  ctx.save();
  ctx.font = "bold 20px sans-serif";
  const width = Math.min(ctx.measureText(label).width + 24, canvas.width - 24);
  ctx.fillStyle = "rgba(20,138,84,0.92)";
  ctx.fillRect(12, 12, width, 38);
  ctx.fillStyle = "#ffffff";
  ctx.fillText(label, 22, 38, width - 20);
  ctx.restore();
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
  if (app.reviewMode === "video") setReviewMode("roi", false);
  if (unavailableIntervalAt(app.frame)) {
    setStatus("This frame is marked target unavailable. Remove the interval before drawing an ROI.");
    return;
  }
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
  const unavailable = unavailableIntervalAt(app.frame);
  if (unavailable) {
    setStatus(
      `Source frame ${app.frame} is inside target-unavailable interval ` +
      `${unavailable.start_frame}-${unavailable.end_frame}. Remove that interval before adding an ROI.`
    );
    return;
  }
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
  markAnnotationDirty();
  renderTimeline();
  draw();
  refreshCurrentPoseReviewStatus();
  syncWorkflowSteps(false);
  setStatus(`Saved ROI keyframe at source frame ${app.frame}.`);
}

function copyPrevious() {
  if (unavailableIntervalAt(app.frame)) {
    setStatus("This frame is marked target unavailable. Remove the interval before copying an ROI.");
    return;
  }
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
  markAnnotationDirty();
  renderTimeline();
  draw();
  refreshCurrentPoseReviewStatus();
  setStatus(`Deleted ROI keyframe at source frame ${app.frame}.`);
}

function startAcceptedInterval() {
  if (!app.hasPreviousPoseReview) {
    setStatus("Generate the YOLOv8 analysis before accepting reviewed skeleton frames.");
    return;
  }
  if (app.pendingAcceptedStart !== null) {
    app.pendingAcceptedStart = null;
    updateAcceptedControls();
    setStatus("Cancelled the pending accepted range.");
    return;
  }
  app.pendingAcceptedStart = app.frame;
  setReviewMode("pose", false);
  updateAcceptedControls();
  setStatus(
    `Accepted range starts at source frame ${app.frame}. ` +
    "Review through the final correct frame, then choose Accept through here."
  );
}

function endAcceptedInterval() {
  if (app.pendingAcceptedStart === null) {
    setStatus("Choose Start accepted range first.");
    return;
  }
  pushHistory();
  let start = Math.min(app.pendingAcceptedStart, app.frame);
  let end = Math.max(app.pendingAcceptedStart, app.frame);
  const touching = app.acceptedIntervals.filter(
    interval => interval.end_frame >= start - 1 && interval.start_frame <= end + 1
  );
  if (touching.length) {
    start = Math.min(start, ...touching.map(interval => interval.start_frame));
    end = Math.max(end, ...touching.map(interval => interval.end_frame));
  }
  app.acceptedIntervals = app.acceptedIntervals.filter(interval => !touching.includes(interval));
  app.unavailableIntervals = subtractFrameRange(app.unavailableIntervals, start, end);
  app.acceptedIntervals.push({
    start_frame: start,
    end_frame: end,
    note: $("acceptedNote").value || ""
  });
  app.acceptedIntervals.sort((a, b) => a.start_frame - b.start_frame);
  app.pendingAcceptedStart = null;
  markAnnotationDirty();
  updateAcceptedControls();
  updateUnavailableControls();
  renderTimeline();
  draw();
  refreshCurrentPoseReviewStatus();
  setStatus(
    `Accepted source frames ${start}-${end} after raw-skeleton review. ` +
    "Save and regenerate analysis to apply the decision."
  );
}

function removeAcceptedInterval() {
  const interval = acceptedIntervalAt(app.frame);
  if (!interval) {
    setStatus("There is no human-accepted range at this frame.");
    return;
  }
  pushHistory();
  app.acceptedIntervals = app.acceptedIntervals.filter(item => item !== interval);
  markAnnotationDirty();
  updateAcceptedControls();
  renderTimeline();
  draw();
  refreshCurrentPoseReviewStatus();
  setStatus(
    `Removed human acceptance for frames ${interval.start_frame}-${interval.end_frame}. ` +
    "The automatic quality decision will apply after regeneration."
  );
}

function updateAcceptedControls() {
  const pending = app.pendingAcceptedStart;
  $("startAccepted").disabled = !app.hasPreviousPoseReview;
  $("startAccepted").textContent = pending === null
    ? "Start accepted range"
    : "Cancel accepted range";
  $("endAccepted").disabled = pending === null;
  const current = acceptedIntervalAt(app.frame);
  $("removeAccepted").disabled = !current;
  const lines = app.acceptedIntervals.length
    ? app.acceptedIntervals.map(interval => `Frames ${interval.start_frame}-${interval.end_frame}: human accepted`)
    : ["No skeleton frames have been manually accepted."];
  if (pending !== null) lines.push(`Pending accepted range start: frame ${pending}`);
  $("acceptedSummary").textContent = lines.join("\n");
  const badge = $("acceptedBadge");
  badge.textContent = pending !== null
    ? `Starting at ${pending}`
    : app.acceptedIntervals.length
      ? `${app.acceptedIntervals.length} accepted`
      : "None";
  badge.classList.toggle("attention", pending !== null);
  if (pending !== null) $("acceptedDetails").open = true;
}

function startUnavailableInterval() {
  if (app.pendingUnavailableStart !== null) {
    app.pendingUnavailableStart = null;
    updateUnavailableControls();
    setStatus("Cancelled the pending target-unavailable interval.");
    return;
  }
  app.pendingUnavailableStart = app.frame;
  updateUnavailableControls();
  setStatus(
    `Unavailable interval starts at source frame ${app.frame}. ` +
    "Move to its final unavailable frame, then choose End interval here."
  );
}

function endUnavailableInterval() {
  if (app.pendingUnavailableStart === null) {
    setStatus("Choose Start excluded interval first.");
    return;
  }
  pushHistory();
  let start = Math.min(app.pendingUnavailableStart, app.frame);
  let end = Math.max(app.pendingUnavailableStart, app.frame);
  const touching = app.unavailableIntervals.filter(
    interval => interval.end_frame >= start - 1 && interval.start_frame <= end + 1
  );
  if (touching.length) {
    start = Math.min(start, ...touching.map(interval => interval.start_frame));
    end = Math.max(end, ...touching.map(interval => interval.end_frame));
  }
  app.unavailableIntervals = app.unavailableIntervals.filter(interval => !touching.includes(interval));
  app.acceptedIntervals = subtractFrameRange(app.acceptedIntervals, start, end);
  app.unavailableIntervals.push({
    start_frame: start,
    end_frame: end,
    reason: $("unavailableReason").value,
    note: $("unavailableNote").value || ""
  });
  app.unavailableIntervals.sort((a, b) => a.start_frame - b.start_frame);
  app.keyframes = app.keyframes.filter(
    keyframe => keyframe.frame_index < start || keyframe.frame_index > end
  );
  app.pendingUnavailableStart = null;
  app.draftBox = null;
  markAnnotationDirty();
  updateUnavailableControls();
  updateAcceptedControls();
  renderTimeline();
  renderWindowSummary();
  draw();
  refreshCurrentPoseReviewStatus();
  setStatus(
    `Marked source frames ${start}-${end} as target unavailable. ` +
    "ROI propagation and pose extraction will be suppressed throughout this interval."
  );
}

function subtractFrameRange(intervals, start, end) {
  return intervals.flatMap(interval => {
    if (interval.end_frame < start || interval.start_frame > end) return [interval];
    const remainder = [];
    if (interval.start_frame < start) {
      remainder.push({...interval, end_frame: start - 1});
    }
    if (interval.end_frame > end) {
      remainder.push({...interval, start_frame: end + 1});
    }
    return remainder;
  }).sort((a, b) => a.start_frame - b.start_frame);
}

function removeUnavailableInterval() {
  const interval = unavailableIntervalAt(app.frame);
  if (!interval) {
    setStatus("There is no target-unavailable interval at this frame.");
    return;
  }
  pushHistory();
  app.unavailableIntervals = app.unavailableIntervals.filter(item => item !== interval);
  markAnnotationDirty();
  updateUnavailableControls();
  renderTimeline();
  draw();
  refreshCurrentPoseReviewStatus();
  setStatus(
    `Removed target-unavailable interval ${interval.start_frame}-${interval.end_frame}. ` +
    "Add ROI corrections around this section before regenerating analysis."
  );
}

function updateUnavailableControls() {
  const pending = app.pendingUnavailableStart;
  $("startUnavailable").textContent = pending === null
    ? "Start excluded interval"
    : "Cancel interval start";
  $("endUnavailable").disabled = pending === null;
  const current = unavailableIntervalAt(app.frame);
  $("removeUnavailable").disabled = !current;
  const lines = app.unavailableIntervals.length
    ? app.unavailableIntervals.map(interval => (
        `Frames ${interval.start_frame}-${interval.end_frame}: ` +
        interval.reason.replaceAll("_", " ").toLowerCase()
      ))
    : ["No target-unavailable intervals marked."];
  if (pending !== null) lines.push(`Pending interval start: frame ${pending}`);
  $("unavailableSummary").textContent = lines.join("\n");
  const badge = $("unavailableBadge");
  badge.textContent = pending !== null
    ? `Starting at ${pending}`
    : app.unavailableIntervals.length
      ? `${app.unavailableIntervals.length} marked`
      : "None";
  badge.classList.toggle("attention", pending !== null || app.unavailableIntervals.length > 0);
  if (pending !== null) $("unavailableDetails").open = true;
  updateSidebarSummary();
}

function updateSidebarSummary() {
  if (!app.currentCase) return;
  const keyframeCount = app.keyframes.length;
  $("roiSummary").textContent = keyframeCount
    ? `${keyframeCount} ROI correction${keyframeCount === 1 ? "" : "s"} recorded`
    : "Draw the first ROI around the documented athlete";

  const injuredSide = $("injuredSide").value;
  $("injurySummary").textContent = injuredSide === "left"
    ? "Left knee recorded by operator"
    : injuredSide === "right"
      ? "Right knee recorded by operator"
      : "Documented injured knee unknown — it will not be inferred from the video";

  const movementEnd = app.movementWindow.movement_end_frame;
  const essentials = [
    keyframeCount > 0,
    injuredSide !== "unknown",
    movementEnd !== null && movementEnd !== undefined
  ].filter(Boolean).length;
  $("annotationProgress").textContent = `${essentials}/3 analysis essentials`;
  refreshWorkflowState();
}

function refreshWorkflowState() {
  if (!app.currentCase) return;
  const keyframeCount = app.keyframes.length;
  const start = movementStartFrame();
  const end = app.movementWindow.movement_end_frame;
  const hasEnd = end !== null && end !== undefined;
  const injuredSide = $("injuredSide").value;
  const hasInjuredKnee = injuredSide !== "unknown";

  $("roiStep").dataset.state = keyframeCount ? "complete" : "current";
  $("injuryStep").dataset.state = injuredSide === "unknown" ? "available" : "complete";
  $("movementStep").dataset.state = hasEnd ? "complete" : keyframeCount ? "current" : "pending";
  $("reviewStep").dataset.state = keyframeCount && hasInjuredKnee && hasEnd
    ? "current"
    : "pending";

  ["roiStep", "injuryStep", "movementStep", "reviewStep"].forEach(id => {
    const summary = $(id).querySelector(":scope > summary");
    if ($(id).dataset.state === "current") summary.setAttribute("aria-current", "step");
    else summary.removeAttribute("aria-current");
  });

  if (hasEnd && start !== null) {
    const duration = app.meta.fps ? Math.max(0, end - start) / app.meta.fps : null;
    $("movementStepSummary").textContent = `Frames ${start}-${end}`
      + (duration !== null ? ` · ${duration.toFixed(2)} s` : "");
  } else {
    $("movementStepSummary").textContent = keyframeCount
      ? "Movement Start recorded · mark Movement End"
      : "Starts with the first ROI correction";
  }
  const essentialsComplete = keyframeCount && hasInjuredKnee && hasEnd;
  $("reviewStepSummary").textContent = !essentialsComplete
    ? "Complete the target ROI, injured knee, and Movement Window first"
    : app.annotationFinalized
      ? app.resultsAvailable
        ? "Analysis available · regenerate after corrections"
        : "Ready to generate analysis"
      : "Essentials complete · validate and save";
}

function syncWorkflowSteps(openCurrent = false) {
  refreshWorkflowState();
  if (!openCurrent) return;
  const steps = ["roiStep", "injuryStep", "movementStep", "reviewStep"];
  const preferredId = app.editMode
    ? "roiStep"
    : steps.find(id => $(id).dataset.state === "current")
      || steps.find(id => $(id).dataset.state !== "complete")
      || "reviewStep";
  openWorkflowStep(preferredId);
}

function updateCaseDetailsBadge() {
  const values = [
    $("injuryDate").value,
    $("matchMinute").value,
    $("caseTeam").value,
    $("caseOpponent").value,
    $("caseCompetition").value,
    $("positionGroup").value !== "unknown" ? $("positionGroup").value : "",
    $("dateOfBirth").value,
  ];
  const count = values.filter(Boolean).length;
  $("caseDetailsBadge").textContent = count ? `${count} recorded` : "Not recorded";
}

function undo() {
  const previous = app.history.pop();
  if (!previous) {
    setStatus("Nothing to undo.");
    return;
  }
  app.keyframes = previous.keyframes;
  app.unavailableIntervals = previous.unavailableIntervals;
  app.pendingUnavailableStart = previous.pendingUnavailableStart;
  app.acceptedIntervals = previous.acceptedIntervals;
  app.pendingAcceptedStart = previous.pendingAcceptedStart;
  markAnnotationDirty();
  updateUnavailableControls();
  updateAcceptedControls();
  renderTimeline();
  draw();
  refreshCurrentPoseReviewStatus();
}

function pushHistory() {
  app.history.push(JSON.parse(JSON.stringify({
    keyframes: app.keyframes,
    unavailableIntervals: app.unavailableIntervals,
    pendingUnavailableStart: app.pendingUnavailableStart,
    acceptedIntervals: app.acceptedIntervals,
    pendingAcceptedStart: app.pendingAcceptedStart
  })));
  if (app.history.length > 25) app.history.shift();
}

function setMovementEnd() {
  app.movementWindow.movement_end_frame = app.frame;
  app.movementWindow.confidence = $("confidence").value;
  app.movementWindow.rationale = $("movementRationale").value || "";
  markAnnotationDirty();
  updateMovementWindowControls();
  renderTimeline();
  draw();
  syncWorkflowSteps(true);
  setStatus(`Marked Movement End at source frame ${app.frame}.`);
}

async function saveSession(finalized) {
  if (app.saveInProgress) return;
  app.movementWindow.confidence = $("confidence").value;
  app.movementWindow.rationale = $("movementRationale").value || "";
  const payload = {
    case_slug: app.currentCase.slug,
    annotator_id: $("annotatorId").value || "researcher_01",
    injured_side: $("injuredSide").value || "unknown",
    case_details: {
      player_name: $("casePlayerName").value || app.currentCase.player_name,
      injury_date: $("injuryDate").value || "",
      competition: $("caseCompetition").value || "",
      team: $("caseTeam").value || "",
      opponent: $("caseOpponent").value || "",
      position_group: $("positionGroup").value || "unknown",
      match_minute: $("matchMinute").value || "",
      date_of_birth: $("dateOfBirth").value || "",
    },
    roi_keyframes: app.keyframes,
    target_unavailable_intervals: app.unavailableIntervals,
    target_accepted_intervals: app.acceptedIntervals,
    movement_window: app.movementWindow,
    notes: $("sessionNotes").value || "",
    finalized,
    revision: app.revision
  };
  const saveButtons = [$("save"), $("finalSave")];
  app.saveInProgress = true;
  saveButtons.forEach(button => button.disabled = true);
  setSaveFeedback(finalized ? "Validating and saving…" : "Saving draft…", "saving");
  try {
    const response = await api("/api/save", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const lines = [
      finalized ? "Saved as ready for validation." : "Saved partial annotation.",
      `ROI keyframes: ${response.validation.summary.target_roi_keyframes}`,
      `Target-unavailable intervals: ${response.validation.summary.target_unavailable_intervals}`,
      `Target-unavailable frames: ${response.validation.summary.target_unavailable_frames}`,
      `Human-accepted intervals: ${response.validation.summary.target_accepted_intervals}`,
      `Human-accepted frames: ${response.validation.summary.target_accepted_frames}`,
      `Movement Start: ${response.validation.summary.movement_start ?? "-"}`,
      `Movement End: ${response.validation.summary.movement_end ?? "-"}`,
      `Movement duration: ${response.validation.summary.movement_duration_ms ?? "-"} ms`,
      `Confidence: ${response.validation.summary.confidence ?? "-"}`,
      `Injured knee: ${response.validation.summary.injured_side ?? "unknown"}`,
      ...response.validation.warnings.map(w => `Warning: ${w}`),
      ...response.validation.errors.map(e => `Error: ${e}`),
      `ROI file: ${response.paths.roi_csv}`,
      `Unavailable intervals file: ${response.paths.target_unavailable_csv}`,
      `Movement Window file: ${response.paths.movement_window_json}`,
      `Event file: ${response.paths.event_json}`,
      `Session file: ${response.paths.session_json}`,
      `Case details file: ${response.paths.case_research_metadata_json}`,
    ];
    app.annotationDirty = false;
    app.revision = Number(response.revision ?? app.revision);
    if (response.validation.ok) {
      updateAnalysisActions(response.human_results_available, finalized);
      setSaveFeedback(finalized ? "Saved and ready for analysis." : "All changes saved.", "saved");
    } else {
      updateAnalysisActions(response.human_results_available, false);
      setSaveFeedback("Saved, but validation needs attention.", "saved");
      $("advancedQa").open = true;
      $("activityDetails").open = true;
    }
    setStatus(lines.join("\n"));
    return response;
  } catch (error) {
    app.annotationDirty = true;
    setSaveFeedback("Save failed — your edits remain on this screen.", "unsaved");
    setStatus(`The annotation could not be saved. ${error.message}`);
    return null;
  } finally {
    app.saveInProgress = false;
    saveButtons.forEach(button => button.disabled = false);
  }
}

function updateAnalysisActions(hasResults, finalized) {
  app.resultsAvailable = Boolean(hasResults);
  app.annotationFinalized = Boolean(finalized);
  const generate = $("generateAnalysis");
  const hasInjuredKnee = $("injuredSide").value !== "unknown";
  generate.disabled = !app.annotationFinalized || !hasInjuredKnee;
  generate.textContent = app.resultsAvailable ? "Regenerate analysis" : "Generate analysis";
  generate.title = !hasInjuredKnee
    ? "Select the documented injured knee before generating analysis."
    : !app.annotationFinalized
      ? "Save as ready for validation before generating analysis."
      : "Run YOLOv8n pose, quality, geometry, dynamics, and movement-story processing.";
  $("viewAnalysis").style.display = app.resultsAvailable ? "inline-block" : "none";
  updateSidebarSummary();
}

function markAnnotationDirty() {
  if (!app.currentCase) return;
  app.annotationDirty = true;
  if (app.hasPreviousPoseReview) app.previousPoseStale = true;
  updatePoseReviewStaleNotice();
  updateAnalysisActions(app.resultsAvailable, false);
  $("generateAnalysis").title = "Save as ready for validation before regenerating changed annotations.";
  setSaveFeedback("Unsaved changes.", "unsaved");
}

function setSaveFeedback(message, state = "") {
  const feedback = $("saveFeedback");
  feedback.textContent = message;
  feedback.classList.remove("unsaved", "saving", "saved");
  if (state) feedback.classList.add(state);
}

function friendlyAnalysisError(error) {
  const message = String(error?.message || "Analysis stopped before results were completed.");
  const normalized = message.toLowerCase();
  if (normalized.includes("injured") && normalized.includes("knee")) {
    return "Select the documented injured knee, then validate and save again.";
  }
  if (normalized.includes("movement") && normalized.includes("end")) {
    return "Mark Movement End, then validate and save again.";
  }
  if (normalized.includes("final") || normalized.includes("validation")) {
    return "Choose Validate & save, resolve the listed issue, then retry analysis.";
  }
  if (normalized.includes("video") || normalized.includes("source")) {
    return "Check that this clip opens correctly, then retry. The saved annotation has not been changed.";
  }
  return `${message} Your saved annotation has not been changed; review Advanced / QA and retry.`;
}

async function generateAnalysis() {
  if (!app.annotationFinalized) {
    setStatus("Save as ready for validation before generating analysis.");
    return;
  }
  const button = $("generateAnalysis");
  button.disabled = true;
  const previousLabel = button.textContent;
  button.hidden = true;
  $("analysisProgress").hidden = false;
  setSaveFeedback("Analysis generation is running.", "saving");
  setStatus(
    "Reviewing the replay while the pose runtime warms up, then running quality checks, measurements, and movement results. " +
    "The first run after a restart can spend several minutes loading the pose runtime."
  );
  try {
    let response = await api("/api/generate-analysis", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({case: app.currentCase.slug})
    });
    while (response.status === "queued" || response.status === "running") {
      const stageMessage = response.stage === "queued"
        ? "Replay queued—the analysis will kick off shortly."
        : "Pose runtime warmed up; the movement analysis is now in play.";
      setSaveFeedback(stageMessage, "saving");
      await new Promise(resolve => setTimeout(resolve, 1500));
      response = await api(
        `/api/analysis-status?case=${encodeURIComponent(app.currentCase.slug)}`
      );
    }
    if (response.status !== "completed") {
      throw new Error(response.error || "Analysis stopped before results were completed.");
    }
    updateAnalysisActions(true, true);
    app.annotationDirty = false;
    setStatus("Analysis generated. Opening results...");
    window.location.href = response.result_url;
  } catch (error) {
    button.disabled = false;
    button.hidden = false;
    button.textContent = previousLabel;
    $("analysisProgress").hidden = true;
    setSaveFeedback("Analysis not generated — annotation is still saved.", "saved");
    $("reviewStep").open = true;
    $("advancedQa").open = true;
    $("activityDetails").open = true;
    setStatus(friendlyAnalysisError(error));
  }
}

async function compareAnnotations() {
  const data = await api(`/api/compare?case=${encodeURIComponent(app.currentCase.slug)}`);
  $("compareStatus").textContent = JSON.stringify(data, null, 2);
}

function toggleReview() {
  if (app.reviewTimer) {
    stopReviewPlayback();
    return;
  }
  $("review").textContent = "Stop review";
  app.reviewTimer = setTimeout(advanceReviewPlayback, 180);
}

async function advanceReviewPlayback() {
  if (!app.reviewTimer) return;
  const next = app.frame + 1;
  if (next >= (app.meta.frame_count || 1)) {
    toggleReview();
    return;
  }
  const startedAt = performance.now();
  await loadFrame(next);
  if (!app.reviewTimer) return;
  const remainingDelay = Math.max(30, 180 - (performance.now() - startedAt));
  app.reviewTimer = setTimeout(advanceReviewPlayback, remainingDelay);
}

function keyframeAt(frame) {
  return app.keyframes.find(k => k.frame_index === frame);
}

function propagatedBox(frame) {
  if (unavailableIntervalAt(frame)) return null;
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

function unavailableIntervalAt(frame) {
  return app.unavailableIntervals.find(
    interval => frame >= interval.start_frame && frame <= interval.end_frame
  ) || null;
}

function acceptedIntervalAt(frame) {
  return app.acceptedIntervals.find(
    interval => frame >= interval.start_frame && frame <= interval.end_frame
  ) || null;
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
  const frameRange = start !== null && end !== null && end !== undefined
    ? `Frames ${start}-${end}`
    : start !== null
      ? `Starts at frame ${start}`
      : "Window not set";
  const durationLabel = duration !== null
    ? `${(duration / 1000).toFixed(2)} s`
    : "Mark an end frame";
  $("windowSummary").innerHTML = `<strong>${frameRange}</strong><span>${durationLabel}</span>`;
  updateSidebarSummary();
}

function renderPoseAnalysisTimeline() {
  const timeline = $("poseAnalysisTimeline");
  if (!timeline) return;
  timeline.innerHTML = "";
  if (app.reviewMode !== "pose" || !app.poseReviewIntervals.length) return;
  const maxFrame = Math.max((app.meta.frame_count || 1) - 1, 1);
  const colors = {
    USED: "#148a54",
    INSUFFICIENT_EVIDENCE: "#c47b00",
    HUMAN_EXCLUDED: "#8f2f3f",
    NO_POSE: "#7c8794"
  };
  app.poseReviewIntervals.forEach(interval => {
    const band = document.createElement("button");
    band.type = "button";
    band.style.position = "absolute";
    band.style.border = "0";
    band.style.borderRadius = "0";
    band.style.height = "100%";
    band.style.padding = "0";
    band.style.top = "0";
    band.style.left = `${Number(interval.start_frame) / maxFrame * 100}%`;
    band.style.width = `${Math.max(Number(interval.end_frame) - Number(interval.start_frame) + 1, 1) / maxFrame * 100}%`;
    band.style.background = colors[interval.state] || "#7c8794";
    band.title = `${interval.label}: frames ${interval.start_frame}-${interval.end_frame}`;
    band.setAttribute("aria-label", band.title);
    band.onclick = () => loadFrame(Number(interval.start_frame));
    timeline.appendChild(band);
  });
  const cursor = document.createElement("div");
  cursor.style.position = "absolute";
  cursor.style.background = "#111827";
  cursor.style.boxShadow = "0 0 0 1px #ffffff";
  cursor.style.height = "100%";
  cursor.style.top = "0";
  cursor.style.width = "2px";
  cursor.style.left = `${app.frame / maxFrame * 100}%`;
  cursor.title = `Current frame ${app.frame}`;
  timeline.appendChild(cursor);
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
  app.unavailableIntervals.forEach(interval => {
    const band = document.createElement("div");
    band.style.position = "absolute";
    band.style.top = "2px";
    band.style.height = "16px";
    band.style.borderRadius = "4px";
    band.style.background = "rgba(143,47,63,0.72)";
    band.style.left = `${interval.start_frame / maxFrame * 100}%`;
    band.style.width = `${Math.max(interval.end_frame - interval.start_frame, 1) / maxFrame * 100}%`;
    band.title = `Target unavailable ${interval.start_frame}-${interval.end_frame}: ${interval.reason}`;
    band.onclick = () => loadFrame(interval.start_frame);
    $("timeline").appendChild(band);
  });
  app.acceptedIntervals.forEach(interval => {
    const band = document.createElement("div");
    band.style.position = "absolute";
    band.style.top = "2px";
    band.style.height = "16px";
    band.style.borderRadius = "4px";
    band.style.background = "rgba(20,138,84,0.72)";
    band.style.left = `${interval.start_frame / maxFrame * 100}%`;
    band.style.width = `${Math.max(interval.end_frame - interval.start_frame, 1) / maxFrame * 100}%`;
    band.title = `Human accepted skeleton ${interval.start_frame}-${interval.end_frame}`;
    band.onclick = () => loadFrame(interval.start_frame);
    $("timeline").appendChild(band);
  });
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

function normalizeUnavailableInterval(interval) {
  return {
    start_frame: Number(interval.start_frame),
    end_frame: Number(interval.end_frame),
    reason: interval.reason || "TARGET_NOT_VISIBLE",
    note: interval.note || ""
  };
}

function normalizeAcceptedInterval(interval) {
  return {
    start_frame: Number(interval.start_frame),
    end_frame: Number(interval.end_frame),
    note: interval.note || ""
  };
}

function setStatus(message) {
  $("status").textContent = message;
}

init().catch(error => setStatus(error.message));
</script>
</body>
</html>
""".replace("__APP_SHELL_CSS__", app_shell_css()).replace(
        "__APP_SITE_HEADER__", app_site_header("Human Annotation")
    )
