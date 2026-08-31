from __future__ import annotations

from urllib.parse import quote

import cv2
import numpy as np

from acl_motion.ui.video_cutter import cut_video_segment, smoke_test
from acl_motion.video.io import read_video_metadata


def test_video_cutter_ui_smoke_has_review_controls() -> None:
    result = smoke_test()

    assert result["html_has_video_player"] is True
    assert result["html_has_mark_in"] is True
    assert result["html_has_mark_out"] is True
    assert result["html_has_five_frame_controls"] is True
    assert result["html_has_reload_player"] is True
    assert result["html_has_player_error_recovery"] is True
    assert result["html_has_cut"] is True
    assert result["writes_files"] is False


def test_video_cutter_exposes_context_clip_role_without_analysis_claims() -> None:
    from acl_motion.ui.video_cutter import render_video_cutter_page

    html = render_video_cutter_page()

    assert 'id="clipRoleSelect"' in html
    assert 'value="REAL_TIME_CONTEXT"' in html
    assert 'id="contextCaseSelect"' in html
    assert "[hidden] { display: none !important; }" in html
    assert "Context only. This clip will not be used for measurements" in html
    assert "clip_role: app.clipRole" in html


def test_video_cutter_can_create_or_extend_a_dated_multi_view_case() -> None:
    from acl_motion.ui.video_cutter import render_video_cutter_page

    html = render_video_cutter_page(main_menu_url="/")

    assert 'id="assignmentModeSelect"' in html
    assert 'id="analysisCaseSelect"' in html
    assert 'id="casePlayerInput"' in html
    assert 'id="caseDateInput" type="date"' in html
    assert 'id="caseTeamInput"' in html
    assert 'id="caseOpponentInput"' in html
    assert 'id="caseCompetitionInput"' in html
    assert 'id="viewLabelInput"' in html
    assert "assign-analysis-clip" in html
    assert "Create or choose the case before opening video" in html
    assert "Every clip you cut in this session will be attached" in html
    assert 'id="beginCaseButton"' in html
    assert 'id="cutAnotherViewButton"' in html
    assert 'id="cancelCutterLink"' in html
    assert 'const requestedReturnRef = requestedParams.get("return");' in html
    assert 'function safeReturnUrl(rawValue)' in html
    assert '"Cancel and return to analysis"' in html
    assert 'id="videoSelect" disabled' in html
    assert 'id="videoWorkspace" class="panel viewer" hidden' in html
    assert 'id="cutControlsPanel" class="panel side" hidden' in html
    assert "function setVideoControlsEnabled(enabled)" in html
    assert "app.videoReady = true;" in html
    assert 'setStatus("Finding the available source videos…");' in html
    assert 'aria-label="Back 5 seconds"' in html
    assert "if (!context) await assignAnalysisClip();" in html
    assert 'id="caseSetupForm"' in html
    assert 'id="activeCaseTitle"' in html
    assert 'id="activeCaseMeta"' in html
    assert '$("caseSetupForm").hidden = true;' in html
    assert '$("caseSetupForm").hidden = false;' in html
    assert "renderActiveCaseSummary(activeCase);" in html
    assert 'id="viewDetailsPanel" class="view-details"' in html
    assert "Optional view and export details" in html
    assert html.index('id="cutButton"') > html.index('id="viewDetailsPanel"')
    assert 'id="assignmentTitle"' in html
    assert 'id="annotateAssignedLink" class="button primary"' in html
    assert "Open cut in player" not in html
    assert 'id="reviewOutputLink"' not in html
    assert "Video view saved to the active case" not in html


def test_video_cutter_main_menu_url_is_configurable() -> None:
    from acl_motion.ui.video_cutter import render_video_cutter_page

    html = render_video_cutter_page(main_menu_url="http://127.0.0.1:8785/")

    assert 'class="app-brand" href="http://127.0.0.1:8785/"' in html


def test_video_cutter_api_base_is_configurable_for_single_port_hosting() -> None:
    from acl_motion.ui.video_cutter import render_video_cutter_page

    html = render_video_cutter_page(
        main_menu_url="/",
        api_base="/video-cutter/api/",
    )

    assert 'const apiBase = "/video-cutter/api";' in html
    assert 'fetch(`${apiBase}/videos`)' in html
    assert '`${apiBase}/metadata?id=${encodeURIComponent(video.id)}`' in html
    assert 'return `${apiBase}/video?id=' in html


def test_video_cutter_deep_link_can_select_case_and_source_video() -> None:
    from acl_motion.ui.video_cutter import render_video_cutter_page

    html = render_video_cutter_page()

    assert 'const requestedCaseId = requestedParams.get("case");' in html
    assert 'const requestedVideoRef = requestedParams.get("video");' in html
    assert "video.name === requestedVideoRef" in html
    assert "video.path === requestedVideoRef" in html
    assert "if (requestedVideo && !app.requestedVideoApplied)" in html
    assert "selectVideo(requestedVideo.id);" in html


def test_video_cutter_requires_explicit_source_selection_and_confirmation() -> None:
    from acl_motion.ui.video_cutter import render_video_cutter_page

    html = render_video_cutter_page()

    assert 'id="videoSearchInput"' in html
    assert "Choose a source video…" in html
    assert "Views already linked to this case" in html
    assert "Other local source videos" in html
    assert "selectVideo((requestedVideo || app.videos[0]).id)" not in html
    assert "function resetVideoSelection" in html
    assert 'id="sourceAssignmentWarning"' in html
    assert 'id="sourceVerifiedInput" type="checkbox" disabled' in html
    assert "Verify the selected athlete and injury event before adding this view." in html
    assert '!$("sourceVerifiedInput").checked' in html
    assert "Verify that this footage shows the correct athlete and injury event." in html
    assert "const confirmed = window.confirm(" in html
    assert "if (!confirmed) return;" in html
    assert "resetVideoSelection(`Choose and verify the source video for the next view" in html


def test_video_inventory_labels_views_that_are_already_registered(tmp_path, monkeypatch) -> None:
    from acl_motion.annotations.models import AnnotationCase
    from acl_motion.ui import video_cutter

    source_path = tmp_path / "known-view.mp4"
    source_path.write_bytes(b"inventory-only")
    case = AnnotationCase(
        slug="known_view",
        case_id="known_case",
        source_id="known_source",
        player_name="Known Player",
        video_path=source_path,
        view_label="Wide replay",
    )
    state = video_cutter.create_video_cutter_state(
        video_roots=(tmp_path,),
        output_dir=tmp_path / "cuts",
    )
    monkeypatch.setattr(video_cutter, "_registered_cases", lambda _state: (case,))

    video = video_cutter.video_cutter_videos_response(state)["videos"][0]

    assert video["registered_view"] is True
    assert video["case_id"] == "known_case"
    assert video["player_name"] == "Known Player"
    assert video["view_label"] == "Wide replay"


def test_video_inventory_does_not_open_every_discovered_video(tmp_path, monkeypatch) -> None:
    from acl_motion.ui import video_cutter

    source_path = tmp_path / "slow-or-damaged.mp4"
    source_path.write_bytes(b"not opened during inventory")
    state = video_cutter.create_video_cutter_state(
        video_roots=(tmp_path,),
        output_dir=tmp_path / "cuts",
    )

    def fail_if_opened(path):
        raise AssertionError(f"inventory unexpectedly opened {path}")

    monkeypatch.setattr(video_cutter, "read_video_metadata", fail_if_opened)

    response = video_cutter.video_cutter_videos_response(state)

    assert response["skipped"] == []
    assert response["videos"] == [
        {
            "id": video_cutter._encode_path(source_path),
            "name": source_path.name,
            "path": str(source_path),
        }
    ]


def test_integrated_cutter_supports_head_requests_for_firefox(tmp_path) -> None:
    from acl_motion.ui.annotation import AnnotationUiState, make_handler
    from acl_motion.ui.video_cutter import (
        create_video_cutter_state,
        open_video_path_response,
    )

    source_path = tmp_path / "firefox-preview.mp4"
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (16, 16),
    )
    try:
        assert writer.isOpened()
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    finally:
        writer.release()
    cutter_state = create_video_cutter_state(
        video_roots=(tmp_path,),
        output_dir=tmp_path / "cuts",
        annotation_output_dir=tmp_path / "annotations",
    )
    video = open_video_path_response({"path": str(source_path)}, cutter_state)["video"]
    state = AnnotationUiState(
        cases=[],
        output_dir=tmp_path / "annotations",
        video_root=tmp_path,
        video_cutter_state=cutter_state,
    )
    handler_type = make_handler(state)
    handler = object.__new__(handler_type)
    sent = {}
    handler.path = f"/video-cutter/api/video?id={quote(video['id'])}"
    handler._send_file = lambda path, **kwargs: sent.update(path=path, **kwargs)

    handler.do_HEAD()

    assert sent["path"] == source_path.resolve()
    assert sent["send_body"] is False


def test_integrated_video_response_avoids_unicode_inline_header(tmp_path) -> None:
    from acl_motion.ui.annotation import AnnotationUiState, make_handler

    source_path = tmp_path / "Cascarino 21⧸5⧸23.mp4"
    source_path.write_bytes(b"video")
    state = AnnotationUiState(
        cases=[],
        output_dir=tmp_path / "annotations",
        video_root=tmp_path,
    )
    handler_type = make_handler(state)
    handler = object.__new__(handler_type)
    sent_headers = []
    handler.headers = {}
    handler.send_response = lambda status: None
    handler.send_header = lambda name, value: sent_headers.append((name, value))
    handler.end_headers = lambda: None

    handler._send_file(source_path, send_body=False)

    assert ("Content-Type", "video/mp4") in sent_headers
    assert not any(name == "Content-Disposition" for name, _ in sent_headers)

    sent_headers.clear()
    handler._send_file(source_path, attachment=True, send_body=False)
    disposition = next(
        value for name, value in sent_headers if name == "Content-Disposition"
    )
    assert disposition.isascii()
    assert "filename*=UTF-8''Cascarino%2021%E2%A7%B85%E2%A7%B823.mp4" in disposition


def test_integrated_video_response_returns_416_for_end_of_file_range(tmp_path) -> None:
    from http import HTTPStatus

    from acl_motion.ui.annotation import AnnotationUiState, make_handler

    source_path = tmp_path / "reviewed-to-the-end.mp4"
    source_path.write_bytes(b"video")
    state = AnnotationUiState(
        cases=[],
        output_dir=tmp_path / "annotations",
        video_root=tmp_path,
    )
    handler_type = make_handler(state)
    handler = object.__new__(handler_type)
    sent = {"headers": []}
    handler.headers = {"Range": "bytes=5-"}
    handler.send_response = lambda status: sent.update(status=status)
    handler.send_header = lambda name, value: sent["headers"].append((name, value))
    handler.end_headers = lambda: None

    handler._send_file(source_path, send_body=False)

    assert sent["status"] == HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE
    assert ("Content-Range", "bytes */5") in sent["headers"]
    assert ("Accept-Ranges", "bytes") in sent["headers"]
    assert ("Content-Length", "0") in sent["headers"]


def test_integrated_video_response_ignores_browser_disconnect(tmp_path) -> None:
    from acl_motion.ui.annotation import AnnotationUiState, make_handler

    source_path = tmp_path / "browser-cancelled-request.mp4"
    source_path.write_bytes(b"video")
    state = AnnotationUiState(
        cases=[],
        output_dir=tmp_path / "annotations",
        video_root=tmp_path,
    )
    handler_type = make_handler(state)
    handler = object.__new__(handler_type)
    handler.headers = {}
    handler.send_response = lambda status: None
    handler.send_header = lambda name, value: None
    handler.end_headers = lambda: None

    class DisconnectedBrowser:
        def write(self, data):
            raise ConnectionResetError("browser moved to another video range")

    handler.wfile = DisconnectedBrowser()

    handler._send_file(source_path)


def test_reload_resumes_before_the_end_of_the_clip() -> None:
    from acl_motion.ui.video_cutter import render_video_cutter_page

    html = render_video_cutter_page()

    assert "const lastPlayableTime = Math.max(0, loadedDuration - frameStep());" in html
    assert "player.currentTime = Math.min(resumeAt, lastPlayableTime);" in html
    assert html.index('player.addEventListener("loadedmetadata"') < html.index(
        "player.src = videoUrl(app.selected.id, Date.now());"
    )


def test_cut_video_segment_with_opencv_fallback(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    try:
        assert writer.isOpened()
        for index in range(20):
            frame = np.full((48, 64, 3), index * 8, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    result = cut_video_segment(
        video_path=source_path,
        output_dir=tmp_path / "cuts",
        start_seconds=0.2,
        end_seconds=1.0,
        output_name="review_clip",
        mode="opencv",
    )
    output_path = tmp_path / "cuts" / "review_clip.mp4"
    metadata = read_video_metadata(output_path)

    assert result["saved"] is True
    assert result["method"] == "opencv"
    assert result["width"] == 64
    assert result["height"] == 48
    assert output_path.exists()
    assert metadata.frame_count > 0
    assert 0.2 <= metadata.duration_seconds <= 1.2
