from __future__ import annotations

import pytest

from acl_motion.analytics.similarity import (
    SimilarityComputationCancelled,
    build_similarity_payload,
)
from acl_motion.annotations.models import AnnotationCase
from acl_motion.persistence import CaseArtifactTransaction, atomic_write_json
from acl_motion.runtime import ensure_supported_runtime
from acl_motion.ui.annotation import (
    AnnotationUiState,
    StateConflictError,
    _delete_case_entry,
    _save_response,
    make_handler,
    render_annotation_page,
)
from acl_motion.ui.comparison import render_comparison_page
from acl_motion.ui.home import render_home_page
from acl_motion.ui.video_cutter import (
    assign_analysis_clip,
    create_video_cutter_state,
    cut_video_response,
    render_video_cutter_page,
)


def test_atomic_replace_failure_preserves_previous_file(tmp_path, monkeypatch) -> None:
    from acl_motion import persistence

    destination = tmp_path / "state.json"
    destination.write_text('{"version": 1}', encoding="utf-8")

    def fail_replace(source, target):
        raise OSError("simulated interrupted replace")

    monkeypatch.setattr(persistence.os, "replace", fail_replace)

    with pytest.raises(OSError, match="interrupted"):
        atomic_write_json(destination, {"version": 2})

    assert destination.read_text(encoding="utf-8") == '{"version": 1}'
    assert list(tmp_path.iterdir()) == [destination]


def test_case_artifact_transaction_rolls_back_partial_bundle(tmp_path) -> None:
    original = tmp_path / "case_session.json"
    created = tmp_path / "case_new.csv"
    unrelated = tmp_path / "other_session.json"
    original.write_text("before", encoding="utf-8")
    unrelated.write_text("keep", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="pipeline failed"),
        CaseArtifactTransaction(tmp_path, "case"),
    ):
        original.write_text("partial", encoding="utf-8")
        created.write_text("partial", encoding="utf-8")
        raise RuntimeError("pipeline failed")

    assert original.read_text(encoding="utf-8") == "before"
    assert not created.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_cut_retry_reuses_request_without_duplicate_work(tmp_path, monkeypatch) -> None:
    from acl_motion.ui import video_cutter

    state = create_video_cutter_state(video_roots=(tmp_path,), output_dir=tmp_path / "cuts")
    calls = []

    def fake_cut(payload, cutter_state, *, api_base):
        calls.append(dict(payload))
        return {"saved": True, "video_id": "clip", "download_url": f"{api_base}/clip"}

    monkeypatch.setattr(video_cutter, "_cut_video_response_unlocked", fake_cut)
    payload = {
        "request_id": "one-operation",
        "video_id": "source",
        "start_seconds": 1,
        "end_seconds": 2,
    }

    first = cut_video_response(payload, state)
    second = cut_video_response(payload, state)

    assert first == second
    assert len(calls) == 1
    with pytest.raises(ValueError, match="already used"):
        cut_video_response({**payload, "end_seconds": 3}, state)


def test_assignment_retry_is_idempotent(tmp_path, monkeypatch) -> None:
    from acl_motion.ui import video_cutter

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    state = create_video_cutter_state(
        video_roots=(tmp_path,),
        output_dir=tmp_path / "cuts",
        annotation_output_dir=tmp_path / "annotations",
    )
    case = AnnotationCase(
        slug="case_view",
        case_id="case_id",
        source_id="source_id",
        player_name="Player",
        video_path=clip,
    )
    calls = []
    monkeypatch.setattr(video_cutter, "_decode_video_id", lambda video_id, cutter_state: clip)

    def fake_register(payload, **kwargs):
        calls.append(dict(payload))
        return case, {"player_name": "Player"}

    monkeypatch.setattr(video_cutter, "register_analysis_clip", fake_register)
    payload = {
        "request_id": "one-operation:assign",
        "video_id": "clip",
        "assignment_mode": "new",
        "player_name": "Player",
        "injury_date": "2026-08-31",
    }

    first, _ = assign_analysis_clip(payload, state, cases=())
    second, _ = assign_analysis_clip(payload, state, cases=())

    assert first == second
    assert len(calls) == 1


def test_stale_save_and_deletion_during_analysis_are_rejected(tmp_path) -> None:
    case = AnnotationCase(
        slug="case_view",
        case_id="case_id",
        source_id="source_id",
        player_name="Player",
        video_path=tmp_path / "clip.mp4",
    )
    state = AnnotationUiState(
        cases=[case],
        output_dir=tmp_path / "human",
        video_root=tmp_path,
        trash_dir=tmp_path / "trash",
        case_revisions={case.case_id: 2},
    )

    with pytest.raises(StateConflictError, match="changed in another tab"):
        _save_response({"case_slug": case.slug, "revision": 1}, state)

    state.analysis_jobs[case.slug] = {
        "case": case.slug,
        "case_id": case.case_id,
        "status": "running",
    }
    with pytest.raises(StateConflictError, match="Analysis is queued or running"):
        _delete_case_entry({"scope": "case", "case_id": case.case_id}, state)


def test_main_page_supports_head_health_checks(tmp_path) -> None:
    state = AnnotationUiState(cases=[], output_dir=tmp_path / "human", video_root=tmp_path)
    handler_type = make_handler(state)
    handler = object.__new__(handler_type)
    sent = {}
    handler.path = "/"
    handler._send_html = lambda html, **kwargs: sent.update(html=html, **kwargs)

    handler.do_HEAD()

    assert "ACL Movement Analytics Lab" in sent["html"]
    assert sent["send_body"] is False


def test_browser_race_and_navigation_guards_are_rendered() -> None:
    annotation = render_annotation_page()
    home = render_home_page()
    comparison = render_comparison_page()
    cutter = render_video_cutter_page()

    assert "caseLoadAbortController" in annotation
    assert "caseLoadVersion" in annotation
    assert 'window.addEventListener("popstate"' in annotation
    assert "revision: app.revision" in annotation
    assert "stopReviewPlayback();" in annotation
    assert "showNoFilteredSelection" in home
    assert "syncCaseUrl" in home
    assert 'window.addEventListener("popstate"' in home
    assert "comparisonAbortController.abort()" in comparison
    assert "syncComparisonUrl" in comparison
    assert "client_id=${encodeURIComponent(app.comparisonClientId)}" in comparison
    assert "request_id=${encodeURIComponent(requestId)}" in comparison
    assert 'window.addEventListener("popstate"' in comparison
    assert 'cutState: "idle"' in cutter
    assert 'setCutState("cutting")' in cutter
    assert 'setCutState("assigning")' in cutter
    assert 'setCutState("assigned")' in cutter
    assert 'setCutState("error")' in cutter
    assert "The cut is safe; retry assignment." in cutter


def test_runtime_policy_accepts_only_python_312() -> None:
    ensure_supported_runtime((3, 12, 0))
    with pytest.raises(RuntimeError, match="requires Python 3.12 exactly"):
        ensure_supported_runtime((3, 11, 9))
    with pytest.raises(RuntimeError, match="requires Python 3.12 exactly"):
        ensure_supported_runtime((3, 13, 0))


def test_similarity_work_can_be_cancelled_before_resampling() -> None:
    records = [
        {
            "case_id": "case_a",
            "feature_name": "feature",
            "body_region": "lower_limb",
            "mean": 1.0,
            "geometry_analytics_eligible": True,
            "geometry_completeness": 1.0,
        }
    ]
    events = [{"case_id": "case_a", "player_name": "Player"}]

    with pytest.raises(SimilarityComputationCancelled, match="superseded"):
        build_similarity_payload(records, events, cancelled=lambda: True)
