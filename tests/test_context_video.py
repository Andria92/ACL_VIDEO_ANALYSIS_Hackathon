from __future__ import annotations

from datetime import UTC, datetime

import pytest

from acl_motion.annotations.models import AnnotationCase
from acl_motion.ui.results import build_context_video_payload, context_video_path
from acl_motion.video.context import (
    CONTEXT_CLIP_ROLE,
    ContextVideoClip,
    context_clip_registry_path,
    context_video_clip_by_id,
    context_video_clips_for_case,
    load_context_video_clips,
    save_context_video_clip,
)


def _clip(tmp_path, *, clip_id: str = "context_01") -> ContextVideoClip:
    video_path = tmp_path / "context.mp4"
    video_path.write_bytes(b"context-video")
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source-video")
    return ContextVideoClip(
        clip_id=clip_id,
        case_id="case_acl",
        video_path=video_path,
        source_video_path=source_path,
        start_seconds=1.25,
        end_seconds=4.75,
        created_at=datetime.now(UTC).isoformat(),
    )


def test_context_clip_registry_preserves_context_only_policy(tmp_path) -> None:
    path = context_clip_registry_path(tmp_path)
    clip = _clip(tmp_path)

    save_context_video_clip(clip, path)
    loaded = load_context_video_clips(path)
    record = loaded[0].to_dict()

    assert loaded == (clip,)
    assert context_video_clips_for_case("case_acl", path) == (clip,)
    assert context_video_clips_for_case("different_case", path) == ()
    assert record["role"] == CONTEXT_CLIP_ROLE
    assert record["use_for_measurements"] is False
    assert record["use_for_movement_narrative"] is False
    assert record["automated_contact_interpretation"] is False


def test_context_clip_lookup_is_scoped_to_the_injury_case(tmp_path) -> None:
    path = context_clip_registry_path(tmp_path)
    clip = _clip(tmp_path)
    save_context_video_clip(clip, path)

    assert context_video_clip_by_id(
        clip.clip_id,
        case_id="case_acl",
        path=path,
    ) == clip
    with pytest.raises(KeyError, match="Unknown real-time context clip"):
        context_video_clip_by_id(
            clip.clip_id,
            case_id="different_case",
            path=path,
        )


def test_results_context_payload_links_video_without_analysis_authority(tmp_path) -> None:
    path = context_clip_registry_path(tmp_path)
    clip = _clip(tmp_path)
    save_context_video_clip(clip, path)
    case = AnnotationCase(
        slug="case_view",
        case_id="case_acl",
        source_id="source_01",
        player_name="Player A",
        video_path=tmp_path / "analysis.mp4",
    )

    payload = build_context_video_payload(case, data_root=tmp_path)

    assert payload["available"] is True
    assert payload["policy"] == {
        "context_only": True,
        "used_for_measurements": False,
        "used_for_movement_narrative": False,
        "automated_contact_interpretation": False,
    }
    assert payload["clips"][0]["video_url"].endswith(
        "case=case_view&clip=context_01"
    )
    assert context_video_path(case, clip_id=clip.clip_id, data_root=tmp_path) == clip.video_path
