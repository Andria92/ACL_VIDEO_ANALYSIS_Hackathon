"""Local case registry for the M5.5 annotation UI."""

from __future__ import annotations

from pathlib import Path

from acl_motion.annotations.models import AnnotationCase

DEFAULT_VIDEO_ROOT = Path("/Users/andriagryffinpro/Desktop/injury_videos")
DEFAULT_IMPORTED_CASES_PATH = Path("data/annotations/human/imported_video_cases_human.json")


def default_annotation_cases(video_root: str | Path = DEFAULT_VIDEO_ROOT) -> tuple[AnnotationCase, ...]:
    """Return the current human-validation cases.

    Development ROI/event paths are retained only for explicit post-save comparison mode.
    They are not surfaced by the annotation UI during independent human annotation.
    """

    root = Path(video_root)
    return (
        AnnotationCase(
            slug="christen_press",
            case_id="christen_press_acl",
            source_id="christen_press_view_01",
            view_id="christen_press_view_01",
            view_label="Primary broadcast view",
            primary_view=True,
            player_name="Christen Press",
            video_path=root / "02_YvnMYc6OdT8_160s-166s.mp4",
            development_roi_path=Path("data/annotations/christen_press_roi_keyframes.csv"),
            development_event_path=Path("data/annotations/christen_press_event_annotation.json"),
            notes="Cleaner baseline human-annotation validation case.",
        ),
        AnnotationCase(
            slug="case_01",
            case_id="case_01_acl_candidate",
            source_id="case_01_view_01",
            view_id="case_01_view_01",
            view_label="Primary broadcast view",
            primary_view=True,
            player_name="Vivianne Miedema",
            video_path=root / "01_YXpvkHw2BP8_135s-145s.mp4",
            notes=(
                "Candidate cleaner baseline clip for human target annotation; "
                "injury laterality supplied by human operator as left knee."
            ),
        ),
        AnnotationCase(
            slug="leah_williamson_broadcast_wide",
            case_id="leah_williamson_acl",
            source_id="leah_williamson_broadcast_wide_01",
            view_id="leah_williamson_broadcast_wide_01",
            view_label="Live broadcast wide",
            primary_view=True,
            perspective="oblique",
            occlusion_level="moderate",
            view_quality="contextual",
            player_name="Leah Williamson",
            video_path=root / "10_leah_williamson_view_01_broadcast_wide_0s-12s.mp4",
            notes=(
                "Primary contextual view from the full Leah Williamson injury video. "
                "The target is small, so pose/landmark support may be limited."
            ),
        ),
        AnnotationCase(
            slug="leah_williamson_replay_close_oblique",
            case_id="leah_williamson_acl",
            source_id="leah_williamson_replay_close_oblique_02",
            view_id="leah_williamson_replay_close_oblique_02",
            view_label="Replay close oblique",
            primary_view=False,
            perspective="oblique",
            occlusion_level="moderate",
            view_quality="short_landmark_candidate",
            slow_motion=True,
            cropped_or_zoomed=True,
            player_name="Leah Williamson",
            video_path=root / "10_leah_williamson_TygjH39bmfU_00m42s814_00m44s647.mp4",
            notes=(
                "Short close replay view of the approach/contact sequence. Because replay "
                "time scale is not registered, timing-dependent measurements should be "
                "treated cautiously."
            ),
        ),
        AnnotationCase(
            slug="leah_williamson_replay_frontal_oblique",
            case_id="leah_williamson_acl",
            source_id="leah_williamson_replay_frontal_oblique_03",
            view_id="leah_williamson_replay_frontal_oblique_03",
            view_label="Replay frontal-like oblique",
            primary_view=False,
            perspective="frontal-like",
            occlusion_level="moderate",
            view_quality="landmark_candidate",
            slow_motion=True,
            cropped_or_zoomed=True,
            player_name="Leah Williamson",
            video_path=root / "10_leah_williamson_TygjH39bmfU_00m48s869_00m52s235.mp4",
            notes=(
                "Frontal-like oblique replay. Potentially useful for projected knee-foot "
                "relationship and bilateral spacing; timing-dependent measures remain "
                "limited unless replay scale is later registered."
            ),
        ),
        AnnotationCase(
            slug="leah_williamson_replay_close_sagittal",
            case_id="leah_williamson_acl",
            source_id="leah_williamson_replay_close_sagittal_04",
            view_id="leah_williamson_replay_close_sagittal_04",
            view_label="Replay close sagittal-like",
            primary_view=False,
            perspective="sagittal-like",
            occlusion_level="moderate_high",
            view_quality="landmark_candidate",
            slow_motion=True,
            cropped_or_zoomed=True,
            player_name="Leah Williamson",
            video_path=root / "10_leah_williamson_TygjH39bmfU_01m33s519_01m37s685.mp4",
            notes=(
                "Close sideline replay candidate. Potentially strongest for projected "
                "lower-limb and trunk geometry, but overlapping players must be treated "
                "as QC evidence rather than ignored."
            ),
        ),
        AnnotationCase(
            slug="ellie_carpenter",
            case_id="ellie_carpenter_acl",
            source_id="ellie_carpenter_view_01",
            view_id="ellie_carpenter_view_01",
            view_label="Primary broadcast view",
            primary_view=True,
            player_name="Ellie Carpenter",
            video_path=root / "07_4v6Y-ZSgziE_from_86s_86s-106s.mp4",
            development_roi_path=Path("data/annotations/ellie_carpenter_roi_keyframes_dense.csv"),
            development_event_path=Path("data/annotations/ellie_carpenter_event_annotation.json"),
            notes="Overlap and occlusion stress case for human annotation.",
        ),
    )


def views_for_case(
    case: AnnotationCase,
    cases: tuple[AnnotationCase, ...] | None = None,
) -> tuple[AnnotationCase, ...]:
    """Return all registered views for the same ACL event as ``case``."""

    siblings = tuple(
        item for item in cases or analysis_annotation_cases() if item.case_id == case.case_id
    )
    return tuple(sorted(siblings, key=_view_sort_key)) or (case,)


def primary_view_for_case(
    case: AnnotationCase,
    cases: tuple[AnnotationCase, ...] | None = None,
) -> AnnotationCase:
    """Return the default view for a case without implying scientific superiority."""

    views = views_for_case(case, cases)
    for view in views:
        if view.primary_view:
            return view
    return views[0]


def case_by_slug(
    slug: str,
    cases: tuple[AnnotationCase, ...] | None = None,
) -> AnnotationCase:
    """Return a registered annotation case by slug."""

    for case in cases or analysis_annotation_cases():
        if case.slug == slug:
            return case
    raise KeyError(f"Unknown annotation case: {slug}")


def analysis_annotation_cases(
    video_root: str | Path = DEFAULT_VIDEO_ROOT,
    *,
    imported_cases_path: str | Path = DEFAULT_IMPORTED_CASES_PATH,
) -> tuple[AnnotationCase, ...]:
    """Return built-in plus locally imported annotation cases for analysis scripts."""

    return (
        *default_annotation_cases(video_root),
        *imported_annotation_cases(video_root, imported_cases_path=imported_cases_path),
    )


def imported_annotation_cases(
    video_root: str | Path = DEFAULT_VIDEO_ROOT,
    *,
    imported_cases_path: str | Path = DEFAULT_IMPORTED_CASES_PATH,
) -> tuple[AnnotationCase, ...]:
    """Load local videos imported through the human annotation UI."""

    import json

    path = Path(imported_cases_path)
    if not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()

    cases = []
    root = Path(video_root)
    for record in payload.get("cases", ()):
        try:
            video_path = Path(str(record["video_path"]))
            if not video_path.is_absolute():
                video_path = root / video_path
            cases.append(
                AnnotationCase(
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
                    player_name=str(record.get("player_name", video_path.stem)),
                    video_path=video_path,
                    notes=str(record.get("notes", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(cases)


def _view_sort_key(case: AnnotationCase) -> tuple[int, str]:
    return (0 if case.primary_view else 1, case.view_label.lower(), case.slug)
