"""One-off audited cleanup for legacy duplicate and split case records."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = WORKSPACE / "data" / "annotations" / "human"
REGISTRY_PATH = ANNOTATION_DIR / "imported_video_cases_human.json"
METADATA_PATH = ANNOTATION_DIR / "case_research_metadata_human.json"

JORDAN_CASE_ID = "jordan_nobbs_acl"
JORDAN_VIEWS = {
    "imported_jordan_nobbs_everton_injury_gz23btdm4ve_00m32s314_00m34s562": {
        "old_case_id": (
            "imported_jordan_nobbs_everton_injury_gz23btdm4ve_"
            "00m32s314_00m34s562_acl_candidate"
        ),
        "view_label": "Tight replay view",
        "primary_view": True,
        "perspective": "oblique",
    },
    "imported_jordan_nobbs_everton_injury_gz23btdm4ve_00m36s300_00m39s611": {
        "old_case_id": (
            "imported_jordan_nobbs_everton_injury_gz23btdm4ve_"
            "00m36s300_00m39s611_acl_candidate"
        ),
        "view_label": "Wide replay view",
        "primary_view": False,
        "perspective": "wide-oblique",
    },
}

DUPLICATES = {
    "imported_10_leah_williamson_tygjh39bmfu_01m33s519_01m37s685": {
        "canonical_video": Path(
            "data/videos/analysis_clips/"
            "10_leah_williamson_TygjH39bmfU_01m33s519_01m37s685.mp4"
        ),
        "duplicate_video": Path(
            "data/videos/analysis_clips/"
            "10_leah_williamson_tygjh39bmfu_01m33s519_01m37s685_2.mp4"
        ),
    },
    "imported_3f55f120_abb1_44f1_9352_f9ebe1ec23ec_00m22s800_00m27s098_2": {
        "canonical_video": Path(
            "data/videos/analysis_clips/"
            "3f55f120_abb1_44f1_9352_f9ebe1ec23ec_00m22s800_00m27s098.mp4"
        ),
        "duplicate_video": Path(
            "data/videos/analysis_clips/"
            "3f55f120_abb1_44f1_9352_f9ebe1ec23ec_00m22s800_00m27s098_2.mp4"
        ),
    },
    "imported_screen_recording_2026_08_22_at_3_26_40_pm_00m08s281_00m11s236_2": {
        "canonical_video": Path(
            "data/videos/analysis_clips/"
            "screen_recording_2026_08_22_at_3_26_40_pm_00m08s281_00m11s236.mp4"
        ),
        "duplicate_video": Path(
            "data/videos/analysis_clips/"
            "screen_recording_2026_08_22_at_3_26_40_pm_00m08s281_00m11s236_2.mp4"
        ),
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _duplicate_artifacts() -> list[Path]:
    slugs = tuple(DUPLICATES)
    roots = (WORKSPACE / "data", WORKSPACE / "artifacts")
    return sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and any(path.name.startswith(f"{slug}_") for slug in slugs)
    )


def _replace_values(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            replacements.get(str(key), str(key)): _replace_values(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_values(item, replacements) for item in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _jordan_artifacts() -> list[Path]:
    slugs = tuple(JORDAN_VIEWS)
    return sorted(
        path
        for path in (WORKSPACE / "data").rglob("*")
        if path.is_file() and any(path.name.startswith(f"{slug}_") for slug in slugs)
    )


def _update_jordan_artifact(path: Path, replacements: dict[str, str]) -> None:
    if path.suffix == ".json":
        _write_json(path, _replace_values(_load_json(path), replacements))
        return
    if path.suffix == ".csv":
        text = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
        return
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        for column in frame.columns:
            if frame[column].dtype == "object" or pd.api.types.is_string_dtype(
                frame[column]
            ):
                frame[column] = frame[column].map(
                    lambda value: replacements.get(value, value)
                    if isinstance(value, str)
                    else value
                )
        frame.to_parquet(path, index=False)


def resume_jordan_rekey() -> None:
    """Idempotently finish the Jordan case-ID rewrite after an interrupted cleanup."""

    replacements = {
        str(config["old_case_id"]): JORDAN_CASE_ID
        for config in JORDAN_VIEWS.values()
    }
    jordan_artifacts = _jordan_artifacts()
    for path in jordan_artifacts:
        _update_jordan_artifact(path, replacements)
    print(f"Jordan artifact re-key complete: {len(jordan_artifacts)} files")


def _updated_registry() -> tuple[dict[str, Any], dict[str, str]]:
    payload = _load_json(REGISTRY_PATH)
    duplicate_slugs = set(DUPLICATES)
    records = [
        record for record in payload.get("cases", []) if record.get("slug") not in duplicate_slugs
    ]
    replacements: dict[str, str] = {}
    found_jordan: set[str] = set()
    for record in records:
        slug = str(record.get("slug", ""))
        if slug not in JORDAN_VIEWS:
            continue
        config = JORDAN_VIEWS[slug]
        old_case_id = str(config["old_case_id"])
        replacements[old_case_id] = JORDAN_CASE_ID
        record["case_id"] = JORDAN_CASE_ID
        record["view_label"] = config["view_label"]
        record["primary_view"] = config["primary_view"]
        record["perspective"] = config["perspective"]
        record["player_name"] = "Jordan Nobbs"
        found_jordan.add(slug)
    if found_jordan != set(JORDAN_VIEWS):
        missing = sorted(set(JORDAN_VIEWS) - found_jordan)
        raise RuntimeError(f"Jordan registry views not found: {missing}")
    payload["cases"] = records
    return payload, replacements


def _updated_metadata(replacements: dict[str, str]) -> dict[str, Any]:
    payload = _load_json(METADATA_PATH)
    cases = dict(payload.get("cases", {}))
    for duplicate in DUPLICATES:
        record = next(
            item
            for item in _load_json(REGISTRY_PATH).get("cases", [])
            if item.get("slug") == duplicate
        )
        cases.pop(str(record["case_id"]), None)

    jordan: dict[str, Any] = {}
    for old_case_id in replacements:
        jordan.update(cases.pop(old_case_id, {}))
    jordan.update(
        {
            "statistical_unit_id": JORDAN_CASE_ID,
            "player_name": "Jordan Nobbs",
        }
    )
    cases[JORDAN_CASE_ID] = jordan
    payload["cases"] = cases
    return payload


def _validate_duplicate_hashes() -> None:
    for slug, paths in DUPLICATES.items():
        canonical = paths["canonical_video"]
        duplicate = paths["duplicate_video"]
        if not canonical.exists() or not duplicate.exists():
            raise RuntimeError(f"Missing canonical or duplicate video for {slug}.")
        if _sha256(canonical) != _sha256(duplicate):
            raise RuntimeError(f"Video hash mismatch for {slug}; cleanup stopped.")


def _trash_destination(trash_root: Path, path: Path) -> Path:
    try:
        relative = path.relative_to(WORKSPACE)
        destination = trash_root / "workspace" / relative
    except ValueError:
        destination = trash_root / "external_videos" / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RuntimeError(f"Trash destination already exists: {destination}")
    return destination


def cleanup(*, apply: bool) -> None:
    _validate_duplicate_hashes()
    registry, replacements = _updated_registry()
    metadata = _updated_metadata(replacements)
    duplicate_artifacts = _duplicate_artifacts()
    duplicate_videos = [
        paths["duplicate_video"] for paths in DUPLICATES.values()
    ]
    jordan_artifacts = _jordan_artifacts()

    print(f"Duplicate records to remove: {len(DUPLICATES)}")
    print(f"Duplicate artifacts to trash: {len(duplicate_artifacts)}")
    print(f"Duplicate source videos to trash: {len(duplicate_videos)}")
    print(f"Jordan artifacts to re-key: {len(jordan_artifacts)}")
    for slug in DUPLICATES:
        print(f"  duplicate: {slug}")
    for slug in JORDAN_VIEWS:
        print(f"  Jordan view retained: {slug}")
    if not apply:
        print("Dry run complete; no files changed.")
        return

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    trash_root = Path.home() / ".Trash" / f"acl-case-cleanup-{stamp}"
    trash_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(REGISTRY_PATH, trash_root / REGISTRY_PATH.name)
    shutil.copy2(METADATA_PATH, trash_root / METADATA_PATH.name)

    for path in [*duplicate_artifacts, *duplicate_videos]:
        shutil.move(str(path), _trash_destination(trash_root, path))

    _write_json(REGISTRY_PATH, registry)
    _write_json(METADATA_PATH, metadata)
    for path in jordan_artifacts:
        _update_jordan_artifact(path, replacements)

    manifest = {
        "cleanup_at": datetime.now(UTC).isoformat(),
        "duplicate_slugs": list(DUPLICATES),
        "jordan_case_id": JORDAN_CASE_ID,
        "jordan_views": list(JORDAN_VIEWS),
        "trashed_artifact_count": len(duplicate_artifacts),
        "trashed_video_count": len(duplicate_videos),
        "rekeyed_jordan_artifact_count": len(jordan_artifacts),
    }
    _write_json(trash_root / "cleanup_manifest.json", manifest)
    print(f"Cleanup applied. Recovery bundle: {trash_root}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the cleanup. Without this flag the script only audits.",
    )
    parser.add_argument(
        "--resume-jordan",
        action="store_true",
        help="Idempotently finish only the Jordan artifact case-ID rewrite.",
    )
    args = parser.parse_args()
    if args.resume_jordan:
        resume_jordan_rekey()
        return 0
    cleanup(apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
