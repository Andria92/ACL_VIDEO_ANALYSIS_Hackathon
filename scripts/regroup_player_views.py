"""Regroup legacy per-clip records into one case per player injury event.

The affected clips were imported before the player-first intake flow existed.  This
migration preserves every view and generated artifact, changing only shared case identity
and player-facing labels.  Run without ``--apply`` for an audit-only dry run.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_ROOT = WORKSPACE / "data"
REGISTRY_PATH = DATA_ROOT / "annotations" / "human" / "imported_video_cases_human.json"
METADATA_PATH = DATA_ROOT / "annotations" / "human" / "case_research_metadata_human.json"

GROUPS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "beth_mead_acl",
        "player_name": "Beth Mead",
        "views": (
            {
                "slug": "imported_04_5bk1vmbhwu0_full_compilation_00m00s000_00m06s635",
                "view_label": "Clip 1 · 00:00.000–00:06.635",
            },
            {
                "slug": "imported_04_5bk1vmbhwu0_full_compilation_00m07s655_00m10s374",
                "view_label": "Clip 2 · 00:07.655–00:10.374",
            },
        ),
    },
    {
        "case_id": "delphine_cascarino_acl",
        "player_name": "Delphine Cascarino",
        "views": (
            {
                "slug": (
                    "imported_delphine_cascarino_out_of_world_cup_injury_vs_psg_21_5_23_"
                    "00m03s351_00m06s647"
                ),
                "view_label": "Clip 1 · 00:03.351–00:06.647",
            },
            {
                "slug": (
                    "imported_delphine_cascarino_out_of_world_cup_injury_vs_psg_21_5_23_"
                    "00m07s579_00m09s769"
                ),
                "view_label": "Clip 2 · 00:07.579–00:09.769",
            },
        ),
    },
    {
        "case_id": "chloe_kelly_acl",
        "player_name": "Chloe Kelly",
        "views": (
            {
                "slug": "imported_chloe_kelly_acl_00m34s693_00m40s216",
                "view_label": "Clip 1 · 00:34.693–00:40.216",
            },
            {
                "slug": "imported_chloe_kelly_acl_00m47s150_00m52s168",
                "view_label": "Clip 2 · 00:47.150–00:52.168",
            },
        ),
    },
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _view_lookup() -> dict[str, dict[str, Any]]:
    return {
        str(view["slug"]): {
            **view,
            "case_id": group["case_id"],
            "player_name": group["player_name"],
            "primary_view": index == 0,
        }
        for group in GROUPS
        for index, view in enumerate(group["views"])
    }


def _updated_registry() -> tuple[dict[str, Any], dict[str, str]]:
    payload = _load_json(REGISTRY_PATH)
    views = _view_lookup()
    found: set[str] = set()
    replacements: dict[str, str] = {}
    for record in payload.get("cases", []):
        slug = str(record.get("slug", ""))
        config = views.get(slug)
        if config is None:
            continue
        old_case_id = str(record["case_id"])
        if old_case_id != str(config["case_id"]):
            replacements[old_case_id] = str(config["case_id"])
        record["case_id"] = config["case_id"]
        record["player_name"] = config["player_name"]
        record["view_label"] = config["view_label"]
        record["primary_view"] = config["primary_view"]
        found.add(slug)
    missing = sorted(set(views) - found)
    if missing:
        raise RuntimeError(f"Expected registry views were not found: {missing}")
    return payload, replacements


def _merge_nonempty(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    merged = dict(target)
    for key, value in source.items():
        current = merged.get(key)
        if key not in merged or current is None or current == "" or current == "unknown":
            merged[key] = value
    return merged


def _updated_metadata(replacements: dict[str, str]) -> dict[str, Any]:
    payload = _load_json(METADATA_PATH)
    records = dict(payload.get("cases", {}))
    grouped: dict[str, dict[str, Any]] = {}
    for old_case_id, new_case_id in replacements.items():
        grouped[new_case_id] = _merge_nonempty(
            grouped.get(new_case_id, {}),
            dict(records.pop(old_case_id, {})),
        )
    player_by_case = {str(group["case_id"]): str(group["player_name"]) for group in GROUPS}
    for case_id, details in grouped.items():
        details["statistical_unit_id"] = case_id
        details["player_name"] = player_by_case[case_id]
        records[case_id] = details
    payload["cases"] = records
    return payload


def _artifact_paths() -> list[Path]:
    slugs = tuple(_view_lookup())
    return sorted(
        path
        for path in DATA_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".csv", ".parquet"}
        and any(path.name.startswith(f"{slug}_") for slug in slugs)
    )


def _replace_json_values(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {
            replacements.get(str(key), str(key)): _replace_json_values(item, replacements)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_json_values(item, replacements) for item in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


def _artifact_needs_update(path: Path, replacements: dict[str, str]) -> bool:
    if path.suffix.lower() in {".json", ".csv"}:
        text = path.read_text(encoding="utf-8")
        return any(old in text for old in replacements)
    frame = pd.read_parquet(path)
    return any(
        frame[column].astype("string").isin(replacements).any()
        for column in frame.columns
        if frame[column].dtype == "object" or pd.api.types.is_string_dtype(frame[column])
    )


def _update_artifact(path: Path, replacements: dict[str, str]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        _write_json(path, _replace_json_values(_load_json(path), replacements))
        return
    if suffix == ".csv":
        text = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
        return
    frame = pd.read_parquet(path)
    for column in frame.columns:
        if frame[column].dtype == "object" or pd.api.types.is_string_dtype(frame[column]):
            frame[column] = frame[column].map(
                lambda value: replacements.get(value, value)
                if isinstance(value, str)
                else value
            )
    frame.to_parquet(path, index=False)


def regroup(*, apply: bool) -> None:
    registry, replacements = _updated_registry()
    metadata = _updated_metadata(replacements)
    changed_artifacts = [
        path for path in _artifact_paths() if _artifact_needs_update(path, replacements)
    ]
    print(f"Player injury cases to consolidate: {len(GROUPS)}")
    print(f"Video views retained: {sum(len(group['views']) for group in GROUPS)}")
    print(f"Generated artifacts to re-key: {len(changed_artifacts)}")
    for group in GROUPS:
        print(f"  {group['player_name']}: {len(group['views'])} views -> {group['case_id']}")
    if not apply:
        print("Dry run complete; no files changed.")
        return

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_root = WORKSPACE / "artifacts" / "case_regroup_backups" / stamp
    for path in [REGISTRY_PATH, METADATA_PATH, *changed_artifacts]:
        destination = backup_root / path.relative_to(WORKSPACE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    _write_json(REGISTRY_PATH, registry)
    _write_json(METADATA_PATH, metadata)
    for path in changed_artifacts:
        _update_artifact(path, replacements)
    _write_json(
        backup_root / "manifest.json",
        {
            "regrouped_at": datetime.now(UTC).isoformat(),
            "replacements": replacements,
            "artifact_count": len(changed_artifacts),
        },
    )
    print(f"Regrouping applied. Recovery bundle: {backup_root}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the regrouping. Without this flag the script performs a dry run.",
    )
    args = parser.parse_args()
    regroup(apply=args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
