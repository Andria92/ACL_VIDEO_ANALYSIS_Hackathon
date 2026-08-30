"""Safely rebuild comparison ranges after the angular-semantics correction."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from acl_motion.analytics.similarity import COMPARISON_STATISTICS_VERSION
from acl_motion.geometry.angular_semantics import (
    ANGULAR_STATISTICS_VERSION,
    angle_type_for_metric,
    angular_difference,
    measurement_range_for_metric,
    range_semantics_for_metric,
)

SUMMARY_SUFFIX = "_case_feature_summary.parquet"


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    summary_dir = data_root / "analytics" / "human"
    dynamic_dir = data_root / "dynamics" / "human"
    semantics_dir = data_root / "semantics" / "human"
    summary_paths = sorted(summary_dir.glob(f"*{SUMMARY_SUFFIX}"))
    if not summary_paths:
        raise FileNotFoundError(f"No case feature summaries found in {summary_dir}.")

    plans: list[tuple[Path, pd.DataFrame, list[dict[str, object]]]] = []
    for summary_path in summary_paths:
        slug = summary_path.name.removesuffix(SUMMARY_SUFFIX)
        dynamic_path = dynamic_dir / f"{slug}_dynamic_features.parquet"
        semantics_path = semantics_dir / f"{slug}_observable_movement_descriptions.json"
        if not dynamic_path.exists():
            raise FileNotFoundError(f"Missing dynamic features for {slug}: {dynamic_path}")
        if not semantics_path.exists():
            raise FileNotFoundError(f"Missing supported-interval evidence for {slug}: {semantics_path}")
        semantics = json.loads(semantics_path.read_text(encoding="utf-8"))
        corrected, changes = corrected_summary(
            pd.read_parquet(summary_path),
            pd.read_parquet(dynamic_path),
            slug=slug,
            supported_intervals=semantics.get("supported_intervals") or [],
        )
        plans.append((summary_path, corrected, changes))

    change_rows = [change for _, _, changes in plans for change in changes]
    report = {
        "angular_statistics_version": ANGULAR_STATISTICS_VERSION,
        "mode": "apply" if args.apply else "dry_run",
        "summary_file_count": len(plans),
        "changed_range_count": sum(bool(item["range_changed"]) for item in change_rows),
        "changed_mean_count": sum(bool(item["mean_changed"]) for item in change_rows),
        "changed_pre_late_change_count": sum(
            bool(item["pre_late_change_changed"]) for item in change_rows
        ),
        "affected_rows": change_rows,
    }
    if not args.apply:
        print(json.dumps(report, indent=2))
        return 0

    generated_at = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = (
        Path(args.backup_dir)
        if args.backup_dir
        else PROJECT_ROOT / "artifacts" / "backups" / f"angular_ranges_{generated_at}"
    )
    if backup_dir.exists() and any(backup_dir.iterdir()):
        raise FileExistsError(f"Backup directory is not empty: {backup_dir}")
    backup_dir.mkdir(parents=True, exist_ok=True)

    for summary_path, corrected, _ in plans:
        shutil.copy2(summary_path, backup_dir / summary_path.name)
        temporary = summary_path.with_suffix(".angular-statistics.tmp.parquet")
        corrected.to_parquet(temporary, index=False)
        os.replace(temporary, summary_path)

    report["backup_dir"] = str(backup_dir)
    report["applied_at"] = datetime.now(UTC).isoformat()
    (backup_dir / "migration_manifest.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "affected_rows"}, indent=2))
    return 0


def corrected_summary(
    summary: pd.DataFrame,
    dynamic: pd.DataFrame,
    *,
    slug: str,
    supported_intervals: list[dict[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Return an updated comparison summary without mutating either input."""

    required_summary = {"feature_name", "mean", "range", "pre_late_change"}
    required_dynamic = {
        "feature_name",
        "feature_value",
        "feature_status",
        "event_relative_ms",
    }
    if not required_summary.issubset(summary.columns):
        missing = sorted(required_summary.difference(summary.columns))
        raise ValueError(f"{slug}: summary is missing columns {missing}.")
    if not required_dynamic.issubset(dynamic.columns):
        missing = sorted(required_dynamic.difference(dynamic.columns))
        raise ValueError(f"{slug}: dynamic features are missing columns {missing}.")

    output = summary.copy()
    output["range_semantics"] = ""
    output["angular_statistics_version"] = ANGULAR_STATISTICS_VERSION
    output["comparison_statistics_version"] = ""
    output["comparison_support_scope"] = ""
    changes: list[dict[str, object]] = []
    interval_mask = _supported_interval_mask(dynamic, supported_intervals)
    if not interval_mask.any():
        return output, changes
    output["comparison_statistics_version"] = COMPARISON_STATISTICS_VERSION
    output["comparison_support_scope"] = "observable_supported_intervals"
    for index, row in output.iterrows():
        metric_name = str(row["feature_name"])
        supported = dynamic.loc[
            dynamic["feature_name"].eq(metric_name)
            & dynamic["feature_status"].eq("SUPPORTED")
            & interval_mask
        ].sort_values("event_relative_ms")
        values = pd.to_numeric(supported["feature_value"], errors="coerce").dropna()
        corrected_mean = float(values.mean()) if not values.empty else None
        old_mean = _finite_or_none(row["mean"])
        output.at[index, "mean"] = corrected_mean if corrected_mean is not None else np.nan
        corrected_range = measurement_range_for_metric(metric_name, values)
        old_range = _finite_or_none(row["range"])
        new_range = _finite_or_none(corrected_range)
        output.at[index, "range"] = new_range if new_range is not None else np.nan
        output.at[index, "range_semantics"] = range_semantics_for_metric(metric_name)

        pre_late = supported.loc[
            supported["event_relative_ms"].ge(-250.0)
            & supported["event_relative_ms"].lt(0.0)
        ]
        pre_late_values = pd.to_numeric(pre_late["feature_value"], errors="coerce").dropna()
        corrected_change = _start_to_end_change(metric_name, pre_late_values)
        old_change = _finite_or_none(row["pre_late_change"])
        output.at[index, "pre_late_change"] = (
            corrected_change if corrected_change is not None else np.nan
        )
        range_changed = not _same_optional_number(old_range, new_range)
        mean_changed = not _same_optional_number(old_mean, corrected_mean)
        change_changed = not _same_optional_number(old_change, corrected_change)
        if mean_changed or range_changed or change_changed:
            changes.append(
                {
                    "slug": slug,
                    "feature_name": metric_name,
                    "range_semantics": range_semantics_for_metric(metric_name),
                    "old_mean": old_mean,
                    "new_mean": corrected_mean,
                    "mean_changed": mean_changed,
                    "old_range": old_range,
                    "new_range": new_range,
                    "range_changed": range_changed,
                    "old_pre_late_change": old_change,
                    "new_pre_late_change": corrected_change,
                    "pre_late_change_changed": change_changed,
                }
            )
    return output, changes


def _supported_interval_mask(
    dynamic: pd.DataFrame,
    supported_intervals: list[dict[str, object]],
) -> pd.Series:
    frames = pd.to_numeric(dynamic["source_frame_index"], errors="coerce")
    mask = pd.Series(False, index=dynamic.index)
    for interval in supported_intervals:
        try:
            start = int(interval["start_frame"])
            end = int(interval["end_frame"])
        except (KeyError, TypeError, ValueError):
            continue
        mask |= frames.between(start, end, inclusive="both")
    return mask


def _start_to_end_change(metric_name: str, values: pd.Series) -> float | None:
    if len(values) < 2:
        return None
    start = float(values.iloc[0])
    end = float(values.iloc[-1])
    angle_type = angle_type_for_metric(metric_name)
    if angle_type is not None:
        return float(angular_difference(start, end, angle_type))
    return end - start


def _finite_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _same_optional_number(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-dir")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
