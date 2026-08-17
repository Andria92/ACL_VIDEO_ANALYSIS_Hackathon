"""Apply Milestone 4.1 dynamic reliability hardening to M4 event features."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from acl_motion.events.dynamic_reliability import (
    RobustDynamicConfig,
    build_dynamic_quality_summary,
    build_dynamic_spike_audit,
    build_dynamic_window_summaries,
    harden_dynamic_reliability,
)


def main() -> int:
    args = parse_args()
    event_df = pd.read_parquet(args.event_features)
    event_df = _ensure_movement_timing(event_df)
    config = RobustDynamicConfig(
        half_window_samples=args.half_window_samples,
        minimum_samples=args.minimum_samples,
    )
    dynamic_df = harden_dynamic_reliability(event_df, config=config)
    spike_audit = build_dynamic_spike_audit(dynamic_df, top_n=args.spike_top_n)
    window_summary = build_dynamic_window_summaries(dynamic_df)
    quality_summary = build_dynamic_quality_summary(
        dynamic_df,
        config=config,
        event_annotation_file=args.event_annotation,
        event_features_file=args.event_features,
    )

    output = Path(args.output)
    quality_output = Path(args.quality_output)
    spike_output = Path(args.spike_audit_output)
    window_output = Path(args.window_output)
    for path in (output, quality_output, spike_output, window_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    dynamic_df.to_parquet(output, index=False)
    spike_audit.to_csv(spike_output, index=False)
    window_summary.to_parquet(window_output, index=False)
    quality_output.write_text(
        json.dumps(_json_ready(quality_summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Wrote {len(dynamic_df)} hardened dynamic rows to {output}")
    print(f"Wrote dynamic quality summary to {quality_output}")
    print(f"Wrote {len(spike_audit)} spike audit rows to {spike_output}")
    print(f"Wrote {len(window_summary)} dynamic window summary rows to {window_output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-features", required=True)
    parser.add_argument("--event-annotation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quality-output", required=True)
    parser.add_argument("--spike-audit-output", required=True)
    parser.add_argument("--window-output", required=True)
    parser.add_argument("--half-window-samples", type=int, default=2)
    parser.add_argument("--minimum-samples", type=int, default=3)
    parser.add_argument("--spike-top-n", type=int, default=10)
    return parser.parse_args()


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _ensure_movement_timing(event_df: pd.DataFrame) -> pd.DataFrame:
    output = event_df.copy()
    if "movement_end_relative_ms" not in output.columns and "event_relative_ms" in output.columns:
        output["movement_end_relative_ms"] = output["event_relative_ms"]
    if "movement_elapsed_ms" not in output.columns:
        start_ms = float(output["timestamp_ms"].min()) if not output.empty else 0.0
        output["movement_elapsed_ms"] = output["timestamp_ms"].astype(float) - start_ms
    return output


if __name__ == "__main__":
    raise SystemExit(main())
