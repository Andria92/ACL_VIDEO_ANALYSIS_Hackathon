"""Compute Milestone 4 event-relative trajectories and first derivatives."""

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

from acl_motion.cases.annotations import load_event_annotation
from acl_motion.events.temporal import (
    DEFAULT_EVENT_WINDOWS,
    build_event_relative_features,
    build_event_summary,
    build_window_summaries,
)


def main() -> int:
    args = parse_args()
    feature_df = pd.read_parquet(args.feature_input)
    annotation = load_event_annotation(args.event_annotation)
    event_df = build_event_relative_features(feature_df, annotation)
    window_summaries = build_window_summaries(
        event_df,
        windows=DEFAULT_EVENT_WINDOWS,
        minimum_completeness=args.minimum_window_completeness,
    )
    summary = build_event_summary(
        event_df,
        window_summaries,
        annotation,
        event_annotation_file=args.event_annotation,
        feature_input_file=args.feature_input,
        minimum_window_completeness=args.minimum_window_completeness,
        windows=DEFAULT_EVENT_WINDOWS,
    )

    output_path = Path(args.output)
    window_output = Path(args.window_output)
    summary_output = Path(args.summary_output)
    for path in (output_path, window_output, summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    event_df.to_parquet(output_path, index=False)
    window_summaries.to_parquet(window_output, index=False)
    summary_output.write_text(
        json.dumps(_json_ready(summary), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Wrote {len(event_df)} event-relative feature rows to {output_path}")
    print(f"Wrote {len(window_summaries)} window summary rows to {window_output}")
    print(f"Wrote event summary to {summary_output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-input", required=True)
    parser.add_argument("--event-annotation", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--window-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--minimum-window-completeness", type=float, default=0.5)
    return parser.parse_args()


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
