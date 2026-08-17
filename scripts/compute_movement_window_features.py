"""Compute human Movement-Window-relative feature trajectories."""

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

from acl_motion.annotations.movement_window import (
    add_movement_timing_columns,
    filter_to_movement_window,
    movement_window_to_event_annotation,
)
from acl_motion.annotations.storage import load_human_annotation_session, load_movement_window_json
from acl_motion.events.temporal import (
    EventWindow,
    build_event_relative_features,
    build_event_summary,
    build_window_summaries,
)

MOVEMENT_WINDOW_CONFIGURATION = (
    EventWindow("FINAL_1500_TO_1000_MS", -1500.0, -1000.0),
    EventWindow("FINAL_1000_TO_750_MS", -1000.0, -750.0),
    EventWindow("FINAL_750_TO_500_MS", -750.0, -500.0),
    EventWindow("FINAL_500_TO_250_MS", -500.0, -250.0),
    EventWindow("FINAL_250_TO_END_MS", -250.0, 1.0),
)


def main() -> int:
    args = parse_args()
    feature_df = pd.read_parquet(args.feature_input)
    session = load_human_annotation_session(args.session)
    movement_window = load_movement_window_json(args.movement_window)
    filtered = filter_to_movement_window(feature_df, movement_window)
    compatibility_event = movement_window_to_event_annotation(session, movement_window)
    event_df = build_event_relative_features(filtered, compatibility_event)
    event_df = add_movement_timing_columns(event_df, movement_window)
    event_df["time_reference"] = "movement_end"
    event_df["event_relative_ms"] = event_df["movement_end_relative_ms"]
    window_summaries = build_window_summaries(
        event_df,
        windows=MOVEMENT_WINDOW_CONFIGURATION,
        minimum_completeness=args.minimum_window_completeness,
    )
    summary = build_event_summary(
        event_df,
        window_summaries,
        compatibility_event,
        event_annotation_file=args.movement_window,
        feature_input_file=args.feature_input,
        minimum_window_completeness=args.minimum_window_completeness,
        windows=MOVEMENT_WINDOW_CONFIGURATION,
    )
    summary["movement_window"] = movement_window.to_dict()
    summary["time_reference"] = {
        "movement_elapsed_ms": "0 ms = Movement Start",
        "movement_end_relative_ms": "0 ms = Movement End",
        "event_relative_ms": "legacy alias for movement_end_relative_ms in this human run",
    }
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
    print(f"Wrote {len(event_df)} movement-window feature rows to {output_path}")
    print(f"Wrote {len(window_summaries)} movement-window summary rows to {window_output}")
    print(f"Wrote movement summary to {summary_output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-input", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--movement-window", required=True)
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
