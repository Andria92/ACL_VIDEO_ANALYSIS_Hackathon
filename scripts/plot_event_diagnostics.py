"""Generate Milestone 4 event-relative diagnostic plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from acl_motion.visualisation.events import (
    plot_dynamic_bilateral_hka,
    plot_event_extrema_timing,
    plot_event_relative_hka,
    plot_event_window_summary,
    plot_hka_angular_velocity,
    plot_whole_body_event_profile,
)


def main() -> int:
    args = parse_args()
    event_df = pd.read_parquet(args.event_features)
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    windows = summary["run_metadata"]["window_configuration"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_event_relative_hka(event_df, output_dir / f"{args.prefix}_event_relative_hka.png"),
        plot_hka_angular_velocity(event_df, output_dir / f"{args.prefix}_hka_angular_velocity.png"),
        plot_dynamic_bilateral_hka(event_df, output_dir / f"{args.prefix}_dynamic_bilateral_hka.png"),
        plot_whole_body_event_profile(
            event_df, output_dir / f"{args.prefix}_whole_body_event_profile.png"
        ),
        plot_event_window_summary(
            event_df,
            output_dir / f"{args.prefix}_event_window_summary.png",
            windows=windows,
        ),
        plot_event_extrema_timing(
            event_df,
            summary,
            output_dir / f"{args.prefix}_event_extrema_timing.png",
        ),
    ]
    for path in paths:
        print(path)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-features", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output-dir", default="data/diagnostics")
    parser.add_argument("--prefix", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
