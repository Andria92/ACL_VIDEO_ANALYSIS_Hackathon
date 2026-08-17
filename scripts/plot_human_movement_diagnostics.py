"""Generate product-oriented diagnostics for a human Movement Window run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from acl_motion.annotations.storage import (
    load_human_annotation_session,
    load_movement_window_json,
)
from acl_motion.visualisation.dynamics import plot_hka_raw_vs_robust_velocity
from acl_motion.visualisation.events import (
    plot_dynamic_bilateral_hka,
    plot_event_relative_hka,
    plot_whole_body_event_profile,
)
from acl_motion.visualisation.human import (
    plot_body_region_evidence_availability,
    plot_movement_window_timeline,
)


def main() -> int:
    args = parse_args()
    session = load_human_annotation_session(args.session)
    movement_window = load_movement_window_json(args.movement_window)
    dynamic_df = pd.read_parquet(args.dynamic_features)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_movement_window_timeline(
            session.roi_keyframes,
            movement_window,
            output_dir / f"{args.prefix}_movement_window.png",
        ),
        plot_body_region_evidence_availability(
            dynamic_df,
            output_dir / f"{args.prefix}_evidence_availability.png",
        ),
        plot_whole_body_event_profile(
            dynamic_df,
            output_dir / f"{args.prefix}_whole_body_movement_profile.png",
        ),
        plot_event_relative_hka(dynamic_df, output_dir / f"{args.prefix}_hka_trajectories.png"),
        plot_dynamic_bilateral_hka(
            dynamic_df,
            output_dir / f"{args.prefix}_bilateral_hka_dynamics.png",
        ),
        plot_hka_raw_vs_robust_velocity(
            dynamic_df,
            output_dir / f"{args.prefix}_robust_dynamics.png",
        ),
    ]
    for output in outputs:
        print(f"Wrote {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--movement-window", required=True)
    parser.add_argument("--dynamic-features", required=True)
    parser.add_argument("--output-dir", default="data/diagnostics/human")
    parser.add_argument("--prefix", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
