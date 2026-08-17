"""Build M5.9.1 case-level Movement Signature prototype for a human case."""

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

from acl_motion.annotations.registry import case_by_slug
from acl_motion.annotations.storage import load_movement_window_json
from acl_motion.signatures import build_case_movement_signature


def main() -> int:
    args = parse_args()
    case = case_by_slug(args.case_slug, None)
    dynamic_df = pd.read_parquet(args.dynamic_features)
    path_df = pd.read_parquet(args.path_features)
    movement_story = json.loads(Path(args.movement_story).read_text(encoding="utf-8"))
    movement_window = load_movement_window_json(args.movement_window).to_dict()
    path_quality = json.loads(Path(args.path_quality).read_text(encoding="utf-8"))
    signature = build_case_movement_signature(
        case_id=case.case_id,
        source_id=case.source_id,
        dynamic_df=dynamic_df,
        path_df=path_df,
        movement_story=movement_story,
        movement_window=movement_window,
        path_quality_summary=path_quality,
    )

    for output in (
        args.long_output,
        args.matrix_output,
        args.registry_output,
        args.summary_output,
    ):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
    signature.long_table.to_csv(args.long_output, index=False)
    signature.matrix_preview.to_csv(args.matrix_output, index=False)
    signature.registry.to_csv(args.registry_output, index=False)
    Path(args.summary_output).write_text(
        json.dumps(
            _json_ready(
                {
                    "run_metadata": signature.run_metadata,
                    "summary": signature.summary,
                    "long_output": args.long_output,
                    "matrix_output": args.matrix_output,
                    "registry_output": args.registry_output,
                }
            ),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote Movement Signature long table to {args.long_output}")
    print(f"Wrote Movement Signature matrix preview to {args.matrix_output}")
    print(f"Wrote clustering feature registry to {args.registry_output}")
    print(f"Wrote Movement Signature summary to {args.summary_output}")
    print(json.dumps(_json_ready(signature.summary), indent=2, allow_nan=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-slug", default="christen_press")
    parser.add_argument("--dynamic-features", default="data/dynamics/human/christen_press_dynamic_features.parquet")
    parser.add_argument("--path-features", default="data/path/human/christen_press_projected_movement_path.parquet")
    parser.add_argument("--movement-story", default="data/phases/human/christen_press_movement_phases.json")
    parser.add_argument("--movement-window", default="data/annotations/human/christen_press_movement_window_human.json")
    parser.add_argument("--path-quality", default="data/quality/human/christen_press_path_quality_summary.json")
    parser.add_argument(
        "--long-output",
        default="data/signatures/human/christen_press_case_movement_signature_long.csv",
    )
    parser.add_argument(
        "--matrix-output",
        default="data/signatures/human/christen_press_case_movement_signature_matrix.csv",
    )
    parser.add_argument(
        "--registry-output",
        default="data/signatures/human/christen_press_clustering_feature_registry.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="data/signatures/human/christen_press_case_movement_signature.json",
    )
    return parser.parse_args()


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
