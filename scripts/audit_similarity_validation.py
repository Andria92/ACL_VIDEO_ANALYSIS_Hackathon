"""Generate the current-library internal validation audit for movement similarity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acl_motion.analytics.exploration import load_exploration_payload
from acl_motion.annotations.registry import default_annotation_cases
from acl_motion.validation.similarity import build_internal_similarity_validation_report


def main() -> int:
    args = parse_args()
    exploration = load_exploration_payload(
        default_annotation_cases(args.video_root),
        summary_dir=args.summary_dir,
        research_metadata_path=args.research_metadata,
        semantics_dir=args.semantics_dir,
    )
    report = build_internal_similarity_validation_report(
        exploration["records"],
        exploration["events"],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote internal similarity validation audit to {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/validation/human/similarity_internal_validation.json",
    )
    parser.add_argument("--summary-dir", default="data/analytics/human")
    parser.add_argument(
        "--research-metadata",
        default="data/annotations/human/case_research_metadata_human.json",
    )
    parser.add_argument("--semantics-dir", default="data/semantics/human")
    parser.add_argument(
        "--video-root",
        default="data/videos/analysis_clips",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
