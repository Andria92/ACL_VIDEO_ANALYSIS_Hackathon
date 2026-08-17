"""Run the local M5.5 human annotation UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acl_motion.ui.annotation import run_annotation_ui, smoke_test


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        result = smoke_test(output_dir=args.output_dir, video_root=args.video_root)
        print(json.dumps(result, indent=2))
        return 0
    run_annotation_ui(
        host=args.host,
        port=args.port,
        output_dir=args.output_dir,
        video_root=args.video_root,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-dir", default="data/annotations/human")
    parser.add_argument(
        "--video-root",
        default="/Users/andriagryffinpro/Desktop/injury_videos",
        help="Local directory containing registered validation clips.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate UI wiring without starting a persistent server or writing annotations.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
