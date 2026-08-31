"""Run the local video review and cutter UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acl_motion.runtime import ensure_supported_runtime
from acl_motion.ui.video_cutter import DEFAULT_VIDEO_ROOTS, run_video_cutter_ui, smoke_test

ensure_supported_runtime()


def main() -> int:
    args = parse_args()
    video_roots = tuple(args.video_root) if args.video_root else DEFAULT_VIDEO_ROOTS
    if args.smoke_test:
        result = smoke_test(output_dir=args.output_dir, video_roots=video_roots)
        print(json.dumps(result, indent=2))
        return 0
    run_video_cutter_ui(
        host=args.host,
        port=args.port,
        video_roots=video_roots,
        output_dir=args.output_dir,
        main_menu_url=args.main_menu_url,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--video-root",
        action="append",
        help=(
            "Directory to scan for source videos. Repeat to add multiple roots. "
            "Defaults to data/videos and the local injury_videos folder."
        ),
    )
    parser.add_argument("--output-dir", default="data/videos/analysis_clips")
    parser.add_argument(
        "--main-menu-url",
        default="http://127.0.0.1:8765/",
        help="URL opened by the Video Cutter's Main menu button.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate UI wiring without starting a persistent server or writing clips.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
