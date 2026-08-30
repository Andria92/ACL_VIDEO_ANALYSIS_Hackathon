"""Compare two independently created human annotation-session JSON files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acl_motion.annotations.storage import load_human_annotation_session
from acl_motion.annotations.validation import compare_independent_annotation_sessions
from acl_motion.video.io import read_video_metadata


def main() -> int:
    args = parse_args()
    metadata = read_video_metadata(args.video)
    report = compare_independent_annotation_sessions(
        load_human_annotation_session(args.first_session),
        load_human_annotation_session(args.second_session),
        frame_count=metadata.frame_count,
        fps=metadata.fps,
    ).to_dict()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote independent-annotation agreement report to {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-session", required=True)
    parser.add_argument("--second-session", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
