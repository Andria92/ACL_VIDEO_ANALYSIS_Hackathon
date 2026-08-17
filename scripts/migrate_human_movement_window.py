"""Migrate an existing human annotation session to Movement Window terminology."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acl_motion.annotations.movement_window import migrate_session_to_movement_window
from acl_motion.annotations.storage import (
    load_human_annotation_session,
    save_human_annotation_session,
)
from acl_motion.video.io import read_video_metadata


def main() -> int:
    args = parse_args()
    session = load_human_annotation_session(args.session)
    metadata = read_video_metadata(session.provenance.video_path)
    migrated = migrate_session_to_movement_window(session, fps=metadata.fps)
    paths = save_human_annotation_session(migrated, args.output_dir, args.slug)
    window = migrated.movement_window
    print(f"Migrated {args.slug} human annotation session.")
    print(f"Movement Start: {window.movement_start_frame}")
    print(f"Movement End: {window.movement_end_frame}")
    print(f"Movement duration ms: {window.duration_ms:.1f}")
    print(f"Movement Window JSON: {paths.movement_window_json}")
    print(f"Compatibility Event JSON: {paths.event_json}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--output-dir", default="data/annotations/human")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
