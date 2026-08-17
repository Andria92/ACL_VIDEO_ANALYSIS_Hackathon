"""Download the default YOLOv8n pose model asset."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt"
DEFAULT_OUTPUT = Path("data/models/yolov8n-pose.pt")


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        print(f"Model already exists: {output}")
        return 0

    print(f"Downloading {args.url} -> {output}")
    urlretrieve(args.url, output)
    print(f"Wrote model asset to {output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
