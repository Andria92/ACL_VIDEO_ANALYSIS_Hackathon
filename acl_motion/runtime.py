"""Runtime policy for the reproducible ACL analysis application."""

from __future__ import annotations

import sys

REQUIRED_PYTHON = (3, 12)


def ensure_supported_runtime(version_info=None) -> None:
    """Reject interpreters outside the project's pinned Python 3.12 line."""

    current = version_info or sys.version_info
    current_pair = (int(current[0]), int(current[1]))
    if current_pair != REQUIRED_PYTHON:
        current_label = ".".join(str(part) for part in current_pair)
        raise RuntimeError(
            "ACL Movement Analytics Lab requires Python 3.12 exactly; "
            f"the current interpreter is Python {current_label}. "
            "Launch it with .venv/bin/python scripts/run_annotation_ui.py."
        )
