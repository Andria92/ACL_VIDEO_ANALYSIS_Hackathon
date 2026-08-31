"""Crash-safe persistence primitives for user-authored application state."""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Self

_PATH_LOCKS: dict[Path, RLock] = {}
_PATH_LOCKS_LOCK = Lock()


def path_lock(path: str | Path) -> RLock:
    """Return one shared re-entrant lock for a persisted file path."""

    resolved = Path(path).resolve()
    with _PATH_LOCKS_LOCK:
        return _PATH_LOCKS.setdefault(resolved, RLock())


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Replace ``path`` atomically after fully flushing a same-directory temp file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_text(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically replace one text file without exposing a partial payload."""

    return atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(
    path: str | Path,
    payload: Any,
    *,
    trailing_newline: bool = False,
) -> Path:
    """Serialize and atomically replace one JSON document."""

    text = json.dumps(payload, indent=2)
    if trailing_newline:
        text += "\n"
    return atomic_write_text(path, text)


def atomic_write_csv(
    path: str | Path,
    *,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Serialize a CSV document in memory and atomically replace its destination."""

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return atomic_write_text(path, buffer.getvalue())


def _fsync_directory(path: Path) -> None:
    """Best-effort directory flush so the rename survives a sudden restart."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


class CaseArtifactTransaction:
    """Rollback all case-prefixed files when a multi-step rebuild raises an error."""

    def __init__(self, root: str | Path, slug: str) -> None:
        self.root = Path(root).resolve()
        self.slug = str(slug)
        self.backup_root: Path | None = None
        self.original_paths: set[Path] = set()

    def __enter__(self) -> Self:
        self.root.mkdir(parents=True, exist_ok=True)
        self.backup_root = Path(
            tempfile.mkdtemp(prefix=f".{self.slug}-rollback-", dir=self.root)
        )
        self.original_paths = self._case_paths()
        for source in self.original_paths:
            relative = source.relative_to(self.root)
            destination = self.backup_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            if exc_type is not None:
                self.rollback()
        finally:
            if self.backup_root is not None:
                shutil.rmtree(self.backup_root, ignore_errors=True)
        return False

    def rollback(self) -> None:
        """Restore the exact pre-transaction case file set."""

        if self.backup_root is None:
            return
        for current in self._case_paths().difference(self.original_paths):
            current.unlink(missing_ok=True)
        for original in self.original_paths:
            backup = self.backup_root / original.relative_to(self.root)
            if backup.exists():
                atomic_write_bytes(original, backup.read_bytes())

    def _case_paths(self) -> set[Path]:
        prefix = f"{self.slug}_"
        return {
            path.resolve()
            for path in self.root.rglob("*")
            if path.is_file()
            and path.name.startswith(prefix)
            and (self.backup_root is None or self.backup_root not in path.parents)
        }
