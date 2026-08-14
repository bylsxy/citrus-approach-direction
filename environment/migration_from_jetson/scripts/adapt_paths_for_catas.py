#!/usr/bin/env python3
"""Atomically adapt copied text files from /home/jetson to /home/catas.

The Jetson originals are never touched.  Every remote file changed by this
script is copied to a timestamped backup tree before replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from datetime import datetime, timezone


OLD = b"/home/jetson"
NEW = b"/home/catas"
MAX_TEXT_SIZE = 32 * 1024 * 1024
SKIP_DIRS = {".git", "build", "devel", "install", "__pycache__"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def candidates(root: Path):
    if root.is_file() and not root.is_symlink():
        yield root
        return
    for base, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        base_path = Path(base)
        for name in filenames:
            path = base_path / name
            if not path.is_symlink() and path.is_file():
                yield path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Copied files or directories below /home/catas to adapt",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=None,
        help="Backup directory (default: ~/migration_backups/path-adaptation-TIMESTAMP)",
    )
    args = parser.parse_args()

    home = Path("/home/catas")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_root = args.backup_root or (
        home / "migration_backups" / f"path-adaptation-{stamp}"
    )
    backup_root.mkdir(parents=True, exist_ok=False)

    records = []
    for root in args.roots:
        resolved_root = root.resolve()
        try:
            resolved_root.relative_to(home)
        except ValueError:
            raise SystemExit(f"refusing path outside {home}: {root}")
        if not root.exists():
            raise SystemExit(f"missing input: {root}")

        for path in candidates(root):
            size = path.stat().st_size
            if size > MAX_TEXT_SIZE:
                continue
            original = path.read_bytes()
            if OLD not in original or b"\0" in original:
                continue
            try:
                original.decode("utf-8")
            except UnicodeDecodeError:
                continue

            changed = original.replace(OLD, NEW)
            relative = path.relative_to(home)
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)

            mode = stat.S_IMODE(path.stat().st_mode)
            with tempfile.NamedTemporaryFile(
                prefix=f".{path.name}.",
                dir=path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(changed)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, path)

            records.append(
                {
                    "path": str(path),
                    "backup": str(backup),
                    "replacements": original.count(OLD),
                    "before_sha256": sha256(original),
                    "after_sha256": sha256(changed),
                }
            )

    manifest = backup_root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "old": OLD.decode(),
                "new": NEW.decode(),
                "changed_file_count": len(records),
                "files": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"changed_file_count={len(records)}")
    print(f"backup_root={backup_root}")
    print(f"manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
