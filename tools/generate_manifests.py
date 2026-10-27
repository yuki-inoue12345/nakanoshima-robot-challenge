#!/usr/bin/env python3
"""Generate provenance and repository SHA256 manifests deterministically."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGIN_MANIFEST = ROOT / "docs" / "ORIGIN_MANIFEST.tsv"
SOURCE_MANIFEST = ROOT / "docs" / "SOURCE_MANIFEST.sha256"

AR_ROOT = Path("/home/inoue-yuki/ar_navigation_project/ar_path_viewer")
CAMERA_ROOT = Path("/home/inoue-yuki/nuc_audit_20260919")
ROBOT_ROOT = Path("/home/inoue-yuki/robot_archives/renya_20261025")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def origin_for(relative: Path) -> Path | None:
    text = relative.as_posix()
    if text == "robot_ws/README.md":
        return None
    camera_test_prefix = "camera_v7/development_20261019/tests/"
    if text.startswith(camera_test_prefix):
        return CAMERA_ROOT / "camera_v7_fix" / relative.name
    mappings = (
        ("ar_ws/src/ar_path_viewer/", AR_ROOT),
        ("camera_v7/snapshots/runtime_20260919/", CAMERA_ROOT / "camera_v7"),
        (
            "camera_v7/snapshots/production_20260928/",
            CAMERA_ROOT / "camera_v7_production_20260928",
        ),
        ("camera_v7/development_20261019/", CAMERA_ROOT / "camera_v7_fix"),
        ("robot_ws/", ROBOT_ROOT),
    )
    for prefix, source_root in mappings:
        if text.startswith(prefix):
            return source_root / text.removeprefix(prefix)
    if text in {
        "docs/DEPLOYMENT_HANDOFF_20261019.md",
        "docs/PERSON_SEARCH_OPERATION_GUIDE_20261019.md",
    }:
        return CAMERA_ROOT / "camera_v7_fix" / relative.name
    return None


def repository_files() -> list[Path]:
    ignored_directories = {
        ".git",
        ".pytest_cache",
        ".venv",
        "__pycache__",
        "build",
        "install",
        "log",
        "outputs",
        "venv",
    }
    ignored_suffixes = {
        ".avi",
        ".db3",
        ".engine",
        ".gguf",
        ".mcap",
        ".mp4",
        ".onnx",
        ".pem",
        ".pt",
        ".pyc",
    }
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not ignored_directories.intersection(path.relative_to(ROOT).parts)
        and path.suffix.lower() not in ignored_suffixes
        and path != SOURCE_MANIFEST
    )


def main() -> None:
    copied_rows: list[tuple[str, str, str]] = []
    mismatches: list[str] = []
    for destination in repository_files():
        relative = destination.relative_to(ROOT)
        source = origin_for(relative)
        if source is None:
            continue
        if not source.is_file():
            mismatches.append(f"missing source: {source}")
            continue
        destination_hash = digest(destination)
        source_hash = digest(source)
        if destination_hash != source_hash:
            mismatches.append(f"hash mismatch: {relative} != {source}")
        copied_rows.append((relative.as_posix(), str(source), source_hash))

    if mismatches:
        raise SystemExit("\n".join(mismatches))

    lines = ["destination\tsource\tsha256"]
    lines.extend("\t".join(row) for row in copied_rows)
    ORIGIN_MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")

    checksum_lines = []
    for path in repository_files():
        relative = path.relative_to(ROOT).as_posix()
        checksum_lines.append(f"{digest(path)}  {relative}")
    SOURCE_MANIFEST.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    print(f"verified copied files: {len(copied_rows)}")
    print(f"repository files hashed: {len(checksum_lines)}")


if __name__ == "__main__":
    main()
