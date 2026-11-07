"""Create and safely unpack the single ZIP uploaded to the grading platform.

Only ONNX and JSON data are allowed. This module uses the Python
standard library; the grading runner separately checks ONNX/input semantics.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import stat
import tempfile
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_FILES = ("controller.onnx", "controller.json")
MIB = 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * MIB
MAX_EXPANDED_BYTES = 256 * MIB
MAX_METADATA_BYTES = MIB
CHUNK_BYTES = MIB


class BundleError(ValueError):
    """An uploaded model package is incomplete, unsafe or malformed."""


def _member_limit(name: str) -> int:
    if name == "controller.json":
        return MAX_METADATA_BYTES
    return MAX_EXPANDED_BYTES


def _check_entries(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    entries = archive.infolist()
    if not entries or len(entries) != len(MODEL_FILES):
        raise BundleError("Unexpected archive member count")
    names = [entry.filename for entry in entries]
    if len(set(names)) != len(names):
        raise BundleError("Duplicate archive member names are not allowed")
    if set(names) != set(MODEL_FILES):
        raise BundleError("ZIP root must contain exactly: " + ", ".join(MODEL_FILES))

    total = 0
    for entry in entries:
        file_type = stat.S_IFMT(entry.external_attr >> 16)
        if entry.is_dir() or file_type not in (0, stat.S_IFREG) or entry.external_attr & 0x10:
            raise BundleError("Only regular files are allowed; links and directories are rejected")
        if entry.orig_filename != entry.filename:
            raise BundleError("Invalid archive member name")
        if entry.flag_bits & 0x41:
            raise BundleError("Encrypted archive entries are not supported")
        if entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
            raise BundleError("Only stored or deflated ZIP entries are supported")
        limit = _member_limit(entry.filename)
        if entry.file_size <= 0 or entry.file_size > limit:
            raise BundleError(f"{entry.filename} is empty or exceeds its expanded size limit")
        total += entry.file_size
    if total > MAX_EXPANDED_BYTES:
        raise BundleError("Archive exceeds its total expanded size limit")
    return entries


def _copy_bounded(source, destination, limit: int) -> int:
    count = 0
    while True:
        chunk = source.read(min(CHUNK_BYTES, limit - count + 1))
        if not chunk:
            return count
        count += len(chunk)
        if count > limit:
            raise BundleError("File exceeds its expanded size limit")
        destination.write(chunk)


def _validate_metadata(path: Path) -> None:
    from submission.controller_contract import read_manifest
    read_manifest(path)


def _validate_directory(directory: Path) -> None:
    total = 0
    for name in MODEL_FILES:
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise BundleError(f"Missing regular export file: {name}")
        size = path.stat().st_size
        if size <= 0 or size > _member_limit(name):
            raise BundleError(f"{name} is empty or exceeds its size limit")
        total += size
    if total > MAX_EXPANDED_BYTES:
        raise BundleError("Export exceeds its total expanded size limit")
    _validate_metadata(directory / "controller.json")


def create_bundle(model_dir: str | Path, output: str | Path) -> Path:
    """Atomically create one ZIP from the two local controller export files."""
    model_dir, output = Path(model_dir), Path(output)
    if output.suffix.lower() != ".zip":
        raise BundleError("Output model package must have a .zip extension")
    temporary = None
    try:
        _validate_directory(model_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=f".{output.name}-", suffix=".tmp", dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            total = 0
            for name in MODEL_FILES:
                with (model_dir / name).open("rb") as source, archive.open(name, "w") as destination:
                    total += _copy_bounded(source, destination, min(_member_limit(name), MAX_EXPANDED_BYTES - total))
        if temporary.stat().st_size > MAX_ARCHIVE_BYTES:
            raise BundleError("Compressed package exceeds the 128 MiB upload limit")
        os.replace(temporary, output)
        temporary = None
        return output
    except (OSError, ValueError, SyntaxError, RecursionError, zipfile.BadZipFile, zlib.error, EOFError, RuntimeError, NotImplementedError) as exc:
        if isinstance(exc, BundleError):
            raise
        raise BundleError(f"Cannot create model package: {exc}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def unpack_bundle(zip_path: str | Path, dest: str | Path) -> Path:
    """Validate data-only ZIP and publish it into a new or empty directory.

    The destination is never merged with existing files. Failed validation
    leaves no partial model directory behind. Local I/O failures remain
    OSError so graders can retry infrastructure faults instead of blaming
    the submitted model.
    """
    zip_path, dest = Path(zip_path), Path(dest)
    try:
        if dest.is_symlink() or (dest.exists() and (not dest.is_dir() or any(dest.iterdir()))):
            raise BundleError("Model destination must be a new or empty directory")
        if zip_path.stat().st_size > MAX_ARCHIVE_BYTES:
            raise BundleError("Compressed package exceeds the 128 MiB upload limit")
        with zipfile.ZipFile(zip_path) as archive:
            entries = _check_entries(archive)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f".{dest.name}-", dir=dest.parent) as temporary:
                staging = Path(temporary) / "model"
                staging.mkdir()
                for entry in entries:
                    with archive.open(entry) as source, (staging / entry.filename).open("xb") as output:
                        count = _copy_bounded(source, output, entry.file_size)
                    if count != entry.file_size:
                        raise BundleError(f"Truncated archive member: {entry.filename}")
                _validate_directory(staging)
                if dest.exists():
                    # rmdir only succeeds if the destination remains empty.
                    dest.rmdir()
                staging.replace(dest)
        return dest
    except (ValueError, SyntaxError, RecursionError, zipfile.BadZipFile, zlib.error, EOFError, RuntimeError, NotImplementedError) as exc:
        if isinstance(exc, BundleError):
            raise
        raise BundleError(f"Invalid model package: {exc}") from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Package ONNX export files into one platform upload; no local simulation is run.")
    parser.add_argument("--model-dir", type=Path, default=Path("submission/model"))
    parser.add_argument("--output", type=Path, default=Path("results/submission_model.zip"))
    args = parser.parse_args(argv)
    try:
        output = create_bundle(args.model_dir, args.output)
    except BundleError as exc:
        parser.exit(1, f"Model package error: {exc}\n")
    print(f"Model package ready: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
