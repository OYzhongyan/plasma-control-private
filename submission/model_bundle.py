"""Create and safely unpack the single ZIP uploaded to the grading platform.

Only ONNX, JSON and numeric NPZ data are allowed. This module uses the Python
standard library; the grading runner separately checks ONNX/input semantics.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
import zipfile
import zlib

MODEL_FILES = ("policy2.onnx", "policy2.metadata.json", "policy2_vecnormalize_stats.npz")
MIB = 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * MIB
MAX_EXPANDED_BYTES = 256 * MIB
MAX_METADATA_BYTES = MIB
MAX_STATS_BYTES = 64 * MIB
MAX_NPZ_MEMBERS = 32
CHUNK_BYTES = MIB


class BundleError(ValueError):
    """An uploaded model package is incomplete, unsafe or malformed."""


def _member_limit(name: str) -> int:
    if name == "policy2.metadata.json":
        return MAX_METADATA_BYTES
    if name == "policy2_vecnormalize_stats.npz":
        return MAX_STATS_BYTES
    return MAX_EXPANDED_BYTES


def _check_entries(archive: zipfile.ZipFile, *, npz: bool = False) -> list[zipfile.ZipInfo]:
    entries = archive.infolist()
    maximum = MAX_NPZ_MEMBERS if npz else len(MODEL_FILES)
    if not entries or len(entries) > maximum:
        raise BundleError("Unexpected archive member count")
    names = [entry.filename for entry in entries]
    if len(set(names)) != len(names):
        raise BundleError("Duplicate archive member names are not allowed")
    if npz:
        if not {"obs_mean.npy", "obs_var.npy"}.issubset(names):
            raise BundleError("Normalization NPZ must contain obs_mean and obs_var")
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*\.npy", name) for name in names):
            raise BundleError("Normalization NPZ must contain only named NPY arrays")
    elif set(names) != set(MODEL_FILES):
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
        limit = MAX_STATS_BYTES if npz else _member_limit(entry.filename)
        if entry.file_size <= 0 or entry.file_size > limit:
            raise BundleError(f"{entry.filename} is empty or exceeds its expanded size limit")
        total += entry.file_size
    if total > (MAX_STATS_BYTES if npz else MAX_EXPANDED_BYTES):
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
    def reject_constant(value):
        raise BundleError(f"Metadata contains a nonstandard JSON number: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise BundleError(f"Metadata contains duplicate JSON key: {key}")
            result[key] = value
        return result

    raw = path.read_bytes()
    if len(raw) > MAX_METADATA_BYTES:
        raise BundleError("Metadata exceeds its size limit")
    value = json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise BundleError("Model metadata must be a JSON object")


def _validate_npy(stream, size: int) -> None:
    """Check an NPY header and payload length without importing or unpickling it."""
    if stream.read(6) != b"\x93NUMPY":
        raise BundleError("Normalization member is not an NPY array")
    version = stream.read(2)
    if version not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
        raise BundleError("Unsupported NPY version")
    length_bytes = 2 if version[0] == 1 else 4
    header_size_bytes = stream.read(length_bytes)
    if len(header_size_bytes) != length_bytes:
        raise BundleError("Truncated NPY header")
    header_size = struct.unpack("<H" if length_bytes == 2 else "<I", header_size_bytes)[0]
    if header_size > 65536 or header_size > size - 8 - length_bytes:
        raise BundleError("NPY header exceeds its size limit")
    header = ast.literal_eval(stream.read(header_size).decode("utf-8" if version[0] == 3 else "latin-1").strip())
    if not isinstance(header, dict) or set(header) != {"descr", "shape", "fortran_order"}:
        raise BundleError("Invalid NPY header fields")
    dtype = header["descr"]
    if not isinstance(dtype, str) or not re.fullmatch(r"[<>=|][biufc][1-9][0-9]*", dtype):
        raise BundleError("Normalization arrays must use plain numeric dtypes, not object or structured dtypes")
    allowed_sizes = {"b": {1}, "i": {1, 2, 4, 8}, "u": {1, 2, 4, 8}, "f": {2, 4, 8, 16}, "c": {8, 16, 32}}
    itemsize = int(dtype[2:])
    if itemsize not in allowed_sizes[dtype[1]]:
        raise BundleError("Unsupported numeric dtype size")
    shape = header["shape"]
    if not isinstance(shape, tuple) or len(shape) > 8 or any(type(dim) is not int or dim < 0 for dim in shape):
        raise BundleError("Invalid normalization array shape")
    if type(header["fortran_order"]) is not bool:
        raise BundleError("Invalid NPY array order")
    data_size = math.prod(shape) * itemsize
    if data_size > MAX_STATS_BYTES or data_size != size - (8 + length_bytes + header_size):
        raise BundleError("Normalization array shape does not match its bounded payload size")
    # Consume the entire bounded payload to check truncation and the ZIP CRC.
    consumed = 0
    while True:
        chunk = stream.read(min(CHUNK_BYTES, data_size - consumed + 1))
        if not chunk:
            break
        consumed += len(chunk)
        if consumed > data_size:
            raise BundleError("Normalization array payload is larger than its declared shape")
    if consumed != data_size:
        raise BundleError("Truncated normalization array payload")


def _validate_stats(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        for entry in _check_entries(archive, npz=True):
            with archive.open(entry) as stream:
                _validate_npy(stream, entry.file_size)


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
    _validate_metadata(directory / "policy2.metadata.json")
    _validate_stats(directory / "policy2_vecnormalize_stats.npz")


def create_bundle(model_dir: str | Path, output: str | Path) -> Path:
    """Atomically create one ZIP from the three trusted local export files."""
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
