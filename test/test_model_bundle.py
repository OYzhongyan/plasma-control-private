"""Model uploads contain only data, and unpack atomically into a fresh directory."""
from __future__ import annotations

import io
import errno
import json
from pathlib import Path
import stat
import struct
import subprocess
import sys
import warnings
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def files():
    manifest = dict(version=1, interface="hfm_controller_v1", observations=[{"name":"Ip", "shape":[1]}],
                    state_dim=0, state_dtype="float32", action_dim=12,
                    action_units="voltage_setpoint", reset_state="zeros")
    return {"controller.onnx": b"ONNX semantics checked by the loader",
            "controller.json": json.dumps(manifest).encode()}


def write_archive(path, entries):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in entries:
                archive.writestr(name, content)
    return path


def test_package_roundtrip_preserves_both_payloads(tmp_path):
    from submission.model_bundle import create_bundle, unpack_bundle
    source = tmp_path / "model"
    source.mkdir()
    expected = files()
    for name, raw in expected.items():
        (source / name).write_bytes(raw)
    (source / "unrelated.pkl").write_bytes(b"must never be included")
    package = create_bundle(source, tmp_path / "upload.zip")
    with zipfile.ZipFile(package) as archive:
        assert set(archive.namelist()) == set(expected)
    destination = unpack_bundle(package, tmp_path / "unpacked")
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == expected


@pytest.mark.parametrize("extra", ["../escape", "/escape", "C:/escape", "model/controller.onnx", "controller.onnx", "exploit.py", "state.pkl"])
def test_rejects_unexpected_paths_and_duplicate_members_without_writes(tmp_path, extra):
    from submission.model_bundle import BundleError, unpack_bundle
    package = write_archive(tmp_path / "bad.zip", [*files().items(), (extra, b"bad")])
    with pytest.raises(BundleError):
        unpack_bundle(package, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "escape").exists()


def test_rejects_symlinks_even_when_the_member_name_is_allowed(tmp_path):
    from submission.model_bundle import BundleError, unpack_bundle
    entries = list(files().items())
    link = zipfile.ZipInfo("controller.onnx")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    package = write_archive(tmp_path / "link.zip", [(link, b"/etc/passwd"), *entries[1:]])
    with pytest.raises(BundleError, match="regular|link"):
        unpack_bundle(package, tmp_path / "out")


def test_rejects_encrypted_entries_before_opening_them(tmp_path):
    from submission.model_bundle import BundleError, unpack_bundle
    package = write_archive(tmp_path / "encrypted.zip", files().items())
    raw = bytearray(package.read_bytes())
    central = raw.index(b"PK\x01\x02")
    struct.pack_into("<H", raw, central + 8, struct.unpack_from("<H", raw, central + 8)[0] | 1)
    package.write_bytes(raw)
    with pytest.raises(BundleError, match="(?i)encrypt"):
        unpack_bundle(package, tmp_path / "out")


@pytest.mark.parametrize("metadata", [b"[]", b'{"value": NaN}', b'{"x": 1, "x": 2}', b"not json"])
def test_rejects_invalid_metadata_and_removes_partial_output(tmp_path, metadata):
    from submission.model_bundle import BundleError, unpack_bundle
    payload = files()
    payload["controller.json"] = metadata
    package = write_archive(tmp_path / "bad.zip", payload.items())
    with pytest.raises(BundleError):
        unpack_bundle(package, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".out-*"))


def test_rejects_pickle_backed_npz_arrays(tmp_path):
    from submission.model_bundle import BundleError, unpack_bundle
    payload = files()
    data = io.BytesIO()
    np.savez(data, obs_mean=np.asarray([object()], dtype=object), obs_var=np.ones(1))
    payload["policy2_vecnormalize_stats.npz"] = data.getvalue()
    package = write_archive(tmp_path / "object.zip", payload.items())
    with pytest.raises(BundleError, match="member|exactly"):
        unpack_bundle(package, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_rejects_npz_with_unbounded_array_shape(tmp_path):
    from submission.model_bundle import BundleError, unpack_bundle
    payload = files()
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False, "shape": (2**60,)})
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("obs_mean.npy", header.getvalue())
        archive.writestr("obs_var.npy", header.getvalue())
    payload["policy2_vecnormalize_stats.npz"] = data.getvalue()
    package = write_archive(tmp_path / "huge.zip", payload.items())
    with pytest.raises(BundleError):
        unpack_bundle(package, tmp_path / "out")


def test_rejects_expansion_limit_before_writing(tmp_path, monkeypatch):
    import submission.model_bundle as bundle
    package = write_archive(tmp_path / "large.zip", files().items())
    monkeypatch.setattr(bundle, "MAX_EXPANDED_BYTES", 10)
    with pytest.raises(bundle.BundleError, match="limit|large"):
        bundle.unpack_bundle(package, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_existing_destination_is_never_merged_or_overwritten(tmp_path):
    from submission.model_bundle import BundleError, unpack_bundle
    package = write_archive(tmp_path / "model.zip", files().items())
    destination = tmp_path / "out"
    destination.mkdir()
    marker = destination / "old-model"
    marker.write_bytes(b"keep")
    with pytest.raises(BundleError):
        unpack_bundle(package, destination)
    assert marker.read_bytes() == b"keep"
    assert list(destination.iterdir()) == [marker]


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.ENOSPC])
def test_local_staging_io_error_is_not_an_invalid_submission(tmp_path, monkeypatch, error_number):
    from submission.model_bundle import unpack_bundle
    package = write_archive(tmp_path / "valid.zip", files().items())
    original_open = Path.open

    def fail_local_write(path, mode="r", *args, **kwargs):
        if path.name == "controller.onnx" and mode == "xb":
            raise OSError(error_number, "local storage unavailable")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_local_write)
    with pytest.raises(OSError) as error:
        unpack_bundle(package, tmp_path / "out")
    assert error.value.errno == error_number
    assert not (tmp_path / "out").exists()


def test_corrupt_compressed_member_is_an_invalid_submission(tmp_path):
    from submission.model_bundle import BundleError, unpack_bundle
    package = write_archive(tmp_path / "corrupt.zip", files().items())
    raw = bytearray(package.read_bytes())
    local = raw.index(b"PK\x03\x04")
    name_size, extra_size = struct.unpack_from("<HH", raw, local + 26)
    start = local + 30 + name_size + extra_size
    # DEFLATE block type 3 is reserved/invalid, independent of payload contents.
    raw[start] = (raw[start] & 0xf8) | 0x07
    package.write_bytes(raw)
    with pytest.raises(BundleError):
        unpack_bundle(package, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_invalid_export_does_not_replace_existing_package(tmp_path):
    from submission.model_bundle import BundleError, create_bundle
    model = tmp_path / "model"
    model.mkdir()
    (model / "controller.onnx").write_bytes(b"missing other export files")
    output = tmp_path / "upload.zip"
    output.write_bytes(b"previous valid archive")
    with pytest.raises(BundleError):
        create_bundle(model, output)
    assert output.read_bytes() == b"previous valid archive"


def test_cli_can_package_from_another_working_directory(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    for name, raw in files().items():
        (source / name).write_bytes(raw)
    output = tmp_path / "submission.zip"
    result = subprocess.run([sys.executable, str(ROOT / "submission/model_bundle.py"), "--model-dir", str(source), "--output", str(output)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.is_file()
