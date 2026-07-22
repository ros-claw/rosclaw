from __future__ import annotations

from pathlib import Path

import rosclaw.runtime.plugins.realsense_doctor as doctor


def _usb_device(root: Path, product_id: str, control: str) -> None:
    root.mkdir(parents=True)
    (root / "idVendor").write_text("8086\n", encoding="utf-8")
    (root / "idProduct").write_text(f"{product_id}\n", encoding="utf-8")
    (root / "power").mkdir()
    (root / "power" / "control").write_text(f"{control}\n", encoding="utf-8")


def test_doctor_recognizes_official_d405_usb_product_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _usb_device(tmp_path / "2-1", "0b5b", "on")
    monkeypatch.setattr(doctor, "_SYSFS_USB", tmp_path)

    assert doctor._read_sysfs_autosuspend() == "on"


def test_doctor_does_not_misidentify_old_incorrect_d405_product_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _usb_device(tmp_path / "2-1", "0b3d", "auto")
    monkeypatch.setattr(doctor, "_SYSFS_USB", tmp_path)

    assert doctor._read_sysfs_autosuspend() is None
