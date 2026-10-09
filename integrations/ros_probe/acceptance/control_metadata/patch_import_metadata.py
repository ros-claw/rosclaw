"""Metadata-only backport for ros2_control4.48.1's component import path.

The simulator plugin loads its actual class with pluginlib, then uses
ResourceManager.import_component. That path skips load_hardware's metadata
registration. Preserve strict introspection checks by retaining the supplied
HardwareInfo during initialization. No interfaces, lifecycle, limits, command
values, scheduling or hardware execution behavior is changed.
"""

import argparse
import hashlib
import json
from pathlib import Path

ORIGINAL_SHA256 = "9430e8121bf46b95a2fa3246c2c0cd6dac11be67982cb0fe6ae75c66d3ba2ecc"
SOURCE = """  bool initialize_hardware(
    const hardware_interface::HardwareComponentParams & params, HardwareT & hardware)
  {
"""
METADATA = """    // ROSClaw metadata-only backport: import_component skips load_hardware.
    // Keep the exact HardwareInfo supplied by the actual plugin loader.
    auto & component_info = hardware_info_map_[params.hardware_info.name];
    component_info.name = params.hardware_info.name;
    component_info.type = params.hardware_info.type;
    component_info.group = params.hardware_info.group;
    component_info.rw_rate = params.hardware_info.rw_rate;
    component_info.plugin_name = params.hardware_info.hardware_plugin_name;
    component_info.is_async = params.hardware_info.is_async;
"""


def patch_source(path):
    path = Path(path)
    raw = path.read_bytes()
    if path.is_symlink() or hashlib.sha256(raw).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("exact original upstream4.48.1 source required; no fuzzy/repeated patch")
    text = raw.decode()
    if text.count(SOURCE) != 1:
        raise ValueError("exact unique original initialization signature required")
    patched = text.replace(SOURCE, SOURCE + METADATA).encode()
    path.write_bytes(patched)
    return {
        "original_sha256": ORIGINAL_SHA256,
        "patched_sha256": hashlib.sha256(patched).hexdigest(),
        "scope": "HardwareInfo metadata only; no control behavior or ABI change",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(patch_source(args.source)))
