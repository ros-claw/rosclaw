"""Run the exact portable 14 stdlib tests; no SDK/DDS/model authority."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixture-root", required=True)
    args = ap.parse_args()
    root = Path(args.fixture_root).resolve()
    expected = {
        "mf02_dds_sync_scenario.py": "030c57cb8e6a90348cda55f61646ee5fca6f51086820b7f048fb71cc37aa8f0c",
        "test_mf02_dds_sync_scenario.py": "30ddb7c9c6ec732bf53188275fb7cf841d7019eb8bb3ca024ed3a3d169d5bf0f",
    }
    assert sys.version_info[:2] in {(3, 11), (3, 12)}
    assert {p.name for p in root.iterdir()} == set(expected)

    def hashes():
        out = {}
        for name, digest in expected.items():
            p = root / name
            assert (
                p.is_file()
                and not p.is_symlink()
                and p.stat().st_nlink == 1
                and p.stat().st_size <= 65536
            )
            out[name] = hashlib.sha256(p.read_bytes()).hexdigest()
            assert out[name] == digest
        return out

    before = hashes()
    denied = []

    def audit(event, values):
        if event == "import" and str(values[0]).split(".")[0] in {
            "rclpy",
            "message_filters",
            "geometry_msgs",
            "rcl_interfaces",
            "mujoco",
            "onnxruntime",
            "torch",
            "numpy",
        }:
            denied.append("SDK_IMPORT")
            raise RuntimeError("MF02_PURE_SDK_IMPORT_FORBIDDEN")
        if event.startswith("socket.") or event in {
            "subprocess.Popen",
            "os.system",
            "os.fork",
            "os.posix_spawn",
            "os.kill",
            "os.killpg",
            "ctypes.dlopen",
        }:
            denied.append(event)
            raise RuntimeError("MF02_PURE_EFFECT_FORBIDDEN")
        if event == "open":
            mode, flags = values[1], values[2]
            writing = (
                any(c in mode for c in "wax+")
                if type(mode) is str
                else bool(
                    type(flags) is int
                    and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
                )
            )
            if writing:
                denied.append("INPUT_WRITE")
                raise RuntimeError("MF02_PURE_INPUT_WRITE_FORBIDDEN")

    sys.dont_write_bytecode = True
    sys.addaudithook(audit)
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "native_mf02_public_tests", root / "test_mf02_dds_sync_scenario.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    assert suite.countTestCases() == 14
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    unchanged = hashes() == before
    passed = (
        result.testsRun == 14
        and result.wasSuccessful()
        and not result.skipped
        and not denied
        and unchanged
    )
    report = {
        "status": "MF02_PORTABLE_EXACT_PURE14_PASS_NOT_DDS"
        if passed
        else "MF02_PORTABLE_PURE_FAIL",
        "ordinary_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "sources_unchanged": unchanged,
        "effect_denials": denied,
        "SDK_DDS_model_calls": 0,
        "original_source_workflow_failure_preserved": True,
        "full_DDS_certificate": False,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
