"""Independent SDK original-Native/ECM collector for a guarded SIM sink.

No ROS client, Runtime, action, lease or permission is created here. The
independent operator enrolls an accepted stopped G1/L1 prefix; every collection
rebuilds current ROS facts, checks source continuity and uses actual clocks.
"""

import argparse
import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path

from simulation_source_stream import SimulationSourceStreamReader, SourceStreamPins

from rosclaw.connectors.ros.context.navigation_readiness import (
    _validate_policy,
    review_current_sim_navigation_health,
)
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.intelligence import build_system_model


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError("bounded original regular file required")
        raw = os.read(fd, limit + 1)
        after = os.fstat(fd)
        if len(raw) != before.st_size or (
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("original observation changed during read")
        return raw
    finally:
        os.close(fd)


class CurrentSimulationFacts:
    def __init__(self, directory, *, assets_directory, enrollment_sha256):
        self.directory = Path(directory)
        info = self.directory.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("independent original evidence directory required")
        self._directory_identity = info.st_dev, info.st_ino
        self.assets_directory = Path(assets_directory)
        assets_info = self.assets_directory.stat(follow_symlinks=False)
        if not stat.S_ISDIR(assets_info.st_mode):
            raise ValueError("explicit immutable original assets directory required")
        self._assets_identity = assets_info.st_dev, assets_info.st_ino
        self.failed_reason = None
        self.reader = None
        raw = _read(self.directory / "execution-facts-enrollment.json", 64_000)
        if type(enrollment_sha256) is not str or _sha(raw) != enrollment_sha256:
            raise ValueError("independent execution facts enrollment SHA mismatch")
        self.enrollment_sha256 = enrollment_sha256
        e = json.loads(raw)
        keys = {
            "schema_version",
            "request",
            "policy",
            "policy_hash",
            "source_pins",
            "guard_inputs",
            "prefix_sha256",
            "initial_L1_sha256",
            "world_sha256",
            "urdf_sha256",
        }
        if (
            type(e) is not dict
            or set(e) != keys
            or e["schema_version"] != "rosclaw.execution_facts_enrollment.v1"
        ):
            raise ValueError("closed SDK execution facts enrollment required")
        # Execution preparation deliberately strengthens the passive P03 policy.
        # Faster capture must satisfy these bounds; stale data is never promoted.
        if (
            type(e["policy"]["max_age_ms"]) not in (int, float)
            or not 0 < e["policy"]["max_age_ms"] <= 500
            or type(e["request"]["max_age_ms"]) not in (int, float)
            or not 0 < e["request"]["max_age_ms"] <= 500
            or e["guard_inputs"]["max_age_ms"] != 500
        ):
            raise ValueError("execution requires independently configured ROS/source age <=500ms")
        if (
            set(e["request"])
            != {"schema_version", "manifest_hash", "body_snapshot_hash", "sim_run_id", "max_age_ms"}
            or e["request"]["schema_version"] != "rosclaw.current_sim_navigation_health_request.v1"
        ):
            raise ValueError("closed immutable Body/run health request required")
        _validate_policy(e["request"], e["policy"], e["policy_hash"])
        initial_raw = _read(self.directory / "execution-initial-L1.json", 2_000_000)
        initial = json.loads(initial_raw)
        if (
            _sha(initial_raw) != e["initial_L1_sha256"]
            or initial["status"] != "READY_FOR_SIM_TEST"
            or initial["source_binding"]["result"] != "MATCHED"
            or any(
                initial[k] != e["request"][k]
                for k in ("manifest_hash", "body_snapshot_hash", "sim_run_id")
            )
        ):
            raise ValueError("independently accepted stopped G1/L1 Body/run required")
        for name, key, limit in (
            ("robot.urdf", "urdf_sha256", 5_000_000),
            ("world.sdf", "world_sha256", 20_000_000),
        ):
            if _sha(_read(self.assets_directory / name, limit)) != e[key]:
                raise ValueError("independent original URDF/World SHA mismatch")
        self.enrollment = e
        self._cached_native_sha256 = self._cached_model = None
        self.reader = SimulationSourceStreamReader(
            self.directory,
            "ecm-original.jsonl",
            pins=SourceStreamPins(**e["source_pins"]),
            accepted_prefix=_read(self.directory / "execution-enrolled-prefix.jsonl", 30_000_000),
            expected_prefix_sha256=e["prefix_sha256"],
            guard_inputs=e["guard_inputs"],
        )

    def collect(self, *, freshness_horizon_ms=0):
        if type(freshness_horizon_ms) is not int or not 0 <= freshness_horizon_ms <= 100:
            raise ValueError("bounded independent freshness lookahead <=100ms required")
        if self.failed_reason is not None:
            raise ValueError("current SIM facts permanently latched: " + self.failed_reason)
        try:
            started = time.monotonic_ns()
            if (
                _sha(_read(self.directory / "execution-facts-enrollment.json", 64_000))
                != self.enrollment_sha256
            ):
                raise ValueError("immutable execution facts Body/run enrollment changed")
            info = self.directory.stat(follow_symlinks=False)
            if (info.st_dev, info.st_ino) != self._directory_identity or not stat.S_ISDIR(
                info.st_mode
            ):
                raise ValueError("original evidence directory replaced")
            info = self.assets_directory.stat(follow_symlinks=False)
            if (info.st_dev, info.st_ino) != self._assets_identity or not stat.S_ISDIR(
                info.st_mode
            ):
                raise ValueError("original asset directory replaced")
            for name, key, limit in (
                ("robot.urdf", "urdf_sha256", 5_000_000),
                ("world.sdf", "world_sha256", 20_000_000),
            ):
                if _sha(_read(self.assets_directory / name, limit)) != self.enrollment[key]:
                    raise ValueError("enrolled original URDF/World bytes changed")
            pointer = json.loads(_read(self.directory / "live-native-pointer.json", 4096))
            if (
                type(pointer) is not dict
                or set(pointer) != {"file", "sha256", "captured_at", "received_monotonic_ns"}
                or type(pointer["file"]) is not str
                or re.fullmatch(r"original-native-[0-9]{3,6}\.json", pointer["file"]) is None
                or type(pointer["received_monotonic_ns"]) is not int
                or not 0 <= started - pointer["received_monotonic_ns"] <= 500_000_000
            ):
                raise ValueError("bounded current original Native pointer required")
            raw = _read(self.directory / pointer["file"], 10_000_000)
            native = json.loads(raw)
            if _sha(raw) != pointer["sha256"] or native["captured_at"] != pointer["captured_at"]:
                raise ValueError("original Native SHA or capture timestamp changed")
            if self._cached_native_sha256 != pointer["sha256"]:
                self._cached_model = build_system_model(
                    RosGraphSnapshot.from_dict(
                        {
                            "ros_version": "ros2",
                            "distro": "jazzy",
                            "endpoint": "fixture://actual-gazebo-source",
                            "captured_at": native["captured_at"],
                            **native["graph"],
                        }
                    ),
                    native=native,
                )
                self._cached_native_sha256 = pointer["sha256"]
            model = self._cached_model
            # Evaluate the stricter age margin once. This does not predict
            # future state; actual review time and age-evaluation time remain
            # distinct in the original report. No second Body/model review is
            # needed on the same immutable bytes.
            horizon_started_ns = time.monotonic_ns()
            health = review_current_sim_navigation_health(
                self.directory / "generic-compiled",
                self.enrollment["request"],
                model=model,
                policy=self.enrollment["policy"],
                policy_hash=self.enrollment["policy_hash"],
                freshness_margin_ms=freshness_horizon_ms,
            )
            source = self.reader.poll()
            ended = time.monotonic_ns()
            if (
                health["status"] != "HEALTHY_SIM_OBSERVATION_ONLY"
                or source["continuous"] is not True
                or not 0 <= ended - started <= 500_000_000
                or ended - pointer["received_monotonic_ns"] > 500_000_000
            ):
                raise ValueError(
                    "current ROS/source expired or rejected: "
                    + json.dumps(
                        {
                            "health": health,
                            "source": source,
                        },
                        sort_keys=True,
                    )[:2000]
                )
            result = {
                "health": health,
                "source": source,
                "health_review_ns": started,
                "completed_ns": ended,
                "native_pointer": pointer,
            }
            if freshness_horizon_ms:
                observed = json.loads(self.reader.last_raw)["observer_monotonic_ns"]
                result.update(
                    age_horizon_check=health,
                    valid_until_monotonic_ns=min(
                        horizon_started_ns + freshness_horizon_ms * 1_000_000,
                        observed + 500_000_000,
                        pointer["received_monotonic_ns"] + 500_000_000,
                    ),
                )
                if result["valid_until_monotonic_ns"] <= ended:
                    raise ValueError("bounded freshness window expired during collection")
            return result
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            self.failed_reason = str(exc)[:2500]
            raise ValueError(
                "current SIM facts permanently latched: " + self.failed_reason
            ) from exc

    def close(self):
        self.failed_reason = self.failed_reason or "CURRENT_FACTS_READER_CLOSED"
        if self.reader is not None:
            self.reader.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrollment-sha256", required=True)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--assets-directory", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.duration <= 10:
        raise ValueError("bounded SDK read-only observation duration <=10s required")
    root = Path("/evidence")
    collector = CurrentSimulationFacts(
        root, assets_directory=args.assets_directory, enrollment_sha256=args.enrollment_sha256
    )
    started = time.monotonic_ns()
    rows, reason = 0, None
    try:
        with (root / "execution-current-facts-original.jsonl").open("x") as trace:
            while True:
                facts = None
                try:
                    facts = collector.collect()
                except (ValueError, OSError, RuntimeError) as exc:
                    reason = str(exc)
                now = time.monotonic_ns()
                trace.write(
                    json.dumps(
                        {
                            "index": rows,
                            "monotonic_ns": now,
                            "facts": facts,
                            "reason": reason,
                            "terminal": reason is not None,
                            "execution_authority": "NONE",
                            "L2": "NOT_RUN",
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
                trace.flush()
                rows += 1
                if reason or now - started >= args.duration * 1e9:
                    break
                time.sleep(0.05)
    finally:
        collector.close()
    outcome = {
        "status": "TERMINAL_LATCHED" if reason else "PASS_CURRENT_FACTS_READONLY_WINDOW",
        "reason": reason,
        "rows": rows,
        "duration_sec": (time.monotonic_ns() - started) / 1e9,
        "enrollment_sha256": args.enrollment_sha256,
        "execution_authority": "NONE",
        "L2": "NOT_RUN",
        "G3_done": False,
    }
    with (root / "execution-current-facts-outcome.json").open("x") as stream:
        stream.write(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome), flush=True)
    if reason:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
