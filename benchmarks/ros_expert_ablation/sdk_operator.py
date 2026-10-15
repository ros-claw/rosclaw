"""External disposable SIM operator: no ROS, Runtime or driver in model clients.

All physical navigation goes through the existing isolated SDK canonical MCP
fixture. Inputs are operator-provided recovery/material paths, never model paths.
"""

import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

import yaml

from benchmarks.ros_expert_ablation.protocol import CASES, SEEDS

SDK_IMAGE = "sha256:51ffb0e0aeb391f0efbef3013c7fc83586794c222b233e3dea688f3015ef1c6b"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class SdkExperiment:
    def __init__(
        self, worktree, case, seed, ordinal, *, recovery_root, healthy_stage, fault_staging_root
    ):
        if case not in CASES or seed not in SEEDS or type(ordinal) is not int or ordinal < 1:
            raise ValueError("closed experiment case, seed and ordinal required")
        self.recovery_root = Path(recovery_root).resolve()
        self.evidence_root = self.recovery_root.parent
        self.healthy_stage = Path(healthy_stage).resolve()
        self.fault_staging_root = Path(fault_staging_root).resolve()
        self.worktree = Path(worktree).resolve()
        self.case = case
        self.seed = seed
        self.ordinal = ordinal
        self.source = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.worktree, text=True
        ).strip()
        self.prefix = f"G5-E{ordinal:02d}-{uuid.uuid4().hex[:10]}"
        self.output = self.recovery_root / (self.prefix + "-operator")
        self.output.mkdir(exist_ok=False)
        self.worlds = []
        self.started = time.monotonic()

    def reference(self, path):
        path = Path(path)
        return {
            "path": str(path.relative_to(self.evidence_root)),
            "sha256": sha(path.read_bytes()),
            "bytes": path.stat().st_size,
        }

    def check_source(self):
        assert (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.worktree, text=True
            ).strip()
            == self.source
        )
        assert not subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=self.worktree
        ).strip()

    def execute(self, argv, log, *, timeout):
        self.check_source()
        with Path(log).open("xb") as stream:
            process = subprocess.Popen(
                argv,
                cwd=self.worktree,
                env=dict(os.environ, PYTHONPATH="src", G3_REVIEW_WORKTREE=str(self.worktree)),
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
                return 124

    def cleanup_owned_world(self, path):
        cid_path = path / "container-id.txt"
        if not cid_path.exists():
            return
        cid = cid_path.read_text().strip()
        raw = subprocess.run(["docker", "inspect", cid], capture_output=True, timeout=10)
        if raw.returncode:
            return
        value = json.loads(raw.stdout)[0]
        assert (
            value["Config"]["Image"] == SDK_IMAGE and value["HostConfig"]["NetworkMode"] == "none"
        )
        assert value["Id"] == cid and any(
            m.get("Destination") == "/evidence" and m.get("Source") == str(path / "execution")
            for m in value["Mounts"]
        )
        if value["State"]["Running"]:
            subprocess.run(
                ["docker", "stop", "-t", "3", cid], capture_output=True, timeout=10, check=True
            )
            (path / "operator-timeout-cleanup-original.json").write_text(
                json.dumps(
                    {
                        "container_id": cid,
                        "World_terminated": True,
                        "independent_physical_stop_claimed": False,
                    }
                )
                + "\n"
            )

    def world(self, case, stage, label, remaining_sec):
        self.check_source()
        prefix = self.prefix + "-" + label
        path = self.recovery_root / (prefix + "-" + case)
        env = dict(
            os.environ,
            PYTHONPATH="src",
            G3_VERIFIED_WORKTREE=str(self.worktree),
            G3_REVIEW_WORKTREE=str(self.worktree),
            G3_VERIFIED_SEED=str(self.seed),
            G3_VERIFIED_OUTPUT_PREFIX=prefix,
            G3_VERIFIED_CONTAINER_PREFIX="reh-" + prefix.lower(),
            G4_VERIFIED_PHASE="before_repair",
            G4_STAGING_DIRECTORY=str(stage),
        )
        log = self.output / (label + "-host-original.log")
        with log.open("xb") as stream:
            process = subprocess.Popen(
                [
                    str(self.worktree / ".venv/bin/python"),
                    str(self.recovery_root / "run_G5_actual_original_task_World_v2.py"),
                    case,
                ],
                cwd=self.worktree,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                rc = process.wait(timeout=min(150, max(1, remaining_sec)))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
                rc = 124
        self.cleanup_owned_world(path)
        self.worlds.append(
            {
                "directory": str(path),
                "case": case,
                "returncode": rc,
                "host_log": self.reference(log),
            }
        )
        return path, rc

    def capture(self):
        stage = (
            self.healthy_stage
            if self.case == "j100_navigation_onboarding"
            else self.fault_staging_root / self.case / "before_repair"
        )
        for preparation in range(2):
            self.before, rc = self.world(
                self.case,
                stage,
                f"initial{preparation:02d}",
                300 - (time.monotonic() - self.started),
            )
            if rc == 0:
                break
            status_path = self.before / "execution/governed-sink-status.json"
            if not status_path.exists():
                raise RuntimeError("initial capture failed without original unadmitted evidence")
            status = json.loads(status_path.read_text())
            assert status["action_id"] is None and all(
                status[k] == 0
                for k in ["lease_updates", "received_commands", "published_velocity_messages"]
            )
            assert not list(
                (self.before / "execution").glob("governed-action-*-ticket-original.json")
            )
        if rc:
            raise RuntimeError(
                "two original no-admission preparation Worlds failed; no model episode fabricated"
            )
        e = self.before / "execution"
        review = json.loads((e / "engineering-original-candidate-review.json").read_text())
        self.files = {
            n: (self.before / "assets" / n).read_text()
            for n in ["nav2.yaml", "bringup.json", "robot.urdf"]
        }
        self.files.update(
            {
                "native.json": (e / "engineering-original-native.json").read_text(),
                "requirements.json": (e / "engineering-original-requirements.json").read_text(),
            }
        )
        self.reviewed_at = review["reviewed_at"]
        self.radius = review["candidate"]["geometry"]["physical_radius_m"]
        marker = json.loads((e / "engineering-capture-ready.json").read_text())
        assert (
            marker["navigation_goals_sent"] == 0
            and marker["navigation_workspace_compiled"] is False
        )
        for sink in [marker["sink_before"], marker["sink_after"]]:
            assert all(
                sink[k] == 0
                for k in ["lease_updates", "received_commands", "published_velocity_messages"]
            )
        return {
            "task_id": sha(self.prefix.encode())[:16],
            "goal": (
                "Select the four observed navigation interfaces, propose immutable onboarding and validate bounded SIM navigation in a new World."
                if self.case == "j100_navigation_onboarding"
                else "Diagnose the supplied staged ROS navigation stack from original observations; propose a minimal allowed configuration repair and independently verify it in a new SIM World."
            ),
            "seed": self.seed,
            "evidence_scope": "SEALED_ORIGINAL_SOURCE_ONLY; not current live L1 or action authority",
            "allowed_files": ["nav2.yaml", "bringup.json"],
            "safety": "URDF/Body/controllers and freshness/lease bounds immutable; canonical rosclawd SIM actions only.",
        }

    def evaluate(self, staging, selected, index, *, remaining_sec):
        if remaining_sec < 40:
            return {
                "verified_success": False,
                "reason": "insufficient original wall budget for new SDK validation",
            }
        if self.case == "j100_navigation_onboarding" and selected is None:
            return {
                "verified_success": False,
                "reason": "explicit observed interface proposal missing",
            }
        stage = self.output / f"proposal-{index:02d}"
        stage.mkdir()
        import shutil

        shutil.copytree(self.before / "assets", stage / "assets")
        for name, value in staging.items():
            (stage / "assets" / name).write_text(
                yaml.safe_dump(value, sort_keys=False)
                if name.endswith(".yaml")
                else json.dumps(value, indent=2) + "\n"
            )
        if selected is not None:
            req = json.loads((stage / "assets/requirements.json").read_text())
            req["interface_names"] = selected
            (stage / "assets/requirements.json").write_text(json.dumps(req, indent=2) + "\n")
        (stage / "preregistration.json").write_text(
            json.dumps(
                {
                    "schema_version": "rosclaw.model_proposed_staging.v1",
                    "source": self.source,
                    "parent_original_capture": str(self.before.relative_to(self.evidence_root)),
                    "physical_acceptance": "NOT_RUN",
                    "proposal_files": {
                        n: sha((stage / "assets" / n).read_bytes())
                        for n in ["nav2.yaml", "bringup.json", "requirements.json"]
                    },
                },
                indent=2,
            )
            + "\n"
        )
        start = time.monotonic()
        path, rc = self.world("positive", stage, f"test{index:02d}", remaining_sec)
        if rc:
            return {
                "verified_success": False,
                "reason": "new SDK World did not produce accepted canonical navigation",
                "raw_World_ref": str(path.relative_to(self.evidence_root)),
                "returncode": rc,
                "model_feedback": {
                    "verified_success": False,
                    "reason": "fresh SDK L1/canonical validation failed; original artifacts preserved",
                },
            }
        review = self.output / f"test{index:02d}-independent-original-review.json"
        age = self.output / f"test{index:02d}-strict-live-stream-review.json"
        for script, out in [
            ("verify_G3_J100_governed_positive_originals_v3.py", review),
            ("review_G3_strict_live_stream_envelope_v3.py", age),
        ]:
            left = remaining_sec - (time.monotonic() - start)
            if left <= 0:
                return {
                    "verified_success": False,
                    "reason": "original validation wall deadline reached",
                }
            checked = self.execute(
                [
                    str(self.worktree / ".venv/bin/python"),
                    str(self.recovery_root / script),
                    str(path),
                    str(out),
                ],
                out.with_suffix(".log"),
                timeout=min(60, left),
            )
            if checked:
                return {
                    "verified_success": False,
                    "reason": "independent original verification failed",
                    "returncode": checked,
                }
        v = json.loads(review.read_text())
        assert v["source"] == self.source and v["status"].startswith("PASS_")
        receipts = sorted((path / "execution").glob("governed-action-*-receipt-original.json"))
        assert len(receipts) == 2 and all(
            json.loads(p.read_text())["receipt"]["final_state"] == "COMPLETED" for p in receipts
        )
        return {
            "verified_success": True,
            "verification_source": "INDEPENDENT_ORIGINAL_SDK_REVIEW",
            "source": self.source,
            "new_world_run": v["run_id"],
            "fresh_l1_ref": self.reference(path / "actual-live-L1-readiness-review.json"),
            "canonical_receipts": [self.reference(p) for p in receipts],
            "independent_stop_ref": self.reference(review),
            "strict_source_envelope": self.reference(age),
            "model_feedback": {
                "verified_success": True,
                "status": "two canonical goals and independent stop/source verification passed",
            },
        }
