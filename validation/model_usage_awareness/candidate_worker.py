"""Trusted bounded build/test driver; no product implementation."""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from candidate_clone import clone_candidate
from public_diagnostic import capture_failure_diagnostic

B = Path(__file__).resolve().parent.parent
operator_dir = B / "operator"
N = "/home/nvidia/.nvm/versions/node/v24.19.0/bin/node"
WORKER_STAGE = "STARTUP"


def main():
    global WORKER_STAGE
    WORKER_STAGE = "PRIVATE_CLONE"
    workspace = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    mode = sys.argv[3]
    s = json.loads((B / "cases/SOURCE/spec.json").read_text())
    candidate = output_dir / "candidate"
    clone_candidate(workspace, candidate)
    home = output_dir / "private_home"
    home.mkdir(mode=0o700)
    env = {
        "PATH": str(Path(N).parent) + ":/usr/bin:/bin",
        "HOME": str(home),
        "ROSCLAW_HOME": str(home),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(candidate / "src") + ":" + str(candidate),
        "TMPDIR": str(output_dir),
        "PI_OFFLINE": "1",
    }
    cases = []
    diagnostics = {"commands": [], "omitted_commands": 0}
    WORKER_STAGE = "DEPENDENCY_LINK_SETUP"
    for pkg, root in s["candidate_dependency_roots"].items():
        (candidate / "packages" / pkg / "node_modules").symlink_to(root, target_is_directory=True)

    def call(label, argv, cwd, seconds=10, extra=None):
        nonlocal diagnostics
        e = dict(env)
        e.update(extra or {})
        p = subprocess.run(argv, cwd=cwd, env=e, capture_output=True, timeout=seconds)
        (output_dir / (label + ".stdout")).write_bytes(p.stdout)
        (output_dir / (label + ".stderr")).write_bytes(p.stderr)
        cases.append({"case_id": label, "passed": p.returncode == 0, "exit": p.returncode})
        if p.returncode != 0:
            diagnostics = capture_failure_diagnostic(
                diagnostics, label, p.returncode, p.stdout, p.stderr
            )
        return p.returncode

    WORKER_STAGE = "COMPILE_AND_OFFLINE_CHECKS"
    root = candidate / "packages/rosclaw-agent"
    tsc = Path(s["candidate_dependency_roots"]["rosclaw-agent"]) / "typescript/bin/tsc"
    code = call("strict_current_TS_compile", [N, str(tsc), "-p", str(root / "tsconfig.json")], root)
    if code == 0:
        call("copy_assets", [N, str(root / "scripts/copy-assets.mjs")], root)
        if mode == "run":
            guard = {
                "NODE_OPTIONS": "--require=" + str(operator_dir / "node_effect_guard.cjs"),
                "COMPACT_EFFECT_LOG": str(output_dir / "node_effects.jsonl"),
            }
            paths = [
                "tool-call-budget-visible",
                "tool-call-path-policy",
                "tool-call-budget-compact-sdk",
                "cli-tool-call-policy",
                "model-usage-awareness",
                "model-override",
                "compaction-stream",
                "compaction-observer",
                "operation-monitor-health",
            ]
            for rel in paths:
                target = root / "dist/test" / f"{rel}.test.js"
                if not target.exists():
                    if rel in [
                        "model-usage-awareness",
                        "tool-call-budget-visible",
                        "cli-tool-call-policy",
                    ]:
                        cases.append(
                            {
                                "case_id": "Node_" + rel,
                                "passed": False,
                                "error": "REQUIRED_TEST_MISSING",
                            }
                        )
                    continue
                extra = dict(guard)
                extra["COMPACT_ALLOWED_CLI"] = (
                    str(root / "dist/src/main.js")
                    if rel in ["cli-tool-call-policy", "model-usage-awareness"]
                    else ""
                )
                if rel == "model-override":
                    assert (
                        hashlib.sha256(
                            (root / "test/model-override.test.ts").read_bytes()
                        ).hexdigest()
                        == "f778eb4c3bc9aa0db68096317a19d7a4d3e3aaef646767f418c704eda18e49e0"
                    ), "TRUSTED_OVERRIDE_TEST_BYTES_CHANGED"
                    extra["COMPACT_ALLOWED_CLI"] = str(root / "dist/src/main.js")
                    extra["COMPACT_TRUSTED_OVERRIDE_ONLY"] = "1"
                call(
                    "Node_" + rel,
                    [N, "--test", "--test-reporter=tap", str(target)],
                    root,
                    12,
                    extra,
                )
                text = (output_dir / ("Node_" + rel + ".stdout")).read_text()
                actual = [
                    {"case_id": "Node_" + rel + "::" + m.group(2), "passed": m.group(1) == "ok"}
                    for m in re.finditer(r"^(ok|not ok) \d+ - (.+)$", text, re.M)
                ]
                cases.extend(actual)
                if rel == "model-usage-awareness":
                    required = json.loads((workspace / "inputs/SOURCE_CONTRACT.json").read_text())[
                        "required_new_test_names"
                    ]
                    observed = {r["case_id"].split("::", 1)[-1]: r["passed"] for r in actual}
                    cases.append(
                        {
                            "case_id": "operator_required_eight_new_semantic_tests",
                            "passed": all(observed.get(n) is True for n in required),
                        }
                    )
            call(
                "operator_real_SDK_usage_causality",
                [N, str(operator_dir / "usage_sdk_oracle.mjs"), str(candidate), str(output_dir)],
                candidate,
                15,
                guard,
            )
            call(
                "Python_CLI_original_and_operator",
                [
                    sys.executable,
                    "-B",
                    str(operator_dir / "static_pytest_worker.py"),
                    str(candidate / "tests/agentd/test_cli_tool_call_policy.py"),
                    str(output_dir / "python_counts.json"),
                    str(operator_dir / "python_usage_oracle.py"),
                ],
                candidate,
                12,
            )
    if mode == "run" and (output_dir / "python_counts.json").exists():
        py = json.loads((output_dir / "python_counts.json").read_text())
        cases.extend(
            {"case_id": "Python::" + r["case_id"], "passed": r["passed"]}
            for r in py["case_results"]
        )
        if py["effect_attempts_rejected"] or py["collection_errors"] or py["skipped"]:
            cases.append({"case_id": "python_guard_collection_skip_boundary", "passed": False})
    effects = (
        [json.loads(line) for line in (output_dir / "node_effects.jsonl").read_text().splitlines()]
        if (output_dir / "node_effects.jsonl").exists()
        else []
    )
    if effects:
        cases.append(
            {
                "case_id": "no_network_or_undeclared_process_attempts",
                "passed": False,
                "attempts": effects,
            }
        )
    operator = (
        json.loads((output_dir / "operator_cases.json").read_text())
        if (output_dir / "operator_cases.json").exists()
        else {}
    )
    if mode == "run":
        cases.extend(operator.get("cases", []))
    assert len(cases) <= 256 and len({c["case_id"] for c in cases}) == len(cases), (
        "FINITE256_UNIQUE_CASE_BOUND"
    )
    result = {
        "public_diagnostics": diagnostics,
        "cases": cases,
        "effect_attempts_rejected": effects,
        "SDK_observation": operator.get("SDK"),
        "scope": "Strict TS compile, guarded offline tests and actual installed SDK/local inert stream; no provider/ROS/physics",
        "passed": sum(c["passed"] for c in cases),
        "failed": sum(not c["passed"] for c in cases),
        "Python_typing_scope": "Original pinned gradual mypy config ignores legacy CLI; no new strict-Python typing certification",
    }
    (output_dir / "counts.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if cases and not result["failed"] else 1


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as error:
        # Do not expose arbitrary exception strings or open dynamic stderr paths.
        out = Path(sys.argv[2])
        out.mkdir(mode=0o700, parents=True, exist_ok=True)
        failure = {
            "stage": WORKER_STAGE,
            "error_class": type(error).__name__,
            "errno": getattr(error, "errno", None),
            "code": "CANDIDATE_WORKER_SETUP_OR_EXECUTION_FAILED",
        }
        causes = []
        cause = error
        for _ in range(4):
            if cause is None:
                break
            causes.append(
                {"error_class": type(cause).__name__, "errno": getattr(cause, "errno", None)}
            )
            cause = cause.__cause__ or cause.__context__
        failure["cause_chain"] = causes
        counts = {
            "cases": [{"case_id": "operator_candidate_worker_incomplete", "passed": False}],
            "passed": 0,
            "failed": 1,
            "effect_attempts_rejected": None,
            "SDK_observation": None,
            "scope": "Candidate failed before complete durable observations; effect counts UNKNOWN",
            "execution_mode": "BOUNDED_CHILD_INCOMPLETE",
            "failure": failure,
        }
        target = out / "counts.json"
        temporary = out / "counts.json.atomic"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(counts, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        exit_code = 1
    raise SystemExit(exit_code)
