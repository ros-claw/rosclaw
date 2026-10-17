"""Local source regressions, authored but not run by the author.

Only the actual YAML digest command runs, in a temporary workspace. Dummy
wheel/sdist bytes give hashing/selection coverage, NOT build or package-validity
credit. No publisher, build script, SDK, or network is invoked.
"""

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml


def workflow():
    path = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def release_step(name):
    return next(s for s in workflow()["jobs"]["release"]["steps"] if s.get("name") == name)


def gh_equal(left, right):
    """Relevant GitHub loose equality, including boolean/numeric coercion."""
    if type(left) is type(right):
        if isinstance(left, str):
            return left.casefold() == right.casefold()
        return left == right

    def number(value):
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (float, int)):
            return value
        if isinstance(value, str):
            try:
                return float(value) if value else 0
            except ValueError:
                return math.nan
        return math.nan

    return number(left) == number(right)


class Expression:
    """Fail-closed evaluator for the actual gates' small GitHub expression subset.

    Not a claim of live Actions execution. Unknown syntax/functions fail tests;
    no hard-coded release decision substitutes for the extracted expressions.
    """

    def __init__(self, source, context):
        self.tokens = []
        token = re.compile(r"\s*(\&\&|\|\||==|[(),]|'[^']*'|[A-Za-z_][A-Za-z_0-9.]*)")
        rest = source.strip()
        while rest:
            match = token.match(rest)
            assert match, f"unsupported expression: {rest}"
            self.tokens.append(match.group(1))
            rest = rest[match.end() :]
        self.index = 0
        self.context = context

    def take(self, expected):
        if self.index < len(self.tokens) and self.tokens[self.index] == expected:
            self.index += 1
            return True
        return False

    def atom(self):
        if self.take("("):
            value = self.disjunction()
            assert self.take(")")
            return value
        token = self.tokens[self.index]
        self.index += 1
        if token.startswith("'"):
            return token[1:-1]
        if token in ("true", "false"):
            return token == "true"
        if self.take("("):
            args = [self.disjunction()]
            while self.take(","):
                args.append(self.disjunction())
            assert self.take(")")
            if token == "toJSON":
                assert len(args) == 1
                return json.dumps(args[0], separators=(",", ":"))
            assert token == "startsWith" and len(args) == 2
            return str(args[0]).casefold().startswith(str(args[1]).casefold())
        assert token in self.context, f"unsupported context: {token}"
        return self.context[token]

    def equality(self):
        value = self.atom()
        while self.take("=="):
            value = gh_equal(value, self.atom())
        return value

    def conjunction(self):
        value = self.equality()
        while self.take("&&"):
            right = self.equality()
            value = right if value else value
        return value

    def disjunction(self):
        value = self.conjunction()
        while self.take("||"):
            right = self.conjunction()
            value = value if value else right
        return value

    def evaluate(self):
        value = self.disjunction()
        assert self.index == len(self.tokens)
        return bool(value)


@pytest.mark.parametrize(
    "event,ref,tag_allowed",
    [
        ("pull_request", "refs/pull/721/merge", False),
        ("pull_request", "refs/tags/v1.2.3", False),
        ("push", "refs/heads/main", False),
        ("push", "refs/heads/develop", False),
        ("push", "refs/heads/v1.2.3", False),
        ("push", "refs/tags/v1.2.3", True),
        ("push", "refs/tags/1.2.3", False),
        ("push", "refs/tags/release-v1.2.3", False),
        ("workflow_dispatch", "refs/heads/main", False),
        ("workflow_dispatch", "refs/tags/v1.2.3", False),
    ],
)
@pytest.mark.parametrize("value", [None, False, True, "false", "true", "", "yes", 0, 1, [], {}])
def test_actual_gate_truth_table(event, ref, tag_allowed, value):
    context = {"github.event_name": event, "github.ref": ref, "inputs.publish_testpypi": value}
    job = Expression(workflow()["jobs"]["release"]["if"], context).evaluate()
    manual = Expression(
        release_step("Publish to TestPyPI (manual rehearsal)")["if"], context
    ).evaluate()
    pypi = Expression(release_step("Publish to PyPI (tag only)")["if"], context).evaluate()
    opt_in = event == "workflow_dispatch" and value is True
    assert job == (tag_allowed or opt_in)
    assert manual == opt_in
    # A publisher step is reachable only when its containing job is admitted.
    assert (job and pypi) == tag_allowed
    assert not (job and manual and pypi)


def test_dispatch_default_and_trigger_authority():
    doc = workflow()
    dispatch = doc["on"]["workflow_dispatch"]["inputs"]["publish_testpypi"]
    assert dispatch["type"] == "boolean"
    assert yaml.safe_load(dispatch["default"]) is False
    assert yaml.safe_load(dispatch["required"]) is False
    assert doc["on"]["push"] == {"branches": ["main", "develop"]}
    assert doc["on"]["pull_request"] == {"branches": ["main", "develop"]}
    assert set(doc["on"]) == {"push", "pull_request", "workflow_dispatch"}
    for name in ("build", "integration-test", "test", "full-regression"):
        assert "if" not in doc["jobs"][name]
    release = doc["jobs"]["release"]
    assert release["needs"] == ["build"]
    assert release["permissions"] == {"id-token": "write"}
    assert release["environment"] == "${{ github.event_name == 'push' && 'pypi' || 'testpypi' }}"
    assert [s.get("name") for s in release["steps"]] == [
        None,
        "Download artifacts",
        "Record artifact digests (traceability)",
        "Publish to TestPyPI (manual rehearsal)",
        "Publish to PyPI (tag only)",
    ]
    assert release_step("Publish to PyPI (tag only)")["if"] == "github.event_name == 'push'"


def fixture_dist(tmp_path, stage=True):
    """Synthetic bytes only: these are deliberately not valid package archives."""
    dist = tmp_path / "dist"
    dist.mkdir()
    payloads = {
        "rosclaw-1.2.3-py3-none-any.whl": b"fixture wheel\x00\xff\n",
        "rosclaw-1.2.3.tar.gz": b"fixture sdist\x00\xfe\n",
    }
    # Reverse creation order to exercise deterministic filename ordering.
    for name, content in reversed(list(payloads.items())):
        (dist / name).write_bytes(content)
    if stage:
        support = dist / "js-stage" / "rosclaw-agent" / "dist"
        support.mkdir(parents=True)
        (support / "main.js").write_bytes(b"fixture JS support, not a distribution")
    return dist, payloads


def run_digest(tmp_path):
    step = release_step("Record artifact digests (traceability)")
    assert step.get("shell", "bash") == "bash"
    # Execute the entire ACTUAL YAML run, not a copied implementation.
    return subprocess.run(  # noqa: PLW1510 - tests inspect both exit outcomes
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        env={"PATH": os.environ["PATH"], "LC_ALL": "C"},
    )


@pytest.mark.parametrize("stage", [True, False])
def test_actual_digest_bytes_and_order(tmp_path, stage):
    dist, payloads = fixture_dist(tmp_path, stage)
    result = run_digest(tmp_path)
    assert result.returncode == 0, result.stderr
    expected = "".join(
        f"{hashlib.sha256(content).hexdigest()}  dist/{name}\n"
        for name, content in sorted(payloads.items())
    )
    assert result.stdout == expected
    assert "js-stage" not in result.stdout
    for name, content in payloads.items():
        assert (dist / name).read_bytes() == content
    if stage:
        assert (dist / "js-stage/rosclaw-agent/dist/main.js").read_bytes() == (
            b"fixture JS support, not a distribution"
        )
    assert {p.name for p in dist.iterdir()} == set(payloads) | ({"js-stage"} if stage else set())


@pytest.mark.parametrize(
    "fault",
    [
        "missing-wheel",
        "missing-sdist",
        "missing-both",
        "extra-file",
        "hidden-file",
        "foreign-wheel",
        "malformed-wheel",
        "extra-wheel",
        "extra-sdist",
        "version-mismatch",
        "wheel-symlink",
        "sdist-symlink",
        "dangling-symlink",
        "artifact-directory",
        "artifact-fifo",
        "stage-symlink",
        "stage-file",
        "stage-fifo",
        "nested-stage-symlink",
        "nested-stage-fifo",
        "foreign-directory",
        "dist-symlink",
        "missing-dist",
    ],
)
def test_actual_digest_fails_closed(tmp_path, fault):
    dist, payloads = fixture_dist(tmp_path)
    wheel = dist / "rosclaw-1.2.3-py3-none-any.whl"
    sdist = dist / "rosclaw-1.2.3.tar.gz"
    stage = dist / "js-stage"
    if fault == "missing-wheel":
        wheel.unlink()
    elif fault == "missing-sdist":
        sdist.unlink()
    elif fault == "missing-both":
        wheel.unlink()
        sdist.unlink()
    elif fault in ("extra-file", "hidden-file", "foreign-wheel", "extra-wheel", "extra-sdist"):
        names = {
            "extra-file": "notes.txt",
            "hidden-file": ".unexpected",
            "foreign-wheel": "foreign-1.2.3-py3-none-any.whl",
            "extra-wheel": "rosclaw-2.0.0-py3-none-any.whl",
            "extra-sdist": "rosclaw-2.0.0.tar.gz",
        }
        (dist / names[fault]).write_bytes(b"unexpected fixture")
    elif fault == "malformed-wheel":
        wheel.rename(dist / "rosclaw-nonsense-py3-none-any.whl")
    elif fault == "version-mismatch":
        sdist.rename(dist / "rosclaw-2.0.0.tar.gz")
    elif fault in ("wheel-symlink", "sdist-symlink", "dangling-symlink"):
        path = sdist if fault == "sdist-symlink" else wheel
        path.unlink()
        target = tmp_path / "target-bytes"
        if fault != "dangling-symlink":
            target.write_bytes(b"symlink target")
        path.symlink_to(target)
    elif fault in ("artifact-directory", "artifact-fifo"):
        wheel.unlink()
        if fault == "artifact-directory":
            wheel.mkdir()
        else:
            os.mkfifo(wheel)
    elif fault in ("stage-symlink", "stage-file", "stage-fifo"):
        stage.rename(tmp_path / "support")
        if fault == "stage-symlink":
            stage.symlink_to(tmp_path / "support", target_is_directory=True)
        elif fault == "stage-file":
            stage.write_bytes(b"not a directory")
        else:
            os.mkfifo(stage)
    elif fault == "nested-stage-symlink":
        (stage / "link").symlink_to(wheel)
    elif fault == "nested-stage-fifo":
        os.mkfifo(stage / "fifo")
    elif fault == "foreign-directory":
        (dist / "unexpected").mkdir()
    elif fault in ("dist-symlink", "missing-dist"):
        dist.rename(tmp_path / "saved-dist")
        if fault == "dist-symlink":
            dist.symlink_to(tmp_path / "saved-dist", target_is_directory=True)
    else:
        raise AssertionError(f"unhandled fixture: {fault}")
    result = run_digest(tmp_path)
    assert result.returncode != 0, (fault, result.stdout, result.stderr)
    assert result.stdout == "", (fault, result.stdout)
    # Failure may not rewrite the surviving built bytes.
    for name, content in payloads.items():
        path = dist / name
        if path.is_file() and not path.is_symlink():
            assert path.read_bytes() == content
