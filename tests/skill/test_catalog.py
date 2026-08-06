"""Tests for the submit-catalog direct-vs-fork flow selection.

The 2026-08-04 incident: an org member's submission went through their
personal fork (shaoxiang/skills) because submit_to_catalog forked
unconditionally.  The catalog flow must branch directly on the catalog
repo when the gh-authenticated user has push access, and fork only for
external contributors — and fork only when none exists yet, without
assuming the fork's name.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from rosclaw.skill import catalog as catalog_mod
from rosclaw.skill.catalog import submit_to_catalog
from rosclaw.skill.models import SkillPackage, SkillYaml


def _make_pkg(tmp_path: Path, name: str = "demo_skill") -> SkillPackage:
    root = tmp_path / name
    root.mkdir()
    (root / "skill.yaml").write_text("schema_version: rosclaw.skill.v1\n")
    pkg = SkillPackage(root)
    pkg.skill = SkillYaml.model_validate(
        {
            "schema_version": "rosclaw.skill.v1",
            "kind": "Skill",
            "metadata": {"name": name, "namespace": "ros-claw", "version": "1.0.0"},
            "identity": {"skill_id": f"ros-claw/{name}", "package_name": f"ros-claw/{name}"},
            "task": {"intent": name},
            "execution": {"entrypoint": {"type": "behavior_tree", "file": "behavior_tree.xml"}},
            "status": {},
        }
    )
    return pkg


class _Recorder:
    """Dispatch fake subprocess results per command shape and record calls."""

    def __init__(
        self,
        *,
        can_push: bool,
        viewer: str = "octocat",
        parents: dict[str, str] | None = None,
        user_repos: list[str] | None = None,
    ):
        self.can_push = can_push
        self.viewer = viewer
        self.parents = dict(parents or {})  # repo full_name -> its parent's full_name
        self.user_repos = list(user_repos or [])
        self.calls: list[list[str]] = []

    def __call__(self, cmd, cwd=None, env=None):
        self.calls.append(list(cmd))
        out = ""
        if cmd[:3] == ["gh", "api", "user"]:
            out = self.viewer + "\n"
        elif cmd[:2] == ["gh", "api"] and any("permissions.push" in c for c in cmd):
            out = ("true" if self.can_push else "false") + "\n"
        elif cmd[:2] == ["gh", "api"] and any("users/" in c and "repos" in c for c in cmd):
            out = "\n".join(self.user_repos) + "\n"
        elif cmd[:2] == ["gh", "api"] and any(".parent.full_name" in c for c in cmd):
            repo = cmd[2].split("repos/")[-1]
            if repo in self.parents:
                out = self.parents[repo] + "\n"
            else:
                raise subprocess.CalledProcessError(1, cmd, "", "Not Found")
        elif cmd[:3] == ["gh", "repo", "fork"]:
            parent = cmd[3]
            self.parents[f"{self.viewer}/{parent.split('/')[-1]}"] = parent  # fork now exists
        elif cmd[:2] == ["gh", "pr"]:
            out = "https://github.com/ros-claw/skills/pull/99\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")


def _run_submit(pkg, recorder):
    class _Report:
        ok = True
        errors: list[str] = []

    with (
        patch.object(catalog_mod, "_run", recorder),
        patch.object(catalog_mod, "_has_gh", return_value=True),
        patch.object(catalog_mod, "_gh_auth_status", return_value=True),
        patch.object(catalog_mod, "validate_package", return_value=_Report()),
    ):
        return submit_to_catalog(pkg)


def test_direct_flow_when_viewer_can_push(tmp_path):
    rec = _Recorder(can_push=True)
    result = _run_submit(_make_pkg(tmp_path), rec)
    clones = [c for c in rec.calls if c[:3] == ["gh", "repo", "clone"]]
    forks = [c for c in rec.calls if c[:3] == ["gh", "repo", "fork"]]
    prs = [c for c in rec.calls if c[:3] == ["gh", "pr", "create"]]
    assert result["flow"] == "direct"
    assert result["fork_repo"] is None
    assert clones and clones[0][3] == "ros-claw/skills"  # clones the catalog, not a fork
    assert not forks  # never forks
    assert prs and f"{rec.viewer}:" not in prs[0][prs[0].index("--head") + 1]


def test_fork_flow_for_external_contributor(tmp_path):
    rec = _Recorder(can_push=False, parents={"octocat/skills": "ros-claw/skills"})
    result = _run_submit(_make_pkg(tmp_path), rec)
    clones = [c for c in rec.calls if c[:3] == ["gh", "repo", "clone"]]
    forks = [c for c in rec.calls if c[:3] == ["gh", "repo", "fork"]]
    prs = [c for c in rec.calls if c[:3] == ["gh", "pr", "create"]]
    assert result["flow"] == "fork"
    assert result["fork_repo"] == "octocat/skills"
    assert clones and clones[0][3] == "octocat/skills"
    assert not forks  # fork already exists -> gh repo fork never called
    assert prs and prs[0][prs[0].index("--head") + 1].startswith("octocat:")


def test_fork_created_only_when_missing(tmp_path):
    rec = _Recorder(can_push=False)  # no fork anywhere
    result = _run_submit(_make_pkg(tmp_path), rec)
    forks = [c for c in rec.calls if c[:3] == ["gh", "repo", "fork"]]
    assert result["flow"] == "fork"
    assert len(forks) == 1  # forked exactly once, after the lookup missed


def test_renamed_fork_is_found_not_assumed(tmp_path):
    # owner/skills exists but is NOT a fork of the catalog; the real fork
    # was renamed to octocat/catalog-fork.
    rec = _Recorder(
        can_push=False,
        parents={"octocat/skills": "someone/else", "octocat/catalog-fork": "ros-claw/skills"},
        user_repos=["octocat/skills", "octocat/catalog-fork"],
    )
    result = _run_submit(_make_pkg(tmp_path), rec)
    assert result["fork_repo"] == "octocat/catalog-fork"
    clones = [c for c in rec.calls if c[:3] == ["gh", "repo", "clone"]]
    assert clones and clones[0][3] == "octocat/catalog-fork"
    forks = [c for c in rec.calls if c[:3] == ["gh", "repo", "fork"]]
    assert not forks


def test_viewer_can_push_false_on_api_error():
    def boom(cmd, cwd=None, env=None):
        raise subprocess.CalledProcessError(1, cmd, "", "rate limited")

    with patch.object(catalog_mod, "_run", boom):
        assert catalog_mod._viewer_can_push("ros-claw/skills") is False
