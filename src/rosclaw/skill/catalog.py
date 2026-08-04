"""Helpers for contributing Skills to the official ros-claw/skills catalog."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from rosclaw.skill.models import SkillPackage
from rosclaw.skill.validators import validate_package

CATALOG_REPO = "ros-claw/skills"
CATALOG_DEFAULT_BRANCH = "main"


def _run(
    cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _has_gh() -> bool:
    return shutil.which("gh") is not None


def _gh_auth_status() -> bool:
    try:
        _run(["gh", "auth", "status"])
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _ensure_git_config(workdir: Path) -> None:
    """Set a fallback git user name/email only if none are configured."""
    for key, value in (("user.name", "ROSClaw Skill Bot"), ("user.email", "noreply@rosclaw.io")):
        try:
            _run(["git", "config", "--global", key], cwd=workdir)
        except subprocess.CalledProcessError:
            _run(["git", "config", "--global", key, value], cwd=workdir)


def _viewer_login() -> str:
    """The gh-authenticated user's login."""
    try:
        return _run(["gh", "api", "user", "-q", ".login"]).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to get GitHub user login: {exc.stderr}") from exc


def _viewer_can_push(repo: str) -> bool:
    """True when the gh-authenticated user has push access to ``repo``.

    Org members with write access should branch directly on the catalog
    repo — the fork detour is for external contributors only.
    """
    try:
        return (
            _run(["gh", "api", f"repos/{repo}", "-q", ".permissions.push"]).stdout.strip() == "true"
        )
    except subprocess.CalledProcessError:
        return False


def _parent_full_name(repo: str) -> str | None:
    try:
        parent = _run(["gh", "api", f"repos/{repo}", "-q", ".parent.full_name"]).stdout.strip()
        return parent or None
    except subprocess.CalledProcessError:
        return None


def _fork_repo_for(catalog_repo: str, owner: str) -> str:
    """The owner's fork of catalog_repo, forking only when none exists.

    ``gh repo fork`` is NOT a reliable no-op across gh versions when the
    fork already exists, and a fork's name is not guaranteed to equal the
    upstream repo name (renamed forks) — look the fork up first, fork only
    when absent, and never assume the name.
    """
    fork_name = catalog_repo.split("/")[-1]
    candidate = f"{owner}/{fork_name}"
    if _parent_full_name(candidate) == catalog_repo:
        return candidate
    try:
        res = _run(["gh", "api", f"users/{owner}/repos?per_page=100", "-q", ".[].full_name"])
        for full in res.stdout.splitlines():
            if full != candidate and _parent_full_name(full) == catalog_repo:
                return full
    except subprocess.CalledProcessError:
        pass
    try:
        _run(["gh", "repo", "fork", catalog_repo, "--clone=false", "--default-branch-only"])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to fork {catalog_repo}: {exc.stderr}") from exc
    if _parent_full_name(candidate) == catalog_repo:
        return candidate
    raise RuntimeError(
        f"Fork of {catalog_repo} not found for {owner} after forking — "
        "check the fork's name and pass it explicitly."
    )


def submit_to_catalog(
    pkg: SkillPackage,
    *,
    dry_run: bool = False,
    catalog_repo: str = CATALOG_REPO,
    base_branch: str = CATALOG_DEFAULT_BRANCH,
    branch_prefix: str = "add",
) -> dict[str, Any]:
    """Submit a local skill to the official catalog by opening a GitHub PR.

    This does **not** require ``ROSCLAW_ADMIN_API_KEY``. It requires only a
    working ``gh`` CLI login (``gh auth login``).  When the authenticated
    user has push access to the catalog repo (org members), the branch is
    pushed directly to the catalog and the PR is same-repo; external
    contributors go through their fork (created only when missing).
    """
    if pkg.skill is None:
        raise RuntimeError("skill.yaml not loaded")

    if not _has_gh():
        raise RuntimeError(
            "`gh` (GitHub CLI) is not installed. "
            "Install it from https://cli.github.com and run `gh auth login`."
        )

    if not _gh_auth_status():
        raise RuntimeError("`gh` is not authenticated. Run `gh auth login` first.")

    # Validate the local skill first.
    report = validate_package(pkg)
    if not report.ok:
        raise RuntimeError("Local skill validation failed:\n  " + "\n  ".join(report.errors))

    name = pkg.skill.metadata.name
    namespace = pkg.skill.metadata.namespace or "ros-claw"
    version = pkg.skill.metadata.version
    display_name = pkg.skill.metadata.display_name or name
    branch_name = f"{branch_prefix}-{name}-v{version}"

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "catalog_repo": catalog_repo,
            "base_branch": base_branch,
            "branch": branch_name,
            "skill_name": f"{namespace}/{name}",
            "version": version,
        }

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp) / "skills"

        # Org members with push access branch directly on the catalog;
        # external contributors go through their fork.
        direct = _viewer_can_push(catalog_repo)
        if direct:
            clone_repo = catalog_repo
            pr_head = branch_name
        else:
            fork_owner = _viewer_login()
            if not fork_owner:
                raise RuntimeError("Could not determine GitHub login from `gh api user`.")
            fork_repo = _fork_repo_for(catalog_repo, fork_owner)
            clone_repo = fork_repo
            pr_head = f"{fork_owner}:{branch_name}"

        # Clone the working repo.
        try:
            _run(["gh", "repo", "clone", clone_repo, str(workdir), "--", "--depth=1"])
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to clone {clone_repo}: {exc.stderr}") from exc

        _ensure_git_config(workdir)

        # Create a feature branch.
        try:
            _run(["git", "checkout", "-b", branch_name], cwd=workdir)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to create branch {branch_name}: {exc.stderr}") from exc

        target_dir = workdir / "skills" / name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(pkg.root, target_dir)

        # Run catalog validation scripts inside the clone.
        try:
            _run([sys.executable, "scripts/validate_skill.py", f"skills/{name}"], cwd=workdir)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Catalog validation failed:\n" + (exc.stdout or "") + (exc.stderr or "")
            ) from exc

        try:
            _run([sys.executable, "scripts/build_registry.py"], cwd=workdir)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Registry build failed:\n" + (exc.stdout or "") + (exc.stderr or "")
            ) from exc

        # Commit changes.
        try:
            _run(["git", "add", "--all"], cwd=workdir)
            _run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"Add {name} skill v{version}\n\nSubmitted from rosclaw CLI.",
                ],
                cwd=workdir,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to commit changes: {exc.stderr}") from exc

        # Push the branch (origin = the catalog itself for org members,
        # the fork for external contributors).
        env = os.environ.copy()
        env["GIT_ASKPASS"] = "echo"
        try:
            _run(["git", "push", "-u", "origin", branch_name], cwd=workdir, env=env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to push branch {branch_name}: {exc.stderr}") from exc

        # Open the PR against the upstream catalog repo.
        try:
            pr_res = _run(
                [
                    "gh",
                    "pr",
                    "create",
                    "--repo",
                    catalog_repo,
                    "--base",
                    base_branch,
                    "--head",
                    pr_head,
                    "--title",
                    f"Add {display_name} skill v{version}",
                    "--body",
                    f"Proposes adding `{namespace}/{name}` v{version} to the official ROSClaw Skills Catalog.\n\n"
                    "Please review validation output before merging.",
                ],
                cwd=workdir,
            )
            pr_url = pr_res.stdout.strip()
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to create pull request: {exc.stderr}") from exc

        return {
            "ok": True,
            "dry_run": False,
            "catalog_repo": catalog_repo,
            "flow": "direct" if direct else "fork",
            "fork_repo": None if direct else fork_repo,
            "branch": branch_name,
            "skill_name": f"{namespace}/{name}",
            "version": version,
            "pr_url": pr_url,
        }
