"""Policy materialization for the LeRobot bridge.

This module resolves a user-supplied ``policy.path`` into a local directory
containing a LeRobot policy config and weights. It is safe to import from the
ROSClaw core interpreter because heavy ML imports are deferred inside functions.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw.firstboot.workspace import get_rosclaw_home

DEFAULT_CACHE_SUBDIR = "cache/lerobot/policies"


@dataclass
class MaterializationResult:
    """Result of resolving a policy path to a local directory."""

    local_path: Path
    network_used: bool = False
    cache_hit: bool = False


class PolicyMaterializationError(Exception):
    """Raised when a policy cannot be materialized locally."""

    def __init__(self, code: str, message: str, details: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def get_policy_cache_dir() -> Path:
    """Return the directory where downloaded policies are cached."""
    return get_rosclaw_home() / DEFAULT_CACHE_SUBDIR


def materialize_policy_path(
    policy_path: str,
    *,
    revision: str = "main",
    allow_network: bool = False,
    force_download: bool = False,
) -> MaterializationResult:
    """Return a local directory containing the policy artifacts.

    ``policy_path`` may be a local directory or a Hugging Face repo id
    (``owner/repo``). When a repo id is given and ``allow_network`` is false,
    the function looks in the local Hugging Face cache first.
    """
    if not policy_path:
        raise PolicyMaterializationError(
            "policy_config_not_found", "policy.path is empty"
        )

    local_path = Path(policy_path)
    if local_path.is_dir():
        _validate_local_policy_dir(local_path)
        return MaterializationResult(
            local_path=local_path.resolve(),
            network_used=False,
            cache_hit=True,
        )

    # Treat as Hugging Face repo id.
    if "/" not in policy_path:
        raise PolicyMaterializationError(
            "policy_config_not_found",
            f"Policy path is neither a local directory nor a HF repo id: {policy_path}",
        )

    return _materialize_hf_repo(
        policy_path,
        revision=revision,
        allow_network=allow_network,
        force_download=force_download,
    )


def _validate_local_policy_dir(policy_dir: Path) -> None:
    """Ensure a local policy directory has the minimal required files."""
    config_candidates = [policy_dir / "config.json", policy_dir / "config.yaml"]
    if not any(p.exists() for p in config_candidates):
        raise PolicyMaterializationError(
            "policy_config_not_found",
            f"No config.json or config.yaml found in {policy_dir}",
        )
    if not (policy_dir / "model.safetensors").exists() and not list(
        policy_dir.rglob("*.safetensors")
    ):
        raise PolicyMaterializationError(
            "policy_config_not_found",
            f"No model.safetensors found in {policy_dir}",
            "Local policy directories must contain a checkpoint to load-test/infer.",
        )


def _materialize_hf_repo(
    repo_id: str,
    *,
    revision: str,
    allow_network: bool,
    force_download: bool,
) -> MaterializationResult:
    """Resolve a HF repo id to a local directory."""
    cache_dir = get_policy_cache_dir()
    sanitized = _sanitize_repo_id(repo_id)
    target_dir = cache_dir / sanitized

    # If we already have a cached copy and are not forcing a re-download, reuse it.
    if target_dir.exists() and not force_download and _looks_like_loadable_policy_dir(target_dir):
        return MaterializationResult(
            local_path=target_dir,
            network_used=False,
            cache_hit=True,
        )

    try:
        import huggingface_hub as hf_hub
    except ImportError as exc:
        raise PolicyMaterializationError(
            "network_disabled",
            "huggingface_hub is not installed and no complete local policy cache was found.",
            str(exc),
        ) from exc

    if not allow_network:
        # Try the HF cache before giving up.
        cached_path = _find_in_hf_cache(hf_hub, repo_id, revision)
        if cached_path is not None:
            return MaterializationResult(
                local_path=cached_path,
                network_used=False,
                cache_hit=True,
            )
        raise PolicyMaterializationError(
            "network_disabled",
            f"Policy '{repo_id}' is not cached locally and allow_network=false.",
        )

    try:
        downloaded = hf_hub.snapshot_download(
            repo_id=repo_id,
            revision=revision,
            cache_dir=str(cache_dir),
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            allow_patterns=[
                "config.json",
                "model.safetensors",
                "train_config.json",
                "README.md",
            ],
            resume_download=not force_download,
        )
    except Exception as exc:  # noqa: BLE001
        raise PolicyMaterializationError(
            "network_download_required",
            f"Failed to download policy '{repo_id}': {exc}",
            str(exc),
        ) from exc

    local_path = Path(downloaded)
    if not _looks_like_loadable_policy_dir(local_path):
        raise PolicyMaterializationError(
            "policy_config_not_found",
            f"Downloaded policy directory does not contain a config and checkpoint: {local_path}",
        )
    return MaterializationResult(
        local_path=local_path,
        network_used=True,
        cache_hit=False,
    )


def _find_in_hf_cache(hf_hub: Any, repo_id: str, revision: str) -> Path | None:
    """Locate a locally cached repo directory using huggingface_hub APIs."""
    try:
        from huggingface_hub.utils import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id == repo_id:
                for rev in repo.revisions:
                    if (
                        rev.commit_hash.startswith(revision) or revision in rev.commit_hash
                    ) and _looks_like_loadable_policy_dir(rev.snapshot_path):
                        return Path(rev.snapshot_path)
    except Exception:  # noqa: BLE001
        pass

    # Fallback: try hf_hub_download with local_files_only to grab config path.
    try:
        config_path = hf_hub.hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
            local_files_only=True,
        )
        candidate = Path(config_path).parent
        if _looks_like_loadable_policy_dir(candidate):
            return candidate
    except Exception:  # noqa: BLE001
        pass

    return None


def _looks_like_policy_dir(path: Path) -> bool:
    return (path / "config.json").exists() or (path / "config.yaml").exists()


def _looks_like_loadable_policy_dir(path: Path) -> bool:
    if not _looks_like_policy_dir(path):
        return False
    return (path / "model.safetensors").exists() or bool(list(path.rglob("*.safetensors")))


def _sanitize_repo_id(repo_id: str) -> str:
    """Convert a HF repo id to a filesystem-safe directory name."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", repo_id)


def set_hf_hub_offline_env() -> None:
    """Set HF_HUB_OFFLINE=1 unless the user has already configured it."""
    if os.environ.get("HF_HUB_OFFLINE") is None:
        os.environ["HF_HUB_OFFLINE"] = "1"
