"""Read-only compiled navigation artifact review; never source admission.

Reuse the bounded, no-follow source-reader approach of the earlier generic
stack source reader. Only the closed compiler inventory is accepted. All
parsing and Body recompilation use captured bytes, without filesystem writes.
"""

import hashlib
import json
import os
import re
import stat
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

import yaml

from rosclaw.body.compiler import EffectiveBodyCompiler
from rosclaw.body.schema import BodyYaml, CalibrationYaml, EffectiveBody, EurdfProfile
from rosclaw.connectors.ros.context.navigation_workspace import (
    _navigation_instance,
    _prepare_navigation_profile,
)
from rosclaw.connectors.ros.intelligence.system_model import RosSystemModel
from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from rosclaw.contracts.common import content_hash

_MANIFEST = "navigation-body-workspace.json"
_DIRECTORIES = ("body", "body/refs", "body/snapshots", "body/refs/generated")
_FILES = {
    "body/refs/eurdf.profile.yaml",
    "body/refs/eurdf.lock",
    "body/body.yaml",
    "body/calibration.yaml",
    "body/refs/effective_body.json",
    "body/refs/robot.urdf",
    "execution_config.json",
    "source-snapshot.json",
    "navigation-requirements.json",
    "fixture-policy.json",
    "navigation-review.json",
}
_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "body_snapshot_hash",
    "source_snapshot_hash",
    "source_urdf_sha256",
    "navigation_review_hash",
    "fixture_policy_hash",
    "requirements_hash",
    "evidence_domain",
    "binding_verified",
    "physical_acceptance_level",
    "usable_for_real_execution",
    "execution_entry",
    "direct_actions_dispatched",
    "file_sha256",
    "manifest_hash",
}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("unique string artifact keys required")
        result[key] = value
    return result


def _json(raw):
    def invalid(value):
        raise ValueError("finite JSON artifact values required: " + value)

    return json.loads(raw, object_pairs_hook=_object, parse_constant=invalid)


class _Yaml(yaml.SafeLoader):
    pass


_Yaml.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    lambda loader, node: _object(loader.construct_pairs(node, deep=True)),
)


def _yaml(raw):
    if any(
        isinstance(t, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken)) for t in yaml.scan(raw)
    ):
        raise ValueError("artifact YAML aliases/anchors are not accepted")
    loader = _Yaml(raw)
    try:
        value = loader.get_single_data()
        if type(value) is not dict:
            raise ValueError("artifact YAML object required")
        return value
    finally:
        loader.dispose()


def _identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


class _Reader:
    """Hold directory identities and check the inventory again before return."""

    def __init__(self, root):
        self.root, self.dirs, self.identities, self.file_metadata = Path(root), {}, {}, {}

    def __enter__(self):
        try:
            self.dirs[""] = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            for name in _DIRECTORIES:
                parent, _, component = name.rpartition("/")
                self.dirs[name] = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=self.dirs[parent],
                )
            self.identities = {name: _identity(os.fstat(fd)) for name, fd in self.dirs.items()}
            self._inventory()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_args):
        for fd in reversed(list(self.dirs.values())):
            os.close(fd)
        self.dirs.clear()

    def _inventory(self):
        for directory, fd in self.dirs.items():
            expected = {
                path[len(directory) + 1 if directory else 0 :].split("/")[0]
                for path in (_FILES | {_MANIFEST} | set(_DIRECTORIES))
                if not directory or path.startswith(directory + "/")
            }
            if set(os.listdir(fd)) != expected:
                raise ValueError("closed compiler workspace inventory required: " + directory)

    def read(self):
        files, total = {}, 0
        for name in sorted(_FILES | {_MANIFEST}):
            parent, _, component = name.rpartition("/")
            fd = os.open(
                component, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.dirs[parent]
            )
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 5_000_000:
                    raise ValueError("bounded nonempty regular workspace member required: " + name)
                raw = stream.read(5_000_001)
                after = os.fstat(stream.fileno())
                if _identity(before) != _identity(after) or len(raw) != before.st_size:
                    raise ValueError("workspace member changed during read: " + name)
                self.file_metadata[name] = _identity(after)
            total += len(raw)
            if total > 30_000_000:
                raise ValueError("workspace exceeds bounded artifact capture budget")
            files[name] = raw
        return files

    def unchanged(self):
        self._inventory()
        for name, metadata in self.file_metadata.items():
            parent, _, component = name.rpartition("/")
            current = os.stat(component, dir_fd=self.dirs[parent], follow_symlinks=False)
            if _identity(current) != metadata:
                raise ValueError("workspace member changed during review: " + name)
        for name in _DIRECTORIES:
            parent, _, component = name.rpartition("/")
            if (
                _identity(os.stat(component, dir_fd=self.dirs[parent], follow_symlinks=False))
                != self.identities[name]
            ):
                raise ValueError("workspace directory changed during review")
        if _identity(self.root.lstat()) != self.identities[""]:
            raise ValueError("workspace root changed during review")


def inspect_navigation_body_workspace(
    directory, *, expected_manifest_hash=None, expected_body_hash=None, now=None
):
    """Verify captured artifacts and replay their historical review, read-only.

    A self-consistent hash is not publisher provenance or a live readiness
    check. Optional expected hashes let a caller pin separately trusted inputs.
    No file, Body status, policy, admission or execution configuration changes.
    """
    try:
        with _Reader(directory) as reader:
            result = _inspect(
                reader.read(), expected_manifest_hash, expected_body_hash, now or datetime.now(UTC)
            )
            reader.unchanged()
            return result
    except (
        KeyError,
        TypeError,
        AttributeError,
        OverflowError,
        RecursionError,
        yaml.YAMLError,
    ) as exc:
        raise ValueError("invalid compiled navigation workspace artifact") from exc


def _inspect(files, expected_manifest_hash, expected_body_hash, now):
    # Every caller still reads the complete no-follow workspace. Cache only an
    # exact-byte static compiler replay, never live health, source or clocks.
    if (
        type(files) is dict
        and set(files) == _FILES | {_MANIFEST}
        and all(type(raw) is bytes for raw in files.values())
        and sum(map(len, files.values())) <= 10_000_000
        and all(
            value is None or type(value) is str
            for value in (expected_manifest_hash, expected_body_hash)
        )
    ):
        result = json.loads(
            _static_integrity_json(
                tuple(sorted(files.items())), expected_manifest_hash, expected_body_hash
            )
        )
        captured = datetime.fromisoformat(result["captured_at"])
        result.update(
            inspected_at=now.isoformat(), snapshot_age_ms=(now - captured).total_seconds() * 1000
        )
        return result
    return _inspect_uncached(files, expected_manifest_hash, expected_body_hash, now)


@lru_cache(maxsize=2)
def _static_integrity_json(original_files, expected_manifest_hash, expected_body_hash):
    """At most two 10MB exact-byte keys; returned JSON is immutable."""
    files = dict(original_files)
    original_reviewed_at = datetime.fromisoformat(
        _json(files["navigation-review.json"])["reviewed_at"]
    )
    result = _inspect_uncached(
        files, expected_manifest_hash, expected_body_hash, original_reviewed_at
    )
    result.pop("inspected_at")
    result.pop("snapshot_age_ms")
    return json.dumps(result, allow_nan=False)


def _inspect_uncached(files, expected_manifest_hash, expected_body_hash, now):
    manifest = _json(files[_MANIFEST])
    if type(manifest) is not dict or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("closed navigation workspace manifest required")
    saved = manifest.pop("manifest_hash")
    if saved != content_hash("navbodyworkspace", manifest):
        raise ValueError("navigation workspace manifest hash mismatch")
    if expected_manifest_hash is not None and saved != expected_manifest_hash:
        raise ValueError("expected navigation workspace manifest hash mismatch")
    if (
        manifest["schema_version"] != "rosclaw.navigation_body_workspace.v1"
        or manifest["status"] != "CONTRACT_COMPILED_NOT_ADMITTED"
        or manifest["evidence_domain"] != "SIMULATION"
        or manifest["binding_verified"] is not False
        or manifest["physical_acceptance_level"] != "NOT_RUN"
        or manifest["usable_for_real_execution"] is not False
        or manifest["direct_actions_dispatched"] is not False
        or manifest["execution_entry"] != "request_action"
        or type(manifest["file_sha256"]) is not dict
        or set(manifest["file_sha256"]) != _FILES
    ):
        raise ValueError("unadmitted navigation-only artifact contract required")
    for name, digest in manifest["file_sha256"].items():
        if (
            type(digest) is not str
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
            or hashlib.sha256(files[name]).hexdigest() != digest
        ):
            raise ValueError("navigation workspace file hash mismatch: " + name)
    model = RosSystemModel.from_dict(_json(files["source-snapshot.json"]))
    review, policy = _json(files["navigation-review.json"]), _json(files["fixture-policy.json"])
    requirements = _json(files["navigation-requirements.json"])
    urdf = files["body/refs/robot.urdf"]
    replay, _, expected_profile = _prepare_navigation_profile(
        model, urdf, requirements, policy=policy, now=datetime.fromisoformat(review["reviewed_at"])
    )
    if replay != review or review["status"] != "PROPOSED":
        raise ValueError("original navigation observation review replay mismatch")
    required = {
        "source_snapshot_hash": model.snapshot_hash,
        "source_urdf_sha256": hashlib.sha256(urdf).hexdigest(),
        "navigation_review_hash": content_hash("navobservationreview", review),
        "fixture_policy_hash": content_hash("navfixturepolicy", policy),
        "requirements_hash": review["requirements_hash"],
    }
    if any(manifest[key] != value for key, value in required.items()):
        raise ValueError("navigation workspace source/review/policy identity mismatch")
    profile_raw, instance_raw, calibration_raw = (
        _yaml(files[name])
        for name in ("body/refs/eurdf.profile.yaml", "body/body.yaml", "body/calibration.yaml")
    )
    cached_raw = _json(files["body/refs/effective_body.json"])
    for data, schema in (
        (profile_raw, EurdfProfile),
        (instance_raw, BodyYaml),
        (calibration_raw, CalibrationYaml),
        (cached_raw, EffectiveBody),
    ):
        if type(data) is not dict or set(data) - set(schema.__dataclass_fields__):
            raise ValueError("unknown compiled Body fields")
    profile, instance, calibration = (
        EurdfProfile.from_dict(profile_raw),
        BodyYaml.from_dict(instance_raw),
        CalibrationYaml.from_dict(calibration_raw),
    )
    cached = EffectiveBody.from_dict(cached_raw)
    # Pure in-memory recompilation: never use a resolver's write/recompile fallback.
    rebuilt = EffectiveBodyCompiler().compile(profile, instance, calibration)
    binding = SimulationNavigationBinding.from_body(cached)
    if (
        rebuilt.compute_hash() != binding.body_snapshot_hash
        or manifest["body_snapshot_hash"] != binding.body_snapshot_hash
    ):
        raise ValueError("cached Body differs from captured compiler inputs")
    if expected_body_hash is not None and expected_body_hash != binding.body_snapshot_hash:
        raise ValueError("expected compiled Body hash mismatch")
    checksum = "sha256:" + hashlib.sha256(files["body/refs/eurdf.profile.yaml"]).hexdigest()
    lock = _yaml(files["body/refs/eurdf.lock"])
    if (
        lock.get("checksum") != checksum
        or instance.model_ref.get("checksum") != checksum
        or lock.get("profile_id") != profile.profile_id
        or lock.get("profile_version") != profile.profile_version
        or lock.get("uri") != cached.eurdf_uri
        or instance.model_ref.get("eurdf_uri") != cached.eurdf_uri
        or profile.assets != {"urdf": "robot.urdf"}
    ):
        raise ValueError("captured e-URDF lock/profile/source identity mismatch")
    if (
        profile_raw != expected_profile.to_dict()
        or instance_raw
        != _navigation_instance(policy, profile.profile_id, cached.eurdf_uri, checksum).to_dict()
        or calibration_raw != CalibrationYaml().to_dict()
        or lock
        != {
            "schema_version": "rosclaw.eurdf_lock.v1",
            "profile_id": profile.profile_id,
            "profile_version": profile.profile_version,
            "uri": cached.eurdf_uri,
            "source": "observed_navigation_SIM_fixture",
            "checksum": checksum,
        }
        or _json(files["execution_config.json"])
        != {
            "executor_kind": "navigation_only",
            "body_id": binding.body_id,
            "body_snapshot_hash": binding.body_snapshot_hash,
        }
    ):
        raise ValueError("source-bound navigation-only compiler contract mismatch")
    return {
        "schema_version": "rosclaw.navigation_workspace_inspection.v1",
        "status": "ARTIFACT_INTEGRITY_VERIFIED_NOT_ADMITTED",
        "manifest_hash": saved,
        "body_snapshot_hash": binding.body_snapshot_hash,
        **required,
        "captured_at": model.captured_at.isoformat(),
        "inspected_at": now.isoformat(),
        "snapshot_age_ms": model.age_ms(now),
        "original_reviewed_at": review["reviewed_at"],
        "verified_output_files": len(_FILES),
        "current_navigation_readiness": "NOT_EVALUATED",
        "physical_source_provenance": "NOT_VERIFIED",
        "binding_verified": False,
        "physical_acceptance_level": "NOT_RUN",
        "usable_for_real_execution": False,
        "direct_actions_dispatched": False,
        "workspace_modified": False,
    }
