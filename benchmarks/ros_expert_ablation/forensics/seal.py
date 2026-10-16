"""Content-addressed public original evidence bundle, excluding SDK private state."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def seal(study_root, workspace_root, output):
    study_root = Path(study_root).resolve()
    workspace = Path(workspace_root).resolve()
    output = Path(output).resolve()
    if study_root == output or study_root in output.parents:
        raise ValueError("output must be outside original study")
    output.mkdir(parents=True, exist_ok=False)
    blobs = output / "blobs"
    blobs.mkdir()
    manifest = {}
    worlds = []

    def add(path):
        path = Path(path)
        relative = path.relative_to(workspace)
        if path.is_symlink() or any(
            part in {"home", "state", "__pycache__", ".git"} for part in relative.parts
        ):
            raise ValueError("private or linked path")
        if (
            path.name.endswith((".sqlite", ".sqlite-wal", ".sqlite-shm", ".key", ".anchor"))
            or path.name == "auth.json"
        ):
            raise ValueError("private credential/state path")
        raw = path.read_bytes()
        original = digest(raw)
        stored = gzip.compress(raw, mtime=0)
        name = original + ".gz"
        if not (blobs / name).exists():
            (blobs / name).write_bytes(stored)
        manifest[str(relative)] = {
            "original_sha256": original,
            "stored_sha256": digest(stored),
            "original_bytes": len(raw),
            "blob": "blobs/" + name,
        }

    for p in study_root.rglob("*"):
        if p.is_file():
            add(p)
    for result in sorted(study_root.glob("episode-*/original-result.json")):
        episode = json.loads(result.read_text())
        for row in episode["original_operator_Worlds"]:
            directory = Path(row["directory"])
            execution = directory / "execution"
            settings = json.loads((directory / "settings.json").read_text())
            worlds.append(settings["run_id"])
            for p in directory.iterdir():
                if p.is_file():
                    add(p)
            for name in [
                "assets",
                "execution/generic-compiled",
                "execution/public-action-artifacts",
            ]:
                tree = directory / name
                if tree.exists():
                    for p in tree.rglob("*"):
                        if p.is_file() and "__pycache__" not in p.parts:
                            add(p)
            for p in execution.iterdir():
                if p.is_file():
                    add(p)
            for name in ["source-freeze.json", "execution_config.json", "daemon_ready.json"]:
                p = execution / "execution-control/daemon" / name
                if p.exists():
                    add(p)
        operator = Path(episode["operator_directory"])
        for p in operator.iterdir():
            if p.is_file():
                add(p)
    registration = json.loads((study_root / "preregistration.json").read_text())
    for reference in (
        registration["helpers"]
        + registration["frozen_training"]
        + [registration["gate_prerequisite"], registration["original_protocol"]]
    ):
        path = workspace / reference["path"]
        if digest(path.read_bytes()) != reference["sha256"]:
            raise ValueError("frozen reference changed")
        add(path)
    training = workspace / registration["frozen_training"][0]["path"]
    for record in json.loads(training.read_text())["records"]:
        for key in ["episode_ref", "original_proof_ref"]:
            add(workspace / record[key])
    recovery = study_root.parent
    for label in ["PR683-source-2e4cd8ae", "PR683-main-04ca79c5"]:
        add(recovery / (label + "-original-workflows-latest-review.json"))
        for path in (recovery / (label + "-original-workflows")).iterdir():
            if path.is_file():
                add(path)
    assert len(worlds) == len(set(worlds)) == 56
    for p, entry in manifest.items():
        raw = (output / entry["blob"]).read_bytes()
        assert digest(raw) == entry["stored_sha256"]
        original = gzip.decompress(raw)
        assert (
            digest(original) == entry["original_sha256"]
            and len(original) == entry["original_bytes"]
        )
        assert digest((workspace / p).read_bytes()) == entry["original_sha256"]
    encoded = json.dumps(manifest, indent=2) + "\n"
    (output / "public-original-manifest.json").write_text(encoded)
    review = {
        "status": "PASS_PUBLIC_ORIGINAL_G5_54_EPISODES_56_WORLDS_SEALED",
        "original_paths": len(manifest),
        "distinct_content_blobs": len(list(blobs.iterdir())),
        "distinct_Worlds": len(worlds),
        "manifest_sha256": digest(encoded.encode()),
        "private_credentials_keys_ledger_or_database_archived": False,
        "deduplication": "equal bytes stored once; original paths and different run identities preserved",
        "originals_modified": False,
        "model_or_World_reruns": 0,
        "v1_done": False,
    }
    (output / "sealed-review.json").write_text(json.dumps(review, indent=2) + "\n")
    return review


def main():
    p = argparse.ArgumentParser()
    p.add_argument("study_root")
    p.add_argument("workspace_root")
    p.add_argument("output")
    a = p.parse_args()
    print(json.dumps(seal(a.study_root, a.workspace_root, a.output)))


if __name__ == "__main__":
    main()
