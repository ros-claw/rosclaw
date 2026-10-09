"""Build the isolated metadata overlay from an exact local source archive.

No download or package upgrade occurs. Preserve the build's original output and
verify the local base tag before and after Docker resolves the Dockerfile.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

BASE_TAG = "rosclaw/ros-expert-generic-runtime:1008-v1"


def inspect_image(reference):
    return json.loads(
        subprocess.check_output(["docker", "image", "inspect", reference], timeout=10)
    )[0]


def build(archive, output, tag):
    source = Path(__file__).resolve().parent
    lock = json.loads((source / "source-lock.json").read_text())
    archive = Path(archive)
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("a regular local upstream archive is required")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != lock["archive_sha256"]:
        raise ValueError("upstream archive differs from the committed source lock")
    if not tag.startswith("rosclaw/ros-expert-generic-runtime-metadata:"):
        raise ValueError("a separate generic metadata image tag is required")
    before = inspect_image(BASE_TAG)
    if before["Id"] != lock["base_image_id"]:
        raise ValueError("local base tag differs from the immutable source lock")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    context = output / "context"
    context.mkdir()
    for path in (source / "source-lock.json", source / "patch_import_metadata.py"):
        shutil.copyfile(path, context / path.name)
    shutil.copyfile(archive, context / "ros2_control-4.48.1.tar.gz")
    shutil.copyfile(source.parent / "Dockerfile.generic-control-metadata", context / "Dockerfile")
    command = [
        "docker",
        "build",
        "--network",
        "none",
        "--pull=false",
        "--progress",
        "plain",
        "--build-arg",
        "BASE_IMAGE=" + BASE_TAG,
        "--tag",
        tag,
        str(context),
    ]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (
        (output / "build.stdout").open("xb") as stdout,
        (output / "build.stderr").open("xb") as stderr,
    ):
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True, timeout=600)
    after = inspect_image(BASE_TAG)
    built = inspect_image(tag)
    if after["Id"] != before["Id"]:
        raise ValueError("base image tag changed during the build")
    base_layers = before["RootFS"]["Layers"]
    if built["RootFS"]["Layers"][: len(base_layers)] != base_layers:
        raise ValueError("built image does not retain the exact original base layers")
    review = {
        "status": "PASS_EXACT_BASE_OFFLINE_METADATA_BUILD",
        "source_lock": lock,
        "base_image_id_before_and_after": before["Id"],
        "built_image_id": built["Id"],
        "built_tag": tag,
        "original_base_layers_preserved": True,
        "context_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(context.iterdir())
        },
        "World_started": False,
        "physical_acceptance": "NOT_RUN",
    }
    (output / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps(review), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    build(args.archive, args.output, args.tag)
