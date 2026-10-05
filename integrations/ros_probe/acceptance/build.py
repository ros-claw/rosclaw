"""Build a pinned upstream simulator in a disposable context, without host apt."""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

SOURCE = "https://github.com/open-navigation/opennav_coverage.git"
REVISION = "65a6598c3587cb947978227c01af421e18576f0a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apt-proxy", default="")
    parser.add_argument(
        "--base-image",
        default="ros@sha256:ec00a06df4f573a86e4336c4585940f7f614646ef22675ace1b3e14ee0a82cb3",
    )
    parser.add_argument("--tag", default="rosclaw/ros-expert-jazzy:acceptance")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="ros-expert-build-") as directory:
        context = Path(directory)
        upstream = context / "opennav_coverage"
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", SOURCE, str(upstream)],
            check=True,
        )
        subprocess.run(["git", "checkout", REVISION], cwd=upstream, check=True)
        shutil.copyfile(Path(__file__).with_name("Dockerfile"), context / "Dockerfile")
        (context / ".dockerignore").write_text("**/.git\n")
        subprocess.run(
            [
                "docker",
                "build",
                "--network",
                "host",
                "--build-arg",
                f"APT_PROXY={args.apt_proxy}",
                "--build-arg",
                f"BASE_IMAGE={args.base_image}",
                "-t",
                args.tag,
                str(context),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
