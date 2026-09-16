"""P0-6: the installer support matrix must be TRUTHFUL.

For every platform the tooling declares *supported*:
  * the SHA256 map in install_1_4.sh must carry a real 64-hex checksum
    (never a placeholder);
  * the asset filename must exist in the upstream v1.4.0 release (network
    check, integration-marked).
No placeholder may appear in the supported matrix.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON = REPO_ROOT / "scripts" / "seekdb" / "_common.sh"
INSTALL = REPO_ROOT / "scripts" / "seekdb" / "install_1_4.sh"

ENGINE_VERSION = "1.4.0"
BUILD = "100000212026082616"


def _asset_name(os_id: str, arch: str) -> str:
    return f"seekdb_{ENGINE_VERSION}-{BUILD}{os_id}_{arch}.deb"


def _supported_pairs() -> list[tuple[str, str]]:
    text = COMMON.read_text()
    m = re.search(r'^SEEKDB_SUPPORTED_PLATFORMS="([^"]+)"', text, flags=re.M)
    assert m, "SEEKDB_SUPPORTED_PLATFORMS not declared in _common.sh"
    return [tuple(p.split("/")) for p in m.group(1).split()]


def _experimental_pairs() -> list[tuple[str, str]]:
    text = COMMON.read_text()
    m = re.search(r'^SEEKDB_EXPERIMENTAL_PLATFORMS="([^"]*)"\s*$', text, flags=re.M)
    assert m is not None, "SEEKDB_EXPERIMENTAL_PLATFORMS not declared in _common.sh"
    body = m.group(1).strip()
    return [tuple(p.split("/")) for p in body.split()] if body else []


def _sha_map() -> dict[str, str]:
    text = INSTALL.read_text()
    return dict(re.findall(r'\["(seekdb_[^"]+\.deb)"\]="([0-9a-f]{64})"', text))


def test_supported_matrix_nonempty_and_scoped():
    pairs = _supported_pairs()
    assert pairs, "no supported platforms declared"
    # PR-SDB-140-5: only Ubuntu 24.04 is formally supported this round.
    assert set(pairs) == {("ubuntu24.04", "amd64"), ("ubuntu24.04", "arm64")}


def test_supported_assets_have_real_sha256():
    sha = _sha_map()
    install_text = INSTALL.read_text()
    assert "__FILL_ME__" not in install_text, "placeholder checksum in installer"
    for os_id, arch in _supported_pairs():
        asset = _asset_name(os_id, arch)
        assert asset in sha, f"supported asset {asset} lacks a SHA256 entry"
        assert re.fullmatch(r"[0-9a-f]{64}", sha[asset])


def test_supported_and_experimental_disjoint():
    assert not (set(_supported_pairs()) & set(_experimental_pairs()))


def test_asset_name_function_covers_declared_platforms():
    """_common.sh seekdb_asset_name() must emit names for every declared
    platform (supported or experimental) and refuse the rest."""
    text = COMMON.read_text()
    case_block = re.search(r"case \"\$os_id\" in\n\s+([a-z0-9.|]+)\)", text)
    assert case_block, "seekdb_asset_name os case not found"
    known = set(case_block.group(1).split("|"))
    for os_id, _arch in _supported_pairs() + _experimental_pairs():
        assert os_id in known, f"{os_id} declared but not handled by seekdb_asset_name"


@pytest.mark.integration
def test_supported_assets_exist_upstream():
    """Network check: every supported asset name is in the GitHub release."""
    import subprocess

    out = subprocess.run(
        [
            "gh",
            "api",
            f"repos/oceanbase/seekdb/releases/tags/v{ENGINE_VERSION}",
            "--jq",
            ".assets[].name",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if out.returncode != 0:
        pytest.skip(f"gh api unavailable: {out.stderr.strip()}")
    upstream = set(out.stdout.split())
    for os_id, arch in _supported_pairs():
        assert _asset_name(os_id, arch) in upstream
