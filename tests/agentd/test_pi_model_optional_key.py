"""Optional credentials must not generate invalid PI models.json."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from rosclaw.agentd.onboarding import configure_model
from rosclaw.agentd.pi_config import write_pi_model_config


def _write(home: Path, ref: str = "") -> dict:
    write_pi_model_config(
        home,
        provider="private-local",
        model="private-model",
        base_url="http://127.0.0.1:9/v1",
        api_key_ref=ref,
    )
    return json.loads((home / "agent/models.json").read_text())["providers"]["private-local"]


def test_new_provider_without_key_omits_field(tmp_path: Path) -> None:
    assert "apiKey" not in _write(tmp_path)


def test_existing_empty_key_is_removed(tmp_path: Path) -> None:
    _write(tmp_path)
    path = tmp_path / "agent/models.json"
    doc = json.loads(path.read_text())
    doc["providers"]["private-local"]["apiKey"] = ""
    path.write_text(json.dumps(doc))
    assert "apiKey" not in _write(tmp_path)


def test_existing_env_reference_is_retained_without_new_reference(tmp_path: Path) -> None:
    _write(tmp_path, "env:PRIVATE_OLD_REFERENCE")
    assert _write(tmp_path)["apiKey"] == "$PRIVATE_OLD_REFERENCE"


def test_explicit_new_env_reference_replaces_existing(tmp_path: Path) -> None:
    _write(tmp_path, "env:PRIVATE_OLD_REFERENCE")
    assert _write(tmp_path, "env:PRIVATE_NEW_REFERENCE")["apiKey"] == "$PRIVATE_NEW_REFERENCE"


def test_local_onboarding_without_key_omits_field(tmp_path: Path) -> None:
    summary = configure_model(
        tmp_path,
        "local",
        base_url="http://127.0.0.1:9/v1",
        model="private-model",
    )
    doc = json.loads((tmp_path / "agent/models.json").read_text())
    assert "apiKey" not in doc["providers"]["local"]
    assert summary["api_key_ref"] == ""


def test_public_sdk_schema_and_auth_are_distinct(tmp_path: Path) -> None:
    sdk = os.environ.get("ROSCLAW_PI_SDK_MODULE")
    if not sdk:
        pytest.skip("set ROSCLAW_PI_SDK_MODULE to installed PI dist/index.js")
    homes = {}
    for name, ref in [
        ("none", ""),
        ("missing", "env:ROSCLAW_PRIVATE_OPTIONAL_KEY_FIXTURE"),
        ("provided", "env:ROSCLAW_PRIVATE_OPTIONAL_KEY_FIXTURE"),
    ]:
        home = tmp_path / name
        _write(home, ref)
        homes[name] = str(home / "agent/models.json")
    # The original invalid form fails schema validation, rather than create().
    original = json.loads(Path(homes["none"]).read_text())
    original["providers"]["private-local"]["apiKey"] = ""
    bad = tmp_path / "empty.json"
    bad.write_text(json.dumps(original))
    homes["empty"] = str(bad)
    script = """
const {ModelRuntime} = await import(process.argv[1]);
let raw = ''; for await (const chunk of process.stdin) raw += chunk.toString();
const fixture = JSON.parse(raw), results = {};
for (const [name, path] of Object.entries(fixture.homes)) {
  delete process.env.ROSCLAW_PRIVATE_OPTIONAL_KEY_FIXTURE;
  if (name === 'provided') process.env.ROSCLAW_PRIVATE_OPTIONAL_KEY_FIXTURE = 'PRIVATE_TEST_ONLY';
  const runtime = await ModelRuntime.create({modelsPath: path,
    authPath: fixture.root + '/absent-auth-' + name + '.json',
    modelsStorePath: fixture.root + '/cache-' + name, allowModelNetwork: false});
  let authResolved = false, authFailure = false;
  try { authResolved = !!(await runtime.getAuth('private-local')); }
  catch { authFailure = true; }
  results[name] = {schemaError: !!runtime.getError(),
    modelFound: !!runtime.getModel('private-local', 'private-model'),
    configured: runtime.hasConfiguredAuth('private-local'),
    check: (await runtime.checkAuth('private-local'))?.type ?? null,
    available: (await runtime.getAvailable('private-local')).map(model => model.id),
    authResolved, authFailure};
}
console.log(JSON.stringify(results));
"""
    env = dict(os.environ)
    env.pop("ROSCLAW_PRIVATE_OPTIONAL_KEY_FIXTURE", None)
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, Path(sdk).resolve().as_uri()],
        input=json.dumps({"root": str(tmp_path), "homes": homes}),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    rows = json.loads(result.stdout)
    assert rows["none"] == {
        "schemaError": False,
        "modelFound": True,
        "configured": False,
        "check": None,
        "available": [],
        "authResolved": False,
        "authFailure": False,
    }
    assert rows["missing"] == {
        "schemaError": False,
        "modelFound": True,
        "configured": False,
        "check": None,
        "available": [],
        "authResolved": False,
        "authFailure": True,
    }
    assert rows["provided"] == {
        "schemaError": False,
        "modelFound": True,
        "configured": True,
        "check": "api_key",
        "available": ["private-model"],
        "authResolved": True,
        "authFailure": False,
    }
    assert rows["empty"] == {
        "schemaError": True,
        "modelFound": False,
        "configured": False,
        "check": None,
        "available": [],
        "authResolved": False,
        "authFailure": False,
    }
