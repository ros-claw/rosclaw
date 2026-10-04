"""Setup retains declared input only for the same explicitly configured route."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from rosclaw.agentd.onboarding import configure_model
from rosclaw.agentd.pi_config import write_pi_model_config


def _configure(home: Path, **changes: str) -> dict:
    options = {
        "provider": "private-test",
        "model": "k3",
        "base_url": "https://example.invalid/v1",
        "api": "openai-completions",
        "api_key_ref": "env:PRIVATE_TEST_ONLY",
    }
    options.update(changes)
    write_pi_model_config(home, **options)
    doc = json.loads((home / "agent/models.json").read_text())
    return doc["providers"][options["provider"]]["models"][-1]


def _seed(home: Path, value: object, **model_overrides: object) -> None:
    _configure(home)
    path = home / "agent/models.json"
    doc = json.loads(path.read_text())
    model = doc["providers"]["private-test"]["models"][0]
    model.update(input=value, **model_overrides)
    path.write_text(json.dumps(doc))


@pytest.mark.parametrize("declared", [["text", "image"], ["text"], ["image"]])
def test_same_route_retains_explicit_input(tmp_path: Path, declared: list[str]) -> None:
    _seed(tmp_path, declared)
    assert _configure(tmp_path)["input"] == declared


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": "other"},
        {"model": "other"},
        {"api": "anthropic-messages"},
        {"base_url": "https://other.invalid/v1"},
    ],
)
def test_changed_route_does_not_inherit_input(tmp_path: Path, changes: dict[str, str]) -> None:
    _seed(tmp_path, ["text", "image"])
    assert "input" not in _configure(tmp_path, **changes)


@pytest.mark.parametrize(
    "overrides",
    [
        {"api": "anthropic-messages"},
        {"baseUrl": "https://other.invalid/v1"},
    ],
)
def test_old_model_route_override_does_not_transfer_input(tmp_path: Path, overrides: dict) -> None:
    _seed(tmp_path, ["text", "image"], **overrides)
    assert "input" not in _configure(tmp_path)


@pytest.mark.parametrize(
    "invalid", [None, "image", [], ["text", "audio"], [True], ["text", "text"], {"input": "image"}]
)
def test_invalid_input_is_not_promoted(tmp_path: Path, invalid: object) -> None:
    _seed(tmp_path, invalid)
    assert "input" not in _configure(tmp_path)


def test_new_custom_named_like_builtin_has_no_inferred_input(tmp_path: Path) -> None:
    assert "input" not in _configure(tmp_path)


def test_ambiguous_duplicate_model_does_not_inherit_input(tmp_path: Path) -> None:
    _seed(tmp_path, ["text", "image"])
    path = tmp_path / "agent/models.json"
    doc = json.loads(path.read_text())
    models = doc["providers"]["private-test"]["models"]
    models.append(dict(models[0]))
    path.write_text(json.dumps(doc))
    assert "input" not in _configure(tmp_path)


def test_public_pi_sdk_offline_resolved_inputs(tmp_path: Path) -> None:
    """Opt-in against an installed public SDK; no credential or model request."""
    sdk = os.environ.get("ROSCLAW_PI_SDK_MODULE")
    if not sdk:
        pytest.skip("set ROSCLAW_PI_SDK_MODULE to installed PI dist/index.js")
    cases = []
    for index, changes in enumerate(
        [
            {},
            {"provider": "other"},
            {"model": "other"},
            {"api": "anthropic-messages"},
            {"base_url": "https://other.invalid/v1"},
        ]
    ):
        home = tmp_path / str(index)
        _seed(home, ["text", "image"])
        before = home / "before.json"
        before.write_bytes((home / "agent/models.json").read_bytes())
        _configure(home, **changes)
        cases.append(
            {
                "path": str(before),
                "provider": "private-test",
                "id": "k3",
                "expected": ["text", "image"],
            }
        )
        cases.append(
            {
                "path": str(home / "agent/models.json"),
                "provider": changes.get("provider", "private-test"),
                "id": changes.get("model", "k3"),
                "expected": ["text"] if changes else ["text", "image"],
            }
        )
    custom = tmp_path / "custom"
    _configure(custom)
    cases.append(
        {
            "path": str(custom / "agent/models.json"),
            "provider": "private-test",
            "id": "k3",
            "expected": ["text"],
        }
    )
    builtin = tmp_path / "builtin"
    configure_model(builtin, "kimi-code")
    settings = json.loads((builtin / "agent/settings.json").read_text())
    assert not (builtin / "agent/models.json").exists()
    cases.append(
        {
            "path": None,
            "provider": settings["defaultProvider"],
            "id": settings["defaultModel"],
            "expected": ["text", "image"],
        }
    )
    script = """
const {ModelRuntime} = await import(process.argv[1]);
let raw = '';
for await (const chunk of process.stdin) raw += chunk.toString();
const fixture = JSON.parse(raw);
const results = [];
for (const item of fixture.cases) {
  const runtime = await ModelRuntime.create({modelsPath: item.path,
    authPath: fixture.root + '/absent-auth.json',
    modelsStorePath: fixture.root + '/cache-' + results.length,
    allowModelNetwork: false, refreshOnCreate: false});
  if (runtime.getError()) throw new Error(runtime.getError());
  const model = runtime.getModel(item.provider, item.id);
  if (!model) throw new Error('Missing fixture model');
  results.push(model.input);
}
console.log(JSON.stringify(results));
"""
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script, Path(sdk).resolve().as_uri()],
        input=json.dumps({"root": str(tmp_path), "cases": cases}),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert json.loads(result.stdout) == [case["expected"] for case in cases]
