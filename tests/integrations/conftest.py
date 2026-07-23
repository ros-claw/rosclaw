"""Shared fixtures for integration tests."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def isolated_rosclaw_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use a per-test ROSClAW_HOME without leaking it to unrelated tests."""
    home = tmp_path / "rosclaw_home"
    monkeypatch.setenv("ROSCLAW_HOME", str(home))
    return home


@pytest.fixture
def minimal_policy_dir(tmp_path: Path) -> Path:
    """Return a path to a minimal LeRobot policy fixture."""
    fixture = Path(__file__).parent.parent / "fixtures" / "lerobot_policy_minimal"
    if fixture.exists():
        return fixture
    # Fallback: create a synthetic config on the fly.
    policy_dir = tmp_path / "lerobot_policy_minimal"
    policy_dir.mkdir()
    (policy_dir / "config.json").write_text(
        '{"policy_type": "act", "input_features": {}, "output_features": {}}',
        encoding="utf-8",
    )
    return policy_dir


@pytest.fixture
def fake_worker_script(tmp_path: Path) -> Path:
    """Return a fake worker_main.py that echoes a valid WorkerResponse."""
    script = tmp_path / "fake_worker_main.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "req_path = Path(sys.argv[sys.argv.index('--request-json') + 1])\n"
        "out_path = Path(sys.argv[sys.argv.index('--output-json') + 1])\n"
        "req = json.loads(req_path.read_text())\n"
        "op = req.get('op', 'inspect')\n"
        "response = {\n"
        "    'schema_version': 'rosclaw.lerobot.worker.v1',\n"
        "    'status': 'ok',\n"
        "    'op': op,\n"
        "    'policy_path': req.get('policy_path', ''),\n"
        "    'real_model_loaded': op in ('load_test', 'infer'),\n"
        "    'real_inference': op == 'infer',\n"
        "    'policy_metadata': {'policy_type': 'act', 'config_found': True},\n"
        "    'timing': {'load_time_sec': 0.1} if op in ('load_test', 'infer') else {},\n"
        "    'runtime': {'python': sys.executable, 'python_version': '3.12.0'},\n"
        "}\n"
        "if op == 'infer':\n"
        "    response['action'] = {\n"
        "        'type': 'raw_lerobot_action',\n"
        "        'values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],\n"
        "        'shape': [7],\n"
        "        'dtype': 'float32',\n"
        "    }\n"
        "out_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "out_path.write_text(json.dumps(response, indent=2))\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


@pytest.fixture
def fake_worker_script_invalid_json(tmp_path: Path) -> Path:
    """Return a fake worker that writes invalid JSON."""
    script = tmp_path / "fake_worker_invalid.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "out_path = Path(sys.argv[sys.argv.index('--output-json') + 1])\n"
        "out_path.write_text('not-json{{')\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


@pytest.fixture
def fake_worker_script_nonzero(tmp_path: Path) -> Path:
    """Return a fake worker that exits with non-zero status."""
    script = tmp_path / "fake_worker_nonzero.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('worker crashed', file=sys.stderr)\n"
        "sys.exit(1)\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


@pytest.fixture
def fake_lerobot_info(tmp_path: Path, monkeypatch):
    """Create a fake `lerobot-info` executable on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "lerobot-info"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('LeRobot info stub')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    return script


@pytest.fixture
def fake_dataset_worker_script(tmp_path: Path) -> Path:
    """Return a fake dataset_worker_main.py that echoes a valid DatasetWorkerResponse."""
    script = tmp_path / "fake_dataset_worker_main.py"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "req_path = Path(sys.argv[sys.argv.index('--request-json') + 1])\n"
        "out_path = Path(sys.argv[sys.argv.index('--output-json') + 1])\n"
        "req = json.loads(req_path.read_text())\n"
        "op = req.get('op', 'export_dataset')\n"
        "response = {\n"
        "    'schema_version': 'rosclaw.lerobot.dataset_worker.v1',\n"
        "    'status': 'ok',\n"
        "    'op': op,\n"
        "    'output_dir': req.get('output_dir', ''),\n"
        "    'repo_id': req.get('repo_id', ''),\n"
        "    'dataset': {\n"
        "        'num_episodes': 1,\n"
        "        'num_frames': 3,\n"
        "        'fps': req.get('fps', 10.0),\n"
        "        'features': {\n"
        "            'observation.state': {'shape': [2], 'dtype': 'float32'},\n"
        "            'action': {'shape': [2], 'dtype': 'float32'},\n"
        "            'observation.images.front': {'shape': [2, 2, 3], 'dtype': 'image'},\n"
        "        },\n"
        "    },\n"
        "    'files': {\n"
        "        'meta_info': True,\n"
        "        'data_files': ['data/chunk-000/file-000.parquet'],\n"
        "        'video_files': [],\n"
        "    },\n"
        "    'validation': {\n"
        "        'load_ok': True,\n"
        "        'index_ok': True,\n"
        "        'num_frames': 3,\n"
        "        'num_episodes': 1,\n"
        "        'sample_keys': ['action', 'observation.state', 'task'],\n"
        "        'sample_image_keys': [],\n"
        "    },\n"
        "    'timing': {'write_time_sec': 0.1},\n"
        "    'runtime': {'python': sys.executable, 'python_version': '3.12.0'},\n"
        "}\n"
        "out_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "out_path.write_text(json.dumps(response, indent=2))\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script

@pytest.fixture
def real_lerobot_runtime_config(isolated_rosclaw_home):
    """If a real LeRobot runtime exists, register it in the isolated home."""
    import sys

    from rosclaw.integrations.lerobot.config import (
        build_lerobot_config,
        save_lerobot_config,
    )
    from rosclaw.integrations.lerobot.runtime import inspect_lerobot_runtime

    # Probe the conventional isolated runtime path used by `rosclaw setup lerobot`.
    candidates = []
    explicit_python = os.environ.get("ROSCLAW_TEST_LEROBOT_PYTHON")
    if explicit_python:
        candidates.append(Path(explicit_python))
    candidates.extend(
        [
            isolated_rosclaw_home.parent.parent / ".venv-lerobot" / "bin" / "python",
            REPOSITORY_ROOT / ".venv-lerobot" / "bin" / "python",
        ]
    )
    for python_exe in candidates:
        if python_exe.exists():
            runtime = inspect_lerobot_runtime(str(python_exe), mode="external")
            if runtime.state in ("ready", "degraded"):
                config = build_lerobot_config(
                    profile="core",
                    mode="external",
                    runtime=runtime,
                    rosclaw_python=sys.executable,
                    rosclaw_version=".",
                )
                save_lerobot_config(config)
                break
