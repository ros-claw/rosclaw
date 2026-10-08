"""Native fixture catalog must expose each fresh source-admitted dynamic mission."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"


def module():
    spec = importlib.util.spec_from_file_location("native_tools_identity", ROOT / "native_tools.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_static_mission_remains_compatible_and_dynamic_uses_fresh_identity():
    m = module()
    assert m.fixture_mission_id({}) == "gazebo-room-cleaning"
    config = {
        "occupancy_binding": {"run_id": "fresh"},
        "dynamic_fixture_admission": {"mission_id": "fresh-d2-mission"},
    }
    assert m.fixture_mission_id(config) == "fresh-d2-mission"


@pytest.mark.parametrize(
    "admission",
    [
        None,
        [],
        "old",
        {},
        {"mission_id": None},
        {"mission_id": ""},
        {"mission_id": 1},
        {"mission_id": "x" * 257},
    ],
)
def test_dynamic_missing_or_malformed_identity_refuses_catalog(admission):
    with pytest.raises(ValueError):
        module().fixture_mission_id(
            {"occupancy_binding": {}, "dynamic_fixture_admission": admission}
        )


def test_actual_fastmcp_catalog_uses_admitted_default_without_running_action(tmp_path, monkeypatch):
    m = module()
    (tmp_path / "body.json").write_text("{}")
    (tmp_path / "execution_config.json").write_text(
        json.dumps(
            {
                "occupancy_binding": {"run_id": "new-run"},
                "dynamic_fixture_admission": {"mission_id": "new-d2-mission"},
            }
        )
    )
    seen = []
    monkeypatch.setattr(m.FastMCP, "run", lambda server: seen.append(server))
    monkeypatch.setattr(sys, "argv", ["native_tools.py", "--directory", str(tmp_path)])
    m.main()
    tool = seen[0]._tool_manager._tools["coverage.execute"]
    assert tool.parameters["properties"]["mission_id"]["default"] == "new-d2-mission"
    assert tool.parameters["additionalProperties"] is False
    with pytest.raises(RuntimeError, match="rosclawd"):
        tool.fn([], "new-d2-mission")
