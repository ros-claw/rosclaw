"""Private stream bounds and instrument argv, no process, Node or service."""

import importlib
import os
from pathlib import Path

import pytest

from tests.connectors.ros.test_backend_probe_evidence import specification


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("owned_probe_controller")


def test_declared_worker_target_cannot_be_active_robot(module):
    policy = specification()
    name = policy["native_policy"]["contact_policy"]["model_name"]
    with pytest.raises(ValueError, match="cannot target"):
        module.worker_arguments(
            policy,
            binary=Path("not_executed"),
            robot_model=name,
            partition="rosclaw_backend_" + "a" * 32,
        )
    argv = module.worker_arguments(
        policy,
        binary=Path("not_executed"),
        robot_model="disjoint_robot",
        partition="rosclaw_backend_" + "a" * 32,
    )
    assert argv[2] == name and argv[-1] == "--owned-runtime"
    assert [float(v) for v in argv[4:7]] == [*policy["probe_xy"], policy["lift_z_m"]]


def test_actual_local_pipe_preserves_extra_line_without_consuming_next_record(module):
    reader, writer = os.pipe()
    with (
        os.fdopen(reader, "rb", buffering=0) as stream,
        os.fdopen(writer, "wb", buffering=0) as sink,
    ):
        output = module.BoundedWorkerOutput(stream)
        try:
            sink.write(b'{"first":1}\n{"second":2}\n')
            assert output.read(0.01) == b'{"first":1}'
            assert output.read(0.01) == b'{"second":2}'
        finally:
            output.close()


def test_actual_local_pipe_refuses_partial_closed_original_record(module):
    reader, writer = os.pipe()
    with os.fdopen(reader, "rb", buffering=0) as stream:
        os.write(writer, b"partial_original_source")
        os.close(writer)
        output = module.BoundedWorkerOutput(stream)
        try:
            with pytest.raises(ValueError, match="incomplete"):
                output.read(0.01)
            assert output.buffer == b"partial_original_source"
        finally:
            output.close()


def test_actual_local_pipe_missing_reply_has_fixed_short_deadline(module):
    reader, writer = os.pipe()
    with os.fdopen(reader, "rb", buffering=0) as stream, os.fdopen(writer, "wb", buffering=0):
        output = module.BoundedWorkerOutput(stream)
        try:
            with pytest.raises(ValueError, match="deadline expired"):
                output.read(0.001)
        finally:
            output.close()
