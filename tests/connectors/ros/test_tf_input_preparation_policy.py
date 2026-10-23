"""SDK preparation policy fails closed before native dependencies are imported."""

from copy import deepcopy

import pytest

from integrations.ros_probe.acceptance.simulation_tf_input_preparation import validate_policy


def settings():
    return {
        "generic_source_observer": True,
        "fault": "none",
        "governed_sink": {},
        "run_id": "owned-run",
        "seed": 101,
        "tf_input_preparation": {
            "schema_version": "rosclaw.tf_input_preparation.v1",
            "run_id": "owned-run",
            "seed": 101,
        },
    }


@pytest.mark.parametrize("seed", [101, 202])
def test_exact_isolated_seed_and_run_required(seed):
    value = settings()
    value["seed"] = value["tf_input_preparation"]["seed"] = seed
    before = deepcopy(value)
    assert validate_policy(value, isolated=True) == value["tf_input_preparation"]
    assert value == before


@pytest.mark.parametrize(
    "mutation",
    [
        "unisolated",
        "foreign_run",
        "extra",
        "seed",
        "bool_seed",
        "engineering",
        "missing_sink",
        "other_fault",
    ],
)
def test_invalid_context_never_reaches_native_access(mutation):
    value = settings()
    isolated = True
    if mutation == "unisolated":
        isolated = False
    elif mutation == "foreign_run":
        value["tf_input_preparation"]["run_id"] = "foreign"
    elif mutation == "extra":
        value["tf_input_preparation"]["activate_inactive_nodes"] = True
    elif mutation == "seed":
        value["seed"] = value["tf_input_preparation"]["seed"] = 303
    elif mutation == "bool_seed":
        value["seed"] = value["tf_input_preparation"]["seed"] = True
    elif mutation == "engineering":
        value["engineering_capture"] = {}
    elif mutation == "missing_sink":
        del value["governed_sink"]
    else:
        value["fault"] = "other"
    with pytest.raises(ValueError, match="closed isolated"):
        validate_policy(value, isolated=isolated)
