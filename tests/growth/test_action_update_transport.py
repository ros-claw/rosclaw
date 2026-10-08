import numpy as np
import pytest

from rosclaw.growth.action_update_transport import bounded_residual_update_transport


def inputs(width=3):
    shape = (5, width)
    return [np.zeros(shape) for _ in range(5)] + [np.tile([-2.0, 2.0], (width, 1))]


def run(values, **kw):
    return bounded_residual_update_transport(*values, residual_cap=0.16, slew_cap=0.012, **kw)


def test_unmasked_update():
    values = inputs()
    values[1][:] = 0.01
    r = run(values)
    assert r["changed_coordinates_fully_masked"] == 0
    assert r["transported_to_intended_l2_ratio"] == 1
    assert r["row_count"] == 5
    assert r["all_supplied_rows_preserved"] is True
    assert r["runtime_execution_authorized"] is False


def test_slew_hides_update():
    values = inputs()
    values[0][:] = 1
    values[1][:] = 2
    values[4][:] = 0.012
    r = run(values)
    assert r["intended_changed_coordinates"] == 15
    assert r["changed_coordinates_fully_masked"] == 15
    assert r["transported_update_rms"] == 0


def test_joint_bound_hides_update():
    values = inputs()
    values[0][:] = 0.01
    values[1][:] = 0.02
    values[5][:, 1] = 0.001
    values[4][:] = 0.001
    r = run(values)
    assert r["changed_coordinates_fully_masked"] == 15
    assert r["behavior_limit_active_coordinates"] == 15


def test_zero_update_has_no_ratio():
    r = run(inputs(1))
    assert r["transported_to_intended_l2_ratio"] is None
    assert r["changed_coordinates_fully_masked"] == 0


@pytest.mark.parametrize("index", range(6))
def test_nonfinite_inputs_rejected(index):
    values = inputs()
    values[index].flat[0] = np.nan
    with pytest.raises(ValueError):
        run(values)


@pytest.mark.parametrize("index", range(6))
def test_incomplete_inputs_rejected(index):
    values = inputs()
    values[index] = values[index][:-1]
    with pytest.raises(ValueError):
        run(values)


def test_measured_behavior_must_match():
    values = inputs()
    values[4][0, 0] = 0.001
    with pytest.raises(ValueError, match="EXACTLY"):
        run(values)


def test_input_hash_binds_every_array():
    a = inputs()
    b = inputs()
    b[1][1, 1] = 0.005
    assert run(a)["input_numeric_hash"] != run(b)["input_numeric_hash"]


def test_does_not_mutate_inputs():
    values = inputs()
    values[1][:] = 0.002
    copies = [v.copy() for v in values]
    run(values)
    assert all(np.array_equal(a, b) for a, b in zip(values, copies, strict=True))


@pytest.mark.parametrize("cap,slew", [(True, 0.01), (0.16, 0), (0.16, 0.2), (np.nan, 0.01)])
def test_invalid_caps(cap, slew):
    with pytest.raises(ValueError):
        bounded_residual_update_transport(*inputs(), residual_cap=cap, slew_cap=slew)


def test_oversized_dimensions_rejected():
    with pytest.raises(ValueError):
        run(inputs(65))
