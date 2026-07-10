# Minimal LeRobot Policy Fixture

This directory contains a synthetic LeRobot-style policy configuration used by
P1 integration tests.  It has **no weights** and is not meant for real
inference; it only validates config/metadata parsing and the worker inspect
path.

## Contents

- `config.json` — minimal LeRobot policy config with `policy_type=act`,
  `input_features`, and `output_features`.

## Usage in tests

```python
def test_inspect_local_policy(worker_runner, minimal_policy_dir):
    response = worker_runner.run(
        WorkerRequest(op="inspect", policy_path=str(minimal_policy_dir))
    )
    assert response.ok
    assert response.policy_metadata["policy_type"] == "act"
```
