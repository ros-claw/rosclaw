# Offline engineering case CLI

The existing ROS CLI exposes the read-only case compiler without a benchmark
controller or a model call:

```bash
.venv/bin/python -m rosclaw.entrypoint ros engineering-case \
  --snapshot original-system.json \
  --urdf original-robot.urdf \
  --requirements original-navigation-requirements.json \
  --reviewed-at 2026-10-16T12:00:00+00:00 \
  --summary-only --json
```

Use the explicit original review clock for reproducible historical analysis. It
does not make historical facts current. Omit `--summary-only` for the full typed
case and its evidence index. Original model, URDF and requirements file SHA256s,
case/snapshot hashes, constraint differences and source-linked conditional
candidates remain visible. No fault label or repair answer is an input.

`--source-review` accepts a bounded original historical review JSON. Optional
`--expected-run-id` and `--expected-body-hash` compare the request with that input;
missing reviews cannot establish a match. The caller-supplied review is explicitly
not an independent current source gate. `MISMATCH` and `UNKNOWN` remain visible.

This command accepts only a sealed offline snapshot; it has no endpoint, live
probe, configuration write, staging selection, action or validation argument.
Inputs are bounded regular files. `--output` creates a new derived JSON file
exclusively and refuses to overwrite any existing path, including source inputs.

Every result remains `SEALED_OFFLINE_REPLAY_NOT_CURRENT_STATE_OR_ADMISSION`, with
`current_authority=NONE` and `current_l1=NOT_EVALUATED`. Even a conditional repair
candidate must enter the existing explicit staging and fresh SDK verification
workflow before it can acquire any physical success evidence. The CLI does not
claim that models reason better or that historical Memory transfers successfully.
