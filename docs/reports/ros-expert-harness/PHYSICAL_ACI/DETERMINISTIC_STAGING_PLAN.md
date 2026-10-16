# Deterministic conditional staging planning

Known original-input mechanisms can be planned without a model call:

```bash
.venv/bin/python -m rosclaw.entrypoint ros engineering-case \
  --snapshot original-system.json --urdf original-robot.urdf \
  --requirements original-navigation-requirements.json \
  --reviewed-at 2026-10-16T12:00:00+00:00 \
  --plan-staging --staging-nav2 original-nav2.yaml \
  --staging-bringup original-bringup.json --json
```

This returns a conditional proposal, minimal changes, exact rollback, original
hash bindings, unknown constraints and the existing new-World/L1/canonical/stop/
source proof obligations. It applies no configuration, invokes no model, starts
no World and grants no motion authority. Existing exclusive `--output` creation
still refuses to overwrite any input. Concrete YAML values and JSON objects are
required; YAML anchors/aliases are rejected before expansion.

`plan_staging_repairs` reuses the existing option and semantic repair compilers.
It checks the unchanged case/model, original URDF bytes and requirements; recomputes
conservative geometry from the original URDF and observations at the explicit
case clock rather than accepting a supplied radius floor. Source mismatch or
missing original Body evidence causes abstention. Unknown constraints stay visible;
they never become guessed repairs or current physical proof. If any observed
violation lacks a supported mechanism, the planner returns reasons and no partial
proposal. Supported combinations use the existing same-baseline compound compiler.

The CLI accepts no endpoint ownership review or process operation. A competing
route without the existing SDK's operator-bound review remains unsupported. A
trusted host already holding that exact bound review can pass it through the pure
API for a future disposable staging proposal. This never authorizes terminating
an original endpoint or mutating live ROS. No new execution gateway is introduced.

The supplied staging files are original task inputs. Relevant parameter readbacks
are checked by the existing compiler; unknown/unobserved configuration and candidate
preconditions remain pending independent verification. No result claims complete
diagnosis, current readiness, physical success or general model/Memory efficacy.
The explicit existing SDK/rosclawd verification workflow remains mandatory.
