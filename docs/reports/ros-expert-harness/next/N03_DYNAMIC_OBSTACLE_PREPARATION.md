
## Native launcher DDS port preflight

The future Native entry now evaluates its registered domain against the actual Linux host ephemeral range before source/merge checks or any process launch. It persists the bounded host preflight only after all admission checks pass. It does not assert container network readiness or diagnose the original B100828 timeout. A conflicting registered domain is refused before any command/process/output-directory creation. Source tests now register domain81; the frozen N02 evaluation protocol remains unchanged. Focused99 passed and full ROS1350 passed/10 integration deselected/one existing warning in29.00s. Actual Native transport/physics remains NOT_RUN.

## Backend container kernel preflight

The independent-instrument backend launcher also checks its own actual kernel
DDS/ephemeral port range before source preparation and before any ROS/World
child. It preserves the original process network-namespace identifier and the
actual kernel range in an exclusive `container-network-preflight.json`; legacy
host field names are normalized to `observed_kernel_ephemeral_range` without
claiming cross-process namespace admission. Source-only `--prepare-only`
remains unlaunched source preparation.

Full ROS1352 passed/10 integration deselected/one existing warning. In the fixed
generic SDK image with network disabled/read-only filesystem/UID1000/no
capabilities, actual domain201 was refused before any preparation; domain81
wrote the actual kernel/namespace record before the deliberate unlaunched
prepare boundary. Original traceback and executed source hashes were retained
and rechecked. No World, ROS Node, robot action or physical acceptance occurred.

N03 physical runs still require the P0 merge gate. After that rebase, the backend
startup must be joined to the new mandatory fresh GetState readiness path;
backend lifecycle-probe integration and real startup remain pending.
