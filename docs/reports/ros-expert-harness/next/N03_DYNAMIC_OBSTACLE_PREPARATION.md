
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


## Fresh direct lifecycle replies in both owned launch paths

The shared startup gate now requires all seven fresh direct GetState replies,
including coverage_server. Both the ordinary brush/physics fixture and the
independent instrument backend launch the read-only lifecycle probe as an owned
mandatory child. Its exit remains a dependency failure; producer-declared
readiness and historic activation logs cannot replace live replies. Startup
timeout preserves the original bounded reply snapshot and SHA-256 without
retrying, resetting a deadline, or granting action/stop authority. Source tests
cover stale reply retention and mandatory child loss. This is source preparation;
no new Native dynamic World acceptance has been run.


## D3 Native completion dispatch correction

The full host entry and scenario-progress decoder already supported D3, but
the Native positive completion branch still rejected every case except D2.
It now permits completed D2 or D3 progress with exact boolean completion and
NOT_VERIFIED fixture status. Negative D4/D5, unfinished perturbations and
producer-claimed acceptance remain rejected. Actual enabled free-cell revisit
diagnostics remain mandatory. D3 final acceptance still separately requires
closed original component replay, four original introduction/withdrawal
confirmations, nonconcurrent blocker masks and no old-mask leakage, canonical
receipts, completed Native root task, Practice/Memory and independent stop.
This fixes source dispatch; no actual D3 task has yet been accepted.
