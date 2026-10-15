# Read-only graph query failures remain explicit UNKNOWN

PR #679 merged normally at `1f9757c5` after three original source workflows and
the independently sealed 14-World source matrix. Actual-main revalidation then
passed six original Worlds before the seventh failed during witness replacement,
before canonical action admission or entity-fault injection. The probe exited
when `get_action_server_names_and_types_by_node` raised an RCLError. The original
failed World and incomplete main matrix remain retained; G3 was not closed.

The probe now enumerates nodes once per snapshot. A failing per-node action read
preserves other original node/action observations, records the actual operation,
node, error type and message, and marks graph completeness false. A RuntimeError
elsewhere in discovery/snapshot collection emits a new explicit UNKNOWN envelope
with no cached graph, lifecycle, transform, signal or navigation facts. Publishing
errors and unrelated programming exceptions are not silently replaced.

The exact SDK exception MRO was inspected: the observed rclpy RCLError derives
from RuntimeError. Current-health validation rejects incomplete graph facts even
when all other observations are otherwise healthy. Existing supplementary SDK
supervision retains permanent failure latching; a later successful read cannot
revive an admitted action or lease. No Body limit, age bound, deadline, ROS write
tool or REAL permission changes. Probe and helper source hashes are included in
SDK source-freeze metadata.

Focused graph/current-health/discovery software checks: 41 passed. Expanded ROS
regression: 1,150 passed, 11 deselected and one existing dependency warning. New
exact-source/actual-main World matrices remain required. These
synthetic exception checks are not counted as physical fault Worlds.
