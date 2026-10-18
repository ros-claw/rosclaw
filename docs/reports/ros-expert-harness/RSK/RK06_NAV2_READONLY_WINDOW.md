# Owned Nav2 original-file observation window

The new host collector reads the already running, owned source-only SDK's
immutable Native frames. It requires the original current pointer within500ms,
SHA/time/completeness/RMW binding, distinct increasing frame identities and at
least five seconds between original capture timestamps. The fixed5.6 second
host interval accounts for the unchanged pointer freshness allowance; actual
intervals remain visible and bounded.

Before and after the window, the sink must still have the same run identity,
zero commands, zero leases, no daemon/action identity and no admission or proof
file, including dangling symlinks. Missing fields and boolean counter values
are rejected. The original before bytes are persisted before waiting, so later
failure retains the beginning of the observation without a success manifest.

The collector starts no ROS client, process, World, Runtime, driver, executor
or action. It changes no SDK state. It copies original evidence into an exclusive
output directory and never retimes it or renews the original case's L1.
An absent runtime Body snapshot remains absent; the source-only capture is not
Body admission, canonical navigation, causal repair or Memory promotion.

This first layer is **not yet a model-callable provider**. A closed owner still
has to prove its container/image/resources/source/config/URDF, keep its existing
capture loop alive, stop its own container, and independently verify the new
target scope and original artifacts. Its actual source-only Gazebo replica
would count as one new physics World, unlike the existing sensor-only provider.
The existing single shared probe budget and four original pilot attestations
remain mandatory for eventual C/E dispatch. No such dispatch is added here.

Current tests use synthetic original files and clocks; they are contract
checks, with zero SDK, ROS, physics or model execution. Frozen-source actual
owned Nav2 observation acceptance is a separate required step.
