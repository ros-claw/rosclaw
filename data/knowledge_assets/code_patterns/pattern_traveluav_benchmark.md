---
pattern_id: pattern_traveluav_benchmark
applicable_symptoms: [traveluav_benchmark]
domain: Planning_Decision
---

# UAV navigation agents fail to generalize across diverse reward settings and long-horizon trajectory planning tasks in complex environments.

**Domain**: `Planning_Decision`

## Fix

Use TravelUAV benchmark with multiple reward configurations and dataset scaling to evaluate and improve VLN agents for aerial navigation.

## Anti-pattern

Evaluating UAV navigation algorithms on a single reward setting or small-scale dataset without considering diverse conditions.

## Cross-domain analogies

- **Perception_Vision** → Use intrinsic-extrinsic decomposition to separate reward structure from environment dynamics.
  - related fix: Use pinhole camera projection model with intrinsic matrix K and extrinsic matrix [R|t] to map 3D world points to 2D image coordinates, enabling local-to-world transformations.
- **Learning_Training** → Use online data aggregation to iteratively collect and correct trajectories under diverse reward conditions.
  - related fix: Iteratively collect new data under the current policy's distribution, query the expert for corrective actions, and aggregate this data into the training set (DAgger).
- **Control_Locomotion** → Closed-loop local replanning with real-time feedback for long-horizon adaptation.
  - related fix: Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Patch

```diff
--- traveluav_benchmark.before.py
+++ traveluav_benchmark.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: UAV navigation agents fail to generalize across diverse reward settings and long-horizon trajectory planning tasks in complex environments.

+# Fix    : Use TravelUAV benchmark with multiple reward configurations and dataset scaling to evaluate and improve VLN agents for aerial navigation.

+# Avoid  : Evaluating UAV navigation algorithms on a single reward setting or small-scale dataset without considering diverse conditions.

```
