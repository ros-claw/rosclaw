---
pattern_id: pattern_last_mile_delivery_robot
applicable_symptoms: [last_mile_delivery_robot]
domain: Planning_Decision
---

# Last-mile delivery robots fail to follow free-form human instructions in unfamiliar urban environments without pre-mapped routes.

**Domain**: `Planning_Decision`

## Fix

UrbanNav framework: language-guided navigation that parses natural-language commands into actionable waypoints for real-time urban navigation.

## Anti-pattern

Relying on pre-mapped routes or fixed navigation policies that cannot generalize to unseen city districts.

## Cross-domain analogies

- **Perception_Vision** → Re-annotate instructions as egocentric, partial-observability waypoint tasks aligned to the robot's local sensor frame.
  - related fix: Re-annotate ScanNet scenes with local occupancy grids aligned to the camera frame, supporting both static and temporal prediction tasks.
- **Learning_Training** → Use output-consistency regularization to penalize deviations from prior instruction-following policies.
  - related fix: Functional regularisation: add a penalty on changes to the network's input-output mapping (e.g., using KL divergence or L2 distance on outputs) when training on new tasks.
- **Control_Locomotion** → Train an end-to-end policy mapping raw sensor data and language directly to actions.
  - related fix: Train a single neural network policy via deep reinforcement learning that maps raw depth camera images directly to motor commands, bypassing hand-crafted perception and control layers.

## Patch

```diff
--- last_mile_delivery_robot.before.py
+++ last_mile_delivery_robot.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Last-mile delivery robots fail to follow free-form human instructions in unfamiliar urban environments without pre-mapped routes.

+# Fix    : UrbanNav framework: language-guided navigation that parses natural-language commands into actionable waypoints for real-time urban navigation.

+# Avoid  : Relying on pre-mapped routes or fixed navigation policies that cannot generalize to unseen city districts.

```
