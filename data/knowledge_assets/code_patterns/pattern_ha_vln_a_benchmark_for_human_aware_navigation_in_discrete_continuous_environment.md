---
pattern_id: pattern_ha_vln_a_benchmark_for_human_aware_navigation_in_discrete_continuous_environment
applicable_symptoms: [ha_vln_a_benchmark_for_human_aware_navigation_in_discrete_continuous_environment]
domain: Planning_Decision
---

# VLN agents fail to navigate safely and socially in environments with dynamic human crowds, leading to collisions or inefficient paths.

**Domain**: `Planning_Decision`

## Fix

HA-VLN benchmark with discrete-continuous environments, dynamic multi-human interactions, real-world validation, and an open leaderboard to evaluate human-aware navigation policies.

## Anti-pattern

Existing VLN benchmarks ignore dynamic human interactions or use only static/scripted humans.

## Cross-domain analogies

- **Perception_Vision** → Use hierarchical decomposition with proximal and distal sub-networks for near and far crowd features.
  - related fix: Hierarchical network with proximal sub-network for near-field features, distal sub-network for far-field context, recurrent temporal modeling, and attention-based fusion to output continuous risk scores.
- **Learning_Training** → Use adaptive normalization and gradient scaling to stabilize crowd-aware path planning without explicit social constraints.
  - related fix: Use adaptive gradient clipping (AGC) and Scaled Weight Standardization to train deep networks without batch normalization.
- **Control_Locomotion** → Use reinforcement learning to map visual observations directly to socially-aware navigation actions in crowds.
  - related fix: Use reinforcement learning to learn a control policy that directly maps sensor observations to actuator commands for plasma shape and position control.

## Patch

```diff
--- ha_vln_a_benchmark_for_human_aware_navigation_in_discrete_continuous_environment.before.py
+++ ha_vln_a_benchmark_for_human_aware_navigation_in_discrete_continuous_environment.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to navigate safely and socially in environments with dynamic human crowds, leading to collisions or inefficient paths.

+# Fix    : HA-VLN benchmark with discrete-continuous environments, dynamic multi-human interactions, real-world validation, and an open leaderboard to evaluate human-aware navigation policies.

+# Avoid  : Existing VLN benchmarks ignore dynamic human interactions or use only static/scripted humans.

```
