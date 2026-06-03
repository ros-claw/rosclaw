---
pattern_id: pattern_cache_enabled_execution
applicable_symptoms: [cache_enabled_execution]
domain: Planning_Decision
---

# Recomputing motion plans from scratch in every deployment causes high computational overhead and slow response times in familiar environments.

**Domain**: `Planning_Decision`

## Fix

Cache task-location trajectories from an exploration phase and retrieve them for reuse, bypassing full planning pipeline when a known task at a known location is encountered.

## Anti-pattern

Recomputing trajectories from scratch each time without caching.

## Cross-domain analogies

- **Perception_Vision** → Active perception-inspired incremental replanning using task-driven memory to prune redundant computation.
  - related fix: Integrate active perception with semantic mapping: agent selects viewpoints to reduce ambiguity while building a task-driven semantic map from RGB-D or lidar data.
- **Learning_Training** → Use adaptive gain scheduling to reuse prior plans, reducing recomputation in familiar settings.
  - related fix: Use adaptive gradient clipping (AGC) and Scaled Weight Standardization to train deep networks without batch normalization.
- **Control_Locomotion** → Use a learned policy with domain randomization to reuse prior plans, avoiding recomputation.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- cache_enabled_execution.before.py
+++ cache_enabled_execution.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Recomputing motion plans from scratch in every deployment causes high computational overhead and slow response times in familiar environments.

+# Fix    : Cache task-location trajectories from an exploration phase and retrieve them for reuse, bypassing full planning pipeline when a known task at a known location is encountered.

+# Avoid  : Recomputing trajectories from scratch each time without caching.

```
