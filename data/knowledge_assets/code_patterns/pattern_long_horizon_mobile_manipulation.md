---
pattern_id: pattern_long_horizon_mobile_manipulation
applicable_symptoms: [long_horizon_mobile_manipulation]
domain: Planning_Decision
---

# Long-horizon mobile manipulation tasks suffer from error accumulation over dozens of action primitives, where early mistakes compound and inference must be maintained without resets.

**Domain**: `Planning_Decision`

## Fix

Use the ODYSSEY benchmark for structured evaluation, which provides standardized tasks that test the interplay of mobility and dexterity under natural language instructions, enabling systematic assessment of long-horizon planning and coordination.

## Anti-pattern

Prior benchmarks lacked structured evaluation for long-horizon mobile manipulation, often focusing on either navigation or manipulation in isolation.

## Cross-domain analogies

- **Perception_Vision** → Train a policy on simulated error-corrupted trajectories to match real execution drift.
  - related fix: Generate synthetic depth images with simulated sensor noise, self-occlusion, and lighting effects to match real sensor distribution.
- **Learning_Training** → Pre-train a latent world model on diverse action sequences to enable closed-loop error correction without task-specific resets.
  - related fix: Pre-train on large-scale image-text-action triplets using self-supervised pretext tasks to learn generic representations that transfer to new navigation tasks.
- **Control_Locomotion** → Closed-loop verification after each primitive to detect and correct errors early.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- long_horizon_mobile_manipulation.before.py
+++ long_horizon_mobile_manipulation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Long-horizon mobile manipulation tasks suffer from error accumulation over dozens of action primitives, where early mistakes compound and inference must be maintained without resets.

+# Fix    : Use the ODYSSEY benchmark for structured evaluation, which provides standardized tasks that test the interplay of mobility and dexterity under natural language instructions, enabling systematic assessment of long-horizon planning and coordination.

+# Avoid  : Prior benchmarks lacked structured evaluation for long-horizon mobile manipulation, often focusing on either navigation or manipulation in isolation.

```
