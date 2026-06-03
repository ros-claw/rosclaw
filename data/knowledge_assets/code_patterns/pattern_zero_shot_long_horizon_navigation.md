---
pattern_id: pattern_zero_shot_long_horizon_navigation
applicable_symptoms: [zero_shot_long_horizon_navigation]
domain: Planning_Decision
---

# Navigation agents fail to reach distant goals in novel environments without prior training or pre-built 3D maps.

**Domain**: `Planning_Decision`

## Fix

Combine semantic reasoning with foundation models to enable zero-shot long-horizon navigation using only onboard sensing.

## Anti-pattern

Traditional SLAM-based or learned navigation methods that require prior environment exposure or offline map building.

## Cross-domain analogies

- **Perception_Vision** → Use text-to-imagination grounding to synthesize goal affordances for zero-shot distant navigation.
  - related fix: Generate synthetic images from landmark text descriptions via a text-to-image diffusion model, and train the agent with an auxiliary grounding loss that aligns instruction representations with imagination embeddings
- **Learning_Training** → Train a planner on abstract topological graphs that generalize across environments.
  - related fix: Train a single policy on shared representations that abstract away physical differences across robot morphologies.
- **Control_Locomotion** → Closed-loop hierarchical decomposition with local real-time replanning from online perception.
  - related fix: Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Patch

```diff
--- zero_shot_long_horizon_navigation.before.py
+++ zero_shot_long_horizon_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation agents fail to reach distant goals in novel environments without prior training or pre-built 3D maps.

+# Fix    : Combine semantic reasoning with foundation models to enable zero-shot long-horizon navigation using only onboard sensing.

+# Avoid  : Traditional SLAM-based or learned navigation methods that require prior environment exposure or offline map building.

```
