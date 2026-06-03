---
pattern_id: pattern_dual_process_theory
applicable_symptoms: [dual_process_theory]
domain: Planning_Decision
---

# VLN agent fails to adapt reasoning depth to task complexity, wasting compute on simple steps or missing multi-step reasoning on hard cases.

**Domain**: `Planning_Decision`

## Fix

Adaptive Chain-of-Thought mechanism that dynamically switches between fast reactive (System 1) and slow deliberative (System 2) reasoning based on task complexity.

## Anti-pattern

Using a fixed-depth chain-of-thought for all navigation steps.

## Cross-domain analogies

- **Perception_Vision** → Fine-tune a visual-language backbone to predict adaptive reasoning depth directly from task complexity.
  - related fix: Fine-tune a long-horizon visual-geometry backbone to predict metric-scale depth and pose directly from images, enabling implicit state estimation and dense geometry reconstruction without external sensors.
- **Learning_Training** → Use a lightweight policy network with adaptive depth, switching to global attention for complex multi-step tasks.
  - related fix: Use a convolutional stem followed by Transformer blocks with global attention to model dependencies across 100kb+ distances
- **Control_Locomotion** → Learn a policy that maps visual inputs to variable reasoning depth via reinforcement learning.
  - related fix: Use reinforcement learning to learn a control policy that directly maps sensor observations to actuator commands for plasma shape and position control.

## Patch

```diff
--- dual_process_theory.before.py
+++ dual_process_theory.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent fails to adapt reasoning depth to task complexity, wasting compute on simple steps or missing multi-step reasoning on hard cases.

+# Fix    : Adaptive Chain-of-Thought mechanism that dynamically switches between fast reactive (System 1) and slow deliberative (System 2) reasoning based on task complexity.

+# Avoid  : Using a fixed-depth chain-of-thought for all navigation steps.

```
