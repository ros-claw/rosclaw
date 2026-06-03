---
pattern_id: pattern_navigation_graph_assumption
applicable_symptoms: [navigation_graph_assumption]
domain: Planning_Decision
---

# VLN agents trained under the Navigation-Graph Assumption fail in continuous environments due to reliance on oracle navigation and perfect localization.

**Domain**: `Planning_Decision`

## Fix

Train and evaluate agents in continuous environments with raw egocentric observations, uncertain localization, and fine-grained motor control, removing the graph assumption.

## Anti-pattern

Using discrete panoramic nodes with known connectivity and oracle navigation.

## Cross-domain analogies

- **Perception_Vision** → Fine-tune a visual backbone to predict metric depth and pose, enabling implicit localization for continuous VLN.
  - related fix: Fine-tune a long-horizon visual-geometry backbone to predict metric-scale depth and pose directly from images, enabling implicit state estimation and dense geometry reconstruction without external sensors.
- **Learning_Training** → Train a shared spatial representation abstracting discrete graph assumptions for continuous navigation.
  - related fix: Train a single policy on shared representations that abstract away physical differences across robot morphologies.
- **Control_Locomotion** → Use reinforcement learning to map raw observations directly to continuous actions, bypassing graph-based assumptions.
  - related fix: Use reinforcement learning to learn a control policy that directly maps sensor observations to actuator commands for plasma shape and position control.

## Patch

```diff
--- navigation_graph_assumption.before.py
+++ navigation_graph_assumption.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents trained under the Navigation-Graph Assumption fail in continuous environments due to reliance on oracle navigation and perfect localization.

+# Fix    : Train and evaluate agents in continuous environments with raw egocentric observations, uncertain localization, and fine-grained motor control, removing the graph assumption.

+# Avoid  : Using discrete panoramic nodes with known connectivity and oracle navigation.

```
