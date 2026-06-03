---
pattern_id: pattern_connectivity_graph
applicable_symptoms: [connectivity_graph]
domain: Planning_Decision
---

# Discrete VLN agents rely on a pre-defined connectivity graph that may contain errors or inconsistencies from raw 3D scans, leading to unrealistic navigation constraints and poor generalization.

**Domain**: `Planning_Decision`

## Fix

Refine the connectivity graph by merging overlapping scans, fixing connectivity errors, and producing a cleaner graph (e.g., from Matterport3D to Habitat-Matterport3D).

## Anti-pattern

Using the raw Matterport3D connectivity graph without refinement.

## Cross-domain analogies

- **Perception_Vision** → Use SLAM-derived trajectories as ground-truth connectivity references to correct graph errors.
  - related fix: Use SLAM-derived trajectories as ground-truth motion tendency references to supervise or condition video world model predictions.
- **Learning_Training** → Randomize connectivity graph parameters during training to improve navigation robustness to graph errors.
  - related fix: Use domain randomization: vary simulation parameters (friction, mass, lighting, delay) randomly during training to improve policy robustness to real-world conditions.
- **Control_Locomotion** → Train an end-to-end policy with domain randomization to bypass reliance on imperfect connectivity graphs.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- connectivity_graph.before.py
+++ connectivity_graph.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Discrete VLN agents rely on a pre-defined connectivity graph that may contain errors or inconsistencies from raw 3D scans, leading to unrealistic navigation constraints and poor generalization.

+# Fix    : Refine the connectivity graph by merging overlapping scans, fixing connectivity errors, and producing a cleaner graph (e.g., from Matterport3D to Habitat-Matterport3D).

+# Avoid  : Using the raw Matterport3D connectivity graph without refinement.

```
