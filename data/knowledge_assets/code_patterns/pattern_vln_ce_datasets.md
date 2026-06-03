---
pattern_id: pattern_vln_ce_datasets
applicable_symptoms: [vln_ce_datasets]
domain: Planning_Decision
---

# Discrete VLN datasets (e.g., Room-to-Room) allow agents to teleport between nodes, which is unrealistic for real-world deployment where agents must execute free-form motion in continuous 3D environments.

**Domain**: `Planning_Decision`

## Fix

Use VLN-CE datasets that provide continuous action spaces (velocity and rotation commands) and realistic 3D environments (Matterport3D, Habitat-Matterport 3D, Gibson) with paired instructions and ground-truth trajectories, evaluated via success rate, SPL, and execution efficiency.

## Anti-pattern

Discrete graph-based navigation with node-to-node teleportation.

## Cross-domain analogies

- **Perception_Vision** → Incremental object-centric anchoring enables continuous real-world grounding from discrete priors.
  - related fix: Incremental object-centric mapping: associate VLM-derived semantic features (captions, embeddings) with LiDAR points via calibrated camera-LiDAR projection, then cluster points into object hypotheses updated frame-by-frame.
- **Learning_Training** → Jointly train on discrete and continuous datasets to learn shared navigation representations.
  - related fix: Jointly train the VLN model on multiple annotated datasets (RxR and R2R) using multitask learning to learn shared visual and linguistic representations, improving robustness and generalization.
- **Control_Locomotion** → Incorporate continuous depth or occupancy maps into the planner to adapt motion in real-time, replacing discrete teleportation.
  - related fix: Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Patch

```diff
--- vln_ce_datasets.before.py
+++ vln_ce_datasets.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Discrete VLN datasets (e.g., Room-to-Room) allow agents to teleport between nodes, which is unrealistic for real-world deployment where agents must execute free-form motion in continuous 3D environments.

+# Fix    : Use VLN-CE datasets that provide continuous action spaces (velocity and rotation commands) and realistic 3D environments (Matterport3D, Habitat-Matterport 3D, Gibson) with paired instructions and ground-truth trajectories, evaluated via success rate, SPL, and execution efficiency.

+# Avoid  : Discrete graph-based navigation with node-to-node teleportation.

```
