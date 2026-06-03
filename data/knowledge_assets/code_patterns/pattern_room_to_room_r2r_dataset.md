---
pattern_id: pattern_room_to_room_r2r_dataset
applicable_symptoms: [room_to_room_r2r_dataset]
domain: Learning_Training
---

# R2R dataset's shortest-path trajectories conflate goal completion with instruction following, failing to test fine-grained route adherence.

**Domain**: `Learning_Training`

## Fix

Use R4R dataset (concatenated R2R paths) to create longer, circuitous trajectories that better differentiate instruction-following agents from goal-seeking ones.

## Anti-pattern

Evaluating VLN agents solely on R2R's direct-to-goal shortest paths.

## Cross-domain analogies

- **Perception_Vision** → Multi-modal fusion of path and instruction signals to verify fine-grained route adherence.
  - related fix: Multi-modal occupancy grid fusion that integrates RGB, depth, lidar, and proprioception with a sensor layout strategy to maximize coverage and handle self-occlusion.
- **Planning_Decision** → Use hierarchical decomposition to separate goal completion from route adherence via language-conditioned subgoals.
  - related fix: Combine visual grounding (e.g., CLIP-based detectors) with semantic mapping and language-conditioned hierarchical exploration policies, as in LOVON.
- **Control_Locomotion** → Use lightweight path sampling to decouple goal completion from instruction following.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- room_to_room_r2r_dataset.before.py
+++ room_to_room_r2r_dataset.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: R2R dataset's shortest-path trajectories conflate goal completion with instruction following, failing to test fine-grained route adherence.

+# Fix    : Use R4R dataset (concatenated R2R paths) to create longer, circuitous trajectories that better differentiate instruction-following agents from goal-seeking ones.

+# Avoid  : Evaluating VLN agents solely on R2R's direct-to-goal shortest paths.

```
