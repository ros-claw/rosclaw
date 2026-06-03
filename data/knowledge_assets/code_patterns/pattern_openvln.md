---
pattern_id: pattern_openvln
applicable_symptoms: [openvln]
domain: Planning_Decision
---

# Vision-language navigation for UAVs fails in data-scarce regimes, with poor long-horizon trajectory planning and low success rates.

**Domain**: `Planning_Decision`

## Fix

Fine-tune a vision-language model via reinforcement learning with rule-based policies and a long-horizon planner that generates actions using value-based rewards.

## Anti-pattern

Standard supervised fine-tuning of VLMs without RL or long-horizon planning under limited data.

## Cross-domain analogies

- **Perception_Vision** → Voxelize the airspace into 3D cells and aggregate sparse UAV views for joint occupancy-trajectory prediction.
  - related fix: Voxelize the physical world into structured 3D cells and aggregate multi-view 2D features into that unified 3D space via 2D-3D spatial sampling, then jointly predict 3D occupancy, room layout, and bounding boxes through multi-task learning.
- **Learning_Training** → Bootstrap with imitation pretraining, then refine via on-policy RL to improve long-horizon planning.
  - related fix: Mixed Imitation and Reinforcement Learning (MIRL): bootstrap policy via off-policy imitation learning, then refine with on-policy RL, gradually shifting weight from imitation to RL.
- **Control_Locomotion** → Use lightweight RL-trained policies with low-frequency execution for direct trajectory commands.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- openvln.before.py
+++ openvln.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Vision-language navigation for UAVs fails in data-scarce regimes, with poor long-horizon trajectory planning and low success rates.

+# Fix    : Fine-tune a vision-language model via reinforcement learning with rule-based policies and a long-horizon planner that generates actions using value-based rewards.

+# Avoid  : Standard supervised fine-tuning of VLMs without RL or long-horizon planning under limited data.

```
