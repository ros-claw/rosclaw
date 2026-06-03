---
pattern_id: pattern_dynamic_multi_human_interactions
applicable_symptoms: [dynamic_multi_human_interactions]
domain: Planning_Decision
---

# Robot navigation fails in crowded environments due to unpredictable human motion and conflicting trajectories.

**Domain**: `Planning_Decision`

## Fix

Use reactive planning with human intent inference and collision avoidance that generalizes beyond scripted motion, as modeled in HAPS 2.0 dataset and HA-VLN 2.0 benchmark.

## Anti-pattern

Static or single-person navigation policies that ignore crowd dynamics.

## Cross-domain analogies

- **Perception_Vision** → Incremental object-centric mapping with semantic clustering enables real-time trajectory prediction and collision avoidance in crowds.
  - related fix: Incremental object-centric mapping: associate VLM-derived semantic features (captions, embeddings) with LiDAR points via calibrated camera-LiDAR projection, then cluster points into object hypotheses updated frame-by-frame.
- **Learning_Training** → Parallelize trajectory prediction across agents with synchronized conflict resolution updates.
  - related fix: Distribute PPO training across multiple workers with synchronized gradient updates (DD-PPO).
- **Control_Locomotion** → Train an end-to-end policy via RL with domain-randomized human trajectories.
  - related fix: Train a single end-to-end neural network policy via large-scale RL in simulation with domain randomization, mapping depth image directly to motor commands

## Patch

```diff
--- dynamic_multi_human_interactions.before.py
+++ dynamic_multi_human_interactions.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Robot navigation fails in crowded environments due to unpredictable human motion and conflicting trajectories.

+# Fix    : Use reactive planning with human intent inference and collision avoidance that generalizes beyond scripted motion, as modeled in HAPS 2.0 dataset and HA-VLN 2.0 benchmark.

+# Avoid  : Static or single-person navigation policies that ignore crowd dynamics.

```
