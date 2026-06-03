---
pattern_id: pattern_end_to_end_learning
applicable_symptoms: [end_to_end_learning]
domain: Learning_Training
---

# Modular pipelines with separate perception, planning, and control components require handcrafted submodules and fail to jointly optimize exploration and navigation, leading to suboptimal behavior in unknown environments.

**Domain**: `Learning_Training`

## Fix

Train a single neural network end-to-end from raw sensor inputs to control outputs using a reward signal (e.g., reinforcement learning), allowing the network to discover internal representations that directly optimize the desired behavior.

## Anti-pattern

Handcrafting separate routines for mapping, path planning, and exploration.

## Cross-domain analogies

- **Perception_Vision** → Use incremental object-centric joint optimization to unify exploration and navigation.
  - related fix: Incremental object-centric mapping: associate VLM-derived semantic features (captions, embeddings) with LiDAR points via calibrated camera-LiDAR projection, then cluster points into object hypotheses updated frame-by-frame.
- **Planning_Decision** → Use a video world model to jointly optimize perception, planning, and control end-to-end.
  - related fix: Use a video world model to simulate future trajectories and select actions that reach the semantic goal, enabling zero-shot mapless navigation.
- **Control_Locomotion** → Pre-train a shared skill library via RL to decouple exploration from navigation planning.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- end_to_end_learning.before.py
+++ end_to_end_learning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Modular pipelines with separate perception, planning, and control components require handcrafted submodules and fail to jointly optimize exploration and navigation, leading to suboptimal behavior in unknown environments.

+# Fix    : Train a single neural network end-to-end from raw sensor inputs to control outputs using a reward signal (e.g., reinforcement learning), allowing the network to discover internal representations that directly optimize the desired behavior.

+# Avoid  : Handcrafting separate routines for mapping, path planning, and exploration.

```
