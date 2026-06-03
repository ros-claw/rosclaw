---
pattern_id: pattern_3d_incremental_object_centric_mapping
applicable_symptoms: [3d_incremental_object_centric_mapping]
domain: Perception_Vision
---

# Dense metric SLAM maps lack object-level semantics and cannot handle dynamic environments where objects move or appear incrementally.

**Domain**: `Perception_Vision`

## Fix

Incremental object-centric mapping: associate VLM-derived semantic features (captions, embeddings) with LiDAR points via calibrated camera-LiDAR projection, then cluster points into object hypotheses updated frame-by-frame.

## Anti-pattern

Traditional dense SLAM methods that build a continuous metric map without object-level reasoning.

## Cross-domain analogies

- **Planning_Decision** → Map semantics as layered value grids for dynamic object integration.
  - related fix: Multi-sourced Value Maps: model key navigation elements (obstacles, goals, instructions) as multiple value layers and combine them into a unified costmap for robot control.
- **Learning_Training** → Use group-relative trajectory sampling to refine object-level semantic SLAM against dynamic scene variations.
  - related fix: Use GRPO (Group Relative Policy Optimization) as a second-stage RL fine-tuning after supervised chain-of-thought alignment, optimizing policy relative to a group of sampled trajectories via group-relative advantage estimation.
- **Control_Locomotion** → Pre-train a library of reusable object-level semantic primitives via self-supervised learning, decoupling perception from mapping.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- 3d_incremental_object_centric_mapping.before.py
+++ 3d_incremental_object_centric_mapping.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Dense metric SLAM maps lack object-level semantics and cannot handle dynamic environments where objects move or appear incrementally.

+# Fix    : Incremental object-centric mapping: associate VLM-derived semantic features (captions, embeddings) with LiDAR points via calibrated camera-LiDAR projection, then cluster points into object hypotheses updated frame-by-frame.

+# Avoid  : Traditional dense SLAM methods that build a continuous metric map without object-level reasoning.

```
