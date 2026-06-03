---
pattern_id: pattern_spatial_scene_graph
applicable_symptoms: [spatial_scene_graph]
domain: Memory_Reasoning
---

# VLN agents fail to generalize to novel environments because they lack a persistent global memory of spatial layout and semantics, leading to inefficient path planning and inability to reason about remote object locations without task-specific training.

**Domain**: `Memory_Reasoning`

## Fix

Use a Spatial Scene Graph (SSG) built incrementally from semantic segmentation and object detection to encode objects, regions, and their spatial relations as nodes and edges, enabling zero-shot global reasoning and planning.

## Anti-pattern

Relying on task-specific training data or local observation memory without a structured global representation.

## Cross-domain analogies

- **Perception_Vision** → Fuse multi-scale spatial memory with attention to prioritize salient layout cues.
  - related fix: Implement a sensing intelligence pipeline that fuses multiple sensor modalities (e.g., vision, depth, IMU) into a coherent representation, with attention mechanisms to prioritize salient input.
- **Planning_Decision** → Parse natural-language commands into persistent spatial-semantic waypoints for global memory.
  - related fix: UrbanNav framework: language-guided navigation that parses natural-language commands into actionable waypoints for real-time urban navigation.
- **Learning_Training** → Use concatenated trajectories to create longer, circuitous exploration tasks that force persistent memory use.
  - related fix: Use R4R dataset (concatenated R2R paths) to create longer, circuitous trajectories that better differentiate instruction-following agents from goal-seeking ones.

## Patch

```diff
--- spatial_scene_graph.before.py
+++ spatial_scene_graph.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to generalize to novel environments because they lack a persistent global memory of spatial layout and semantics, leading to inefficient path planning and inability to reason about remote object locations without task-specific training.

+# Fix    : Use a Spatial Scene Graph (SSG) built incrementally from semantic segmentation and object detection to encode objects, regions, and their spatial relations as nodes and edges, enabling zero-shot global reasoning and planning.

+# Avoid  : Relying on task-specific training data or local observation memory without a structured global representation.

```
