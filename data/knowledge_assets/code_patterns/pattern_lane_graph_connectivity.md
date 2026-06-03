---
pattern_id: pattern_lane_graph_connectivity
applicable_symptoms: [lane_graph_connectivity]
domain: Planning_Decision
---

# Navigation systems lack a structured representation linking low-level perception to high-level drivable space, causing inefficient hierarchical reasoning.

**Domain**: `Planning_Decision`

## Fix

Use lane graph connectivity to partition the environment into hierarchically organized regions, bridging object-level maps with topological structure.

## Anti-pattern

Flat object-level maps without topological connectivity

## Cross-domain analogies

- **Perception_Vision** → Hierarchical open-vocabulary graph linking perception to drivable space enables efficient navigation reasoning.
  - related fix: Open-vocabulary hierarchical 3D graph representation combining VLM-based instance/caption extraction from images, incremental LiDAR projection, and lane graph connectivity segmentation.
- **Learning_Training** → Constrain high-level planning to preserve low-level perceptual mapping via output regularization.
  - related fix: Functional regularisation: add a penalty on changes to the network's input-output mapping (e.g., using KL divergence or L2 distance on outputs) when training on new tasks.
- **Control_Locomotion** → Multi-expert distillation with depth-based exteroception maps perception directly to drivable space for hierarchical reasoning.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- lane_graph_connectivity.before.py
+++ lane_graph_connectivity.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation systems lack a structured representation linking low-level perception to high-level drivable space, causing inefficient hierarchical reasoning.

+# Fix    : Use lane graph connectivity to partition the environment into hierarchically organized regions, bridging object-level maps with topological structure.

+# Avoid  : Flat object-level maps without topological connectivity

```
