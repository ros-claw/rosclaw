---
pattern_id: pattern_multi_sourced_value_maps
applicable_symptoms: [multi_sourced_value_maps]
domain: Planning_Decision
---

# Instruction-guided navigation fails to translate high-level linguistic plans into low-level robot actions, leading to poor trajectory execution.

**Domain**: `Planning_Decision`

## Fix

Multi-sourced Value Maps: model key navigation elements (obstacles, goals, instructions) as multiple value layers and combine them into a unified costmap for robot control.

## Anti-pattern

Using a single costmap or direct language-to-action mapping without explicit value layering.

## Cross-domain analogies

- **Perception_Vision** → Multi-modal fusion of linguistic, geometric, and proprioceptive feedback with hierarchical action decomposition.
  - related fix: Multi-modal occupancy grid fusion that integrates RGB, depth, lidar, and proprioception with a sensor layout strategy to maximize coverage and handle self-occlusion.
- **Learning_Training** → Use hierarchical decomposition with specialized experts for high-level planning and low-level action execution.
  - related fix: Train three specialized RL experts (reaching, squeezing, avoiding) and combine them via dynamic weighting based on individual capability performance.
- **Control_Locomotion** → Map language instructions and state to actions via a learned policy for closed-loop execution.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- multi_sourced_value_maps.before.py
+++ multi_sourced_value_maps.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Instruction-guided navigation fails to translate high-level linguistic plans into low-level robot actions, leading to poor trajectory execution.

+# Fix    : Multi-sourced Value Maps: model key navigation elements (obstacles, goals, instructions) as multiple value layers and combine them into a unified costmap for robot control.

+# Avoid  : Using a single costmap or direct language-to-action mapping without explicit value layering.

```
