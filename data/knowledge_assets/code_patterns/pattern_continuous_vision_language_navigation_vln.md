---
pattern_id: pattern_continuous_vision_language_navigation_vln
applicable_symptoms: [continuous_vision_language_navigation_vln]
domain: Planning_Decision
---

# VLN agents fail to execute continuous motion from language instructions due to gap between high-level planning and low-level control

**Domain**: `Planning_Decision`

## Fix

Modular architecture with two-stage process: coarse path generation from language, then low-level controller for smooth trajectory following

## Anti-pattern

Discrete graph-based navigation that ignores continuous motion constraints

## Cross-domain analogies

- **Perception_Vision** → Hierarchical decomposition with proximal and distal sub-networks bridges high-level planning and low-level control.
  - related fix: Hierarchical network with proximal sub-network for near-field features, distal sub-network for far-field context, recurrent temporal modeling, and attention-based fusion to output continuous risk scores.
- **Learning_Training** → Use human demonstration data to train a behavioral cloning policy bridging high-level plans to low-level actions.
  - related fix: Use large-scale human demonstration dataset (Habitat-Web) collected via web-based interface to train a behavioral cloning policy for object-goal navigation.
- **Control_Locomotion** → Use multi-expert distillation with DAgger to bridge planning and control via iterative imitation and fine-tuning.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- continuous_vision_language_navigation_vln.before.py
+++ continuous_vision_language_navigation_vln.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to execute continuous motion from language instructions due to gap between high-level planning and low-level control

+# Fix    : Modular architecture with two-stage process: coarse path generation from language, then low-level controller for smooth trajectory following

+# Avoid  : Discrete graph-based navigation that ignores continuous motion constraints

```
