---
pattern_id: pattern_unitree_go2
applicable_symptoms: [unitree_go2]
domain: Planning_Decision
---

# Quadruped robot navigation stack fails to achieve real-time performance and zero-shot generalization in dynamic environments

**Domain**: `Planning_Decision`

## Fix

Open-architecture navigation stack with ROS2 real-time control, LOVON framework for long-range object navigation, and zero-shot navigation via general-purpose perception and planning modules

## Anti-pattern

Platform-specific fine-tuning or task-specific training for each new environment

## Cross-domain analogies

- **Perception_Vision** → Integrate learned visual-semantic representations into a hierarchical planning framework for real-time adaptive navigation.
  - related fix: VISR: a framework integrating visual perception with semantic reasoning using learned representations
- **Learning_Training** → Use large-scale synthetic training data to augment real-time quadruped navigation policies.
  - related fix: Use large-scale synthetic data generation (e.g., ScaleVLN with 4M+ instructions) to augment training.
- **Control_Locomotion** → Train an end-to-end policy mapping noisy sensor streams directly to navigation actions via domain randomization.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- unitree_go2.before.py
+++ unitree_go2.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Quadruped robot navigation stack fails to achieve real-time performance and zero-shot generalization in dynamic environments

+# Fix    : Open-architecture navigation stack with ROS2 real-time control, LOVON framework for long-range object navigation, and zero-shot navigation via general-purpose perception and planning modules

+# Avoid  : Platform-specific fine-tuning or task-specific training for each new environment

```
