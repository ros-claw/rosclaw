---
pattern_id: pattern_isaaclab
applicable_symptoms: [isaaclab]
domain: Learning_Training
---

# Sim-to-real gap causes policy collapse on unseen terrain for legged robot navigation

**Domain**: `Learning_Training`

## Fix

Use IsaacLab simulation benchmark with realistic scenes and low-level control primitives to evaluate and transfer navigation policies to real-world robots

## Anti-pattern

Evaluating navigation policies only in simulation without realistic scene diversity or real-world support

## Cross-domain analogies

- **Perception_Vision** → Use first-person occupancy datasets to pre-train sim-to-real terrain policies.
  - related fix: EmbodiedOcc-ScanNet: a large-scale egocentric occupancy dataset derived from ScanNet with voxel-level occupancy labels from first-person perspective.
- **Planning_Decision** → Interactive prompting enables closed-loop verification to adapt policies to real-world terrain.
  - related fix: March-in-Chat (MiC): interactive prompting that allows the agent to ask clarifying questions and receive human responses during navigation.
- **Control_Locomotion** → Use standardized terrain benchmarks with precise proprioceptive and exteroceptive metrics to train sim-to-real policies.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- isaaclab.before.py
+++ isaaclab.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Sim-to-real gap causes policy collapse on unseen terrain for legged robot navigation

+# Fix    : Use IsaacLab simulation benchmark with realistic scenes and low-level control primitives to evaluate and transfer navigation policies to real-world robots

+# Avoid  : Evaluating navigation policies only in simulation without realistic scene diversity or real-world support

```
