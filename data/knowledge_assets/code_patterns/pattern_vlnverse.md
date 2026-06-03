---
pattern_id: pattern_vlnverse
applicable_symptoms: [vlnverse]
domain: Learning_Training
---

# Sim-to-real gap causes policy collapse when transferring navigation policies from simulation to real world due to lack of realistic motion dynamics and physics in existing VLN benchmarks.

**Domain**: `Learning_Training`

## Fix

Use full-kinematics agents with a robust physics engine to enable realistic motion dynamics and high-fidelity simulation, reducing sim-to-real gap.

## Anti-pattern

Existing VLN benchmarks use simplified kinematics or no physics, leading to poor sim-to-real transfer.

## Cross-domain analogies

- **Perception_Vision** → Derive a large-scale dataset of realistic motion dynamics from real-world navigation traces for sim-to-real policy training.
  - related fix: EmbodiedOcc-ScanNet: a large-scale egocentric occupancy dataset derived from ScanNet with voxel-level occupancy labels from first-person perspective.
- **Planning_Decision** → Closed-loop semantic verification bridges simulation dynamics to real-world physics.
  - related fix: Online planning with LLM (GPT-4-Turbo) and open-vocabulary semantic scene graphs, integrating LiDAR-SLAM, RGBD semantic mapping (FC-CLIP), and ROS2 navigation stack with exploration.
- **Control_Locomotion** → Train an end-to-end policy on real-world sensor data to bypass simulated dynamics mismatch.
  - related fix: Train a single neural network policy via deep reinforcement learning that maps raw depth camera images directly to motor commands, bypassing hand-crafted perception and control layers.

## Patch

```diff
--- vlnverse.before.py
+++ vlnverse.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Sim-to-real gap causes policy collapse when transferring navigation policies from simulation to real world due to lack of realistic motion dynamics and physics in existing VLN benchmarks.

+# Fix    : Use full-kinematics agents with a robust physics engine to enable realistic motion dynamics and high-fidelity simulation, reducing sim-to-real gap.

+# Avoid  : Existing VLN benchmarks use simplified kinematics or no physics, leading to poor sim-to-real transfer.

```
