---
pattern_id: pattern_mm_nav
applicable_symptoms: [mm_nav]
domain: Planning_Decision
---

# Visual navigation agents fail to generalize across diverse environments and tasks, especially when relying solely on visual observations without depth sensors.

**Domain**: `Planning_Decision`

## Fix

Train a multi-view VLA student model via teacher-student paradigm using RL experts with privileged depth information, and dynamically balance online data collection per capability (reaching, squeezing, avoiding) based on student performance.

## Anti-pattern

Training navigation policies directly from visual observations without privileged depth or dynamic data balancing leads to poor generalization.

## Cross-domain analogies

- **Perception_Vision** → Use 3D-GS to generate photorealistic training views from sparse real captures, enabling depth-free generalization.
  - related fix: Construct high-fidelity datasets using 3D Gaussian Splatting (3D-GS) to generate photorealistic novel-view synthetic images from sparse real captures, preserving fine-grained textures and lighting details.
- **Learning_Training** → Use back-translation to generate synthetic depth from visual data, augmenting training diversity.
  - related fix: Use back translation: generate new paths and instructions from unlabeled trajectory data via a learned translator, combined with environmental dropout for visual perturbations.
- **Control_Locomotion** → Train an end-to-end policy mapping visual inputs directly to navigation actions, bypassing hand-crafted perception modules.
  - related fix: Train a single neural network policy via deep reinforcement learning that maps raw depth camera images directly to motor commands, bypassing hand-crafted perception and control layers.

## Patch

```diff
--- mm_nav.before.py
+++ mm_nav.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Visual navigation agents fail to generalize across diverse environments and tasks, especially when relying solely on visual observations without depth sensors.

+# Fix    : Train a multi-view VLA student model via teacher-student paradigm using RL experts with privileged depth information, and dynamically balance online data collection per capability (reaching, squeezing, avoiding) based on student performance.

+# Avoid  : Training navigation policies directly from visual observations without privileged depth or dynamic data balancing leads to poor generalization.

```
