---
pattern_id: pattern_proximal_alternating_minimization_network_pan
applicable_symptoms: [proximal_alternating_minimization_network_pan]
domain: Planning_Decision
---

# Real-time motion planning with many point-level constraints is computationally infeasible for online control.

**Domain**: `Planning_Decision`

## Fix

Plug-and-play proximal alternating-minimization network (PAN) that embeds physical constraints into network computations, enabling efficient alternating minimization for collision-free trajectory generation.

## Anti-pattern

Standard optimization solvers that do not exploit learned proximal updates or constraint embedding.

## Cross-domain analogies

- **Perception_Vision** → Reformulate global constraints into a local, ego-centric sliding window for tractable online optimization.
  - related fix: Re-annotate ScanNet scenes with local occupancy grids aligned to the camera frame, supporting both static and temporal prediction tasks.
- **Learning_Training** → Use back translation to generate simplified constraint subsets from offline data for real-time planning.
  - related fix: Use back translation: generate new paths and instructions from unlabeled trajectory data via a learned translator, combined with environmental dropout for visual perturbations.
- **Control_Locomotion** → Train a model-free policy with domain randomization to fuse sparse constraints into learned low-level actions.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- proximal_alternating_minimization_network_pan.before.py
+++ proximal_alternating_minimization_network_pan.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Real-time motion planning with many point-level constraints is computationally infeasible for online control.

+# Fix    : Plug-and-play proximal alternating-minimization network (PAN) that embeds physical constraints into network computations, enabling efficient alternating minimization for collision-free trajectory generation.

+# Avoid  : Standard optimization solvers that do not exploit learned proximal updates or constraint embedding.

```
