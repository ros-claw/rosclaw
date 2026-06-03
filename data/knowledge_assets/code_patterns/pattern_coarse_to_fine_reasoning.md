---
pattern_id: pattern_coarse_to_fine_reasoning
applicable_symptoms: [coarse_to_fine_reasoning]
domain: Planning_Decision
---

# Multi-floor navigation decisions fail when relying solely on geometric frontier ranking, ignoring semantic cues like doorways or stairs.

**Domain**: `Planning_Decision`

## Fix

Coarse-to-Fine Reasoning: first rank frontiers geometrically, then refine with LLM-driven semantic analysis of frontier properties (e.g., room type, connectivity) to select navigation goals.

## Anti-pattern

Using only geometric frontier ranking without semantic context.

## Cross-domain analogies

- **Perception_Vision** → Use 3D-GS to generate semantically-rich synthetic views of doorways and stairs from sparse real floorplans.
  - related fix: Construct high-fidelity datasets using 3D Gaussian Splatting (3D-GS) to generate photorealistic novel-view synthetic images from sparse real captures, preserving fine-grained textures and lighting details.
- **Learning_Training** → Use full-kinematics simulation to incorporate semantic cues like doorways into motion dynamics for multi-floor planning.
  - related fix: Use full-kinematics agents with a robust physics engine to enable realistic motion dynamics and high-fidelity simulation, reducing sim-to-real gap.
- **Control_Locomotion** → Fuse semantic visual features with geometric frontier ranking using RL-trained policy for adaptive multi-floor navigation.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- coarse_to_fine_reasoning.before.py
+++ coarse_to_fine_reasoning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Multi-floor navigation decisions fail when relying solely on geometric frontier ranking, ignoring semantic cues like doorways or stairs.

+# Fix    : Coarse-to-Fine Reasoning: first rank frontiers geometrically, then refine with LLM-driven semantic analysis of frontier properties (e.g., room type, connectivity) to select navigation goals.

+# Avoid  : Using only geometric frontier ranking without semantic context.

```
