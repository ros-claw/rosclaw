---
pattern_id: pattern_snav
applicable_symptoms: [snav]
domain: Planning_Decision
---

# Navigation agents fail to generalize across novel scenes and lack explicit spatial reasoning for object layouts and traversable pathways.

**Domain**: `Planning_Decision`

## Fix

Integrate explicit spatial reasoning modules that process geometric and semantic cues from depth/RGB to output motion commands respecting obstacles and path efficiency, fine-tuned on habitat-sim rendered navigation data with video-QA mixing and height/lighting perturbations.

## Anti-pattern

Prior navigation agents (e.g., Habitat-based navigators, RL-based planners) that rely on extensive pre-training per environment or lack structured spatial reasoning.

## Cross-domain analogies

- **Perception_Vision** → Multi-modal fusion of spatial priors with proprioceptive layout reasoning to generalize scene traversability.
  - related fix: Multi-modal occupancy grid fusion that integrates RGB, depth, lidar, and proprioception with a sensor layout strategy to maximize coverage and handle self-occlusion.
- **Learning_Training** → Use a pretrained vision-language model to generate diverse, spatially grounded training trajectories for explicit layout reasoning.
  - related fix: Use Marky, a multilingual instruction generator that produces visually grounded instruction-trajectory pairs at scale (4.2M pairs) by leveraging pretrained vision-language models and spatial alignment.
- **Control_Locomotion** → Train an end-to-end policy that fuses visual input with state to learn spatial reasoning for navigation.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- snav.before.py
+++ snav.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation agents fail to generalize across novel scenes and lack explicit spatial reasoning for object layouts and traversable pathways.

+# Fix    : Integrate explicit spatial reasoning modules that process geometric and semantic cues from depth/RGB to output motion commands respecting obstacles and path efficiency, fine-tuned on habitat-sim rendered navigation data with video-QA mixing and height/lighting perturbations.

+# Avoid  : Prior navigation agents (e.g., Habitat-based navigators, RL-based planners) that rely on extensive pre-training per environment or lack structured spatial reasoning.

```
