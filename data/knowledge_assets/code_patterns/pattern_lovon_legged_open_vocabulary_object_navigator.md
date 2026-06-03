---
pattern_id: pattern_lovon_legged_open_vocabulary_object_navigator
applicable_symptoms: [lovon_legged_open_vocabulary_object_navigator]
domain: Planning_Decision
---

# Legged robots fail to navigate to open-vocabulary objects in unseen environments due to lack of semantic grounding and adaptive locomotion.

**Domain**: `Planning_Decision`

## Fix

Hierarchical framework: VLM-based high-level planner selects sub-goals via visual grounding, MPC-based low-level controller executes adaptive locomotion.

## Anti-pattern

End-to-end RL policies that overfit to training objects and terrains.

## Cross-domain analogies

- **Perception_Vision** → Train a multimodal model on simulated data to ground semantics and adapt locomotion.
  - related fix: Use deep learning models (e.g., CNNs) trained on simulated galaxy merger images to automatically classify merger stages.
- **Learning_Training** → Use self-occlusion-aware depth simulation to generate synthetic semantic-locomotion training data.
  - related fix: Synthetic depth image generation with self-occlusion-aware ray casting and noise-aware modeling calibrated from real depth sensor characteristics.
- **Control_Locomotion** → Use blocked-action heuristics to iteratively adjust locomotion and re-query semantic grounding.
  - related fix: Trial-and-error heuristic: when an action is blocked, systematically try alternative actions until a traversable path is found or state is exhausted.

## Patch

```diff
--- lovon_legged_open_vocabulary_object_navigator.before.py
+++ lovon_legged_open_vocabulary_object_navigator.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Legged robots fail to navigate to open-vocabulary objects in unseen environments due to lack of semantic grounding and adaptive locomotion.

+# Fix    : Hierarchical framework: VLM-based high-level planner selects sub-goals via visual grounding, MPC-based low-level controller executes adaptive locomotion.

+# Avoid  : End-to-end RL policies that overfit to training objects and terrains.

```
