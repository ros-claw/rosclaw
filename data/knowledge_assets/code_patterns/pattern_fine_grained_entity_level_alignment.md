---
pattern_id: pattern_fine_grained_entity_level_alignment
applicable_symptoms: [fine_grained_entity_level_alignment]
domain: Planning_Decision
---

# VLN agent ignores landmark cues in long instructions, leading to navigation errors in complex environments with similar landmarks.

**Domain**: `Planning_Decision`

## Fix

Fine-grained entity-level alignment: map each entity phrase (e.g., 'the red chair') to a specific visual landmark independently, rather than aligning the whole instruction globally.

## Anti-pattern

Global instruction alignment that treats the entire instruction as a single embedding, losing per-entity grounding.

## Cross-domain analogies

- **Perception_Vision** → Use spherical attention constraints to regularize landmark cue weighting, preventing drift in long-horizon navigation.
  - related fix: Apply spherical geometry-aware constraints to regularize sampling offsets, leveraging panoramic ray properties for distortion-aware alignment without explicit undistortion.
- **Learning_Training** → Use human demonstration data to train landmark-attention policy for instruction following.
  - related fix: Use large-scale human demonstration dataset (Habitat-Web) collected via web-based interface to train a behavioral cloning policy for object-goal navigation.
- **Control_Locomotion** → Use RL to map visual observations directly to navigation actions, bypassing landmark reliance.
  - related fix: Use reinforcement learning to learn a control policy that directly maps sensor observations to actuator commands for plasma shape and position control.

## Patch

```diff
--- fine_grained_entity_level_alignment.before.py
+++ fine_grained_entity_level_alignment.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent ignores landmark cues in long instructions, leading to navigation errors in complex environments with similar landmarks.

+# Fix    : Fine-grained entity-level alignment: map each entity phrase (e.g., 'the red chair') to a specific visual landmark independently, rather than aligning the whole instruction globally.

+# Avoid  : Global instruction alignment that treats the entire instruction as a single embedding, losing per-entity grounding.

```
