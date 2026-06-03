---
pattern_id: pattern_panoramic_augmentation_aug
applicable_symptoms: [panoramic_augmentation_aug]
domain: Perception_Vision
---

# BEV-panoramic feature inconsistency degrades generalization across visual domains in humanoid navigation and manipulation tasks.

**Domain**: `Perception_Vision`

## Fix

Panoramic Augmentation (AUG): a lightweight, plug-and-play data augmentation block combining cross-view transformations and semantic alignment to enforce BEV-panoramic feature consistency during training, with no trainable parameters.

## Anti-pattern

Standard data augmentation without cross-view semantic alignment.

## Cross-domain analogies

- **Planning_Decision** → Learn atomic-level feature decomposition to align BEV and panoramic representations.
  - related fix: Actional Atomic-Concept Learning (AACL): learn atomic-level action representations from language instructions to improve navigation and grounding.
- **Learning_Training** → Distill privileged BEV features into panoramic student via guidance loss to enforce consistency.
  - related fix: Privileged Information Guidance (PIG): train a diffusion policy with privileged depth and collision information during training, then distill into a student policy that uses only RGB observations via a guidance loss.
- **Control_Locomotion** → Train a separate BEV consistency critic to override primary features when cross-domain mismatch exceeds a threshold.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- panoramic_augmentation_aug.before.py
+++ panoramic_augmentation_aug.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: BEV-panoramic feature inconsistency degrades generalization across visual domains in humanoid navigation and manipulation tasks.

+# Fix    : Panoramic Augmentation (AUG): a lightweight, plug-and-play data augmentation block combining cross-view transformations and semantic alignment to enforce BEV-panoramic feature consistency during training, with no trainable parameters.

+# Avoid  : Standard data augmentation without cross-view semantic alignment.

```
