---
pattern_id: pattern_r2r_last_dataset
applicable_symptoms: [r2r_last_dataset]
domain: Planning_Decision
---

# VLN agents fail to ground final navigation actions (e.g., stop) to language cues, relying on pattern matching instead of semantic understanding.

**Domain**: `Planning_Decision`

## Fix

Use R2R-Last benchmark to isolate last-action prediction from full path, training with Actional Atomic-Concept Learning (AACL) to align language tokens with discrete navigational commands.

## Anti-pattern

Standard full-path instruction following masks poor action grounding by allowing pattern matching of action sequences.

## Cross-domain analogies

- **Perception_Vision** → Use panoramic semantic grounding to align language cues with holistic spatial action boundaries.
  - related fix: Use panoramic scene parsing on equirectangular or cubemap representations to extract floorplans, wall boundaries, and free-space regions from a single 360° RGB image.
- **Learning_Training** → Pretrain on stop-relevant language cues, then fine-tune on embodied stop-action data.
  - related fix: Two-stage curriculum: pretrain on large-scale web-scraped image-text pairs (Conceptual Captions) then fine-tune on embodied path-instruction data.
- **Control_Locomotion** → Pre-train a library of grounded navigation primitives via RL, decoupling action grounding from high-level planning.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- r2r_last_dataset.before.py
+++ r2r_last_dataset.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to ground final navigation actions (e.g., stop) to language cues, relying on pattern matching instead of semantic understanding.

+# Fix    : Use R2R-Last benchmark to isolate last-action prediction from full path, training with Actional Atomic-Concept Learning (AACL) to align language tokens with discrete navigational commands.

+# Avoid  : Standard full-path instruction following masks poor action grounding by allowing pattern matching of action sequences.

```
