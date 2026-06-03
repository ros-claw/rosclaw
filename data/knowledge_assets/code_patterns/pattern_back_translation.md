---
pattern_id: pattern_back_translation
applicable_symptoms: [back_translation]
domain: Learning_Training
---

# Navigational agents trained on limited human-annotated data fail to generalize to unseen environments due to insufficient training distribution.

**Domain**: `Learning_Training`

## Fix

Use back translation: generate new paths and instructions from unlabeled trajectory data via a learned translator, combined with environmental dropout for visual perturbations.

## Anti-pattern

Relying solely on human-annotated instruction-path pairs for training.

## Cross-domain analogies

- **Perception_Vision** → Fuse diverse data sources into a unified training distribution to improve generalization.
  - related fix: Propose a multimodal occupancy perception system that fuses vision, depth, and other sensor data into a unified occupancy representation for humanoid robots.
- **Planning_Decision** → Hierarchical decomposition of task into semantic and low-level modules enables broader generalization from limited data.
  - related fix: LOVON framework: hierarchical planning with long-range open-vocabulary object navigation, using a high-level semantic planner and low-level locomotion controller.
- **Control_Locomotion** → Use camera images to augment training distribution, enabling real-time adaptation to unseen environments.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- back_translation.before.py
+++ back_translation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigational agents trained on limited human-annotated data fail to generalize to unseen environments due to insufficient training distribution.

+# Fix    : Use back translation: generate new paths and instructions from unlabeled trajectory data via a learned translator, combined with environmental dropout for visual perturbations.

+# Avoid  : Relying solely on human-annotated instruction-path pairs for training.

```
