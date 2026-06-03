---
pattern_id: pattern_semantic_reasoning
applicable_symptoms: [semantic_reasoning]
domain: Planning_Decision
---

# Agents fail to interpret semantic cues (objects, scenes, spatial relationships) in video streams, leading to poor target-approaching behavior in path planning tasks.

**Domain**: `Planning_Decision`

## Fix

Use Target-Bench benchmark to evaluate and improve semantic reasoning by measuring target-approaching metrics in path planning tasks.

## Anti-pattern

Relying solely on raw coordinate-based navigation without semantic understanding.

## Cross-domain analogies

- **Perception_Vision** → Use spherical geometry-aware constraints to regularize semantic feature alignment in video streams for distortion-free path planning.
  - related fix: Apply spherical geometry-aware constraints to regularize sampling offsets, leveraging panoramic ray properties for distortion-aware alignment without explicit undistortion.
- **Learning_Training** → Use self-supervised pseudo-labeling of semantic cues from video to train robust path planning without manual annotation.
  - related fix: Use unsupervised adversarial training with self-supervised learning (e.g., rotation prediction) to generate pseudo-labels for robust training against adversarial perturbations.
- **Control_Locomotion** → Train a vision-language policy mapping video and state to actions for real-time semantic path planning.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- semantic_reasoning.before.py
+++ semantic_reasoning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Agents fail to interpret semantic cues (objects, scenes, spatial relationships) in video streams, leading to poor target-approaching behavior in path planning tasks.

+# Fix    : Use Target-Bench benchmark to evaluate and improve semantic reasoning by measuring target-approaching metrics in path planning tasks.

+# Avoid  : Relying solely on raw coordinate-based navigation without semantic understanding.

```
