---
pattern_id: pattern_visual_locomotion
applicable_symptoms: [visual_locomotion]
domain: Control_Locomotion
---

# Blind locomotion policies fail to step over or avoid obstacles because they lack visual perception of upcoming terrain.

**Domain**: `Control_Locomotion`

## Fix

Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Anti-pattern

Blind locomotion using only proprioceptive sensing.

## Cross-domain analogies

- **Perception_Vision** → Use shared transformer layers with modality-specific encoders to fuse vision and proprioception for terrain-aware locomotion.
  - related fix: Use a multimodal versatile network (MMV) with shared transformer layers and modality-specific encoders to learn joint embeddings across modalities.
- **Planning_Decision** → Online visual-language mapping inspires online terrain mapping from vision for adaptive locomotion.
  - related fix: Online visual-language mapping that builds and updates a semantic map from visual observations, combined with an LLM-based instruction parser and DD-PPO local controller.
- **Learning_Training** → Progressive distillation of visual priors from a high-resolution teacher to a low-resolution student policy.
  - related fix: Two-stage progressive knowledge distillation: first distill from large teacher to medium student, then from medium to small student, achieving 1/7 model size with same accuracy.

## Patch

```diff
--- visual_locomotion.before.py
+++ visual_locomotion.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Blind locomotion policies fail to step over or avoid obstacles because they lack visual perception of upcoming terrain.

+# Fix    : Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

+# Avoid  : Blind locomotion using only proprioceptive sensing.

```
