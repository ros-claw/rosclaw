---
pattern_id: pattern_zero_shot_generalization_in_navigation
applicable_symptoms: [zero_shot_generalization_in_navigation]
domain: Planning_Decision
---

# Navigation policies fail when deployed in unseen environments with novel layouts or object types due to overfitting to training scenes.

**Domain**: `Planning_Decision`

## Fix

Use structured reasoning (e.g., Chain-of-Thought with Critical Layers) that decouples navigation strategy from environment-specific features.

## Anti-pattern

Policies relying on memorization of specific environments.

## Cross-domain analogies

- **Perception_Vision** → Use incremental object-centric mapping to build adaptive navigation policies from online scene features.
  - related fix: Incremental object-centric mapping: associate VLM-derived semantic features (captions, embeddings) with LiDAR points via calibrated camera-LiDAR projection, then cluster points into object hypotheses updated frame-by-frame.
- **Learning_Training** → Apply random feature masking to policy inputs to force reliance on cross-scene geometric cues.
  - related fix: Apply dropout to panoramic image features (36 views per node) with rate 0.3–0.5 during training, randomly masking a subset of view angle features to force reliance on language instructions.
- **Control_Locomotion** → Use lightweight policy distillation from simulation to reduce overfitting to specific scenes.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- zero_shot_generalization_in_navigation.before.py
+++ zero_shot_generalization_in_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation policies fail when deployed in unseen environments with novel layouts or object types due to overfitting to training scenes.

+# Fix    : Use structured reasoning (e.g., Chain-of-Thought with Critical Layers) that decouples navigation strategy from environment-specific features.

+# Avoid  : Policies relying on memorization of specific environments.

```
