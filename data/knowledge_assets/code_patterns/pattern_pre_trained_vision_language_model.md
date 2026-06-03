---
pattern_id: pattern_pre_trained_vision_language_model
applicable_symptoms: [pre_trained_vision_language_model]
domain: Perception_Vision
---

# Robotic systems fail to generalize to novel objects or commands without task-specific fine-tuning, requiring extensive retraining for each new environment.

**Domain**: `Perception_Vision`

## Fix

Use a pre-trained vision-language model (e.g., CLIP) trained on large-scale image-text pairs to learn a shared embedding space, enabling zero-shot transfer to unseen tasks and objects.

## Anti-pattern

Training task-specific visual models from scratch for each new robotic task, which lacks generalization and requires large labeled datasets.

## Cross-domain analogies

- **Planning_Decision** → Use graph-based semantic priors to enable zero-shot generalization to novel objects.
  - related fix: Scenario-oriented object navigation with graph-based exploration: build a semantic graph of explored regions, use a high-level policy to select frontier nodes based on object-context priors, and a low-level policy to navigate to chosen nodes.
- **Learning_Training** → Unified multi-modal embedding space enables zero-shot generalization to novel objects and commands.
  - related fix: Unified multi-task model co-trained on all VLNVerse benchmark tasks (goal-oriented navigation, language-guided exploration, instruction following) using shared transformer-based cross-modal attention and common visual-linguistic embedding space.
- **Control_Locomotion** → Use a lightweight, simulation-trained policy for direct perception-to-action mapping without task-specific fine-tuning.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- pre_trained_vision_language_model.before.py
+++ pre_trained_vision_language_model.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Robotic systems fail to generalize to novel objects or commands without task-specific fine-tuning, requiring extensive retraining for each new environment.

+# Fix    : Use a pre-trained vision-language model (e.g., CLIP) trained on large-scale image-text pairs to learn a shared embedding space, enabling zero-shot transfer to unseen tasks and objects.

+# Avoid  : Training task-specific visual models from scratch for each new robotic task, which lacks generalization and requires large labeled datasets.

```
