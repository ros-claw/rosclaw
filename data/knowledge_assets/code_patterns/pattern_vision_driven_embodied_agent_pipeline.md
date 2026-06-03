---
pattern_id: pattern_vision_driven_embodied_agent_pipeline
applicable_symptoms: [vision_driven_embodied_agent_pipeline]
domain: Planning_Decision
---

# MLLMs fail to produce executable action sequences from visual observations in embodied tasks

**Domain**: `Planning_Decision`

## Fix

Four-stage pipeline: Visual State Description → Reflection and Reasoning → Language Plan Generation → Executable Plan Generation

## Anti-pattern

Direct end-to-end action prediction without intermediate structured reasoning

## Cross-domain analogies

- **Perception_Vision** → Use cross-view semantic alignment to enforce consistency between visual observations and action sequences during training.
  - related fix: Panoramic Augmentation (AUG): a lightweight, plug-and-play data augmentation block combining cross-view transformations and semantic alignment to enforce BEV-panoramic feature consistency during training, with no trainable parameters.
- **Learning_Training** → Use self-supervised pseudo-label generation to produce executable action sequences from unlabeled visual data.
  - related fix: Use unsupervised adversarial training with self-supervised learning (e.g., rotation prediction) to generate pseudo-labels for robust training against adversarial perturbations.
- **Control_Locomotion** → Use diffusion policies to discretize high-level action spaces and model multi-modal plan distributions for executable sequences.
  - related fix: Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.

## Patch

```diff
--- vision_driven_embodied_agent_pipeline.before.py
+++ vision_driven_embodied_agent_pipeline.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: MLLMs fail to produce executable action sequences from visual observations in embodied tasks

+# Fix    : Four-stage pipeline: Visual State Description → Reflection and Reasoning → Language Plan Generation → Executable Plan Generation

+# Avoid  : Direct end-to-end action prediction without intermediate structured reasoning

```
