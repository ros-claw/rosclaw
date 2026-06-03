---
pattern_id: pattern_navfom
applicable_symptoms: [navfom]
domain: Planning_Decision
---

# Single navigation model fails to generalize across different robot embodiments and tasks without per-task fine-tuning

**Domain**: `Planning_Decision`

## Fix

Unified architecture with identifier tokens encoding embodiment and temporal context, plus dynamic token sampling under budget constraints

## Anti-pattern

Task-specific fine-tuning for each robot platform and navigation task

## Cross-domain analogies

- **Perception_Vision** → Use language-based task embeddings instead of embodiment-specific parameters for policy input.
  - related fix: Replace visual features with language-based representations (e.g., captions from a vision-language model) for navigation policy input.
- **Learning_Training** → Train embodiment-aware navigation policies using calibrated sensor models and occlusion-aware simulation across diverse platforms.
  - related fix: Synthetic depth image generation with self-occlusion-aware ray casting and noise-aware modeling calibrated from real depth sensor characteristics.
- **Control_Locomotion** → Use a lightweight, embodiment-agnostic policy trained via RL across diverse morphologies.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- navfom.before.py
+++ navfom.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Single navigation model fails to generalize across different robot embodiments and tasks without per-task fine-tuning

+# Fix    : Unified architecture with identifier tokens encoding embodiment and temporal context, plus dynamic token sampling under budget constraints

+# Avoid  : Task-specific fine-tuning for each robot platform and navigation task

```
