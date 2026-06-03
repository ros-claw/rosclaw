---
pattern_id: pattern_high_level_action_prediction
applicable_symptoms: [high_level_action_prediction]
domain: Planning_Decision
---

# High-level action prediction in VLN-CE ignores crucial spatial reasoning in low-level movements, leading to poor obstacle avoidance and imprecise turning.

**Domain**: `Planning_Decision`

## Fix

Joint training of high-level action prediction with low-level action training, where the model learns both coarse goals and fine-grained motor commands simultaneously.

## Anti-pattern

Using only high-level action prediction without low-level action training.

## Cross-domain analogies

- **Perception_Vision** → Jointly process visual and textual data for cross-modal spatial reasoning in low-level movements.
  - related fix: Use a Vision-Language Model (VLM) that jointly processes visual and textual data for cross-modal reasoning, as in NavForesee.
- **Learning_Training** → Pre-train low-level motion encoder on spatial trajectory data, then fine-tune with high-level planner.
  - related fix: Pre-train on large-scale image-text-action triplets via self-supervised learning, then fine-tune on downstream VLN tasks
- **Control_Locomotion** → Use diffusion policies to model multi-modal low-level spatial actions for robust obstacle avoidance.
  - related fix: Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.

## Patch

```diff
--- high_level_action_prediction.before.py
+++ high_level_action_prediction.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: High-level action prediction in VLN-CE ignores crucial spatial reasoning in low-level movements, leading to poor obstacle avoidance and imprecise turning.

+# Fix    : Joint training of high-level action prediction with low-level action training, where the model learns both coarse goals and fine-grained motor commands simultaneously.

+# Avoid  : Using only high-level action prediction without low-level action training.

```
