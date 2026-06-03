---
pattern_id: pattern_language_guided_navigation
applicable_symptoms: [language_guided_navigation]
domain: Planning_Decision
---

# Language-guided navigation agents fail to follow noisy, ambiguous instructions in dynamic urban environments due to poor grounding of spatial references and landmark generalization.

**Domain**: `Planning_Decision`

## Fix

Use a vision-language model (VLM) with spatial attention and landmark grounding, trained on diverse urban scenes with synthetic instruction augmentation.

## Anti-pattern

Relying on fixed template-based instruction parsing or static landmark databases that cannot handle novel environments or noisy speech.

## Cross-domain analogies

- **Perception_Vision** → Jointly predict spatial semantics and landmarks from a shared representation to ground ambiguous instructions.
  - related fix: Multi-task learning jointly predicting 3D occupancy, room layout, and object bounding boxes from a shared volumetric representation
- **Learning_Training** → Use closed-loop policy rollouts with corrective human feedback to improve instruction grounding.
  - related fix: Iteratively collect new data under the current policy's distribution, query the expert for corrective actions, and aggregate this data into the training set (DAgger).
- **Control_Locomotion** → Distill multi-expert spatial grounding policies via DAgger and RL fine-tuning with noisy instruction inputs.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- language_guided_navigation.before.py
+++ language_guided_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Language-guided navigation agents fail to follow noisy, ambiguous instructions in dynamic urban environments due to poor grounding of spatial references and landmark generalization.

+# Fix    : Use a vision-language model (VLM) with spatial attention and landmark grounding, trained on diverse urban scenes with synthetic instruction augmentation.

+# Avoid  : Relying on fixed template-based instruction parsing or static landmark databases that cannot handle novel environments or noisy speech.

```
