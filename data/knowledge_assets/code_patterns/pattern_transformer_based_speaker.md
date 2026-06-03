---
pattern_id: pattern_transformer_based_speaker
applicable_symptoms: [transformer_based_speaker]
domain: Learning_Training
---

# Navigation instructions generated from path representations lack diversity and referential validity, causing poor path reconstruction by a listener model.

**Domain**: `Learning_Training`

## Fix

Train a Transformer-based Speaker jointly with a Listener in a Double Back-Translation loop, where the Speaker generates instructions from paths and the Listener reconstructs paths from instructions, enforcing instruction-path consistency through iterative refinement.

## Anti-pattern

Single-direction instruction generation without back-translation consistency loss.

## Cross-domain analogies

- **Perception_Vision** → Combine two complementary path representations into a single training input to improve referential diversity and validity.
  - related fix: Dual-view visual prompt: combine two complementary spatial views into a single prompt at inference time, applied on top of a VLA model.
- **Planning_Decision** → Use closed-loop verification to evaluate instruction diversity and referential validity during generation.
  - related fix: NavSpace benchmark with spatial intelligence instructions and evaluation metrics that test object-relationship and layout reasoning.
- **Control_Locomotion** → Multi-expert distillation with closed-loop fine-tuning could diversify path instructions via iterative listener feedback.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- transformer_based_speaker.before.py
+++ transformer_based_speaker.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation instructions generated from path representations lack diversity and referential validity, causing poor path reconstruction by a listener model.

+# Fix    : Train a Transformer-based Speaker jointly with a Listener in a Double Back-Translation loop, where the Speaker generates instructions from paths and the Listener reconstructs paths from instructions, enforcing instruction-path consistency through iterative refinement.

+# Avoid  : Single-direction instruction generation without back-translation consistency loss.

```
