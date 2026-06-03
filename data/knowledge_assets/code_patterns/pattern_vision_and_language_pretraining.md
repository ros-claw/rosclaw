---
pattern_id: pattern_vision_and_language_pretraining
applicable_symptoms: [vision_and_language_pretraining]
domain: Learning_Training
---

# VLN agents fail to align visual and textual representations, leading to poor navigation on R2R and RxR benchmarks

**Domain**: `Learning_Training`

## Fix

Pretrain a vision-language model on large-scale web data to learn cross-modal alignment, then fine-tune on navigation tasks

## Anti-pattern

Training from scratch on navigation data alone without pretraining

## Cross-domain analogies

- **Perception_Vision** → Use actional atomic concepts as grounded units to align visual and textual representations.
  - related fix: Use actional atomic concepts (natural language phrases combining atomic action and object) as a compact grounded unit to bridge visual and linguistic features.
- **Planning_Decision** → Use a sequential decision-making framework balancing alignment gain, localization cost, and safety.
  - related fix: Use Decision-Driven Semantic Object Exploration (DD-SOE) algorithm, which provides a sequential decision-making framework that balances semantic information gain, localization cost, and safety to guide exploration behavior.
- **Control_Locomotion** → Use diffusion policies to model multi-modal action distributions for aligning visual-textual representations.
  - related fix: Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.

## Patch

```diff
--- vision_and_language_pretraining.before.py
+++ vision_and_language_pretraining.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to align visual and textual representations, leading to poor navigation on R2R and RxR benchmarks

+# Fix    : Pretrain a vision-language model on large-scale web data to learn cross-modal alignment, then fine-tune on navigation tasks

+# Avoid  : Training from scratch on navigation data alone without pretraining

```
