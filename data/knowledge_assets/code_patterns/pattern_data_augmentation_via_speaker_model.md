---
pattern_id: pattern_data_augmentation_via_speaker_model
applicable_symptoms: [data_augmentation_via_speaker_model]
domain: Learning_Training
---

# VLN training data is scarce and expensive to collect, leading to overfitting and poor generalization to unseen environments.

**Domain**: `Learning_Training`

## Fix

Train a speaker model to generate synthetic instruction–trajectory pairs from unannotated visual paths, then augment the original training set with these synthetic pairs.

## Anti-pattern

Relying solely on human-annotated instruction–trajectory pairs without data augmentation.

## Cross-domain analogies

- **Perception_Vision** → Use large-scale pretrained vision-language embeddings to enable zero-shot transfer in VLN, reducing data needs.
  - related fix: Use a pre-trained vision-language model (e.g., CLIP) trained on large-scale image-text pairs to learn a shared embedding space, enabling zero-shot transfer to unseen tasks and objects.
- **Planning_Decision** → Use hierarchical decomposition to break VLN into subtasks with synthetic benchmarks for each.
  - related fix: Comprehensive survey categorizing embodied navigation into sensing, social interaction, and motion intelligence, with taxonomies and benchmarks for each sub-area.
- **Control_Locomotion** → Distill multiple VLN experts via DAgger with synthetic data and RL fine-tuning.
  - related fix: Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Patch

```diff
--- data_augmentation_via_speaker_model.before.py
+++ data_augmentation_via_speaker_model.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN training data is scarce and expensive to collect, leading to overfitting and poor generalization to unseen environments.

+# Fix    : Train a speaker model to generate synthetic instruction–trajectory pairs from unannotated visual paths, then augment the original training set with these synthetic pairs.

+# Avoid  : Relying solely on human-annotated instruction–trajectory pairs without data augmentation.

```
