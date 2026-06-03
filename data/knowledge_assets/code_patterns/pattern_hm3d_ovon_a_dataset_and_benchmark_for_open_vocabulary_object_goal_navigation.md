---
pattern_id: pattern_hm3d_ovon_a_dataset_and_benchmark_for_open_vocabulary_object_goal_navigation
applicable_symptoms: [hm3d_ovon_a_dataset_and_benchmark_for_open_vocabulary_object_goal_navigation]
domain: Planning_Decision
---

# Open-vocabulary object-goal navigation agents fail to generalize to novel object categories not seen during training.

**Domain**: `Planning_Decision`

## Fix

Use a large vision-language model (e.g., CLIP) to encode open-vocabulary goal descriptions and fuse with spatial memory for navigation.

## Anti-pattern

Closed-vocabulary object-goal navigation with fixed category sets.

## Cross-domain analogies

- **Perception_Vision** → Use cross-modal joint embedding to map novel object descriptions to visual features without retraining.
  - related fix: Use a Vision-Language Model (VLM) that jointly processes visual and textual data for cross-modal reasoning, as in NavForesee.
- **Learning_Training** → Two-stage training: supervised pretraining on seen categories, then RL fine-tuning with reward for novel object discovery.
  - related fix: Two-stage training: first supervised fine-tuning on expert demonstrations, then reinforcement fine-tuning with policy gradient (e.g., PPO) to maximize task completion reward
- **Control_Locomotion** → Train an end-to-end policy with domain-randomized object embeddings for zero-shot generalization.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- hm3d_ovon_a_dataset_and_benchmark_for_open_vocabulary_object_goal_navigation.before.py
+++ hm3d_ovon_a_dataset_and_benchmark_for_open_vocabulary_object_goal_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Open-vocabulary object-goal navigation agents fail to generalize to novel object categories not seen during training.

+# Fix    : Use a large vision-language model (e.g., CLIP) to encode open-vocabulary goal descriptions and fuse with spatial memory for navigation.

+# Avoid  : Closed-vocabulary object-goal navigation with fixed category sets.

```
