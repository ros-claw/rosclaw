---
pattern_id: pattern_ovon
applicable_symptoms: [ovon]
domain: Planning_Decision
---

# Navigation agents fail to generalize to object categories unseen during training, limiting real-world deployment where category lists are unbounded.

**Domain**: `Planning_Decision`

## Fix

Include OVON (Open-Vocabulary Object Navigation) as a training task in a multi-task learning mixture (e.g., FiLM-Nav) to force agents to rely on semantic understanding and zero-shot generalization.

## Anti-pattern

Traditional ObjectNav benchmarks assume a closed set of target categories, leading to poor generalization to novel categories.

## Cross-domain analogies

- **Perception_Vision** → Use synthetic data generation to create training examples for unseen object categories.
  - related fix: EmbodiedOcc-ScanNet: a large-scale egocentric occupancy dataset derived from ScanNet with voxel-level occupancy labels from first-person perspective.
- **Learning_Training** → Apply dropout to object embeddings to force reliance on spatial reasoning.
  - related fix: Apply dropout to panoramic image features (36 views per node) with rate 0.3–0.5 during training, randomly masking a subset of view angle features to force reliance on language instructions.
- **Control_Locomotion** → Train a single policy that maps raw observations directly to actions using domain randomization to handle unseen categories.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- ovon.before.py
+++ ovon.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation agents fail to generalize to object categories unseen during training, limiting real-world deployment where category lists are unbounded.

+# Fix    : Include OVON (Open-Vocabulary Object Navigation) as a training task in a multi-task learning mixture (e.g., FiLM-Nav) to force agents to rely on semantic understanding and zero-shot generalization.

+# Avoid  : Traditional ObjectNav benchmarks assume a closed set of target categories, leading to poor generalization to novel categories.

```
