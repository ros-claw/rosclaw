---
pattern_id: pattern_simulator_and_dataset
applicable_symptoms: [simulator_and_dataset]
domain: Learning_Training
---

# VLN agents fail to generalize across diverse environments due to limited scene and instruction diversity in existing datasets.

**Domain**: `Learning_Training`

## Fix

Use large-scale synthetic data generation (e.g., ScaleVLN with 4M+ instructions) to augment training.

## Anti-pattern

Training only on small human-annotated datasets like R2R (21K instructions, 90 scenes).

## Cross-domain analogies

- **Perception_Vision** → Use learned semantic representations to augment training data diversity for robust generalization.
  - related fix: VISR: a framework integrating visual perception with semantic reasoning using learned representations
- **Planning_Decision** → Use graph constraints to structure training data, enforcing diversity via spatial-instruction coverage.
  - related fix: Represent instructions as graph constraints (landmark nodes + spatial edges) and prune action space via constraint satisfaction at each step
- **Control_Locomotion** → Use large-scale simulation with domain randomization to generate diverse scenes and instructions for training.
  - related fix: Train a single end-to-end neural network policy via large-scale RL in simulation with domain randomization, mapping depth image directly to motor commands

## Patch

```diff
--- simulator_and_dataset.before.py
+++ simulator_and_dataset.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to generalize across diverse environments due to limited scene and instruction diversity in existing datasets.

+# Fix    : Use large-scale synthetic data generation (e.g., ScaleVLN with 4M+ instructions) to augment training.

+# Avoid  : Training only on small human-annotated datasets like R2R (21K instructions, 90 scenes).

```
