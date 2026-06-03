---
pattern_id: pattern_synthetic_instructions
applicable_symptoms: [synthetic_instructions]
domain: Learning_Training
---

# Human-annotated instruction datasets for embodied AI are limited in scale, causing poor language grounding and spatial reasoning in imitation learning agents.

**Domain**: `Learning_Training`

## Fix

Use Marky to programmatically generate 4.2 million synthetic instruction–trajectory pairs from structured environment representations and action sequences.

## Anti-pattern

Relying solely on human-annotated datasets like RxR or R2R, which are orders of magnitude smaller.

## Cross-domain analogies

- **Perception_Vision** → Use actional atomic concepts to auto-generate grounded instruction datasets from limited human annotations.
  - related fix: Use actional atomic concepts (natural language phrases combining atomic action and object) as a compact grounded unit to bridge visual and linguistic features.
- **Planning_Decision** → Use multi-task learning to jointly predict auxiliary spatial and semantic labels from limited instructions.
  - related fix: Incorporate volumetric environment representations and multi-task learning (e.g., depth estimation, semantic segmentation) to enrich the agent's grasp of both geometric and semantic scene properties.
- **Control_Locomotion** → Use domain-randomized synthetic data generation to scale instruction diversity.
  - related fix: Train a single end-to-end neural network policy via large-scale RL in simulation with domain randomization, mapping depth image directly to motor commands

## Patch

```diff
--- synthetic_instructions.before.py
+++ synthetic_instructions.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Human-annotated instruction datasets for embodied AI are limited in scale, causing poor language grounding and spatial reasoning in imitation learning agents.

+# Fix    : Use Marky to programmatically generate 4.2 million synthetic instruction–trajectory pairs from structured environment representations and action sequences.

+# Avoid  : Relying solely on human-annotated datasets like RxR or R2R, which are orders of magnitude smaller.

```
