---
pattern_id: pattern_vision_and_language_navigation_vln_evaluation
applicable_symptoms: [vision_and_language_navigation_vln_evaluation]
domain: Planning_Decision
---

# Existing reference-based metrics (BLEU, ROUGE, METEOR, CIDEr) correlate poorly with human wayfinding success in VLN evaluation.

**Domain**: `Planning_Decision`

## Fix

Use Instruction-Trajectory Compatibility Model, a reference-free model that scores how well an instruction aligns with the agent's actual trajectory.

## Anti-pattern

Using BLEU, ROUGE, METEOR, or CIDEr as evaluation metrics for grounded navigation instructions.

## Cross-domain analogies

- **Perception_Vision** → Derive large-scale human-annotated wayfinding success labels from embodied datasets to train a learned evaluation metric.
  - related fix: EmbodiedOcc-ScanNet: a large-scale egocentric occupancy dataset derived from ScanNet with voxel-level occupancy labels from first-person perspective.
- **Learning_Training** → Use group-relative trajectory advantage to evaluate navigation plans against diverse human-like paths.
  - related fix: Use GRPO (Group Relative Policy Optimization) as a second-stage RL fine-tuning after supervised chain-of-thought alignment, optimizing policy relative to a group of sampled trajectories via group-relative advantage estimation.
- **Control_Locomotion** → Train an end-to-end policy via large-scale RL with domain randomization on path trajectories.
  - related fix: Train a single end-to-end neural network policy via large-scale RL in simulation with domain randomization, mapping depth image directly to motor commands

## Patch

```diff
--- vision_and_language_navigation_vln_evaluation.before.py
+++ vision_and_language_navigation_vln_evaluation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Existing reference-based metrics (BLEU, ROUGE, METEOR, CIDEr) correlate poorly with human wayfinding success in VLN evaluation.

+# Fix    : Use Instruction-Trajectory Compatibility Model, a reference-free model that scores how well an instruction aligns with the agent's actual trajectory.

+# Avoid  : Using BLEU, ROUGE, METEOR, or CIDEr as evaluation metrics for grounded navigation instructions.

```
