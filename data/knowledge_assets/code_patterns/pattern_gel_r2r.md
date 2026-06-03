---
pattern_id: pattern_gel_r2r
applicable_symptoms: [gel_r2r]
domain: Perception_Vision
---

# VLN models fail to associate fine-grained entity references in instructions with specific visual landmarks, leading to poor navigation accuracy.

**Domain**: `Perception_Vision`

## Fix

Pre-train on GEL-R2R, a dataset with grounded entity-level annotations, to enable cross-modal alignment at the entity level.

## Anti-pattern

Using only coarse instruction-following without fine-grained visual grounding.

## Cross-domain analogies

- **Planning_Decision** → Use multi-constraint agent profiling to generate diverse entity-landmark associations for robust reference grounding.
  - related fix: CapNav benchmark: evaluate VLMs on 5 agent types with distinct constraints across 45 scenes, 473 tasks, and 2,365 QA pairs to test capability-aware navigation.
- **Learning_Training** → Bootstrapping with imitation on paired landmarks, then refining with RL on unlabeled scenes.
  - related fix: Mixed Imitation and Reinforcement Learning (MIRL): bootstrap policy via off-policy imitation learning, then refine with on-policy RL, gradually shifting weight from imitation to RL.
- **Control_Locomotion** → Train a separate verification policy that overrides the VLN model when entity-reference confidence is low.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- gel_r2r.before.py
+++ gel_r2r.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN models fail to associate fine-grained entity references in instructions with specific visual landmarks, leading to poor navigation accuracy.

+# Fix    : Pre-train on GEL-R2R, a dataset with grounded entity-level annotations, to enable cross-modal alignment at the entity level.

+# Avoid  : Using only coarse instruction-following without fine-grained visual grounding.

```
