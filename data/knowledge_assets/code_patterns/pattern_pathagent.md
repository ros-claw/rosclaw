---
pattern_id: pattern_pathagent
applicable_symptoms: [pathagent]
domain: Planning_Decision
---

# Path planning in complex environments fails to integrate visual context and affordances, leading to suboptimal trajectory selection.

**Domain**: `Planning_Decision`

## Fix

Use PathAgent to overlay trajectory candidates on image input and reason the most probable path by evaluating environmental information via visual affordances prompting.

## Anti-pattern

Traditional path planners that ignore visual affordances or operate solely on metric maps without image-space reasoning.

## Cross-domain analogies

- **Perception_Vision** → Use a cross-attention bottleneck to compress visual context into a fixed-size latent affordance array for efficient planning.
  - related fix: Use a cross-attention bottleneck: project arbitrary input to a fixed-size latent array via cross-attention, then process with iterative self-attention in latent space.
- **Learning_Training** → Use closed-loop verification between planner and affordance predictor to iteratively refine trajectories.
  - related fix: Train a Transformer-based Speaker jointly with a Listener in a Double Back-Translation loop, where the Speaker generates instructions from paths and the Listener reconstructs paths from instructions, enforcing instruction-path consistency through iterative refinement.
- **Control_Locomotion** → Use a safety-critic to override visual affordance planner when trajectory risk exceeds threshold.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- pathagent.before.py
+++ pathagent.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Path planning in complex environments fails to integrate visual context and affordances, leading to suboptimal trajectory selection.

+# Fix    : Use PathAgent to overlay trajectory candidates on image input and reason the most probable path by evaluating environmental information via visual affordances prompting.

+# Avoid  : Traditional path planners that ignore visual affordances or operate solely on metric maps without image-space reasoning.

```
