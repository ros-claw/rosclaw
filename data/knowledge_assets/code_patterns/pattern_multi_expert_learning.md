---
pattern_id: pattern_multi_expert_learning
applicable_symptoms: [multi_expert_learning]
domain: Learning_Training
---

# Monolithic RL policy struggles to balance reaching, squeezing, and avoiding behaviors, leading to poor performance in complex manipulation and avoidance scenarios.

**Domain**: `Learning_Training`

## Fix

Train three specialized RL experts (reaching, squeezing, avoiding) and combine them via dynamic weighting based on individual capability performance.

## Anti-pattern

Training a single monolithic RL policy without task decomposition.

## Cross-domain analogies

- **Perception_Vision** → Fuse multiple behavioral objectives into a unified policy representation via hierarchical decomposition.
  - related fix: Propose a multimodal occupancy perception system that fuses vision, depth, and other sensor data into a unified occupancy representation for humanoid robots.
- **Planning_Decision** → Use hierarchical decomposition to break the monolithic policy into specialized sub-policies for reaching, squeezing, and avoiding.
  - related fix: Enable the agent to actively request and interpret multimodal instructions (natural language and visual cues) from a human assistant when uncertain.
- **Control_Locomotion** → Decompose the monolithic policy into a hierarchical benchmark of sub-tasks for reaching, squeezing, and avoiding.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- multi_expert_learning.before.py
+++ multi_expert_learning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Monolithic RL policy struggles to balance reaching, squeezing, and avoiding behaviors, leading to poor performance in complex manipulation and avoidance scenarios.

+# Fix    : Train three specialized RL experts (reaching, squeezing, avoiding) and combine them via dynamic weighting based on individual capability performance.

+# Avoid  : Training a single monolithic RL policy without task decomposition.

```
