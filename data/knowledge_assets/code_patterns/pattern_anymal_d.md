---
pattern_id: pattern_anymal_d
applicable_symptoms: [anymal_d]
domain: Control_Locomotion
---

# Quadruped locomotion policies fail to generalize to unseen unstructured terrains and lack agility for parkour.

**Domain**: `Control_Locomotion`

## Fix

Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

## Anti-pattern

Single-expert policies trained only in simulation without real-world data augmentation.

## Cross-domain analogies

- **Perception_Vision** → Use multi-sensor fusion and predictive reacquisition to handle blind zones in terrain perception for agile locomotion.
  - related fix: Active mitigation strategies such as multi-sensor fusion or predictive reacquisition to handle blind zones
- **Planning_Decision** → Fine-tune a locomotion policy on simulated terrain experience with visual history and goal conditioning.
  - related fix: Fine-tune a pre-trained VLM on simulated embodied experience to act as a navigation policy, conditioning on visual history and goals (as in FiLM-Nav).
- **Learning_Training** → Train locomotion policies on egocentric terrain video with expert actions via behavioral cloning.
  - related fix: Use VLN-Ego dataset: large-scale egocentric video + expert action pairs from Habitat simulator for imitation learning (behavioral cloning) of navigation policies directly from first-person observations.

## Patch

```diff
--- anymal_d.before.py
+++ anymal_d.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Quadruped locomotion policies fail to generalize to unseen unstructured terrains and lack agility for parkour.

+# Fix    : Multi-expert distillation with DAgger and RL fine-tuning, using depth images as exteroceptive input.

+# Avoid  : Single-expert policies trained only in simulation without real-world data augmentation.

```
