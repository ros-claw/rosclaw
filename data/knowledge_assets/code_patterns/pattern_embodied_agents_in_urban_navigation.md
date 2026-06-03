---
pattern_id: pattern_embodied_agents_in_urban_navigation
applicable_symptoms: [embodied_agents_in_urban_navigation]
domain: Planning_Decision
---

# Embodied agents fail to follow noisy or ambiguous natural language instructions in dynamic urban environments, leading to navigation errors.

**Domain**: `Planning_Decision`

## Fix

Use visual language models (VLMs) to ground language instructions in visual observations, combined with dynamic path planning that adapts to changing street scenes.

## Anti-pattern

Relying solely on static map representations without real-time visual grounding or re-planning.

## Cross-domain analogies

- **Perception_Vision** → Use a transformer-based decoder to predict a structured action occupancy field from language and sensor inputs.
  - related fix: Learn an occupancy network that predicts 3D occupancy and semantics from multi-camera images using a transformer-based 3D decoder.
- **Learning_Training** → Hierarchical instruction refinement: distill complex commands through intermediate abstraction layers for noise robustness.
  - related fix: Two-stage progressive knowledge distillation: first distill from large teacher to medium student, then from medium to small student, achieving 1/7 model size with same accuracy.
- **Control_Locomotion** → Incorporate real-time environmental perception into the language grounding policy to dynamically adapt navigation actions.
  - related fix: Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Patch

```diff
--- embodied_agents_in_urban_navigation.before.py
+++ embodied_agents_in_urban_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Embodied agents fail to follow noisy or ambiguous natural language instructions in dynamic urban environments, leading to navigation errors.

+# Fix    : Use visual language models (VLMs) to ground language instructions in visual observations, combined with dynamic path planning that adapts to changing street scenes.

+# Avoid  : Relying solely on static map representations without real-time visual grounding or re-planning.

```
