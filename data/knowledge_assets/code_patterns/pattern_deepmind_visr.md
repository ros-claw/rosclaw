---
pattern_id: pattern_deepmind_visr
applicable_symptoms: [deepmind_visr]
domain: Perception_Vision
---

# Embodied agents fail to leverage visual semantic reasoning for complex tasks

**Domain**: `Perception_Vision`

## Fix

VISR: a framework integrating visual perception with semantic reasoning using learned representations

## Anti-pattern

End-to-end learning without explicit semantic reasoning

## Cross-domain analogies

- **Planning_Decision** → Overlay trajectory candidates on visual inputs to prompt semantic reasoning for task-relevant affordances.
  - related fix: Use PathAgent to overlay trajectory candidates on image input and reason the most probable path by evaluating environmental information via visual affordances prompting.
- **Learning_Training** → Use human demonstration data to train a behavioral cloning policy for visual semantic reasoning.
  - related fix: Use large-scale human demonstration dataset (Habitat-Web) collected via web-based interface to train a behavioral cloning policy for object-goal navigation.
- **Control_Locomotion** → Pre-train a library of reusable visual semantic reasoning skills via reinforcement learning, decoupling perception from task planning.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- deepmind_visr.before.py
+++ deepmind_visr.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Embodied agents fail to leverage visual semantic reasoning for complex tasks

+# Fix    : VISR: a framework integrating visual perception with semantic reasoning using learned representations

+# Avoid  : End-to-end learning without explicit semantic reasoning

```
