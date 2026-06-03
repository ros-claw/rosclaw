---
pattern_id: pattern_sim_to_real_adaptation
applicable_symptoms: [sim_to_real_adaptation]
domain: Learning_Training
---

# Policies trained in simulation fail when deployed on real robots due to the reality gap (unmodeled dynamics, sensor noise, actuator delays).

**Domain**: `Learning_Training`

## Fix

Use domain randomization: vary simulation parameters (friction, mass, lighting, delay) randomly during training to improve policy robustness to real-world conditions.

## Anti-pattern

Training policies in a fixed simulation environment without randomization, leading to brittle transfer.

## Cross-domain analogies

- **Perception_Vision** → Use deformable attention to selectively sample critical simulation-to-reality discrepancies.
  - related fix: Use deformable cross-attention with learned sampling points to selectively attend to relevant image features instead of the entire grid.
- **Planning_Decision** → Use adaptive reasoning depth to switch between sim and real dynamics based on deployment complexity.
  - related fix: Adaptive Chain-of-Thought mechanism that dynamically switches between fast reactive (System 1) and slow deliberative (System 2) reasoning based on task complexity.
- **Control_Locomotion** → Use camera images to augment proprioception, closing the sim-to-real perception gap.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- sim_to_real_adaptation.before.py
+++ sim_to_real_adaptation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Policies trained in simulation fail when deployed on real robots due to the reality gap (unmodeled dynamics, sensor noise, actuator delays).

+# Fix    : Use domain randomization: vary simulation parameters (friction, mass, lighting, delay) randomly during training to improve policy robustness to real-world conditions.

+# Avoid  : Training policies in a fixed simulation environment without randomization, leading to brittle transfer.

```
