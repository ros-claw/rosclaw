---
pattern_id: pattern_extreme_parkour_policy
applicable_symptoms: [extreme_parkour_policy]
domain: Control_Locomotion
---

# Low-cost robot with imprecise actuation and jittery low-frequency depth camera fails to execute dynamic parkour maneuvers reliably

**Domain**: `Control_Locomotion`

## Fix

Train a single end-to-end neural network policy via large-scale RL in simulation with domain randomization, mapping depth image directly to motor commands

## Anti-pattern

Traditional modular perception and planning systems that rely on precise sensing and high-frequency actuation

## Cross-domain analogies

- **Perception_Vision** → Use atomic motion primitives as grounded units to bridge noisy actuation and depth inputs.
  - related fix: Use actional atomic concepts (natural language phrases combining atomic action and object) as a compact grounded unit to bridge visual and linguistic features.
- **Planning_Decision** → Hierarchical decomposition with closed-loop verification between planning and execution.
  - related fix: Four-stage pipeline: Visual State Description → Reflection and Reasoning → Language Plan Generation → Executable Plan Generation
- **Learning_Training** → Use a convolutional stem for local smoothing then transformer attention for long-horizon planning.
  - related fix: Use a convolutional stem followed by Transformer blocks with global attention to model dependencies across 100kb+ distances

## Patch

```diff
--- extreme_parkour_policy.before.py
+++ extreme_parkour_policy.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Low-cost robot with imprecise actuation and jittery low-frequency depth camera fails to execute dynamic parkour maneuvers reliably

+# Fix    : Train a single end-to-end neural network policy via large-scale RL in simulation with domain randomization, mapping depth image directly to motor commands

+# Avoid  : Traditional modular perception and planning systems that rely on precise sensing and high-frequency actuation

```
