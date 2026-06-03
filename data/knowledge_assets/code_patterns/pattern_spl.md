---
pattern_id: pattern_spl
applicable_symptoms: [spl]
domain: Planning_Decision
---

# Navigation agents achieve high success rate but take overly long, inefficient paths, wasting time and energy.

**Domain**: `Planning_Decision`

## Fix

Use SPL (Success weighted by Path Length) metric to jointly penalize path inefficiency alongside success, computed as S_i * L_i* / max(L_i, L_i*).

## Anti-pattern

Using only success rate (SR) as the evaluation metric, which ignores path length efficiency.

## Cross-domain analogies

- **Perception_Vision** → Train a neural network on optimal path examples to prune inefficient subpaths.
  - related fix: Use deep learning models (e.g., CNNs) trained on simulated galaxy merger images to automatically classify merger stages.
- **Learning_Training** → Randomize path cost weights during training to encourage robust efficiency.
  - related fix: Use domain randomization: vary simulation parameters (friction, mass, lighting, delay) randomly during training to improve policy robustness to real-world conditions.
- **Control_Locomotion** → Use a lightweight, low-frequency policy to prune suboptimal path branches during online replanning.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- spl.before.py
+++ spl.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation agents achieve high success rate but take overly long, inefficient paths, wasting time and energy.

+# Fix    : Use SPL (Success weighted by Path Length) metric to jointly penalize path inefficiency alongside success, computed as S_i * L_i* / max(L_i, L_i*).

+# Avoid  : Using only success rate (SR) as the evaluation metric, which ignores path length efficiency.

```
