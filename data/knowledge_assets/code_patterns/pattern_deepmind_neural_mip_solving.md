---
pattern_id: pattern_deepmind_neural_mip_solving
applicable_symptoms: [deepmind_neural_mip_solving]
domain: Planning_Decision
---

# Classical MIP solvers are slow on large combinatorial optimization problems.

**Domain**: `Planning_Decision`

## Fix

Use learned neural components (e.g., branching policies, primal heuristics) to accelerate MIP solving.

## Anti-pattern

Relying solely on handcrafted heuristics and branch-and-bound without learned guidance.

## Cross-domain analogies

- **Perception_Vision** → Use variance-based filtering to discard low-quality subproblems, reducing MIP solver load.
  - related fix: Laplacian Variance Filtering (LVF): compute variance of Laplacian of each frame; discard or deweight frames with low variance (high blur) to reduce jitter-induced motion blur.
- **Learning_Training** → Use expert solutions on subproblems sampled under current solver distribution to iteratively refine the MIP model.
  - related fix: Iteratively collect new data under the current policy's distribution, query the expert for corrective actions, and aggregate this data into the training set (DAgger).
- **Control_Locomotion** → Replace heavy MIP solvers with lightweight learned heuristics trained offline for fast online decisions.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- deepmind_neural_mip_solving.before.py
+++ deepmind_neural_mip_solving.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Classical MIP solvers are slow on large combinatorial optimization problems.

+# Fix    : Use learned neural components (e.g., branching policies, primal heuristics) to accelerate MIP solving.

+# Avoid  : Relying solely on handcrafted heuristics and branch-and-bound without learned guidance.

```
