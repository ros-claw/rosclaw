---
pattern_id: pattern_pytrajectory
applicable_symptoms: [pytrajectory]
domain: Control_Locomotion
---

# Trajectory optimization for underactuated systems fails to converge or produces infeasible solutions when using naive collocation methods.

**Domain**: `Control_Locomotion`

## Fix

Use direct collocation with B-spline parameterization and automatic differentiation to transcribe the optimal control problem into a nonlinear program.

## Anti-pattern

Shooting methods that require explicit integration and suffer from high sensitivity to initial guesses.

## Cross-domain analogies

- **Perception_Vision** → Jointly predict state feasibility, dynamics, and constraints from a shared latent trajectory representation.
  - related fix: Multi-task learning jointly predicting 3D occupancy, room layout, and object bounding boxes from a shared volumetric representation
- **Planning_Decision** → Use hierarchical scene graph construction to structure state space for collocation.
  - related fix: Use hierarchical scene graph construction from a semantic object map to provide structured, open-vocabulary environment context to the LLM, enabling multi-step plan generation and real-time re-planning.
- **Learning_Training** → Use structured environment priors to generate synthetic collocation constraints, ensuring feasibility.
  - related fix: Use Marky to programmatically generate 4.2 million synthetic instruction–trajectory pairs from structured environment representations and action sequences.

## Patch

```diff
--- pytrajectory.before.py
+++ pytrajectory.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Trajectory optimization for underactuated systems fails to converge or produces infeasible solutions when using naive collocation methods.

+# Fix    : Use direct collocation with B-spline parameterization and automatic differentiation to transcribe the optimal control problem into a nonlinear program.

+# Avoid  : Shooting methods that require explicit integration and suffer from high sensitivity to initial guesses.

```
