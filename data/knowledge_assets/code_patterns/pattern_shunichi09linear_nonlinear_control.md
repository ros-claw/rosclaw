---
pattern_id: pattern_shunichi09linear_nonlinear_control
applicable_symptoms: [shunichi09linear_nonlinear_control]
domain: Control_Locomotion
---

# Implementing nonlinear control algorithms from scratch is complex and error-prone, especially when requiring gradient or Hessian computations.

**Domain**: `Control_Locomotion`

## Fix

Use the PythonLinearNonlinearControl library which provides implementations of linear and nonlinear control algorithms (MPC, CEM, MPPI, iLQR, DDP, NMPC) with varying requirements for model gradients and Hessians, using only basic libraries (scipy, numpy).

## Anti-pattern

Implementing custom nonlinear controllers without a library often leads to bugs and high development overhead.

## Cross-domain analogies

- **Perception_Vision** → Derive a large-scale dataset of precomputed gradients and Hessians from simulation to train a neural surrogate for nonlinear control.
  - related fix: EmbodiedOcc-ScanNet: a large-scale egocentric occupancy dataset derived from ScanNet with voxel-level occupancy labels from first-person perspective.
- **Planning_Decision** → Replace manual gradient derivation with a precomputed annotated Jacobian map updated online.
  - related fix: Replace historical frame sequences with an Annotated Semantic Map (ASM) that is constructed at episode start and updated each timestep, using a VLM to add textual labels for key regions.
- **Learning_Training** → Use driving data to auto-generate control code and gradient approximations for nonlinear controllers.
  - related fix: Use driving videos to automatically generate navigation instructions and action labels for data augmentation.

## Patch

```diff
--- shunichi09linear_nonlinear_control.before.py
+++ shunichi09linear_nonlinear_control.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Implementing nonlinear control algorithms from scratch is complex and error-prone, especially when requiring gradient or Hessian computations.

+# Fix    : Use the PythonLinearNonlinearControl library which provides implementations of linear and nonlinear control algorithms (MPC, CEM, MPPI, iLQR, DDP, NMPC) with varying requirements for model gradients and Hessians, using only basic libraries (scipy, numpy).

+# Avoid  : Implementing custom nonlinear controllers without a library often leads to bugs and high development overhead.

```
