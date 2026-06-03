---
pattern_id: pattern_soon
applicable_symptoms: [soon]
domain: Planning_Decision
---

# VLN agents fail to generalize under distributional shifts and environment changes (e.g., lighting, object placements, layout modifications).

**Domain**: `Planning_Decision`

## Fix

Use the SOON benchmark within the EvolveNav framework to evaluate and train VLN models for robustness to evolving conditions.

## Anti-pattern

Standard VLN benchmarks that assume static environments.

## Cross-domain analogies

- **Perception_Vision** → Train VLN agents on procedurally perturbed environments to match real-world distribution shifts.
  - related fix: Generate synthetic depth images with simulated sensor noise, self-occlusion, and lighting effects to match real sensor distribution.
- **Learning_Training** → Use realistic simulation benchmarks with low-level primitives to evaluate and transfer navigation policies under distributional shifts.
  - related fix: Use IsaacLab simulation benchmark with realistic scenes and low-level control primitives to evaluate and transfer navigation policies to real-world robots
- **Control_Locomotion** → Train an end-to-end policy mapping visual observations directly to navigation actions, bypassing hand-crafted perception and planning modules.
  - related fix: Train a single neural network policy via deep reinforcement learning that maps raw depth camera images directly to motor commands, bypassing hand-crafted perception and control layers.

## Patch

```diff
--- soon.before.py
+++ soon.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to generalize under distributional shifts and environment changes (e.g., lighting, object placements, layout modifications).

+# Fix    : Use the SOON benchmark within the EvolveNav framework to evaluate and train VLN models for robustness to evolving conditions.

+# Avoid  : Standard VLN benchmarks that assume static environments.

```
