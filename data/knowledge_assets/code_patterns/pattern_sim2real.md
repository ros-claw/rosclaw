---
pattern_id: pattern_sim2real
applicable_symptoms: [sim2real]
domain: Learning_Training
---

# Sim-to-real gap causes policy collapse on unseen terrain due to idealized physics and perfect sensors in simulation

**Domain**: `Learning_Training`

## Fix

Use video-only input modality (no depth or map) combined with domain randomization to eliminate sensor fidelity and geometry transfer gaps

## Anti-pattern

Traditional Sim2Real approaches relying on map or depth inputs

## Cross-domain analogies

- **Perception_Vision** → Pre-train on grounded, varied terrain data to align policy features with real-world physics.
  - related fix: Pre-train on GEL-R2R, a dataset with grounded entity-level annotations, to enable cross-modal alignment at the entity level.
- **Planning_Decision** → Use language pretraining to encode diverse terrain variations and stochastic sampling to robustly handle sensor noise.
  - related fix: Use language pretraining (BERT) to encode instructions and stochastic sampling during decoding to improve robustness to instruction variations
- **Control_Locomotion** → Use domain randomization on terrain physics and sensor noise during training to bridge the sim-to-real gap.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- sim2real.before.py
+++ sim2real.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Sim-to-real gap causes policy collapse on unseen terrain due to idealized physics and perfect sensors in simulation

+# Fix    : Use video-only input modality (no depth or map) combined with domain randomization to eliminate sensor fidelity and geometry transfer gaps

+# Avoid  : Traditional Sim2Real approaches relying on map or depth inputs

```
