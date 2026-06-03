---
pattern_id: pattern_habitat_simulator
applicable_symptoms: [habitat_simulator]
domain: World_Physics
---

# Sim-to-real gap causes policy collapse on unseen terrain

**Domain**: `World_Physics`

## Fix

Use photorealistic 3D environment simulation with configurable sensors and physics integration from Habitat Simulator

## Anti-pattern

Training on low-fidelity synthetic environments without diverse scene datasets

## Cross-domain analogies

- **Perception_Vision** → Use synthetic terrain images from diffusion models to align policy with unseen physics via auxiliary loss.
  - related fix: Generate synthetic visual imaginations from segmented instruction phrases using a text-to-image diffusion model, and train with an auxiliary loss that aligns imaginations with their corresponding referring expressions.
- **Planning_Decision** → Benchmark with terrain-relationship metrics to test sim-to-real physics transfer.
  - related fix: NavSpace benchmark with spatial intelligence instructions and evaluation metrics that test object-relationship and layout reasoning.
- **Learning_Training** → Pretrain on diverse simulated terrain data before real-world RL adaptation.
  - related fix: Use supervised fine-tuning (SFT) on expert demonstration trajectories to bootstrap a behavioral prior before reinforcement learning

## Patch

```diff
--- habitat_simulator.before.py
+++ habitat_simulator.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Sim-to-real gap causes policy collapse on unseen terrain

+# Fix    : Use photorealistic 3D environment simulation with configurable sensors and physics integration from Habitat Simulator

+# Avoid  : Training on low-fidelity synthetic environments without diverse scene datasets

```
