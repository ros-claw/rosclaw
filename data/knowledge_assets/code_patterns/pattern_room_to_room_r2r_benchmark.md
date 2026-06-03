---
pattern_id: pattern_room_to_room_r2r_benchmark
applicable_symptoms: [room_to_room_r2r_benchmark]
domain: Planning_Decision
---

# VLN agents fail to generalize to unseen environments and often take inefficient paths, leading to low success rates and poor path efficiency.

**Domain**: `Planning_Decision`

## Fix

Use the Room-to-Room (R2R) benchmark with Success Rate weighted by Path Length (SPL) as the primary metric to evaluate and compare VLN agents on realistic, unseen indoor environments.

## Anti-pattern

Evaluating agents only on seen environments or using simple success rate without considering path efficiency.

## Cross-domain analogies

- **Perception_Vision** → Pretrain a shared visual-language embedding for zero-shot navigation in unseen environments.
  - related fix: Use a pre-trained vision-language model (e.g., CLIP) trained on large-scale image-text pairs to learn a shared embedding space, enabling zero-shot transfer to unseen tasks and objects.
- **Learning_Training** → Train a latent world model for VLN to simulate future paths and rewards, enabling mental exploration.
  - related fix: Train a neural world model that predicts future latent states and rewards from current observations and actions, enabling model-based planning and mental simulation without direct environment interaction.
- **Control_Locomotion** → Train a policy with domain-randomized visual inputs and commands to enable closed-loop visual adaptation for path efficiency.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- room_to_room_r2r_benchmark.before.py
+++ room_to_room_r2r_benchmark.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents fail to generalize to unseen environments and often take inefficient paths, leading to low success rates and poor path efficiency.

+# Fix    : Use the Room-to-Room (R2R) benchmark with Success Rate weighted by Path Length (SPL) as the primary metric to evaluate and compare VLN agents on realistic, unseen indoor environments.

+# Avoid  : Evaluating agents only on seen environments or using simple success rate without considering path efficiency.

```
