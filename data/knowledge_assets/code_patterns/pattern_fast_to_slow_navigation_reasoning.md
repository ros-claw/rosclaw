---
pattern_id: pattern_fast_to_slow_navigation_reasoning
applicable_symptoms: [fast_to_slow_navigation_reasoning]
domain: Planning_Decision
---

# Real-time goal selection in visual-language navigation is computationally expensive when using VLM for every candidate.

**Domain**: `Planning_Decision`

## Fix

Two-stage FSR: fast matching (lightweight similarity) to narrow candidates, then VLM-driven refinement only when fast stage yields low confidence.

## Anti-pattern

Using VLM for all candidate evaluations without fast pre-filtering.

## Cross-domain analogies

- **Perception_Vision** → Hierarchical decomposition: coarse-to-fine candidate pruning reduces VLM calls.
  - related fix: Use a coarse-to-fine pyramid (e.g., U-Net or FPN) that downsamples to capture coarse layout and upsamples to recover fine details, then fuse or sequentially feed multi-scale features.
- **Learning_Training** → Distribute candidate evaluation across parallel workers with synchronized VLM scoring.
  - related fix: Distribute PPO training across multiple workers with synchronized gradient updates (DD-PPO).
- **Control_Locomotion** → Use reinforcement learning to learn a lightweight policy mapping visual inputs directly to goal selection, bypassing VLM per candidate.
  - related fix: Use reinforcement learning to learn a control policy that directly maps sensor observations to actuator commands for plasma shape and position control.

## Patch

```diff
--- fast_to_slow_navigation_reasoning.before.py
+++ fast_to_slow_navigation_reasoning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Real-time goal selection in visual-language navigation is computationally expensive when using VLM for every candidate.

+# Fix    : Two-stage FSR: fast matching (lightweight similarity) to narrow candidates, then VLM-driven refinement only when fast stage yields low confidence.

+# Avoid  : Using VLM for all candidate evaluations without fast pre-filtering.

```
