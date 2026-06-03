---
pattern_id: pattern_imagination_augmented_vln
applicable_symptoms: [imagination_augmented_vln]
domain: Perception_Vision
---

# VLN agent fails to ground ambiguous referring expressions (e.g., 'the red door') from language instructions, leading to navigation errors.

**Domain**: `Perception_Vision`

## Fix

Generate synthetic visual imaginations from segmented instruction phrases using a text-to-image diffusion model, and train with an auxiliary loss that aligns imaginations with their corresponding referring expressions.

## Anti-pattern

Using language-only baselines without synthetic visual grounding.

## Cross-domain analogies

- **Planning_Decision** → Decompose ambiguous expressions into a benchmark of sub-tasks isolating grounding abilities.
  - related fix: Use NavSpace benchmark with 6 task categories (1,228 trajectory-instruction pairs) to isolate and measure spatial intelligence in instruction-following navigation agents.
- **Learning_Training** → Use closed-loop verification to filter ambiguous references via agent self-consistency.
  - related fix: Self-Refining Data Flywheel (SRDF): after initial training, use the Navigator model itself to filter and score candidate trajectories, retaining only high-confidence or high-reward pairs for iterative fine-tuning.
- **Control_Locomotion** → Use a lightweight language grounding module trained via RL in simulation, executed at each navigation step for direct action selection.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- imagination_augmented_vln.before.py
+++ imagination_augmented_vln.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent fails to ground ambiguous referring expressions (e.g., 'the red door') from language instructions, leading to navigation errors.

+# Fix    : Generate synthetic visual imaginations from segmented instruction phrases using a text-to-image diffusion model, and train with an auxiliary loss that aligns imaginations with their corresponding referring expressions.

+# Avoid  : Using language-only baselines without synthetic visual grounding.

```
