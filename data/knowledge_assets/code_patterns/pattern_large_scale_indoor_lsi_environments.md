---
pattern_id: pattern_large_scale_indoor_lsi_environments
applicable_symptoms: [large_scale_indoor_lsi_environments]
domain: Planning_Decision
---

# Navigators lose sense of direction in large indoor spaces with sparse signage, even with map-based wayfinding.

**Domain**: `Planning_Decision`

## Fix

Use semantic signage (e.g., 'Oncology Wing') as navigation cues for instruction-based wayfinding.

## Anti-pattern

Conventional map-based wayfinding alone fails due to dynamic orientation shifts.

## Cross-domain analogies

- **Perception_Vision** → Use synthetic visual imaginations of landmarks to train an auxiliary directional alignment loss.
  - related fix: Generate synthetic visual imaginations from segmented instruction phrases using a text-to-image diffusion model, and train with an auxiliary loss that aligns imaginations with their corresponding referring expressions.
- **Learning_Training** → Train a privileged map-aware planner, then distill into a sign-agnostic policy via guidance loss.
  - related fix: Privileged Information Guidance (PIG): train a diffusion policy with privileged depth and collision information during training, then distill into a student policy that uses only RGB observations via a guidance loss.
- **Control_Locomotion** → End-to-end learned navigation from raw visual input bypassing explicit map and signage reliance.
  - related fix: Train a single neural network policy via deep reinforcement learning that maps raw depth camera images directly to motor commands, bypassing hand-crafted perception and control layers.

## Patch

```diff
--- large_scale_indoor_lsi_environments.before.py
+++ large_scale_indoor_lsi_environments.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigators lose sense of direction in large indoor spaces with sparse signage, even with map-based wayfinding.

+# Fix    : Use semantic signage (e.g., 'Oncology Wing') as navigation cues for instruction-based wayfinding.

+# Avoid  : Conventional map-based wayfinding alone fails due to dynamic orientation shifts.

```
