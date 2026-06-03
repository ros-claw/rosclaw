---
pattern_id: pattern_msgnav
applicable_symptoms: [msgnav]
domain: Planning_Decision
---

# Zero-shot embodied navigation fails to generalize to unseen environments and arbitrary language goals without RL training, and suffers from the last-mile problem where the agent cannot precisely localize the target object when nearby.

**Domain**: `Planning_Decision`

## Fix

Build a Multi-modal 3D Scene Graph (M3DSG) with key subgraph selection, adaptive vocabulary update, closed-loop reasoning, and visibility-based viewpoint decision to resolve the last-mile problem.

## Anti-pattern

Reinforcement learning training with fixed object classes and no scene graph

## Cross-domain analogies

- **Perception_Vision** → Use a shared vision-language embedding space for zero-shot goal grounding and precise localization.
  - related fix: Use a pre-trained vision-language model (e.g., CLIP) trained on large-scale image-text pairs to learn a shared embedding space, enabling zero-shot transfer to unseen tasks and objects.
- **Learning_Training** → Use privileged depth and collision data during training to distill a student policy for precise last-mile localization from RGB.
  - related fix: Privileged Information Guidance (PIG): train a diffusion policy with privileged depth and collision information during training, then distill into a student policy that uses only RGB observations via a guidance loss.
- **Control_Locomotion** → Use terrain-like perceptual cues (depth/object maps) to dynamically adjust navigation policy near goals.
  - related fix: Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Patch

```diff
--- msgnav.before.py
+++ msgnav.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Zero-shot embodied navigation fails to generalize to unseen environments and arbitrary language goals without RL training, and suffers from the last-mile problem where the agent cannot precisely localize the target object when nearby.

+# Fix    : Build a Multi-modal 3D Scene Graph (M3DSG) with key subgraph selection, adaptive vocabulary update, closed-loop reasoning, and visibility-based viewpoint decision to resolve the last-mile problem.

+# Avoid  : Reinforcement learning training with fixed object classes and no scene graph

```
