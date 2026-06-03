---
pattern_id: pattern_navdp_learning_sim_to_real_navigation_diffusion_policy_with_privileged_informati
applicable_symptoms: [navdp_learning_sim_to_real_navigation_diffusion_policy_with_privileged_informati]
domain: Learning_Training
---

# Sim-to-real gap causes navigation policy to fail in real-world deployment due to lack of depth and collision awareness.

**Domain**: `Learning_Training`

## Fix

Privileged Information Guidance (PIG): train a diffusion policy with privileged depth and collision information during training, then distill into a student policy that uses only RGB observations via a guidance loss.

## Anti-pattern

Standard end-to-end imitation learning from RGB images without explicit depth or collision cues.

## Cross-domain analogies

- **Perception_Vision** → Project sensory data into a top-down depth grid to distill collision awareness for sim-to-real transfer.
  - related fix: Project sensory data into a bird's-eye-view grid representation, distilled from visual foundation models, to enable structured spatial reasoning for planning.
- **Planning_Decision** → Augment training with multi-height depth views to bridge sim-to-real perception gaps.
  - related fix: ScaleVLN: retrieve visual information from different heights (e.g., robot dog, vacuum cleaner) to augment current viewpoint.
- **Control_Locomotion** → Train a separate domain-adaptation policy with a safety critic to override the nominal policy when depth uncertainty exceeds a threshold.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- navdp_learning_sim_to_real_navigation_diffusion_policy_with_privileged_informati.before.py
+++ navdp_learning_sim_to_real_navigation_diffusion_policy_with_privileged_informati.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Sim-to-real gap causes navigation policy to fail in real-world deployment due to lack of depth and collision awareness.

+# Fix    : Privileged Information Guidance (PIG): train a diffusion policy with privileged depth and collision information during training, then distill into a student policy that uses only RGB observations via a guidance loss.

+# Avoid  : Standard end-to-end imitation learning from RGB images without explicit depth or collision cues.

```
