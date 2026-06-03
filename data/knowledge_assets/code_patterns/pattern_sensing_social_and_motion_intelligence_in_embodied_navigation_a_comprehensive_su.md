---
pattern_id: pattern_sensing_social_and_motion_intelligence_in_embodied_navigation_a_comprehensive_su
applicable_symptoms: [sensing_social_and_motion_intelligence_in_embodied_navigation_a_comprehensive_su]
domain: Planning_Decision
---

# Embodied navigation agents fail to integrate sensing, social cues, and motion intelligence for robust real-world deployment.

**Domain**: `Planning_Decision`

## Fix

Comprehensive survey categorizing embodied navigation into sensing, social interaction, and motion intelligence, with taxonomies and benchmarks for each sub-area.

## Anti-pattern

Prior surveys focus on isolated aspects (e.g., only vision-language or object-goal navigation) without holistic integration.

## Cross-domain analogies

- **Perception_Vision** → Use Laplacian variance filtering to pre-stabilize multi-modal sensory inputs before fusion.
  - related fix: Apply Laplacian Variance Filtering to stabilize camera feed before detection.
- **Learning_Training** → Use closed-loop verification with pretrained multimodal models to generate diverse social-motion training scenarios.
  - related fix: Use Marky, a multilingual instruction generator that produces visually grounded instruction-trajectory pairs at scale (4.2M pairs) by leveraging pretrained vision-language models and spatial alignment.
- **Control_Locomotion** → Use a lightweight hierarchical policy with low-level reactive control fused to high-level social and sensory reasoning.
  - related fix: Use a lightweight MLP or RNN policy trained via RL in simulation, executed at 50-100 Hz for direct joint-level torque/position commands.

## Patch

```diff
--- sensing_social_and_motion_intelligence_in_embodied_navigation_a_comprehensive_su.before.py
+++ sensing_social_and_motion_intelligence_in_embodied_navigation_a_comprehensive_su.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Embodied navigation agents fail to integrate sensing, social cues, and motion intelligence for robust real-world deployment.

+# Fix    : Comprehensive survey categorizing embodied navigation into sensing, social interaction, and motion intelligence, with taxonomies and benchmarks for each sub-area.

+# Avoid  : Prior surveys focus on isolated aspects (e.g., only vision-language or object-goal navigation) without holistic integration.

```
