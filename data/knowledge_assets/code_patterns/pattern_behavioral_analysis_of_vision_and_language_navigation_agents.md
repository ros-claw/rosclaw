---
pattern_id: pattern_behavioral_analysis_of_vision_and_language_navigation_agents
applicable_symptoms: [behavioral_analysis_of_vision_and_language_navigation_agents]
domain: Planning_Decision
---

# VLN agents ignore landmark cues in long instructions

**Domain**: `Planning_Decision`

## Fix

Behavioral analysis framework to identify agent reliance on visual vs. language cues; use targeted diagnostic tasks to expose failure modes

## Anti-pattern

Standard VLN evaluation metrics (e.g., success rate) mask agent's lack of language grounding

## Cross-domain analogies

- **Perception_Vision** → Use predictive reacquisition to actively re-query visual landmarks when instruction cues are missed.
  - related fix: Active mitigation strategies such as multi-sensor fusion or predictive reacquisition to handle blind zones
- **Learning_Training** → Pre-train on instruction-landmark-action triplets with self-supervised masking to force attention to landmark cues.
  - related fix: Pre-train on large-scale image-text-action triplets using self-supervised pretext tasks to learn generic representations that transfer to new navigation tasks.
- **Control_Locomotion** → Train a policy with domain randomization over visual inputs to force landmark-conditioned closed-loop navigation.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- behavioral_analysis_of_vision_and_language_navigation_agents.before.py
+++ behavioral_analysis_of_vision_and_language_navigation_agents.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents ignore landmark cues in long instructions

+# Fix    : Behavioral analysis framework to identify agent reliance on visual vs. language cues; use targeted diagnostic tasks to expose failure modes

+# Avoid  : Standard VLN evaluation metrics (e.g., success rate) mask agent's lack of language grounding

```
