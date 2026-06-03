---
pattern_id: pattern_deepmind_option_keyboard
applicable_symptoms: [deepmind_option_keyboard]
domain: Learning_Training
---

# Reinforcement learning agents struggle to combine primitive skills for complex tasks, requiring retraining for new task combinations.

**Domain**: `Learning_Training`

## Fix

Option Keyboard: a hierarchical RL framework where a high-level policy selects and composes pre-trained low-level skills (options) via a learned combination mechanism, enabling zero-shot generalization to new tasks.

## Anti-pattern

Training a monolithic policy from scratch for each new task combination.

## Cross-domain analogies

- **Perception_Vision** → Use multi-skill fusion and predictive skill reacquisition to handle unseen task combinations.
  - related fix: Active mitigation strategies such as multi-sensor fusion or predictive reacquisition to handle blind zones
- **Planning_Decision** → Hierarchical decomposition with coarse-to-fine skill selection enables zero-shot task composition.
  - related fix: Hierarchical Multi-Modal Scene Graph (HMSG) combining geometric, semantic, and topological maps for coarse-to-fine localization, plus Fast and Slow Reasoning (FSR) using VLM for goal selection.
- **Control_Locomotion** → Use a safety-critic to gate skill combination, overriding unsafe primitive blends during task composition.
  - related fix: Train a separate RL safety-shielding policy with a safety critic and intervention logic that overrides the nominal controller when risk exceeds a threshold.

## Patch

```diff
--- deepmind_option_keyboard.before.py
+++ deepmind_option_keyboard.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Reinforcement learning agents struggle to combine primitive skills for complex tasks, requiring retraining for new task combinations.

+# Fix    : Option Keyboard: a hierarchical RL framework where a high-level policy selects and composes pre-trained low-level skills (options) via a learned combination mechanism, enabling zero-shot generalization to new tasks.

+# Avoid  : Training a monolithic policy from scratch for each new task combination.

```
