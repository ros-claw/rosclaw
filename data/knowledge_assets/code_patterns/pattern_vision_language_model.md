---
pattern_id: pattern_vision_language_model
applicable_symptoms: [vision_language_model]
domain: Perception_Vision
---

# VLMs fail to ground language instructions in visual scenes for embodied navigation, leading to poor instruction following.

**Domain**: `Perception_Vision`

## Fix

Use a Vision-Language Model (VLM) that jointly processes visual and textual data for cross-modal reasoning, as in NavForesee.

## Anti-pattern

Using separate vision and language models without alignment.

## Cross-domain analogies

- **Planning_Decision** → Hybrid map with multimodal transformer aligns visual-language representations for grounded navigation.
  - related fix: Build a hybrid map combining a local metric map (for fine-grained near-field geometry) and a global topological map (for long-range connectivity), and pre-train a multimodal transformer (BEVBert) with map-based objectives to align visual and language modalities.
- **Learning_Training** → Use synthetic visual-language pairs with closed-loop verification to improve grounding.
  - related fix: Use large-scale synthetic data generation (e.g., ScaleVLN with 4M+ instructions) to augment training.
- **Control_Locomotion** → Map sensor observations directly to actions via RL to bypass explicit grounding failures.
  - related fix: Use reinforcement learning to learn a control policy that directly maps sensor observations to actuator commands for plasma shape and position control.

## Patch

```diff
--- vision_language_model.before.py
+++ vision_language_model.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLMs fail to ground language instructions in visual scenes for embodied navigation, leading to poor instruction following.

+# Fix    : Use a Vision-Language Model (VLM) that jointly processes visual and textual data for cross-modal reasoning, as in NavForesee.

+# Avoid  : Using separate vision and language models without alignment.

```
