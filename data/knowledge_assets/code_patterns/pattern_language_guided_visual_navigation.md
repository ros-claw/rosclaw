---
pattern_id: pattern_language_guided_visual_navigation
applicable_symptoms: [language_guided_visual_navigation]
domain: Planning_Decision
---

# Language-guided navigation agents struggle to share general knowledge across task variants (e.g., high-level search vs. low-level step-by-step) while exploiting task-specific cues, leading to poor adaptation to varying instruction precision and visual context.

**Domain**: `Planning_Decision`

## Fix

State-Adaptive Mixture of Experts (SAME): adaptively selects expert modules based on current state and instruction, enabling shared navigation knowledge with task-specific exploitation.

## Anti-pattern

Using a single monolithic policy for all instruction types without dynamic expert selection.

## Cross-domain analogies

- **Perception_Vision** → Integrate learned hierarchical representations to fuse general knowledge with task-specific cues.
  - related fix: VISR: a framework integrating visual perception with semantic reasoning using learned representations
- **Learning_Training** → Apply stochastic feature masking to force reliance on task-agnostic language cues.
  - related fix: Apply dropout to panoramic image features (36 views per node) with rate 0.3–0.5 during training, randomly masking a subset of view angle features to force reliance on language instructions.
- **Control_Locomotion** → Fuse visual and language inputs into a single policy that dynamically weights task-specific cues.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- language_guided_visual_navigation.before.py
+++ language_guided_visual_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Language-guided navigation agents struggle to share general knowledge across task variants (e.g., high-level search vs. low-level step-by-step) while exploiting task-specific cues, leading to poor adaptation to varying instruction precision and visual context.

+# Fix    : State-Adaptive Mixture of Experts (SAME): adaptively selects expert modules based on current state and instruction, enabling shared navigation knowledge with task-specific exploitation.

+# Avoid  : Using a single monolithic policy for all instruction types without dynamic expert selection.

```
