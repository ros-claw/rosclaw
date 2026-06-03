---
pattern_id: pattern_deepmind_galaxy_mergers
applicable_symptoms: [deepmind_galaxy_mergers]
domain: Perception_Vision
---

# Galaxy merger classification is slow and inaccurate when done manually or with traditional methods.

**Domain**: `Perception_Vision`

## Fix

Use deep learning models (e.g., CNNs) trained on simulated galaxy merger images to automatically classify merger stages.

## Anti-pattern

Manual classification by astronomers or rule-based morphological analysis.

## Cross-domain analogies

- **Planning_Decision** → Use chain-of-thought hierarchical decomposition to classify mergers via stepwise morphological subgoals.
  - related fix: Use chain-of-thought reasoning to decompose tasks into step-by-step subgoals before predicting actions.
- **Learning_Training** → Use dynamic weighting of specialized merger classifiers based on per-class confidence.
  - related fix: Train three specialized RL experts (reaching, squeezing, avoiding) and combine them via dynamic weighting based on individual capability performance.
- **Control_Locomotion** → Use standardized benchmark tasks requiring precise spatial reasoning to train galaxy merger classifiers.
  - related fix: Use EB-Manipulation benchmark to evaluate and train agents on low-level actions (joint torques, end-effector poses) with standardized tasks that require precise perception and spatial reasoning.

## Patch

```diff
--- deepmind_galaxy_mergers.before.py
+++ deepmind_galaxy_mergers.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Galaxy merger classification is slow and inaccurate when done manually or with traditional methods.

+# Fix    : Use deep learning models (e.g., CNNs) trained on simulated galaxy merger images to automatically classify merger stages.

+# Avoid  : Manual classification by astronomers or rule-based morphological analysis.

```
