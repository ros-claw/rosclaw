---
pattern_id: pattern_frontier_based_exploration
applicable_symptoms: [frontier_based_exploration]
domain: Planning_Decision
---

# Exploration policy lacks a compressed representation of unknown space, causing inefficient navigation in vision-language tasks.

**Domain**: `Planning_Decision`

## Fix

Use frontier cells as discrete spatial hypotheses queried via frontier_queries to guide exploration without full map reconstruction.

## Anti-pattern

Traditional occupancy-grid-based frontier exploration without semantic grounding.

## Cross-domain analogies

- **Perception_Vision** → Use language-based compressed representations to replace raw visual features for exploration policy input.
  - related fix: Replace visual features with language-based representations (e.g., captions from a vision-language model) for navigation policy input.
- **Learning_Training** → Jointly train exploration policy on diverse environments to learn shared spatial representations for robust navigation.
  - related fix: Jointly train the VLN model on multiple annotated datasets (RxR and R2R) using multitask learning to learn shared visual and linguistic representations, improving robustness and generalization.
- **Control_Locomotion** → Closed-loop local mapping from sensory data to compress unknown space for efficient exploration.
  - related fix: Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Patch

```diff
--- frontier_based_exploration.before.py
+++ frontier_based_exploration.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Exploration policy lacks a compressed representation of unknown space, causing inefficient navigation in vision-language tasks.

+# Fix    : Use frontier cells as discrete spatial hypotheses queried via frontier_queries to guide exploration without full map reconstruction.

+# Avoid  : Traditional occupancy-grid-based frontier exploration without semantic grounding.

```
