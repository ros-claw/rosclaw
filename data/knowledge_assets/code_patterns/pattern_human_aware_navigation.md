---
pattern_id: pattern_human_aware_navigation
applicable_symptoms: [human_aware_navigation]
domain: Planning_Decision
---

# Navigation agents ignore human presence and social norms, causing collisions and discomfort in crowded environments.

**Domain**: `Planning_Decision`

## Fix

Incorporate social-awareness constraints (e.g., personal space adherence, dynamic human interaction modeling) into path planning.

## Anti-pattern

Standard path planning without social constraints.

## Cross-domain analogies

- **Perception_Vision** → Apply Laplacian variance filtering to stabilize social perception before decision-making.
  - related fix: Apply Laplacian Variance Filtering to stabilize camera feed before detection.
- **Learning_Training** → Generate synthetic social navigation data by combining crowd simulations with LLM-generated social norms and augmenting with diverse human trajectories.
  - related fix: ScaleVLN: large-scale synthetic data generation by combining 3D scene graphs with LLM-generated instructions and augmenting with panoramic views and object-level grounding.
- **Control_Locomotion** → Train a social navigation policy that maps human-centric observations and robot state to collision-avoiding actions.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- human_aware_navigation.before.py
+++ human_aware_navigation.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation agents ignore human presence and social norms, causing collisions and discomfort in crowded environments.

+# Fix    : Incorporate social-awareness constraints (e.g., personal space adherence, dynamic human interaction modeling) into path planning.

+# Avoid  : Standard path planning without social constraints.

```
