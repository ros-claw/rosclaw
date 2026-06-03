---
pattern_id: pattern_waypoint_predictor
applicable_symptoms: [waypoint_predictor]
domain: Planning_Decision
---

# Waypoint predictor generates unrealistic or infeasible waypoints because it neglects object semantics and passibility attributes.

**Domain**: `Planning_Decision`

## Fix

Augment waypoint predictor with semantic and passibility features from the environment (e.g., obstacle labels, terrain traversability) to filter or score candidate waypoints.

## Anti-pattern

Existing waypoint predictors ignore object semantics and passibility, leading to infeasible predictions.

## Cross-domain analogies

- **Perception_Vision** → Inject dual-view semantic-passibility verification into waypoint generation to enforce feasibility.
  - related fix: Dual-view visual prompt: combine two complementary spatial views into a single prompt at inference time, applied on top of a VLA model.
- **Learning_Training** → Train an end-to-end network from raw inputs to waypoints to jointly learn semantics and feasibility.
  - related fix: Train a single neural network end-to-end from raw sensor inputs to control outputs using a reward signal (e.g., reinforcement learning), allowing the network to discover internal representations that directly optimize the desired behavior.
- **Control_Locomotion** → Augment waypoint predictor with semantic passibility maps as visual context.
  - related fix: Train a visual locomotion policy that maps camera images and proprioception to joint actions, enabling real-time foot placement adaptation to obstacles.

## Patch

```diff
--- waypoint_predictor.before.py
+++ waypoint_predictor.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Waypoint predictor generates unrealistic or infeasible waypoints because it neglects object semantics and passibility attributes.

+# Fix    : Augment waypoint predictor with semantic and passibility features from the environment (e.g., obstacle labels, terrain traversability) to filter or score candidate waypoints.

+# Avoid  : Existing waypoint predictors ignore object semantics and passibility, leading to infeasible predictions.

```
