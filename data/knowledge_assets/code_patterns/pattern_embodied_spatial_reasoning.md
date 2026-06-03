---
pattern_id: pattern_embodied_spatial_reasoning
applicable_symptoms: [embodied_spatial_reasoning]
domain: Planning_Decision
---

# Navigation agents assume a generic point robot, failing to account for agent size, shape, and mobility constraints, leading to infeasible paths or collisions.

**Domain**: `Planning_Decision`

## Fix

Capability-conditioned navigation (CapNav): integrate agent-specific physical constraints (e.g., dimensions, turning radius) into spatial reasoning and path planning.

## Anti-pattern

Treating the agent as a point robot without considering its embodiment.

## Cross-domain analogies

- **Perception_Vision** → Hierarchical decomposition with proximal-distal sub-networks and attention fusion to encode agent-specific shape and mobility constraints.
  - related fix: Hierarchical network with proximal sub-network for near-field features, distal sub-network for far-field context, recurrent temporal modeling, and attention-based fusion to output continuous risk scores.
- **Learning_Training** → Randomize agent morphology parameters during planning to generate robust, constraint-aware paths.
  - related fix: Use domain randomization: vary simulation parameters (friction, mass, lighting, delay) randomly during training to improve policy robustness to real-world conditions.
- **Control_Locomotion** → Train vision-conditioned RL policy with domain randomization over agent morphology to learn feasible, collision-free paths.
  - related fix: Train a model-free RL policy (PPO) with domain randomization that fuses egocentric camera images and velocity commands (or mid-level language actions) to output low-level joint actions, enabling real-time visual adaptation for obstacle avoidance.

## Patch

```diff
--- embodied_spatial_reasoning.before.py
+++ embodied_spatial_reasoning.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation agents assume a generic point robot, failing to account for agent size, shape, and mobility constraints, leading to infeasible paths or collisions.

+# Fix    : Capability-conditioned navigation (CapNav): integrate agent-specific physical constraints (e.g., dimensions, turning radius) into spatial reasoning and path planning.

+# Avoid  : Treating the agent as a point robot without considering its embodiment.

```
