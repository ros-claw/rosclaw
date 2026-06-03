---
pattern_id: pattern_context_free_grammar_cfg_for_vln
applicable_symptoms: [context_free_grammar_cfg_for_vln]
domain: Planning_Decision
---

# VLN agents are evaluated only with coarse metrics like success rate, failing to reveal which instruction types (e.g., landmarks, directions, actions) cause failures.

**Domain**: `Planning_Decision`

## Fix

Use a Context-Free Grammar (CFG) to decompose navigation instructions into hierarchical categories (landmarks, directions, actions) and evaluate each category independently.

## Anti-pattern

Single scalar metrics (e.g., success rate, navigation error) that mask compositional instruction understanding.

## Cross-domain analogies

- **Perception_Vision** → Multi-modal fusion of evaluation metrics to decompose instruction types and blind spots in agent failures.
  - related fix: Multi-modal occupancy grid fusion that integrates RGB, depth, lidar, and proprioception with a sensor layout strategy to maximize coverage and handle self-occlusion.
- **Learning_Training** → Use causal decomposition to isolate which instruction types cause planning failures.
  - related fix: Use causal representation learning (e.g., Causal VAEs, independent mechanism analysis) and causal model-based RL to learn structural causal models that support interventions and counterfactuals.
- **Control_Locomotion** → Use closed-loop verification with domain-randomized instruction perturbations to expose failure modes per instruction type.
  - related fix: Train a single reinforcement-learning-based policy that directly maps noisy depth images to motor commands, using domain randomization and sim-to-real transfer to handle sensor artifacts and actuation imprecision.

## Patch

```diff
--- context_free_grammar_cfg_for_vln.before.py
+++ context_free_grammar_cfg_for_vln.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents are evaluated only with coarse metrics like success rate, failing to reveal which instruction types (e.g., landmarks, directions, actions) cause failures.

+# Fix    : Use a Context-Free Grammar (CFG) to decompose navigation instructions into hierarchical categories (landmarks, directions, actions) and evaluate each category independently.

+# Avoid  : Single scalar metrics (e.g., success rate, navigation error) that mask compositional instruction understanding.

```
