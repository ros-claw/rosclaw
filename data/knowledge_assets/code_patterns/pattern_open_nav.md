---
pattern_id: pattern_open_nav
applicable_symptoms: [open_nav]
domain: Planning_Decision
---

# Zero-shot VLN agents using closed-source LLMs incur high token costs and data breach risks, and struggle to decompose long instructions into actionable steps without task-specific training.

**Domain**: `Planning_Decision`

## Fix

Use open-source LLMs with spatial-temporal chain-of-thought reasoning that decomposes navigation into instruction comprehension, progress estimation, and decision-making, enhanced by fine-grained object and spatial knowledge.

## Anti-pattern

Closed-source LLM approaches (e.g., GPT-4) for VLN that require task-specific training or incur token costs and data breach risks.

## Cross-domain analogies

- **Perception_Vision** → Re-annotate instructions with local, camera-aligned sub-goal grids for stepwise decomposition.
  - related fix: Re-annotate ScanNet scenes with local occupancy grids aligned to the camera frame, supporting both static and temporal prediction tasks.
- **Learning_Training** → Use concatenated sub-instructions to create longer, circuitous task decompositions for zero-shot LLM planning.
  - related fix: Use R4R dataset (concatenated R2R paths) to create longer, circuitous trajectories that better differentiate instruction-following agents from goal-seeking ones.
- **Control_Locomotion** → Pre-train a library of reusable action primitives via RL to decouple instruction decomposition from task planning.
  - related fix: Pre-train a library of versatile locomotion and interaction behaviors via reinforcement learning, decoupling skill acquisition from task planning.

## Patch

```diff
--- open_nav.before.py
+++ open_nav.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Zero-shot VLN agents using closed-source LLMs incur high token costs and data breach risks, and struggle to decompose long instructions into actionable steps without task-specific training.

+# Fix    : Use open-source LLMs with spatial-temporal chain-of-thought reasoning that decomposes navigation into instruction comprehension, progress estimation, and decision-making, enhanced by fine-grained object and spatial knowledge.

+# Avoid  : Closed-source LLM approaches (e.g., GPT-4) for VLN that require task-specific training or incur token costs and data breach risks.

```
