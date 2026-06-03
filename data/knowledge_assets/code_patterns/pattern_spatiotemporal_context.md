---
pattern_id: pattern_spatiotemporal_context
applicable_symptoms: [spatiotemporal_context]
domain: Memory_Reasoning
---

# VLN agent fails to interpret commands referencing objects/locations not visible in current frame, and confuses similar-looking features from different viewpoints.

**Domain**: `Memory_Reasoning`

## Fix

Maintain a structured memory of historical visual observations weighted by temporal recency and spatial novelty, so that past frames influence current reasoning and planning.

## Anti-pattern

Using only current-frame visual input without temporal memory.

## Cross-domain analogies

- **Perception_Vision** → Use closed-loop verification to confirm landmark salience before referencing it in instructions.
  - related fix: Prefer instruction design using landmarks that are visually salient and detectable by the agent's sensor suite (e.g., large objects, distinct colors).
- **Planning_Decision** → Closed-loop hierarchical verification with confidence scoring resolves ambiguous references.
  - related fix: Closed-loop hierarchical chain-of-thought: decompose navigation into multi-turn QA with confidence scoring for each step, fine-tune InternVL2 (2B) with LoRA on simulation data.
- **Learning_Training** → Use egocentric video replay to train memory-based grounding of unseen references.
  - related fix: Use VLN-Ego dataset: large-scale egocentric video + expert action pairs from Habitat simulator for imitation learning (behavioral cloning) of navigation policies directly from first-person observations.

## Patch

```diff
--- spatiotemporal_context.before.py
+++ spatiotemporal_context.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent fails to interpret commands referencing objects/locations not visible in current frame, and confuses similar-looking features from different viewpoints.

+# Fix    : Maintain a structured memory of historical visual observations weighted by temporal recency and spatial novelty, so that past frames influence current reasoning and planning.

+# Avoid  : Using only current-frame visual input without temporal memory.

```
