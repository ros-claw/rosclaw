---
pattern_id: pattern_cooperative_localization
applicable_symptoms: [cooperative_localization]
domain: Memory_Reasoning
---

# Single-agent localization fails to resolve spatial ambiguity in collaborative settings, leading to incorrect grounding when agents must communicate via dialog.

**Domain**: `Memory_Reasoning`

## Fix

Use a dialog-based cooperative localization framework where an observer and locator exchange natural language utterances, with spatial reasoning and mutual grounding across turns.

## Anti-pattern

Single-agent localization without dialog interaction.

## Cross-domain analogies

- **Perception_Vision** → Train multi-agent dialog on synthetic data with simulated ambiguity and noise.
  - related fix: Generate synthetic depth images with simulated sensor noise, self-occlusion, and lighting effects to match real sensor distribution.
- **Planning_Decision** → Use multi-height spatial sampling to resolve ambiguous agent locations via collaborative dialog.
  - related fix: ScaleVLN: retrieve visual information from different heights (e.g., robot dog, vacuum cleaner) to augment current viewpoint.
- **Learning_Training** → Use synthetic dialog-trajectory pairs from structured environment representations to resolve spatial ambiguity via programmatic generation.
  - related fix: Use Marky to programmatically generate 4.2 million synthetic instruction–trajectory pairs from structured environment representations and action sequences.

## Patch

```diff
--- cooperative_localization.before.py
+++ cooperative_localization.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Single-agent localization fails to resolve spatial ambiguity in collaborative settings, leading to incorrect grounding when agents must communicate via dialog.

+# Fix    : Use a dialog-based cooperative localization framework where an observer and locator exchange natural language utterances, with spatial reasoning and mutual grounding across turns.

+# Avoid  : Single-agent localization without dialog interaction.

```
