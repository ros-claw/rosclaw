---
pattern_id: pattern_topological_graph_with_visitation_records
applicable_symptoms: [topological_graph_with_visitation_records]
domain: Memory_Reasoning
---

# VLN agent explores randomly without systematic coverage, failing to revisit or avoid already visited areas during long-horizon navigation.

**Domain**: `Memory_Reasoning`

## Fix

Build a topological graph incrementally from waypoints, annotating each node with a visitation count; use this graph to inform a language model for context-aware navigation commands.

## Anti-pattern

Using a flat memory of visited positions without topological connectivity, leading to inefficient exploration.

## Cross-domain analogies

- **Perception_Vision** → Use a pretrained spatial memory embedding to enable zero-shot coverage awareness.
  - related fix: Use a pre-trained vision-language model (e.g., CLIP) trained on large-scale image-text pairs to learn a shared embedding space, enabling zero-shot transfer to unseen tasks and objects.
- **Planning_Decision** → Cross-modal attention can align current observations with a memory of visited locations to guide systematic coverage.
  - related fix: Use a cross-modal attention mechanism to fuse visual features and language embeddings at each step, enabling the agent to align instruction phrases with visual landmarks.
- **Learning_Training** → Use dynamic weighting of exploration, exploitation, and revisitation experts based on area coverage performance.
  - related fix: Train three specialized RL experts (reaching, squeezing, avoiding) and combine them via dynamic weighting based on individual capability performance.

## Patch

```diff
--- topological_graph_with_visitation_records.before.py
+++ topological_graph_with_visitation_records.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent explores randomly without systematic coverage, failing to revisit or avoid already visited areas during long-horizon navigation.

+# Fix    : Build a topological graph incrementally from waypoints, annotating each node with a visitation count; use this graph to inform a language model for context-aware navigation commands.

+# Avoid  : Using a flat memory of visited positions without topological connectivity, leading to inefficient exploration.

```
