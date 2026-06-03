---
pattern_id: pattern_hierarchical_scene_graph_construction
applicable_symptoms: [hierarchical_scene_graph_construction]
domain: Memory_Reasoning
---

# Flat scene graphs lose spatial/functional abstraction, causing LLM planners to fail on long-horizon tasks requiring multi-level reasoning.

**Domain**: `Memory_Reasoning`

## Fix

Build hierarchical scene graph incrementally from semantic object map, with layers for objects, regions, rooms, and functional zones, updated online as new observations arrive.

## Anti-pattern

Traditional scene graphs flatten object relationships without hierarchical abstraction.

## Cross-domain analogies

- **Perception_Vision** → Use shared hierarchical transformers with domain-specific encoders for multi-level scene abstraction.
  - related fix: Use a multimodal versatile network (MMV) with shared transformer layers and modality-specific encoders to learn joint embeddings across modalities.
- **Planning_Decision** → Use hierarchical closed-loop verification to ground abstract scene graph reasoning in real-time visual feedback.
  - related fix: Use a cooperative dialog framework where an oracle provides step-by-step guidance and the agent can ask clarifying questions, grounding instructions in real-time visual observations.
- **Learning_Training** → Use iterative policy aggregation to build hierarchical scene graphs from corrective LLM queries.
  - related fix: Iteratively collect new data under the current policy's distribution, query the expert for corrective actions, and aggregate this data into the training set (DAgger).

## Patch

```diff
--- hierarchical_scene_graph_construction.before.py
+++ hierarchical_scene_graph_construction.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Flat scene graphs lose spatial/functional abstraction, causing LLM planners to fail on long-horizon tasks requiring multi-level reasoning.

+# Fix    : Build hierarchical scene graph incrementally from semantic object map, with layers for objects, regions, rooms, and functional zones, updated online as new observations arrive.

+# Avoid  : Traditional scene graphs flatten object relationships without hierarchical abstraction.

```
