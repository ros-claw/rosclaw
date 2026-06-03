---
pattern_id: pattern_lana
applicable_symptoms: [lana]
domain: Planning_Decision
---

# VLN agents are typically task-specific, requiring separate pipelines for instruction following and generation, which doubles complexity and limits bidirectional human-robot communication.

**Domain**: `Planning_Decision`

## Fix

Unified architecture with shared route and language encoders feeding two decoders (action prediction and instruction generation), trained jointly on both objectives via pretrain-then-fine-tune.

## Anti-pattern

Task-specific VLN agents that only perform instruction following without generation capability.

## Cross-domain analogies

- **Perception_Vision** → Use shared transformer layers with modality-specific encoders for unified instruction following and generation.
  - related fix: Use a multimodal versatile network (MMV) with shared transformer layers and modality-specific encoders to learn joint embeddings across modalities.
- **Learning_Training** → Two-stage curriculum: pretrain on joint instruction-following and generation tasks, then fine-tune on bidirectional dialogue.
  - related fix: Two-stage curriculum: pretrain on large-scale web-scraped image-text pairs (Conceptual Captions) then fine-tune on embodied path-instruction data.
- **Control_Locomotion** → Use diffusion policies to model multi-modal action distributions for unified instruction following and generation.
  - related fix: Use diffusion policies to model multi-modal action distributions and discretize continuous action spaces for low-level action prediction.

## Patch

```diff
--- lana.before.py
+++ lana.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agents are typically task-specific, requiring separate pipelines for instruction following and generation, which doubles complexity and limits bidirectional human-robot communication.

+# Fix    : Unified architecture with shared route and language encoders feeding two decoders (action prediction and instruction generation), trained jointly on both objectives via pretrain-then-fine-tune.

+# Avoid  : Task-specific VLN agents that only perform instruction following without generation capability.

```
