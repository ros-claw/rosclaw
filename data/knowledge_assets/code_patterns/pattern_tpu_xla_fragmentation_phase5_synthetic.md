---
pattern_id: pattern_tpu_xla_fragmentation_phase5_synthetic
applicable_symptoms: [tpu_xla_fragmentation_phase5_synthetic]
domain: Systems_Compute
---

# Repeated re-traces of dynamic-shape inputs fragment the HBM pool on TPU, causing allocation failures despite low actual usage.

**Domain**: `Systems_Compute`

## Fix

Pad batches to a small set of canonical shapes and force a single trace shape via jit with pre-allocated buffers and XLA flags, optionally tuning XLA_TPU_BUFFER_PADDING_RATIO.

## Anti-pattern

Doubling HBM via topology reshape fails because fragmentation re-emerges with long-tail shapes.

## Cross-domain analogies

- **Perception_Vision** → Project sensory data into a fixed-shape grid to eliminate dynamic-shape re-traces and prevent HBM fragmentation.
  - related fix: Project sensory data into a bird's-eye-view grid representation, distilled from visual foundation models, to enable structured spatial reasoning for planning.
- **Planning_Decision** → Cross-modal attention aligns fragmented shape traces into coherent memory layouts.
  - related fix: Cross-modal transformer with attention aligning topological map features and language instruction embeddings to output subgoal sequences.
- **Learning_Training** → Pre-allocating diverse tensor shape templates reduces fragmentation by standardizing memory layouts.
  - related fix: Provide large-scale, diverse offline datasets (e.g., RL Unplugged) with standardized evaluation protocols for fair comparison.

## Patch

```diff
--- tpu_xla_fragmentation_phase5_synthetic.before.py
+++ tpu_xla_fragmentation_phase5_synthetic.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Repeated re-traces of dynamic-shape inputs fragment the HBM pool on TPU, causing allocation failures despite low actual usage.

+# Fix    : Pad batches to a small set of canonical shapes and force a single trace shape via jit with pre-allocated buffers and XLA flags, optionally tuning XLA_TPU_BUFFER_PADDING_RATIO.

+# Avoid  : Doubling HBM via topology reshape fails because fragmentation re-emerges with long-tail shapes.

```
