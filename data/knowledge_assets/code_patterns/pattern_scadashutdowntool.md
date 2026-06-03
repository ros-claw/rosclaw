---
pattern_id: pattern_scadashutdowntool
applicable_symptoms: [scadashutdowntool]
domain: Systems_Compute
---

# SCADA controllers can be maliciously scanned, fuzzed, and rewritten without authentication, leading to potential shutdown or damage.

**Domain**: `Systems_Compute`

## Fix

Use SCADAShutdownTool in safe-mode (read-only) for security testing; for production, enforce authentication and access controls on Modbus/TCP registers.

## Anti-pattern

Relying on default or unprotected controller registers without security measures.

## Cross-domain analogies

- **Perception_Vision** → Hierarchical open-vocabulary verification segments controller commands into authenticated, semantically validated layers.
  - related fix: Open-vocabulary hierarchical 3D graph representation combining VLM-based instance/caption extraction from images, incremental LiDAR projection, and lane graph connectivity segmentation.
- **Planning_Decision** → State-adaptive expert selection could authenticate SCADA commands per context to block unauthorized actions.
  - related fix: State-Adaptive Mixture of Experts (SAME): adaptively selects expert modules based on current state and instruction, enabling shared navigation knowledge with task-specific exploitation.
- **Learning_Training** → Pretrain a secure baseline controller on benign traffic data, then fine-tune on authenticated SCADA protocols.
  - related fix: Pretrain a vision-language model on large-scale web data to learn cross-modal alignment, then fine-tune on navigation tasks

## Patch

```diff
--- scadashutdowntool.before.py
+++ scadashutdowntool.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: SCADA controllers can be maliciously scanned, fuzzed, and rewritten without authentication, leading to potential shutdown or damage.

+# Fix    : Use SCADAShutdownTool in safe-mode (read-only) for security testing; for production, enforce authentication and access controls on Modbus/TCP registers.

+# Avoid  : Relying on default or unprotected controller registers without security measures.

```
