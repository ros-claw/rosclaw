---
pattern_id: pattern_plcinject
applicable_symptoms: [plcinject]
domain: Systems_Compute
---

# Siemens S7 PLCs lack a built-in mechanism to inject custom function block calls into existing cyclic OBs without manual re-engineering.

**Domain**: `Systems_Compute`

## Fix

Use Snap7 library to upload OB1, parse its MC7 code, insert a CALL instruction to a user-provided block (e.g., FC1000), and download the patched OB1 along with the new block to the PLC.

## Anti-pattern

Manual re-engineering of PLC logic or using vendor-specific tools that require stopping the PLC.

## Cross-domain analogies

- **Perception_Vision** → Use language-based abstraction to decouple custom logic from cyclic OB hardware dependencies.
  - related fix: Replace visual features with language-based representations (e.g., captions from a vision-language model) for navigation policy input.
- **Planning_Decision** → Use hierarchical decomposition to inject low-level control blocks into cyclic OBs via a continuous execution layer.
  - related fix: Use continuous action spaces with low-level controllers (e.g., PID) and train with reinforcement learning or imitation learning on continuous trajectories.
- **Learning_Training** → Insert a transformer-like global attention layer to dynamically inject custom blocks into cyclic OBs.
  - related fix: Use a convolutional stem followed by Transformer blocks with global attention to model dependencies across 100kb+ distances

## Patch

```diff
--- plcinject.before.py
+++ plcinject.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Siemens S7 PLCs lack a built-in mechanism to inject custom function block calls into existing cyclic OBs without manual re-engineering.

+# Fix    : Use Snap7 library to upload OB1, parse its MC7 code, insert a CALL instruction to a user-provided block (e.g., FC1000), and download the patched OB1 along with the new block to the PLC.

+# Avoid  : Manual re-engineering of PLC logic or using vendor-specific tools that require stopping the PLC.

```
