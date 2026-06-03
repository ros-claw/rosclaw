---
pattern_id: pattern_ics_mem_collect
applicable_symptoms: [ics_mem_collect]
domain: Systems_Compute
---

# Industrial control system devices lack APIs for programmatic memory access, forcing manual probing for anomalies or malware.

**Domain**: `Systems_Compute`

## Fix

Develop custom APIs (e.g., for GE D20MX) and JTAG interfaces to enable automated memory collection and analysis.

## Anti-pattern

Manual memory probing without APIs

## Cross-domain analogies

- **Perception_Vision** → Derive labeled memory-access datasets from existing control-system logs to train anomaly detectors.
  - related fix: EmbodiedOcc-ScanNet: a large-scale egocentric occupancy dataset derived from ScanNet with voxel-level occupancy labels from first-person perspective.
- **Planning_Decision** → Hierarchical decomposition of high-level queries into stepwise subgoal probes for memory access.
  - related fix: Use Hierarchical Chain-of-Thought (H-CoT) prompting to decompose high-level goals into stepwise subgoals (e.g., region → furniture → object), enabling compositional reasoning for navigation and manipulation.
- **Learning_Training** → Model industrial device behavior with occlusion-aware ray casting to infer hidden memory states from observable signals.
  - related fix: Synthetic depth image generation with self-occlusion-aware ray casting and noise-aware modeling calibrated from real depth sensor characteristics.

## Patch

```diff
--- ics_mem_collect.before.py
+++ ics_mem_collect.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Industrial control system devices lack APIs for programmatic memory access, forcing manual probing for anomalies or malware.

+# Fix    : Develop custom APIs (e.g., for GE D20MX) and JTAG interfaces to enable automated memory collection and analysis.

+# Avoid  : Manual memory probing without APIs

```
