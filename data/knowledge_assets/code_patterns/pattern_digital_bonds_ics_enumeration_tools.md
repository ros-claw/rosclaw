---
pattern_id: pattern_digital_bonds_ics_enumeration_tools
applicable_symptoms: [digital_bonds_ics_enumeration_tools]
domain: Systems_Compute
---

# ICS devices are fragile and can crash or respond unexpectedly to unexpected traffic during enumeration

**Domain**: `Systems_Compute`

## Fix

Use legitimate protocol or application commands (e.g., BACnet, Modbus, S7) to discover and enumerate devices without exploiting or crashing them

## Anti-pattern

Using exploit or crash-based techniques for ICS device discovery

## Cross-domain analogies

- **Perception_Vision** → Use predictive reacquisition to re-enumerate ICS devices after unexpected traffic.
  - related fix: Active mitigation strategies such as multi-sensor fusion or predictive reacquisition to handle blind zones
- **Planning_Decision** → Unified online memory construction and joint optimization for robust traffic handling.
  - related fix: Unified framework that constructs spatial memory online from RGB-D frames, jointly optimizes object grounding and frontier selection, and learns end-to-end trajectories via pre-training on large-scale data.
- **Learning_Training** → Use synthetic traffic generation to augment enumeration robustness testing.
  - related fix: Use large-scale synthetic data generation (e.g., ScaleVLN with 4M+ instructions) to augment training.

## Patch

```diff
--- digital_bonds_ics_enumeration_tools.before.py
+++ digital_bonds_ics_enumeration_tools.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: ICS devices are fragile and can crash or respond unexpectedly to unexpected traffic during enumeration

+# Fix    : Use legitimate protocol or application commands (e.g., BACnet, Modbus, S7) to discover and enumerate devices without exploiting or crashing them

+# Avoid  : Using exploit or crash-based techniques for ICS device discovery

```
