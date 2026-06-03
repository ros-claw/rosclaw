---
pattern_id: pattern_s7scan
applicable_symptoms: [s7scan]
domain: Systems_Compute
---

# Siemens PLCs in industrial networks are not enumerated for security auditing, leaving unknown firmware versions and protection settings that may be vulnerable.

**Domain**: `Systems_Compute`

## Fix

Use s7scan tool that sends S7 'Read SZL' requests over TCP/IP or LLC to identify PLCs and retrieve firmware, hardware, protection settings, and network configuration.

## Anti-pattern

Older plcscan tool lacked LLC support, protection configuration display, and robust TSAP checking.

## Cross-domain analogies

- **Perception_Vision** → Hierarchical decomposition: coarse network scans enumerate PLCs, then fine-grained audits verify firmware and settings.
  - related fix: Use a coarse-to-fine pyramid (e.g., U-Net or FPN) that downsamples to capture coarse layout and upsamples to recover fine details, then fuse or sequentially feed multi-scale features.
- **Planning_Decision** → Use a hierarchical 3D scene graph to encode PLC network topology, firmware, and access controls for zero-shot security auditing.
  - related fix: Use a multi-modal 3D scene graph that encodes object categories, spatial relations, and hierarchical structure, combined with a large language model for zero-shot goal reasoning and path planning.
- **Learning_Training** → Use counterfactual network path simulations to contrast with actual PLC enumerations, highlighting critical security gaps.
  - related fix: Use counterfactual trajectory demonstrations: generate and analyze alternative paths that could have been taken, then contrast them with expert trajectories to focus learning on the most critical features for navigation cost inference.

## Patch

```diff
--- s7scan.before.py
+++ s7scan.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Siemens PLCs in industrial networks are not enumerated for security auditing, leaving unknown firmware versions and protection settings that may be vulnerable.

+# Fix    : Use s7scan tool that sends S7 'Read SZL' requests over TCP/IP or LLC to identify PLCs and retrieve firmware, hardware, protection settings, and network configuration.

+# Avoid  : Older plcscan tool lacked LLC support, protection configuration display, and robust TSAP checking.

```
