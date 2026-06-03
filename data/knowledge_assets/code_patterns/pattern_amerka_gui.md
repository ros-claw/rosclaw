---
pattern_id: pattern_amerka_gui
applicable_symptoms: [amerka_gui]
domain: Perception_Vision
---

# Unsecured IoT/ICS devices exposed on the internet are easily discoverable and exploitable, posing critical infrastructure risk.

**Domain**: `Perception_Vision`

## Fix

Use Shodan, BinaryEdge, and WHOISXMLAPI to passively scan for ICS/IoT devices, geolocate them via Google Maps and device indicators, and report to CERT.

## Anti-pattern

Relying solely on Shodan without additional passive intelligence sources or geolocation indicators.

## Cross-domain analogies

- **Planning_Decision** → Use graph-based exploration and semantic priors to prioritize patching high-risk exposed devices.
  - related fix: Scenario-oriented object navigation with graph-based exploration: build a semantic graph of explored regions, use a high-level policy to select frontier nodes based on object-context priors, and a low-level policy to navigate to chosen nodes.
- **Learning_Training** → Use diverse real-world telemetry data to automatically generate synthetic attack scenarios for training intrusion detection systems.
  - related fix: Use driving videos to automatically generate navigation instructions and action labels for data augmentation.
- **Control_Locomotion** → Use dead-time dominant tuning to anchor exposure thresholds for IoT/ICS devices.
  - related fix: Use dead-time dominant tuning rules (e.g., Cohen-Coon, Lambda tuning) and apply gain-anchored tuning for integrating processes.

## Patch

```diff
--- amerka_gui.before.py
+++ amerka_gui.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Unsecured IoT/ICS devices exposed on the internet are easily discoverable and exploitable, posing critical infrastructure risk.

+# Fix    : Use Shodan, BinaryEdge, and WHOISXMLAPI to passively scan for ICS/IoT devices, geolocate them via Google Maps and device indicators, and report to CERT.

+# Avoid  : Relying solely on Shodan without additional passive intelligence sources or geolocation indicators.

```
