---
pattern_id: pattern_sayconav_utilizing_large_language_models_for_adaptive_collaboration_in_decentral
applicable_symptoms: [sayconav_utilizing_large_language_models_for_adaptive_collaboration_in_decentral]
domain: Planning_Decision
---

# Decentralized multi-robot navigation fails to adapt to dynamic obstacles and heterogeneous robot capabilities in unknown environments.

**Domain**: `Planning_Decision`

## Fix

Three-layer LLM-based architecture: top-level LLM generates collaborative strategies, mid-level LLM produces step-by-step action plans per robot, bottom-level maps plans to motor commands; uses local 3D scene graphs from RGBD images.

## Anti-pattern

Centralized planning or fixed coordination protocols that cannot handle heterogeneity or dynamic changes.

## Cross-domain analogies

- **Perception_Vision** → Use cross-modal alignment to correlate robot capabilities with obstacle dynamics via contrastive pretraining.
  - related fix: Cross-modal alignment pretraining using contrastive or attention-based losses to align visual object features with language tokens.
- **Learning_Training** → Use occlusion-aware and noise-calibrated sensor models to generate heterogeneous robot capability profiles for robust decentralized planning.
  - related fix: Synthetic depth image generation with self-occlusion-aware ray casting and noise-aware modeling calibrated from real depth sensor characteristics.
- **Control_Locomotion** → Incorporate real-time local environment maps into decentralized policies to dynamically adapt navigation and collision avoidance.
  - related fix: Incorporate terrain information (depth images or elevation maps) into the control policy to dynamically adapt gait, foothold placement, and body posture.

## Patch

```diff
--- sayconav_utilizing_large_language_models_for_adaptive_collaboration_in_decentral.before.py
+++ sayconav_utilizing_large_language_models_for_adaptive_collaboration_in_decentral.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Decentralized multi-robot navigation fails to adapt to dynamic obstacles and heterogeneous robot capabilities in unknown environments.

+# Fix    : Three-layer LLM-based architecture: top-level LLM generates collaborative strategies, mid-level LLM produces step-by-step action plans per robot, bottom-level maps plans to motor commands; uses local 3D scene graphs from RGBD images.

+# Avoid  : Centralized planning or fixed coordination protocols that cannot handle heterogeneity or dynamic changes.

```
