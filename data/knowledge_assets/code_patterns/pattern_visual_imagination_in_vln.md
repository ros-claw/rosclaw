---
pattern_id: pattern_visual_imagination_in_vln
applicable_symptoms: [visual_imagination_in_vln]
domain: Perception_Vision
---

# VLN agent ignores landmark cues in long instructions, leading to poor grounding in unseen environments

**Domain**: `Perception_Vision`

## Fix

Generate synthetic images from landmark text descriptions via a text-to-image diffusion model, and train the agent with an auxiliary grounding loss that aligns instruction representations with imagination embeddings

## Anti-pattern

Relying solely on language understanding without visual imagination

## Cross-domain analogies

- **Planning_Decision** → Hierarchical decomposition: coarse landmark identification then fine-grained grounding.
  - related fix: Modular architecture with two-stage process: coarse path generation from language, then low-level controller for smooth trajectory following
- **Learning_Training** → Use video-only input with domain randomization to force landmark-agnostic path integration, eliminating instruction grounding gaps.
  - related fix: Use video-only input modality (no depth or map) combined with domain randomization to eliminate sensor fidelity and geometry transfer gaps
- **Control_Locomotion** → Train an end-to-end policy with domain-randomized instruction grounding via large-scale simulation.
  - related fix: Train a single end-to-end neural network policy via large-scale RL in simulation with domain randomization, mapping depth image directly to motor commands

## Patch

```diff
--- visual_imagination_in_vln.before.py
+++ visual_imagination_in_vln.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: VLN agent ignores landmark cues in long instructions, leading to poor grounding in unseen environments

+# Fix    : Generate synthetic images from landmark text descriptions via a text-to-image diffusion model, and train the agent with an auxiliary grounding loss that aligns instruction representations with imagination embeddings

+# Avoid  : Relying solely on language understanding without visual imagination

```
