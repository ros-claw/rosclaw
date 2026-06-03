---
pattern_id: pattern_llm_driven_contextual_analysis
applicable_symptoms: [llm_driven_contextual_analysis]
domain: Planning_Decision
---

# Navigation systems fail to make fine-grained, context-aware decisions in cluttered or semantically ambiguous environments, relying only on coarse path planning.

**Domain**: `Planning_Decision`

## Fix

Use LLM-driven contextual analysis as a fine-grained reasoning stage in a coarse-to-fine pipeline: a high-level planner generates a coarse trajectory, then an LLM interprets local scene semantics to suggest precise adjustments (e.g., foot placement, speed, orientation).

## Anti-pattern

Traditional cost-map-based planners that ignore semantic context and cannot handle ambiguous terrain or dynamic obstacles at a tactical level.

## Cross-domain analogies

- **Perception_Vision** → Fuse multimodal semantic and geometric cues into a unified occupancy grid for fine-grained, context-aware navigation decisions.
  - related fix: Propose a multimodal occupancy perception system that fuses vision, depth, and other sensor data into a unified occupancy representation for humanoid robots.
- **Learning_Training** → Train a privileged model with dense scene context, then distill it into a lightweight planner via guidance loss.
  - related fix: Privileged Information Guidance (PIG): train a diffusion policy with privileged depth and collision information during training, then distill into a student policy that uses only RGB observations via a guidance loss.
- **Control_Locomotion** → Closed-loop local replanning with real-time sensor feedback enables context-aware fine-grained decisions.
  - related fix: Closed-loop controller that reconciles a local metric map with high-level navigation commands, generating continuous local trajectories from monocular depth and traversability estimates in real-time.

## Patch

```diff
--- llm_driven_contextual_analysis.before.py
+++ llm_driven_contextual_analysis.after.py
@@ -1,2 +1,4 @@
-# --- BEFORE (vulnerable to the symptom below) ---

+# --- AFTER (ROSCLAW heuristic graft) ---

 # Symptom: Navigation systems fail to make fine-grained, context-aware decisions in cluttered or semantically ambiguous environments, relying only on coarse path planning.

+# Fix    : Use LLM-driven contextual analysis as a fine-grained reasoning stage in a coarse-to-fine pipeline: a high-level planner generates a coarse trajectory, then an LLM interprets local scene semantics to suggest precise adjustments (e.g., foot placement, speed, orientation).

+# Avoid  : Traditional cost-map-based planners that ignore semantic context and cannot handle ambiguous terrain or dynamic obstacles at a tactical level.

```
