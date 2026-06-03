---
pattern_id: gradient_clipping
safety_label: Numerical_Instability
applicable_symptoms: [gradient_clipping]
domain: Learning_Training
source: curated
---

# NaN/Inf in loss or weights after a step explodes the gradient

**Domain**: `Learning_Training`
**Safety label**: `Numerical_Instability`

## Fix

Apply `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`. If NaN still appears, halve the learning rate and verify the loss does not feed `log(<=0)` or divide by zero.

## Anti-pattern

Catching the NaN after it surfaces and zeroing it out — the bad direction has already corrupted the optimizer's moment buffers (Adam's m and v). Restart training instead of patching.

## Cross-domain analogies (curated)

- **Control_Locomotion** → Gradient clipping is the SGD analogue of an output limiter on a controller — bound the magnitude of every actuation.
  - related fix: Set `max_norm` analogous to `tau_max`: derived from a physical/training-stability limit, not guessed.

## Patch

```diff
--- gradient_clipping.before.py+++ gradient_clipping.after.py@@ -1,2 +1,3 @@ loss.backward()
-optimizer.step()                # no clipping
+torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
+optimizer.step()

```
