---
pattern_id: sliding_window_kv_cache
safety_label: Memory_Exhaustion
applicable_symptoms: [sliding_window_kv_cache]
domain: Memory_Reasoning
source: curated
---

# Unbounded KV-cache growth during long-horizon LLM rollouts causes CUDA OOM

**Domain**: `Memory_Reasoning`
**Safety label**: `Memory_Exhaustion`

## Fix

Cap the per-layer KV tensor at a fixed window N (e.g. 256–512 tokens). On each forward, evict the oldest key/value rows. Keep an optional global-attention sink (the first M tokens) to preserve task context.

## Anti-pattern

Increasing `--gpu-memory-utilization` or moving to a larger GPU — this only buys one more batch before the same overflow returns at a longer trajectory.

## Cross-domain analogies (curated)

- **Control_Locomotion** → Like an anti-windup clamp on an integrator: keep the size of the accumulating state finite, however long the run.
  - related fix: Treat the KV-cache as the integral term of attention; bound it the same way a PID bounds the integrator.

## Patch

```diff
--- sliding_window_kv_cache.before.py+++ sliding_window_kv_cache.after.py@@ -1,3 +1,6 @@-k_cache.append(k_new)            # grows forever
-v_cache.append(v_new)
+k_cache = (k_cache + [k_new])[-W:]   # sliding window of size W
+v_cache = (v_cache + [v_new])[-W:]
+if attention_sink_tokens:
+    k_cache = sink_keys + k_cache    # keep the global sink
+    v_cache = sink_vals + v_cache
 attn = compute_attention(q, k_cache, v_cache)

```
