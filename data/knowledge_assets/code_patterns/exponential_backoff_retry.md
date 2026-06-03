---
pattern_id: exponential_backoff_retry
safety_label: Communication_Timeout
applicable_symptoms: [exponential_backoff_retry]
domain: Systems_Compute
source: curated
---

# Network/RPC timeout cascades cause request storms after a partial outage

**Domain**: `Systems_Compute`
**Safety label**: `Communication_Timeout`

## Fix

Wrap network calls with exponential backoff (base 0.5 s, factor 2, jitter ±30 %), cap retries at 5, and add a circuit-breaker that opens when error rate > 50 % over the last 20 calls.

## Anti-pattern

Tight `while True: retry()` loops — these turn a transient blip into a thundering herd.

## Cross-domain analogies (curated)

- **Control_Locomotion** → Same idea as PID gain scheduling: tighten effort when the system responds, back off when it stalls.
  - related fix: Lower the retry rate when failure rate climbs, just as a controller lowers gain in unstable regions.

## Patch

```diff
--- exponential_backoff_retry.before.py+++ exponential_backoff_retry.after.py@@ -1,5 +1,7 @@-while True:
+for attempt in range(MAX_ATTEMPTS):
     try:
         return rpc.call()
     except Timeout:
-        continue                    # tight retry
+        delay = BASE * (2 ** attempt) * random.uniform(0.7, 1.3)
+        time.sleep(min(delay, MAX_BACKOFF))
+raise RpcUnreachable()

```
