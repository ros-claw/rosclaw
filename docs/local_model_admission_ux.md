# Local model admission UX

## Outcomes

`MODEL_REQUEST_ALLOWANCE_EXHAUSTED` is a local reservation refusal, not provider billing quota, auth failure, network outage, or evidence of a hang. `MODEL_REQUEST_USAGE_UNKNOWN` means unresolved local usage prevents further dispatch. The existing `MODEL_REQUEST_STREAM_USAGE_UNKNOWN` terminal maps to the latter presentation code; its original raw activity remains available. Authoritative cancellation and output-length terminal semantics take precedence.

Both local codes survive the Harness error event with `retryable: false` under unchanged authority. The normal SDK error/end/idle path is retained; idle does not mean successful delivery. Classification and deterministic presentation make no extra model request. Genuine provider quota, auth, rate and network errors retain their classifications and recovery cards.

Chinese product text explains that further model calls are blocked, existing *saved* progress can be inspected/retained, unfinished delivery remains unfinished, and not every edit is guaranteed saved. Further work requires external newly authorized budget/purpose. UNKNOWN additionally requires independent accounting disposition: merely raising allowance does not clear it. No new authorization or existing recovery control is claimed. `/model`, resending and compaction do not remove same-runtime restrictions. There is no `/budget` command or automatic retry/reset/refill/new session.

## UI and scope

The existing card gate deduplicates local errors without appending its provider-oriented `/model` rescue suffix or task-recoverable claim. Its existing display resets on success/model switch remain unchanged; display reset is not accounting recovery. The extension shows the local refusal as an error and clears only the provider-paused diagnosis, not admission state. No provider-health measurement is implied. The activity entry retains the existing schema and stable code (including the legacy `rosclaw.provider_error` entry name).

Only classification/presentation, the Harness error union, and the adapter's terminal mapping change. Admission/installer, arithmetic, limits/reserves, gate ownership/lifetime, retry/cachewarming policy, summary refusal, UNKNOWN settlement, persistence and session teardown are untouched. This is local admission, not a verified provider-cost bound. Historical external author-budget workflow failures remain failures, not product/provider-hang evidence.

## Source-authored regressions — UNEXECUTED

`local-model-admission-ux.test.ts` exercises actual classifier, card gate and adapter: both local codes, provider controls, cancellation/length precedence, dedupe, safe progress language and typed error/idle outcomes. It captures the real extension callback without emitting unrelated lifecycle/mirror hooks and checks notifications, activity, no provider pause and no explanation-turn dispatch.

Finite offline SDK cases install the actual admission wrapper, use only synthetic streams and an offline model catalog, then verify error-before-idle, nonretryable local codes, no false assistant completion, unchanged dispatch count and refusal after same-gate model switching. Temporary directories are owned fixtures and cleaned up; there is no ROS/hardware/provider/network call. No subprocess/print-mode fixture is included, so nonzero CLI print behavior is not covered here.

All ten fixed inputs were read. No process, compilation, SDK run or test was executed by the author. These tests and source require independent review, type checking and bounded execution; no PASS is claimed. Existing reference tests were not edited. No recoverable read/edit errors occurred.
