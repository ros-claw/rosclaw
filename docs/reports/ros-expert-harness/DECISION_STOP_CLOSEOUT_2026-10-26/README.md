# Decision calibration and failure-stop closeout — 2026-10-26

The input-preparation package from PR #763 is closed. This is a separate decision batch and an original-only audit of the seed-202 failure; it creates no simulation World or navigation goal. Baseline `5744f44a64a8da32c9454ccc4d4cae3a659e4d5e`; raw captures were produced by their separately recorded sources. Results and exact original references are in [RESULTS.json](RESULTS.json).

| Completion object | Evidence and limit |
| --- | --- |
| Conditional decision inputs | Previously completed; original archive and report remain unchanged |
| Runtime TF restoration constraint review | Previously recorded in stationary Worlds; no cold-start or new motion claim |
| New model decision calibration | Attempted, stopped on client timeout; no terminal model decision, no accepted proposal |
| Seed-202 historical task | FAILED on covariance violation before TF injection; not attributable to a model or TF repair |
| Stop after that failure | Independently confirmed from existing post-failure observations; original receipt remains unchanged |
| Covariance root cause / seed-202 motion qualification | Not established |
| Model-to-new-World physical loop | NOT_RUN / NO_GO |

## What the original failure proves

At UTC `2026-10-24T00:03:48.059411Z`, sealed model 249 records AMCL x variance `0.0006431202223559561`, above the unchanged `0.000625` limit by approximately 2.90%. Model 248 is within the limit; the next model, 250, is within it again. Only one of 257 original model snapshots captures the violation. This brackets a sampled excursion; it does not establish continuous duration, an AMCL root cause or actual loss of control. It occurs during the second, yaw-goal task before any TF fault injection.

The old executor exception branch wrote `physical_stop_verified=false` as a literal before final cleanup. It did not run the normal independent stop observation path. That field means the failed receipt did not prove stopping; it does not measure continued movement.

The linked original-only audit confirms stopped state after the failure: witness lines 529–531 at wall UTC `00:03:48.212068` through `00:03:48.310188`, SIM time `27.22` through `27.32`, record a revoked SIM lease and zero planar/angular speed, with complete fresh independent contacts/identity evidence. The original immutable Body stop limits are approximately `0.002 m/s` and `0.00785 rad/s`; the required original 0.1 s holding interval is met. The full ECM chain replays continuously, and 21 later original ECM samples through SIM `29.4` have unchanged pose. The observed sink becomes terminal; no accepted nonzero control or lease renewal is recorded after its final nonzero command and subsequent revoke.

Task FAILED and independently confirmed subsequent stop coexist. No old receipt is edited or promoted to COMPLETED. Exact cancellation acknowledgement and heartbeat-cessation timestamps are unavailable; they are not invented. Wall UTC, kernel monotonic nanoseconds and ROS/Gazebo SIM seconds remain separate: witness rows preserve explicit observed links rather than an inferred clock conversion. Pose-estimation error samples are paired at their original different SIM timestamps and include sample age; they are not synchronous ground-truth error measurements.

## Minimal exception-cleanup correction

Navigation exceptions latch stopping and stop heartbeats, then reuse the existing SIM emergency shutdown and bounded independent stop observation. Lease enable/renewal remains fenced; a failed executor cannot rearm for another action. The failure artifact and returned FAILED result preserve cancellation/cleanup errors and distinguish `physical_stop_status=CONFIRMED` from `NOT_CONFIRMED`, alongside the measured boolean. Failure evidence remains REQUESTED / NOT_VERIFIED even when stopping is confirmed. The existing final shutdown attempt remains in place. No motion budget is extended, and the stop check does not require failed localization health to recover. Missing/stale physics, failed revocation or absent stop proof remains unconfirmed. Software tests cover those paths, a cancel disconnection with a confirmed stop, immutable artifacts and refusal of a subsequent action. This is software verification, not a fresh physical acceptance.

## New decision batch and its stopping result

The new batch freezes the same model `gpt-6.1-sol`, medium effort, two environment seeds 101/202, seven planned categories, at most four decisions / 300 s per episode, 120 s per client call, 48,000 prompt bytes, 64,000 observed input tokens per episode and at most 56 client invocations. Memory, native tools, SDK and physical validation providers are disabled. Prompt, schema, host code, inputs, budget and stop rules are frozen before dispatch. Cases expose one sealed observation and existing expert candidates, without phase/gold labels or restoration outcomes. Candidate use would demonstrate selection, not independent mechanism discovery.

Ten of fourteen planned cells have registered inputs: P1/N1/N6 historical originals, N2 an explicitly registered composition of incompatible original run bindings, and N3 an explicit original-clock-expiry control. These derived controls are not natural captured failures. Four N4/N5 cells lack independently admitted original authority-conflict/consumer-ambiguity instances in the selected registry. They are retained as missing; software fixtures are not relabelled as native captures. Registration explicitly allows cell-wise evaluation of available evidence, and explicitly prevents a partial matrix from qualifying physical entry.

The first seed-101 P1 episode dispatched four client invocations. Three completed turns inspect the task, request expansion using a case `fact_` reference where the API requires a previously returned common-view `acievidence_` reference (host rejection), then read a transform page. The fourth client invocation reaches its 120 s timeout and is terminated; no final model report or proposal exists. Under the registered provider/client-failure rule, the batch stops without retries, prompt changes, seed substitution or further model calls. Thirteen other planned episodes are unattempted; four also have the independently recorded missing-original condition. No positive or negative case is counted as a correct decision. The rejected read request is not an unsafe accepted proposal and is not correct abstention. The timeout and incomplete matrix do not prove inability to understand TF repair.

Known usage is the subtotal from three completed turns; the timed-out invocation reports no usage, so total tokens/cost are unknown. Native tool execution, Memory calls, new Worlds and navigation goals are all zero. Raw model events remain local; this directory publishes only brief public summaries and hashes, not private reasoning. The full model calibration and seed-202 current-stack motion qualification have not passed. A new model physical loop is therefore NOT_RUN / NO_GO; no further preparation or physical rerun is used to conceal that result.

Earlier preparation receipts (seven COMPLETED and one FAILED) belong to different versions; they are accounting records, not eight same-distribution trials and not a navigation success-rate estimate.
