# RK06 new pilot protocol and entry review

The frozen G5 54-experiment negative study, its A/B/C definitions, and its Memory
admission stay unchanged. This module defines a separate D/A/B/C/E experiment.
It provides closed protocol validation, reproducible paired allocation and a
SHA-pinned entry review. It performs no model or physical execution.

| Arm | System available to the evaluation |
| --- | --- |
| D | Current deterministic mechanisms and application compiler; no LLM |
| A | Same frozen model and original fact access, generic engineering tools |
| B | Same model/facts plus OperationalGraph, constraints and typed ACI |
| C | B plus competing hypotheses and bounded active read-only probes |
| E | C plus separately audited cross-Body mechanism Memory |

A is an experimental same-model tool baseline. It is not the Codex product.
Every paired arm has identical original fact access. B/C/E tool capabilities
are separate protocol fields; historical G5 C does not become the new C.
All motion still requires the same canonical MCP/rosclawd boundary.

The protocol requires five task strata: known-mechanism sanity, a new root
cause, a multi-node chain, a new-Body/task combination and independent physical
results. Cases, evaluation episodes, provider settings, model identity, Memory
policy, harness source and benchmark implementation SHA must be frozen before
outcomes. Novel cases must be selected after the Harness freeze; existing
counterfactual development cases are not renamed unseen test cases.

The pilot fixes five cases and one seed, allocating 25 planned episodes.
Each model episode permits at most 12 decisions and 8 read-only probes.
Each episode permits at most one read-only probe World plus one independently
admitted validation World, with two canonical goals in the validation World.
The existing typed evaluator separately enforces one probe and one validation.
Known-mechanism first valid proposal must be within 3 decisions. Allocation creates no authority or actual episodes.
Changing the implementation, source, split or gate-review bytes invalidates the
entry proof. A passing entry review only permits a bounded pilot; it never
authorizes expanding to a formal multi-seed study.

Four independently audited, SHA-pinned reviews must bind the same source and
protocol: exact source/main delivery; navigation on a new Body selected after
an unchanged Core freeze; cross-Body Memory transfer with wrong-cause rejection;
and the post-freeze evaluation-case registry. Missing, corrupt, incompatible or
failed evidence yields NO_GO before any model call or new World.

| Metric | Required original evidence and unit |
| --- | --- |
| Valid hypotheses / falsified hypotheses | Original hypothesis IDs, supporting and contradicting probe results; count and ratio |
| Read-only probes per decision | Actual typed probe invocation/result events; count |
| First valid proposal | Original accepted staging proposal and decision ordinal; missing stays unknown |
| Context efficiency | Actual provider input/cached/output usage and valid proposals; never substitute zero for missing usage |
| Staging change amount | Exact before/after file bytes and allowed semantic diff; changed bytes/fields |
| New-World entry | Original SDK ownership, source, run and admission records; distinct World count |
| Physical success | Fresh L1, two canonical COMPLETED receipts, independent goal/stop/contact/source reviews |
| Correct abstention | Original missing driver/evidence/contradiction and absence of action authority |
| Human interventions / boundary bypass | Original operator and execution audit; count, bypass target0 |
| Cross-Body Memory benefit | Retrieval→context→selection→outcome trace plus paired C/E independent outcomes; a retrieved lesson alone is insufficient |

Entry NO_GO, any boundary bypass, exceeding budgets, regression below D on known
mechanisms, missing actual provider usage for efficiency claims, and failure to
form competing hypotheses plus a discriminating probe on the novel-cause pilot
prevent formal expansion. Record all failures and correct abstentions. Physical
readiness, proposal validity, model efficiency and task completion are separate
outcomes.

Current status: implementation/contract validation only. PR736 and PR737 exact
main delivery, the post-current-freeze positive new-Body gate and an independently
frozen novel-case/provider registry are incomplete. W200's actual covariance
failure and R100/DO150's original-XML abstentions remain negative results. There
is no new model pilot, study success, or causal Memory benefit claim.

Use `.venv/bin/python -m benchmarks.ros_expert_ablation.rsk_pilot_protocol
--protocol FILE --evidence FILE --root ROOT --output NEW_FILE` with
`PYTHONPATH=src:.`. The command only reviews original files and emits a plan and
NO_GO/conditional eligibility; it cannot start the pilot.

## Typed model-facing integration

`rsk_physical_aci.py` composes the existing PhysicalACI for the new B/C/E roles
without changing the older G5 A/B/C meanings. New E may query operator-audited
cross-Body Memory, insert abstract concepts into context and request a typed
current repair. No historical record payload, source, evaluator, permit or
budget can be supplied in model arguments. The current proposal is independently
recompiled through the existing staging whitelist. Validation remains a separate
explicit call through the existing host SDK evaluator; no evaluator means no
execution. B cannot attach an active probe evaluator, C cannot attach Memory,
and the old G5 Memory operation is disabled in this new facade.

The initial sourceff131f0b v1 was a file-only prototype with one total World and
no model-facing Memory facade. Its independent CLI smoke originals are retained.
V2 binds both protocol and ACI implementation SHA and explicitly budgets one
read-only probe World plus one validation World. No scientific pilot or model
call used the prototype; its old schema remains historical documentation only.

The new facade provides phase-bound `decision_schema()` and
`call_model_decision()` for an RSK client. The legacy G5 `run_typed_episode`
runner is unchanged and does not implement the new E arm. Provider invocation
and scientific pilot execution remain gated; interface contract tests and
operator-constructed replay decisions are not LLM results.
