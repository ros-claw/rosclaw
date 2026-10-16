# Opt-in typed current-state runner

This development runner does not replace the original immutable G5 study.
`aci_runner.run_typed_episode` reconstructs a bounded current-state frame and a
workbook of original facts actually read, rather than replaying an ever-growing
transcript. Brief model notes remain explicitly untrusted. Whole records are
omitted with counts when the workbook is full; originals remain in the trace.
Operator artifact locators are opaque in the model view to avoid factory-label
leakage. Public tasks must contain no allocated fault-case label or answer.

All arms share original evidence reads, typed staging and explicit SDK validation.
B adds Expert constraints and predefined investigation. C adds automatically
queried, provenance-admitted Memory, filtered again for the current violated
signature. Same-case exclusion remains unchanged. Provenance admission alone is
not applicability, model use, causal benefit or permission. A known-case pilot
may expose no applicable mechanisms; report this instead of weakening admission.

The provider schema has an object root, nested `anyOf`, closed objects, all fields
required, and nullable declared defaults. The host removes null only for declared
optional cursor/offset/limit fields and validates the original strict call types.
The projection follows the official Structured Outputs contract:
https://developers.openai.com/api/docs/guides/structured-outputs
The pinned CLI/model route and actual event/usage artifacts remain authoritative;
hidden revision, numeric temperature and cost remain unknown.

Budgets remain at most 300 seconds including capture, 12 model/tool turns, 48,000
prompt bytes and 64,000 provider-reported input tokens. Native tool items fail
closed. Only an accepted new SDK validation within budget counts as success.
Finish is report-only. Capture, context/Memory, frame, model, host tool, probe and
SDK validation wall times are measured separately; missing provider usage is not
estimated. Original client events, decisions, results and verification refs are
retained. Synthetic test clients are not model evaluation evidence.

`aci_probe.SdkReadOnlyProbe` uses the existing SDK operator in at most one fresh
disposable replica World and at most two preparation Worlds total. It verifies
zero navigation requests and empty command/lease sinks. Its sealed capture is
fresh at capture time, not current live Self, Body or full L1, and does not rebind
the original case. Unsupported authority/source checks stay unknown. Body or
source admission still requires the independent full SDK validation. No Agent
ROS/DDS connection, Runtime, driver, native shell or hardware authority is added.

Before any real model call, independently preregister the frozen source, schema,
pinned client hash, route, budget and scope. Compatibility smoke tests are not
pilot episodes. A subsequent randomized pilot and deterministic-only baseline
need their own protocol, immutable allocation and original failure retention.
The previous 54 episodes must never be relabeled or pooled into a new study.
