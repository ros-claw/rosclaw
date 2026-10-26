# Action-scoped observation facts (local candidate, 2026-10-26)

Status: local candidate. No new runtime, Agent loop, safety authority or physical acceptance is introduced.

Compiled Body `provider_interfaces.ros_observation_contracts` binds a topic to its exact type, publisher, installed package version and, for conditional telemetry, the selected plugin parameter. Optionality is never inferred from a topic name or Boolean type. Native probe exposes installed package versions and existing current graph/parameters; this is installation evidence, not remote binary attestation.

Supported publication categories: periodic, command, static, event, conditional, optional. Required topics, explicit rates and critical sensor types cannot be downgraded. Missing/mismatched contracts retain conservative legacy behavior. Old auxiliary values remain UNKNOWN. Conditional contracts record execution phase from fresh Action status and lifecycle; lifecycle ACTIVE alone is not an executing goal. Independent controller failure diagnostics and native watchdogs remain active.

Full diagnostics retain all issues. Only Body-bound affected-capability scopes narrow readiness vetoes. Body, global hazards and the selected capability's observation requirements participate in its dependency fingerprint. Full audit context remains separately available.

Migration 040/041 adds optional per-capability hashes to existing context leases. Empty maps preserve legacy full-hash semantics. Authority-produced scopes are captured when issuing a lease; caller/model-provided scopes are not accepted. Identity, writer, mode, Body, revision, TTL, approval and exact grant checks remain in their existing authorities. Relevant facts are rechecked before dispatch; changes reject with NEEDS_REPLAN, without consuming a grant or replaying an action. The patch does not extend TTLs or add automatic action retries.

Qualification requires paired healthy/failure tests, clean installed-artifact checks, same-policy SDK/runtime physical controls, then complete H1 and held-out trials. Component regressions alone are not qualification. Critical controller/sensor stop and cancellation remain owned by existing daemon/native control, not the LLM.
