# Know / How v2 core integration

Status: implemented in the 1.2.0 working tree.

## Ownership boundary

| Domain | Owns | Must not own |
| --- | --- | --- |
| `rosclaw-know` | external sources, immutable snapshots, source-linked wiki, knowledge units, retrieval indexes, Reference Packs, knowledge-governance feedback | robot experience, action authorization, physical success claims |
| `rosclaw-how` | current-context interpretation and cited advisory output | a retrieval index, Memory records, ROS actions, safety authority |
| Core `rosclaw` | optional transports, context projection, tool exposure, bounded orchestration, events and health | Know indexing algorithms or How recommendation algorithms |
| Memory | robot experience and receipts from prior operation | external-world knowledge corpus |
| Practice | sandbox evaluation and knowledge-use feedback | rewriting source provenance or bypassing normal execution policy |

The data flow is deliberately one-way at the cognition boundary:

```text
external source -> Know snapshot/wiki/unit -> ReferencePackV2
                                               |
Body + Software + Runtime + labelled Memory ---+-> HowAdviceBundleV2
                                                        |
                                                        +-> Native Agent (advisory)
                                                        +-> KnowledgeUsageFeedbackV1

physical commands -> normal ROSClaw policy/safety/daemon path only
```

No raw trajectory, video, sensor stream, secret, Memory store handle or action
authorization is accepted by the v2 context and event adapters.

## Runtime modes

- `disabled` is the rollback-safe default. Core loads neither optional package and makes no network connection.
- `service` uses the versioned HTTP APIs. Core never starts or supervises the external processes implicitly.
- `inprocess` imports `rosclaw-know` and `rosclaw-how` lazily. Only Know receives the explicitly configured Know store; How receives a Reference Pack client.

`KnowledgeServiceManager` degrades missing packages, URLs and unavailable services into explicit health state. It does not fall back to Memory as a knowledge source. The older in-core Know/How modules remain available only when v2 is disabled.

Environment controls:

- `ROSCLAW_KNOWLEDGE_MODE` for both `RuntimeConfig` and the standalone v2 CLI adapter (`disabled`, `service`, `inprocess`).
- `ROSCLAW_KNOW_URL`, `ROSCLAW_HOW_V2_URL`, API-key variables and `ROSCLAW_KNOWLEDGE_TIMEOUT` for service mode.
- `ROSCLAW_KNOW_STORE_MODE` and `ROSCLAW_KNOW_SEEKDB_PATH` for in-process Know.

## Agent and MCP surface

The registered tools are intentionally small:

- `rosclaw_know_research`: bounded read-only research.
- `rosclaw_know_build_reference_pack`: evidence-cited retrieval.
- `rosclaw_know_open_reference_pack`: progressive disclosure by opaque ID.
- `rosclaw_how_advice`: DISCOVER, CONSULT, DIAGNOSE or CATALYZE advice.

Every tool is S0/read-only. `rosclaw_how_advice` is additionally marked advisory. None can create an `ActionEnvelope`, call a robot driver or grant approval.

## Feedback and observability

Only allowlisted lifecycle events containing identifiers, counts, versions and statuses cross the EventBus adapter. Feedback records whether a unit was presented, opened or useful; a receipt or Practice reference is only a reference. Feedback cannot claim that a physical action succeeded and cannot carry raw Memory content.

Dashboard status includes v2 mode and component health. A How or Know failure therefore remains observable without changing estop, safety policy, daemon or driver behavior.
