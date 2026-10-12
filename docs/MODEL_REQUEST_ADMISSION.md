# Model-request admission

## Contract and limits

Admission is opt-in local reservation enforcement, not a provider billing contract. Before dispatch, the local ledger synchronously reserves request capacity; inclusive terminal usage settles that reservation. UNKNOWN usage fails closed and blocks further dispatch. `strict_billing` is rejected. A provider request can exceed its local reservation: neither a hard token/billing guarantee nor a legal provider max-output promise is made. The output limit applies to the local ledger; it is not forwarded as a provider output cap.

Opt-in settings disable both retry levels (agent retry = 0 and `provider.maxRetries` = 0) and disable cache warming. Tests verify settings/options, not an actual provider retry loop or network behavior. The inert counting `modelRuntime.streamSimple` fixture is not actual network, billing, or provider retry proof. A 100 ms eligible-warming observation does not establish full scheduling/TTL behavior.

## Dispatch and lifecycle coverage

A01–A08 exercise actual SDK fixtures through the shared production installer, with an inert underlying stream. They do not prove a full CLI lifecycle or provider network execution.

Public `compact()` is explicitly unsupported: the production installer refuses it before SDK abort/wait/auth. The stream wrapper checks `isCompacting`. The branch-summary seam is wrapped but lacks a dedicated positive regression.

The ledger is allocated outside the session factory and installed after the compaction observer for each session. A05 covers spend retention across `newSession`; resume/fork are not individually tested.

Result promises are observed asynchronously. Without a terminal event or an already-available result, failure becomes UNKNOWN. Asynchronous result-only streams may therefore fail closed; universal result-only support is not promised.

