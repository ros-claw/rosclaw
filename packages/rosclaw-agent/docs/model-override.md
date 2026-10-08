# Explicit provider/model selection

All four public surfaces accept a **pair**:

```text
rosclaw chat --provider openai-codex --model gpt-6.1-sol
rosclaw chat --continue --provider openai-codex --model gpt-6.1-sol
rosclaw continue --provider openai-codex --model gpt-6.1-sol
rosclaw resume <approved ID/title> --provider openai-codex --model gpt-6.1-sol
```

Changing global defaults to ChatGPT does **not** migrate a recorded Kimi branch.
Ordinary continue/resume retain SDK recorded-branch selection behavior (including
its existing unavailable-model diagnostics). Only an explicit pair requests an
override. It creates a new private fork, never edits the source model selection.
Use the new session identity printed at startup for subsequent continuation.

Native model authority validates the exact physical provider/model and available
credentials offline before the Python launcher creates its kernel/mission.
Partial pairs, unknown targets and unavailable auth fail with typed errors. There
is no Python model catalog, silent provider fallback, automatic login, OAuth
refresh, model catalog network fetch or quota retry during selection. Configure
credentials separately. A quota error is not permission to retry: select an
approved source with the explicit command above instead.

The source file is read-only input, including legacy sessions. Its SDK migration
occurs only in the new fork; no mutation-capable open of the original is used.
A live or unknown source writer is refused: first obtain a separately approved
stable frozen snapshot. No PID injection or manipulation occurs. The new fork
retains the selected branch, strips extension custom authority and discarded
branches, records parent provenance, and selects the target before SDK runtime
restoration. Recorded thinking (high included) is preserved when supported;
unsupported effort is an error, never silently clamped. Default effort is used
only if there is no recorded thinking selection.

The new UUID enters the normal Native coordinator transaction and gets its own
writer, mission/lease/catalog scope. Tasks are lazy: selection without a prompt
does not prove a Task binding. Old task/current artifact references and signed
authority are not restored as current claims. Historical tool records are
context, **not executable commands**. Source model changes and history do not
trigger provider requests during preparation.

Startup plainly reports actual selected provider/model/effort, source identity,
and new restored session identity. Normal UI chrome reports the current model.
Global settings/auth/models are not rewritten by explicit selection; runtime
settings (quiet UI, hidden thinking, retries disabled) are in-memory only.

## Disclosure boundary

Selecting an override is not blanket authorization to disclose original history.
Use only a source whose selected branch has been separately approved for the
target provider. This source-only implementation/test episode authorizes only
synthetic fixtures. It does not access original6157, credentials/thinking from
original sessions, migrate actual original history, perform paid summarization,
or certify historical review. Actual source migration requires later Root
approval. There is no automatic context compaction as part of offline selection.

## Verification

Native-owned Python tests exercise all parser surfaces, paired flags and exact
launcher authority/error handling with process/network guards. Node tests use
real offline ModelRuntime and SDK restoration, read-only current/legacy files,
branching and ownership, and isolated mirror identities. The exact public check
and regression also exercise compiled private CLI selection and existing writer,
MainExit, close/event, atomic/stale/residual controls plus all Node tests. They
persist opaque full logs with hashes and compact diagnostic summaries. A final
SOURCE_PASS requires both current check and regression passing on the final
seven-source hashes; otherwise deliver truthful SOURCE_FAIL.
