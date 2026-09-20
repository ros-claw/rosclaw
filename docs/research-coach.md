# Research Coach contracts

This is a task-neutral research layer, not another simulator, learner, runtime
executor, or promotion gate. A downstream task supplies authenticated evidence;
Growth suggests the next research direction without authorizing execution.

## Contracts

- `ResearchHypothesis` binds a falsifiable claim, prediction, fixed conditions,
  and evaluation contract before comparison.
- `ExperimentFamily` separates a display identity from a pinned mechanism
  contract. Renaming the family does not reset a plateau for that mechanism.
- `ResearchBudget` reuses `DreamBudget`. `research_budget_available` checks a
  proposed resource allocation against recorded usage and experiment count.
  It is a pure preflight, not a concurrent reservation. Actual leases and
  durable execution accounting remain the existing `DreamScheduler`'s job.
- `ResearchCampaign` binds the four distinct train/development/retention/sealed
  identities. Unmaterialized banks may explicitly remain `None` during planning;
  do not invent a commitment to make the constructor pass. Sealed consumption
  and the team-integration route require all four bindings. Different hashes
  alone do not prove the datasets are disjoint.
- `PlateauSignal` requires three consecutive, known blind non-improvements
  under the same mechanism and evaluator. Unknown blind results do not count;
  reused bank commitments and duplicate experiment evidence are rejected.
  Once observed in supplied history, a later claimed gain cannot erase STOP.
- `PivotDecision` is a diagnosis. Its training and promotion authority are
  always false, including when it recommends team integration.

## Routing

STOP_FAMILY takes precedence over further research in that family. Retention
failure routes to stability/plasticity work. Failed oracle assay controls route
to `SEARCH_OR_ASSAY`: first investigate whether the declared optimizer and
measurement setup can recover known feasible examples. An unknown assay leaves
a failed oracle at `NEED_EVIDENCE`; only explicitly passing controls allow that
negative result to route to action-space or environment diagnosis.
Successful oracle with failed imitation
routes to representation/DAgger; successful imitation with failed closed-loop
performance routes to credit/on-policy work. A development/blind gap routes to
coverage/curriculum. Team integration requires all individual-stage judgments,
including blind and retention, to be explicitly true.

Missing observations remain unknown. They are not converted into failed skills
or successful gates. A bounded failed oracle is not proof of physical
impossibility. The caller must validate raw evidence and authenticated event
ordering before using these pure contracts.

`ResearchObservation.oracle_assay_pass` is an upstream, evidence-backed judgment,
not a score inferred here. The downstream experiment must predeclare the control
population, matching conditions, search budget and acceptance rule, and verify
the original receipts. A hand-picked successful trajectory does not certify a
random searcher's sensitivity, and a failed search is not an impossibility
proof. Explicitly failed controls also block later-stage routing even if a
caller supplies an apparently passing oracle score. STOP and retention failure
retain their precedence. This field adds no optimizer, task-specific threshold,
training permission or automatic activation path.

## One-use sealed bookkeeping

`SealedUseLedger` reuses the continual service's append-only, hash-chained,
fsync-backed event storage. A single-writer lock and in-process mutex prevent
duplicate consumption. Candidate, evaluator, campaign and separate development
and retention receipt hashes are recorded **before** a dataset or score is
revealed. A crash after consumption burns that bank; there is no reset method.
Ambiguous writes latch failure; recovery reads committed events rather than
retrying blindly.

This is **not a security boundary or completed sealed evaluation service**:

- It does not authenticate upstream receipt contents or grant file access.
- It does not make a learner-accessible dataset private.
- Same-UID users can remove a state directory; the ledger does not defend
  against an administrator erasing history.
- A separate evaluator must own the data, verify development and retention
  results, consume the commitment, execute the pinned exam, and publish a
  bounded receipt. No dataset is opened by these modules.

State belongs outside source checkouts. Tests use invented commitments in
temporary directories; no real task holdout is consumed during testing.

## Current integration ceiling

The generic contracts and durable one-use bookkeeping are implemented and
tested. They do not yet drive an end-to-end task campaign or automatically stop
external training jobs. Existing Dream, Growth, safety and promotion gates are
not bypassed. Soccer-specific thresholds, states, scenes, policy classes and
data stay in the downstream application.
