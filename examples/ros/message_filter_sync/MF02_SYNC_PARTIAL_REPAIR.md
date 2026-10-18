# MF02 generated two-topic DDS synchronization

Source-only repair proposal; tests and processes NOT_RUN. Runtime NOT_ADMITTED.
Historical source rejection and all earlier failures remain unchanged. The reported
noContext ABI check is not Context/Node/DDS or synchronization evidence.

## Interfaces and evidence
`validate(rows, complete=True, contract=..., receipt=...)` returns only
METADATA_ONLY. The external contract binds exact case, lowercase 32-hex nonce
(including digit-leading values), image-qualified GID byte length and framework
inventory. Endpoint GIDs retain integer bytes, not a guessed 48/32-hex width.
The synthetic test's three-byte contract is deliberately fictional, never an
image qualification. Missing image contract fails closed.

The final receipt contains the exact canonical JSONL hash and clock samples
before/after final fsync. COMPLETE in a file alone is provisional; final write,
fsync, close and receipt verification must all succeed inside whole30. Terminal
SYNC separately records actual fsync timing against the original phase deadline.
Root must independently authenticate receipt/capture provenance and outer270.

BOOT fixes work20/whole30. Every RX requires a correlated post-dispatch queue
snapshot before another publication/lifecycle step. Selection checks reject
replaced/consumed sources; the runtime snapshots actual ATS queues, not a model.
F disconnects ATS input connections after durable timeout, retains live DDS RX
journaling for LATE, then destroys subscriptions/publishers before G. Recovery
uses the same context but new phase/epoch. E requires probe/control RX, strong
compatibility/discovery and the original bounded observation window.
Partial mode preserves failed phase prefixes without fabricating CLOSE/G.
Cleanup attempts executor, node and context separately and records failures.

## Admission still required
Payload caps remain two topics/two publishers/three subscriptions. Node's
`/parameter_events` publisher is a separate proposed framework surface with
ParameterEvent type, count one, Node.__init__ origin and observed object identity.
No framework surface is authorized here. The v2 observer seam below associates
fresh runtime identities with the unchanged static contract externally; it never
adds fresh identities to that contract. Static metadata and `admitted=True` are
not signatures or authorization. Root must review installed
callback/serialize sources, closure/domain/owner-to-endpoint binding, independent
CDR decoding, framework constructor effects, process/container absence and clocks.
No runtime launcher is supplied; the CLI without contract/receipt cannot accept a
complete journal. This is fail-closed, not a ready-to-run admission wrapper.

### v2 binding and failure-prefix protocol (source proposal)
`framework_descriptor()` supplies the exact one-element structural signature:
ParameterEvent, `/parameter_events`, `Node.__init__`. This is a canonical descriptor,
NOT a cryptographic signature. The externally sealed contract contains this static
list only, never a future PID/address. `framework_link(entities, contract)` checks
that exact projection and hashes canonical contract, descriptor and dynamic list.
FRAMEWORK raw includes real retained Python object labels plus this association;
the final receipt repeats the association and binds the entire raw file hash.
Object labels are process-local correlation coordinates, not credentials.

`run_worker(binding, output, observe_framework=...)` requires a future trusted
wrapper-installed observer before SDK imports. After Node construction, the hook
receives the actual node, retained publisher tuple, copied entity projection and
immutable canonical contract string. It must independently verify constructor
origin, object identity (`is`/retained address in the observed process), owner and
exact inventory against its own installed profiler observations, persist an
externally authenticated observation bound to case/nonce/contract hash and process
incarnation, then return the exact association. Missing hook, None, unknown origin
or mismatching coordinates fail closed. Contract is copied, never extended or
re-signed; mutation of the caller contract during the hook is rejected.

The hook is a TRUST BOUNDARY, not an authentication implementation: ordinary
callers can fabricate hooks/metadata, but cannot thereby obtain admission. Root's
future wrapper must select/pin the observer independently of caller data and verify
the sealed contract/GO out of band before invocation. No signer or launcher is
implemented here. `validate` verifies association consistency only and returns
METADATA_ONLY even with a synthetically perfect receipt. Runtime acceptance must
fail closed without the independent authenticated observer record matched to raw,
receipt, process incarnation, image GID qualification, callback profile, CDR decoder
and whole-clock/absence evidence. The source marker is a provenance requirement,
not proof that the source was actually external. Payload caps are unchanged.

ENTER/ALLOCATING records the next phase before cutoff checks or graph allocation.
OPEN still means the full graph exists; CLOSE still requires actual terminal sync
and release. No partial allocation invents OPEN/CLOSE. The verifier tracks ENTER
independently from active/closed graphs: initialization failure uses A, active
failure uses its entered phase, and post-CLOSE/pre-OPEN failure uses the explicitly
entered next phase. Entry cannot skip/repeat phases or resume after ERROR/cleanup.
Cleanup failure remains PARTIAL with false release flags, not a fabricated success;
FINAL requires one cleanup attempt. Truncated prefixes may omit cleanup/FINAL.
No clock, release, per-RX queue, E wait or F late-isolation gate is relaxed.

Targeted synthetic controls resequence and remap all surviving references, assert
common/event schemas and payload/capture prerequisites, then demand the specific
missing-RX, per-RX-QUEUE or late-PAIR rejection. They are NOT_RUN metadata tests.
Historical 27522 and 44919 MODEL_REQUEST_ALLOWANCE_EXHAUSTED, original whole FAIL
and runtime nonacceptance remain unchanged; 44919 was not a normal STOP. Reported
Root pure7 acceptance is limited component evidence. Reported noContext ABI52B /
serialize1 / deserialize1 / zero Context/DDS is ABI-only, not this scenario DDS.
Actual runtime remains NOT_ADMITTED; future credentials require a separate purpose.

## Pure tests
The readable full A–G fixture is synthetic metadata, not actual DDS or valid CDR.
Mutation controls cover references, snapshots, probe/discovery, typing, identity,
hashes, epochs, cleanup, prefix status and deadlines. Existing unequal-raw/equal-
fields checks test only capture metadata, not legal CDR padding or decoding.
No test was executed by the author; independent review and execution are required
before source acceptance. No simulation or physical success is claimed.

## QoS premise source3 candidate (source only; NOT_RUN / NOT_DDS_PASS)

### Known, hypothesized, unknown
The supplied closed-facts summary reports 103 saved rows from the original frozen
source, with the first oracle mismatch at A / seq4 / DISCOVERY: WARNING/WARNING,
not the required OK/OK. That is a failed run, not a hung model. Normal STOP/quit
and container removal do not make that run pass. Original quota, observer and SDK
precheck failures remain; the external experimental-schema `pattern` delivery
rejection is separate and is not repaired by this candidate.

The supplied exact-image qos.py text shows nine constructor fields, their setters,
and get_c_qos_profile forwarding all nine to rmw_qos_profile_t. Omitted fields use
the predefined default dictionary. The matching qos_profiles.h default explicitly
uses SYSTEM_DEFAULT liveliness. The API documents that default/unknown policies
can produce WARNING. These are source facts; the export constructed no actual
QoSProfile and proves neither runtime defaults nor the unique WARNING cause.
The supplied source does not include Duration implementation or the compiled RMW
compatibility implementation. Actual conversion, middleware behavior and failure
causality remain unverified.

### Chosen explicit premise and consumption
Only the two existing direct profile RHS and necessary local imports change.
Both retain depth=8, KEEP_LAST, their original RELIABLE/BEST_EFFORT split, and
VOLATILE. Both now explicitly use AUTOMATIC liveliness, deadline/lifespan/lease
Duration(nanoseconds=60000000000), and avoid_ros_namespace_conventions=False.
This supplies every field consumed by the shown API, without a preset/default,
UNKNOWN, BEST_AVAILABLE, keyword splat or new SDK operation.

AUTOMATIC avoids introducing application-level manual liveliness assertions,
which the frozen worker does not perform. Equal offered/requested deadline and
lease remove a deliberate inequality; the reliable/best profiles differ only in
reliability. Finite positive 60 seconds avoids duration default/zero sentinel
ambiguity and exceeds the entire original whole30 window: samples should not
expire within this scenario, and no periodic 60-second deadline premise should
be needed within it. This is a conservative candidate choice, not a measured
middleware guarantee. It does not extend any work20/whole30 or phase clock,
256-row/1MiB bound, or authorization. False preserves ROS topic conventions.

Consumption remains the frozen path: reliable/best -> pubq -> create_publisher;
reliable -> Subscriber(qos_profile=...) for both ATS inputs; best -> phase-E
probe create_subscription. The exact same objects feed the three genuine
qos_check_compatible calls (pubq[0]/reliable, pubq[1]/reliable, best/best).
qos.py forwards durations through get_c_duration and policies to the native
profile before its native compatibility call. No wrapper changes its return.
The ordinary phases still require OK/OK; E still requires ERROR/OK/OK from its
intentional BEST_EFFORT publisher versus RELIABLE subscriber mismatch. WARNING
is never promoted to OK. The framework parameter-events profile is not changed.

### Pure controls and falsification
Appended tests parse the actual two constructor expressions into literal records,
not SDK objects. They check all nine arguments, the original four policies,
identical non-reliability fields, finite duration values beyond whole30, and False.
Negative mutations reject missing fields, SYSTEM_DEFAULT/UNKNOWN/BEST_AVAILABLE
policies and zero/negative durations. Synthetic journal controls still reject
WARNING and removal of E's ERROR; they are not RMW results or CDR/DDS evidence.
A source guard restores only the permitted RHS/import seams in memory and checks
the original full source SHA256, freezing every other byte, including consumption,
strict validate, matrix, graph, observer, cleanup and budgets. Existing test AST
and documentation byte prefixes are retained. Root must independently check these
prefixes, scope and effectguard; no local test outcome is claimed.

The hypothesis is falsified as a sufficient repair if the same-image real API
still returns WARNING for these fully explicit profiles, rejects Duration values,
or resolves/consumes different policies. An ERROR for reliable/reliable or
best/best also defeats the candidate. An OK for best/reliable defeats the required
phase-E negative control; it must not be hidden by changing the oracle. Even
successful compatibility checks alone would not prove live discovery, delivery,
ATS selection, CDR authenticity, observer provenance or release.

### Separate future checks needed (not authorized or performed here)
1. Root executes all old/new pure tests and independent source/effect review.
2. Under separate same-image SDK admission, construct the original and candidate
   profiles, record each effective field and native duration conversion, and save
   unmodified compatibility status AND reason for reliable/reliable,
   best/reliable and best/best. Compare original versus explicit premises; retain
   warning/error reasons, image/RMW identity and authenticated observations.
   Varying one premise in that separately scoped diagnostic can discriminate
   liveliness from other causes; no causal claim is made by this source patch.
3. A separately admitted fresh DDS purpose must observe actual endpoint policies
   (not only projected reliability), ownership and framework constructor effects,
   installed callback/serialize identity, independent CDR decoding and full A-G
   raw/queue/lifecycle receipts inside the unchanged clocks and caps. Match raw,
   final durability and external observer records to that process incarnation,
   and verify outer process/container absence. Old case/nonce/GO is not reusable.

Author used read/edit only. No Python, tests, SDK, DDS, Docker, simulation or other
process was run. The three edited files are a source-component candidate pending
independent verification, not acceptance, runtime permission or DDS_PASS.
