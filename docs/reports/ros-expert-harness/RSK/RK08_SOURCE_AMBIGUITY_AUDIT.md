# Source-consumer ambiguity falsification

Before adding an execution provider for `SOURCE_CONSUMER_CONTRACT`, adversarial
checks found that XML first-match selection could incorrectly establish the
static declared contract. Multiple SDF worlds, duplicate URDF/SDF links and
repeated friction coefficients could yield SATISFIED; alias variants could also
yield VIOLATED and a compatibility candidate despite unresolved input semantics.
On parent main `6dfc64d2`, the initial 25-case adversarial suite had 20 failures
and five already-correct abstentions. The original failed run is preserved.

The review now requires one SDF world, unique nonempty link identities, unique
nonempty SDF collision identities per link, and unambiguous surface/friction/ode
containers and mu/mu2/fdir1 fields. Ambiguous source correspondence yields UNKNOWN
and no candidate. Distinct named collisions and other named models remain valid.
This is a precondition check, not an implementation of an SDK's duplicate-field
parsing policy. The independent stdlib SDK adapter remains unchanged.

Acceptance includes adversarial tests and replay of four historical World
originals through the real CLI. Additional offline metamorphic checks duplicate
fields/identities or change only namespace prefixes on derived copies of those
originals. Expanded XML equality is checked for alias-only variants. An equivalent
alias can violate a declared literal lookup while preserving expanded XML.
Original files and historical physical results are never modified.

These checks establish bounded static diagnosis and safe abstention. They do not
establish a live backend's consumption, authentication of operator-declared
parser knowledge, autonomous discovery of an unfamiliar mechanism, Memory benefit,
or physical success. No new World, model call, ROS connection or hardware access
is part of this audit. SOURCE_CONSUMER_CONTRACT remains an operator source-review
proposal; no native provider or motion authority is added. Runtime, daemon,
controller, safety thresholds and old experiment outcomes remain unchanged.

RK08 is an internal evidence-batch identifier, not a new roadmap milestone.
