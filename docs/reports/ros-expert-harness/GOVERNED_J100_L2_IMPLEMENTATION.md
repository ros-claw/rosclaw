# Governed generic SIM navigation: implementation in progress

P03 merged normally in PR #678 at `f361423b040312777676f94ce81b92a21bc7ddf8`.
Its three source and four main original workflows passed; twelve fresh actual-main
Worlds passed independent raw recomputation and read-only artifact inspection.
P03 is passive observation only. It does not authorize L2 motion.

This next implementation adds a daemon-configured supplementary observation
guard at admission, lease enable/renewal, goal dispatch, execution polling and
completion. Loss sets the executor's permanent stopping state; lease revocation
and cancellation remain available on failure. The configured guard never replaces
canonical MCP request_action, Session, Lease, Permit, executor or Receipt checks.

SDK execution supervision proofs bind Body/hash, typed workspace manifest hash,
World SHA, run, source epoch, daemon generation and action. HMAC proofs have a
maximum 250 ms lifetime, monotonic sequence/source cursor checks and permanent
rejection on expiry, tampering, replay, identity change or clock rollback. Keys
are fixture-private; proofs are supplementary observations, never permits.

The SDK final velocity gate requires both a current daemon proof and independent
current ROS/ECM observations. It rejects nonfinite, nonplanar or excessive Twist
values against all unchanged compiled Body limits. Lease loss and revocation
are terminal for that action. The gate is a pure state machine: an actual SDK ROS
sink and its zero-command/watchdog cleanup still need to be composed and accepted.

The independent current-facts collector rebuilds the system model from original
Native bytes, checks SHA/time/Body/run/immutable URDF and World bytes, and polls
the bounded source reader. Execution preparation strengthens the passive policy
to <=500 ms ROS data age. An explicit read-only probe cadence option supports
100 ms snapshots; its default remains one second. A faster snapshot is not itself
evidence of fresh sensor, lifecycle, parameter, localization or map observations.

Contract tests are synthetic software checks. Current-facts World tests are
read-only tests. Neither closes L2. Still required: explicit SDK admission,
actual independently guarded unique velocity sink, daemon proof production,
three fresh J100 seeds with position and yaw actions through canonical MCP,
all ten execution fault cases, raw independent physical verification and normal
source/main PR integration. G4 and G5 remain unaccepted.
