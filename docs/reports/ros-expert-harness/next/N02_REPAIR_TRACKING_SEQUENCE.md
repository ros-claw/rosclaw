# Opt-in repair tracking sequence candidate

Source50a2's preregistered108002 development pairs passed independent canonical
coverage, phase, actual corner dispatch and physical-stop replay. Both nine-pose
boundary actions succeeded. Waffle used52.3184m/529.52SIMs versus its own baseline
67.6464m/749.92SIMs:22.66%/29.39% reductions. Burger used61.2678m/810.04SIMs versus
91.6344m/1114.88SIMs:33.14%/27.34%. Neither met the joint30% target. These are
single training pairs, not series medians or P0 acceptance. Historical negative
pairs and the halted original formal series remain retained.

Independent phase totals, including transition/wait samples, are
Waffle93.84/73.20/362.48SIMs and Burger105.48/68.36/636.20SIMs for main/boundary/repair.
Waffle needed34 repair requests/43 targets and had3 zero-gain requests. Burger
needed47 single-target requests, with1 zero-gain request and1 canceled45-second
timeout. Its observed stationary turning223.20SIMs was35.08% of repair duration.
Two Burger requests estimated near5 seconds actually took44–45 seconds and
accumulated35–46 radians of rotation. These observations motivate an experiment;
they do not establish a controller cause or predict a new route's performance.
Waffle's nine two-target repair actions all succeeded on this seed.

## Explicit hypothesis

`pose_aware_robust_tracking_sequence` is opt-in and restricted to the existing
Waffle/Burger one-cell inset fixture presets. It retains bounded two-target
robust footprint search and actual measured missed-cell replanning. A sequence
estimate pays the existing dispatch overhead once per continuous Nav2 action,
rather than once per intermediate target. Drive/turn estimates and search
budget remain unchanged. The audit records this distinct cost model and its
predicted dispatch cost. It remains a static prediction, not observed duration.

Two-target repair dispatches use a separate source-bound
`/evidence/repair-tracking-through-poses.xml`. Its100mm intermediate pruning
matches the existing controller lookahead; the final goal checker and global
precise25mm BT remain unchanged. The existing validated installed-tree
transformation preserves planner rate, recovery and control nodes. Boundary BT
bytes remain unchanged. A single target or budget fallback uses the existing
NavigateToPose path. The mission/45-second repair deadlines, goal-count/retry
bounds, Body/velocity/collision/lease guards and fixed coverage denominator stay.

Every intermediate target remains a request, not a proven arrival. Pruning
grants no cleaning credit; actual independent brush observations determine
remaining holes and final98% acceptance. The hypothesis may trade tracking
progress against additional missed cells and therefore still needs new paired
physical tests. No claim of improvement is made here.

## Binding and acceptance

The generated experiment and preregistered protocol must agree on explicit
tracking enablement,100mm radius, BT SHA256, shared-overhead flag and profile's
selected repair strategy before World startup. Missing/mismatched/incorrectly
typed metadata fails closed. Legacy strategies reject tracking metadata.
The daemon checks strategy/experiment consistency before initializing its
runtime. Its executor verifies bounded non-symlink BT bytes at startup and
again before each two-target dispatch; host absolute paths are not sent to Nav2.

Preparation retained three expected red failures before implementation.
Uncommitted scoped behavioral/ROS/type checks are preparation only. Final-source
SDK generation and startup refusal probes, Native, full strict regression,
original remote workflows, fresh SIM faults and fresh preregistered paired
coverage/phase/stop replay remain separate required gates. No new physical seed
is registered by this document and no formal holdout is resumed or replaced.
