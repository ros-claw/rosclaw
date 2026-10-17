# Frozen-clock diagnostic RPC example

ROSClaw Native Agent authored the exact module, pure regression suite and source
notes in this directory. This is a standalone ROS 2 diagnostic GetParameters RPC
example and transcript oracle. It is independent of the Parameter Atomic12 oracle.

Three actual RPCs exercise one context/client/executor while both ROS clocks stay
at zero: FAST succeeds, LATE reaches a local timeout while its real future remains
pending, and RECOVERY succeeds before the LATE response is observed and quarantined.
A late response cannot revive the timed-out terminal. Each terminal records the
actual monotonic samples immediately before and after os.fsync returns, checked
against its original deadline. This does not demonstrate server-side cancellation.

The independent 2026-10-17 isolated Jazzy run accepted 48 transcript rows, three
actual RPCs and 25 samples showing both ROS clocks at 0 ns. The genuine Native
orchestration completed with status and output, two delivered artifacts and STOP;
the owned container was absent after cleanup. See runtime_verification_20261017.json
for the bounded, redacted result. No raw runtime session is included.

The closed pure suite passed 25 methods and 49 subtests (124 reports in Root's
observer). CI runs pytest 9.1.1 on Python 3.11 and 3.12; report counts depend on the
runner. Tests fabricate transcripts and inject clock/sync functions. They do not
launch ROS, a model or containers and cannot independently establish live execution.

From a Python environment containing pytest 9.1.1, run only this pure file:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -I -B -m pytest -q -c /dev/null --noconftest -p no:cacheprovider examples/ros/frozen_clock_rpc/test_frozen_clock_rpc_scenario.py

minimal_fixture.tar contains only the module and test; readable_source3.tar adds
the exact author's notes. manifest.json pins all published sibling files and exact
archive membership. verify_publication.py checks regular-only allowlisted tar
members, bounded bytes and hashes before writing them into a new extraction root;
it neither imports the Native module nor executes tests. The dedicated workflow
verifies all three copies are byte-identical, then runs only the fresh extracted
minimal fixture once per Python version.

The source notes preserve their original source-only status at authoring time;
the later independent runtime acceptance is recorded separately. Repair author
episode 72986 exhausted its output allowance before final STOP and remains a
workflow FAIL. Independent software acceptance and the new successful orchestration
do not rewrite that failure. The original 20 test semantics remain preserved.

This proves a bounded diagnostic service protocol component and actual tmpfs fsync
return timing. It does not prove physical-media power-loss durability, host artifact
deadlines, robot motion, full CAP80 coverage, or all physical-AI capabilities of
PI 1.3. The opt-in runtime entry requires separately supervised isolation; CI does
not invoke it. No new package installation or live runtime authority is conferred.
