# Local message-filter SDK diagnostic

ROSClaw Native Agent authored this finite ROS 2 `message_filters` example and its tests. It exercises ordinary synchronization, the strict slop boundary, queue eviction, out-of-order arrival, duplicate timestamps and caller-owned epoch reset using generated PointStamped messages.

The repaired runtime emits `LOCAL_MESSAGE_FILTER_EVIDENCE_V2`. Each feed and callback retains its own actual CDR bytes and hash; source identity and decoded fields are checked separately. Valid alignment padding may differ across serializations. The original byte-equality defect and failed runs remain recorded. The unchanged author document includes its historical V1 text followed by the V2 repair appendix.

Local verification collected 32 pure methods and 17 subtests without importing ROS. A separate ROSClaw Native run completed one genuine managed SDK operation and two artifact deliveries: 19 feeds, six callbacks, seven graphs and 31 serialization/deserialization pairs. Its enclosing Native supervision closed successfully.

The independent verifier decoded all 31 retained representations and checked the six-case queue matrix. Its saved component result is verified, but printing its full module inventory exceeded the supervisor's 8 KiB stdout cap. That original decoder parent remains `STOP_UNCONFIRMED / SUPERVISED_CHILD_STDIO_CAP`, even though the child exited 0 and captured cleanup found no remaining child or signals. Publication did not replay that SDK purpose to change its result. See `runtime_verification_20261017.json` for these separate outcomes.

This is generated local SDK evidence. It provides no DDS, sensor accuracy, hardware, navigation or general framework certification. Pure tests intentionally use synthetic, non-CDR metadata and cannot establish SDK success.

## Reproduce the pure tests

Install Python 3.11 or 3.12 and `pytest==9.1.1`, then from the repository root choose fresh temporary paths:

```sh
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -I -B examples/ros/message_filter_sdk/verify_publication.py --extract-root /tmp/mf01-fresh
python -I -B -m pytest -q -c /dev/null --noconftest -p no:cacheprovider --basetemp /tmp/mf01-pytest-fresh /tmp/mf01-fresh/minimal_fixture/test_message_filter_scenario.py
```

Use pytest import to collect all 32 methods: the preserved original `unittest.main` guard precedes the appended V2 tests. The archive has exactly three regular members: module, test, and `inputs/INPUT_CASES.json`. Verification checks every published byte and validates the bounded archive before exclusive extraction. Running the SDK requires separately reviewed ROS dependencies and process supervision; these pure commands do not run it.
