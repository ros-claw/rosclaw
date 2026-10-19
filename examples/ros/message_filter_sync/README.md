# MF02 message-filter synchronization: exact Native source and portable pure tests

This experiment preserves three ROSClaw Native-authored files exactly. The portable replay runs 14 stdlib `unittest` methods against synthetic metadata and source-premise controls. It imports no ROS SDK and does not start DDS, a model, a container, a robot, or hardware. The `minimal_fixture.tar` contains only the two unmodified Python files; `exact_native_source3.tar` also contains the unmodified Native document. Readable `.py.txt` files are exact byte copies.

The scenario's two QoS premises now explicitly select liveliness and duration fields. The existing oracle, case matrix, consumption, graph, ownership, cleanup, work20/whole30, 256-row and 1MiB caps remain unchanged. WARNING remains a rejection; phase E retains its required reliability ERROR. Native selected the values; this publication does not provide a new control algorithm.

Root separately verified pure14 and one fresh Native-owned isolated DDS execution: one real operation, watcher continuation, terminal status/output, two genuine artifacts, 103 metadata rows, normal STOP/quit, and owned-container removal. An independent stdlib decoder matched all 44 recorded PointStamped capture fields (17 PUB, 17 RX, 10 PAIR), rejected eight corruptions, and accepted arbitrary alignment padding. These are `serialize_message` bytes retained from message objects, not DDS network packet captures. The public bounded JSON records these facts without raw rows, process IDs, session identifiers, credentials, private paths or bindings. The original image and runtime are not included, so these tests do not publicly reproduce that live DDS run.

**Limits:** this is not a full DDS certificate, cryptographic authentication, full SDK-inventory proof, robot/hardware approval, physical result, or general framework capability certification. The observer was same-process retained-object instrumentation. Changing multiple explicit QoS fields and seeing the warning disappear does not establish its unique cause. The original WARNING run remains failed. The source-author post-edit coverage failure and the external closed-review evaluator failure remain failures; later independent component reviews are separate evidence, not rescoring those workflows.

The original Native document deliberately retains its author-time SOURCE_ONLY/NOT_ADMITTED/NOT_RUN statements and local provenance labels. It is an exact historical source artifact, not current authority or a replacement for the bounded evidence JSON. No Native source, old assertion, test method, or document is rewritten for publication.

## Portable replay

Use Python 3.11 or 3.12; no third-party package is required. Choose a fresh absolute extraction path whose parent already exists:

```sh
python -I -B examples/ros/message_filter_sync/verify_publication.py --extract-root /absolute/new-mf02-fixture
python -I -B examples/ros/message_filter_sync/replay_pure.py --fixture-root /absolute/new-mf02-fixture/minimal_fixture
```

The verifier bounds and validates every archive member before creating files, rejects links/duplicates/unsafe paths, and extracts leaves read-only. The replay rejects ROS/SDK imports and network/subprocess/signaling effects, requires 14 tests with no failures/errors/skips, and verifies both source files remain unchanged. Its JSON result reports only the local pure replay; it cannot grant DDS authority. The dedicated CI matrix is Python 3.11/3.12, stdlib only, five minutes per job. Root must freshly extract/replay before publishing this candidate; preparation itself did not execute tests.

## Root replay after preparation

The final public helpers passed a fresh bounded replay on CPython 3.11 and 3.12: all 14 ordinary tests on each interpreter, no failures, errors, skips or denied effects, and source hashes unchanged. This is 14 distinct tests and 28 executions, with no new DDS or SDK calls. `ROOT_LOCAL_PORTABLE_DUAL_REPLAY.json` records the result. Earlier preparation-time pending labels remain historical. Root changed only the external helpers' import formatting, set-comprehension spelling and unused loop name; their remaining AST matches the original helpers and Native3 bytes remain exact. Remote CI is still pending.
