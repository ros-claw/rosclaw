# Reference Humble image: rosapi action type compatibility

Status: **SOURCE_AUTHORED_NOT_RUNTIME_VALIDATED**. Installer and offline tests
are authored, not executed by Native authoring. Native authoring did use
PI/provider calls; this is not a zero-provider session. Standalone installer
and runtime checks, builds, pure tests, ROS, Docker, physics and hardware tests
were NOT_RUN by Native authoring. Root independent checks are separate evidence,
not author-session execution or a current-revision Ruff/test/Docker PASS.

## Confirmed failure, not a client workaround

The supplied PR690 baseline is `65cff31949abcd20646de5fc555dfe93edb37a82`.
Historical PR690 Docker result remains **3 PASS / 1 FAIL**. Captured logs show
`rosapi/proxy.py:369` calling `get_action_names_and_types(node=_node)` through
`ros2action/api/__init__.py:61`. That wrapper calls
`node.get_action_names_and_types()`, which the Humble `Rosapi` node lacks.
The resulting AttributeError killed rosapi_node **pid 42, exit 1**. The supplied
causal trace explicitly excludes provider authentication as the root cause.
No prior failure, elapsed cost or paid probe is erased by this source change;
no additional paid execution was requested here. Numerical historical cost
figures are not supplied in the copied causal trace/LAST_CHECK (the latter is
empty), so they are unknown, not zero or reconstructed.

The fix changes exactly one import in the image's pinned installed proxy:

```diff
-from ros2action.api import get_action_names_and_types
+from rclpy.action import get_action_names_and_types
```

The original `get_action_type` implementation, callback call with `node=`,
empty-string semantics, first type selection, service wire format and license
remain byte-for-byte unchanged. No action type is synthesized. No global
rclpy/Node monkeypatch, dependency downgrade, process restart workaround,
compiler relaxation, client fallback or ROS2-profile bypass is introduced.

## API evidence and its boundary

Captured installed `rclpy/action/__init__.py` exports the standalone function
from `.graph` (SHA256
`55d0e5f72cd905d0320a418d01c9dac102ec6ba14ccb942a9bd9b790cc63641b`).
Official primary source for rclpy tag **3.3.21** defines
`get_action_names_and_types(node: Node)`, accepts keyword `node`, and uses
`with node.handle: return node.handle.get_action_names_and_types()`:

- <https://raw.githubusercontent.com/ros2/rclpy/3.3.21/rclpy/rclpy/action/graph.py>
  SHA256 `72b22224fd9c1fc8f0bec9c0d7743a29f260ab28ed39b3fd0a1fce8e3d9d33a2`.
- Official Humble ros2cli wrapper:
  <https://raw.githubusercontent.com/ros2/ros2cli/humble/ros2action/ros2action/api/__init__.py>
  SHA256 `562458954c01149bad44dc05e0ae7fa01e25e8cb5b0544044798f49d78dfe74c`.

These official files are **not actual installed file bodies**. Actual installed
`rclpy/action/graph.py` and full `ros2action/api/__init__.py` were NOT_CAPTURED.
API selection rests on official tag/branch source, actual installed export and
actual crash stack, as permitted by the supplied API review. This is not
runtime proof that the rebuilt container works.

## Fixed build-only installation

The Dockerfile adds only adjacent COPY and RUN instructions before its original
entrypoint COPY. All original lines/order, APT instructions, entrypoint and
compose settings are preserved. There is no extra network installation.
The CLI has no arguments, target overrides, version overrides or environment
escape hatches. It operates on this fixed relative path under the build root:

`opt/ros/humble/local/lib/python3.10/dist-packages/rosapi/proxy.py`

Before any write it reads the image's `/var/lib/dpkg/status` and requires exact
installed status and package versions:

| Package | Exact version |
| --- | --- |
| ros-humble-rosapi | 2.0.8-1jammy.20260915.220619 |
| ros-humble-rclpy | 3.3.21-1jammy.20260724.022150 |

The complete original proxy must hash to
`7042b78b804ec0b69a597dda48007d679448ce3b85c89d87c4a6c7b4bca208e7`
(13,935 bytes), with exactly the expected import and callback layout.
Unsupported versions, hashes, layouts, duplicate package records, missing files,
symlink components, hardlinked targets and nonregular targets fail clearly.
Idempotence accepts only the exact one-line patched bytes: reversing that one
import must reconstruct the full pinned original SHA. An already patched file
with any unrelated change is refused, and package gating still applies.

Replacement uses a same-directory temporary file and atomic replace, preserving
mode and ownership, with cleanup on failure and a pre-replace identity/content
recheck. Rejected inputs are never intentionally written. This is for a private
build filesystem, **not a security boundary against concurrent hostile writers**;
it is not a live-host repair utility. The callable image-root parameter exists
for owned temporary offline fixtures; the CLI always uses `/` inside the image.
Never apply this installer to a host/remote ROS installation.

APT remains unchanged and can resolve newer packages. Such an image must fail
closed at the compatibility step rather than silently broaden the pin. A future
package revision requires separate source capture/review and a reviewed pin
update, not disabling the check.

## Authored offline tests (NOT_RUN)

The new test module embeds the complete captured proxy with its BSD license,
checks length/full SHA, and copies it into an owned pytest temporary image root.
It invokes the actual installer function and CLI logic (CLI root redirected to
the owned fixture), never a production ROS/provider API. Importing the actual
original/patched proxy is isolated with exact inert dependency module names.
Its `get_action_type` is executed from those actual bytes, not replaced with a
handwritten equivalent.

Authored assertions cover:

- original wrapper fails on a node with no legacy bound method;
- the one-line patch binds the standalone API and uses an inert node handle;
- actual proxy returns the graph stub's turtlesim type, unknown/empty types
  return empty string, and multiple types retain first-type behavior;
- action-name metadata in the inert case retains `/turtle1/rotate_absolute`
  (not an end-to-end compiler or production provenance proof);
- exact byte delta, ownership/mode preservation and no-write idempotence;
- unsupported versions, status, missing/duplicate packages, hash/import/layout
  drift and altered already-patched files refuse without changing proxy bytes;
- symlink, hardlink, parent-link, status-link, directory and missing-file refusal;
- atomic replace failure leaves original bytes and cleans temporary files;
- CLI return codes, fixed root and rejection of override arguments.

Inert API semantics do **not** prove DDS discovery, real package import success,
rosapi service survival, service serialization, compiler behavior or Docker.
No old tests were changed or weakened; no new skip/skipif/xfail was added.
Client transport/resolver/discovery/compiler and all other tracked source remain
outside this change's writable scope.

## Deferred verification / STOP

Root review must inspect the full security/functional diff and run the original
155 pure tests plus this module in the separately authorized validation phase.
Formatting, static checking and those tests are **NOT_RUN**, not PASS.
Root's later Docker phase must build the image, retain the original four smoke
tests unchanged (reachability, doctor, discovery, compiler), exercise action-type
lookup, and observe rosapi PID/liveness and traceback absence across requests,
including unknown-action behavior. Original compiler assertions must remain.
Those four tests alone do not substitute for rosapi-survival observation.

There is no runtime completion claim or new execution authority. The evidence
report records final source hashes obtained from native file registration;
delivery references those four source artifacts plus the report, and delivery
itself is the sixth progress artifact. **STOP at source authoring.**
