# Public and operator output scope

A native SOURCE episode permitted reports and logs in prose, but its supervisor
allowed only `native/test_stdout.txt` and `native/test_stderr.txt`. The agent
wrote `native/source_tests.log`, and the supervisor stopped the episode after
model work. That old incomplete episode and its independent test failure stay
unchanged.

New packet builders can call `validate_episode_write_scope` before creating a
clock, credentials profile or model worker. Publish an immutable JSON list of
all writable relative paths, including source, metadata, reports and logs.
Compare it with independently admitted operator policy and reject overlap with
protected inputs. Exact order is irrelevant; duplicate, ambiguous, traversing
and absolute paths are rejected. Free-form prose is not a machine path list.

This helper is opt-in. It does not change existing runner policy, grant motion
or filesystem authority, constrain OS writes, or replace runtime source hashes,
symlink checks and file monitoring. Callers must not derive independent policy
from the candidate's proposed output list.
