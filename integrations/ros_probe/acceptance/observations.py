"""Read completed auxiliary witness snapshots without racing the live writer.

This does not parse or repair canonical mission trajectories. A partial live
append may be ignored; a malformed completed record must never be skipped.
Consumers retain their independent freshness and completeness checks.
"""

import json


def latest_completed_observation(path):
    with path.open("rb") as stream:
        stream.seek(0, 2)
        offset = max(0, stream.tell() - 65536)
        stream.seek(offset)
        if offset:
            stream.readline()  # The bounded tail may begin inside an older record.
        data = stream.read()
    completed, separator, _unfinished = data.rpartition(b"\n")
    if not separator:
        raise ValueError("no completed independent observation")
    sample = json.loads(completed.rsplit(b"\n", 1)[-1])
    if not isinstance(sample, dict):
        raise ValueError("independent observation must be a JSON object")
    return sample
