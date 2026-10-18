"""Preserve original native bytes before slow derived SDK reviews.

This is an SDK-local artifact sink, never a ROS client or an admission source.
Readers still independently verify and evaluate the original data and its age.
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path


def persist_native_capture(directory, raw, *, index, received_monotonic_ns, publish_pointer):
    if (
        type(raw) is not bytes
        or not 0 < len(raw) <= 5_000_000
        or type(index) is not int
        or not 0 <= index <= 999999
        or type(received_monotonic_ns) is not int
        or received_monotonic_ns <= 0
        or type(publish_pointer) is not bool
    ):
        raise ValueError("bounded original bytes, index and original receive clock required")
    native = json.loads(raw)
    stamp = datetime.fromisoformat(native["captured_at"])
    if stamp.tzinfo is None:
        raise ValueError("aware original native capture time required")
    directory = Path(directory)
    target = directory / f"original-native-{index:03}.json"
    with target.open("xb") as stream:
        stream.write(raw)
    if publish_pointer:
        pointer = {
            "file": target.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "captured_at": native["captured_at"],
            "received_monotonic_ns": received_monotonic_ns,
        }
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(pointer, stream)
        try:
            os.replace(temporary, directory / "live-native-pointer.json")
        finally:
            temporary.unlink(missing_ok=True)
    return native
