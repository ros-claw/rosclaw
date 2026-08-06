"""Fixture workers for process-stdio adapter tests.

Each is a small script speaking (or deliberately breaking) the
``rosclaw.worker_adapter.v1`` envelope. Written to tmp_path by tests.
"""

from __future__ import annotations

GOOD_WORKER = """
import json, sys, os

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()

def recv():
    return json.loads(sys.stdin.readline())

recv()  # handshake
send({"type": "ready", "protocol": os.environ.get("ROSCLAW_WORKER_PROTOCOL", "rosclaw.worker_adapter.v1")})
msg = recv()
order = msg["order"]
# Env scrub check: no API keys should be visible in our environment.
leaked = [k for k in os.environ if "KEY" in k or "SECRET" in k]
send({"type": "heartbeat", "progress_seq": 1, "checkpoint": "half"})
send({"type": "result", "result": {
    "schema_version": "rosclaw.work_result.v1",
    "work_order_id": order["work_order_id"],
    "worker_id": "%(worker_id)s",
    "lease_id": order["lease"]["lease_id"],
    "status": "COMPLETED",
    "summary": "analysis done; env leaked vars: " + ",".join(leaked),
    "artifacts": [{"schema_version": "rosclaw.result_artifact.v1",
                   "ref": "artifact://text/stdio-1", "media_type": "text/plain"}],
    "claims": [{"schema_version": "rosclaw.result_claim.v1",
                "claim": "produced analysis", "evidence_refs": ["artifact://text/stdio-1"]}]
}})
"""

GARBAGE_WORKER = """
import sys
sys.stdout.write("this is not json\\n")
sys.stdout.flush()
"""

WRONG_ORDER_WORKER = """
import json, sys
sys.stdout.write(json.dumps({"type": "result", "result": {}}) + "\\n")
sys.stdout.flush()
"""

SECRET_ENV_WORKER = """
import json, sys, os
def send(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()
def recv():
    return json.loads(sys.stdin.readline())
recv()
send({"type": "ready"})
msg = recv()
order = msg["order"]
creds = msg.get("credentials", {})
send({"type": "result", "result": {
    "schema_version": "rosclaw.work_result.v1",
    "work_order_id": order["work_order_id"],
    "worker_id": "%(worker_id)s",
    "lease_id": order["lease"]["lease_id"],
    "status": "COMPLETED",
    "summary": "credential seen: " + str(creds.get("token", "none")),
    "artifacts": [{"schema_version": "rosclaw.result_artifact.v1",
                   "ref": "artifact://text/x", "media_type": "text/plain"}],
    "claims": [{"schema_version": "rosclaw.result_claim.v1",
                "claim": "x", "evidence_refs": ["artifact://text/x"]}]
}})
"""

HANG_WORKER = """
import json, sys, time
def send(obj):
    sys.stdout.write(json.dumps(obj) + "\\n")
    sys.stdout.flush()
recv = lambda: json.loads(sys.stdin.readline())
recv()
send({"type": "ready"})
recv()
time.sleep(3600)
"""
