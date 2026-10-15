"""SDK-only negative context signer; never an Agent tool or valid admission.

The daemon selects this fixture only for a SHA-pinned wrong_run_id test in an
isolated SIM. It deliberately signs an invalid run context with its own fixture
key so the real sink must reject identity even after successful authentication.
Original Body, source facts, action request and execution proof pins stay intact.
"""

import hashlib
import hmac
import json
import time

from simulation_execution_fault_policy import parse_governed_before_dispatch_fault_policy
from simulation_execution_proof import _canonical
from simulation_navigation_supervision import SdkNavigationExecutionSupervision


class RejectedRunContextSupervision(SdkNavigationExecutionSupervision):
    def __init__(self, *, operator_fault_raw, operator_fault_sha256, **kwargs):
        binding, facts = kwargs["binding"], kwargs["facts"]
        policy = parse_governed_before_dispatch_fault_policy(
            operator_fault_raw,
            expected_sha256=operator_fault_sha256,
            run_id=binding.run_id,
            body_snapshot_hash=binding.body_snapshot_hash,
            world_sha256=facts.enrollment["world_sha256"],
            action_id=kwargs["allowed_action_ids"][0],
        )
        if policy["case"] != "wrong_run_id":
            raise ValueError("explicit wrong-run operator fixture policy required")
        self._operator_fault_sha256 = operator_fault_sha256
        self._wrong_run_injected = False
        super().__init__(**kwargs)

    def _write_private(self, name, raw):
        if name == "admission.json":
            if self._wrong_run_injected:
                raise ValueError("never retry a rejected wrong-run context")
            envelope = json.loads(raw)
            original = envelope["payload"]
            if original["run_id"] != self.binding.run_id:
                raise ValueError("original SDK context must bind the actual run before injection")
            invalid = {**original, "run_id": self.binding.run_id + "-wrong-run"}
            envelope = {
                "payload": invalid,
                "hmac_sha256": hmac.new(self._key, _canonical(invalid), hashlib.sha256).hexdigest(),
            }
            changed = _canonical(envelope)
            self._record(
                {
                    "event": "OPERATOR_WRONG_RUN_CONTEXT_INJECTED",
                    "policy_sha256": self._operator_fault_sha256,
                    "original_context_sha256": hashlib.sha256(raw).hexdigest(),
                    "actual_rejected_context": envelope,
                    "actual_rejected_context_sha256": hashlib.sha256(changed).hexdigest(),
                    "monotonic_ns": time.monotonic_ns(),
                }
            )
            raw = changed
            self._wrong_run_injected = True
        return super()._write_private(name, raw)
