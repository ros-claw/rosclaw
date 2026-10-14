"""SDK SIM supervision proof contract; never a Session, Lease or Permit.

The isolated daemon signs short-lived observations after independently checking
current ROS/ECM facts. The SDK sink must ALSO check its own current observations
and canonical lease. This module alone cannot admit or execute an action.
Keys belong to isolated fixture processes, never model tools or public evidence.
"""

import hashlib
import hmac
import json
import re
from dataclasses import asdict, dataclass

MAX_PROOF_BYTES = 8192
MAX_PROOF_LIFETIME_NS = 250_000_000
MAX_OBSERVATION_AGE_NS = 500_000_000


def _hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _identity(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value) is not None


def _integer(value):
    return type(value) is int and 0 < value < 2**63


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate proof field")
        result[key] = value
    return result


@dataclass(frozen=True)
class ExecutionProofPins:
    body_id: str
    body_snapshot_hash: str
    manifest_hash: str
    run_id: str
    world_sha256: str
    observer_epoch: str
    daemon_instance_id: str
    action_id: str

    def __post_init__(self):
        if (
            any(not _hash(getattr(self, k)) for k in ("body_snapshot_hash", "world_sha256"))
            or type(self.manifest_hash) is not str
            or re.fullmatch(r"navbodyworkspace_[0-9a-f]{64}", self.manifest_hash) is None
            or any(
                not _identity(getattr(self, k))
                for k in ("body_id", "run_id", "daemon_instance_id", "action_id")
            )
            or type(self.observer_epoch) is not str
            or re.fullmatch(r"[0-9a-f]{32}", self.observer_epoch) is None
        ):
            raise ValueError("immutable Body/World/run/source/daemon/action pins required")


def _validate_payload(payload, pins, now_ns):
    required = set(asdict(pins)) | {
        "schema_version",
        "sequence",
        "source_sample",
        "health_sha256",
        "source_review_ns",
        "health_review_ns",
        "issued_ns",
        "expires_ns",
    }
    if type(payload) is not dict or set(payload) != required:
        raise ValueError("closed execution supervision proof required")
    if payload["schema_version"] != "rosclaw.sim_execution_supervision.v1":
        raise ValueError("unknown supervision proof schema")
    if any(payload[k] != v for k, v in asdict(pins).items()):
        raise ValueError("Body/World/run/source/daemon/action identity changed")
    if (
        not _integer(now_ns)
        or any(
            not _integer(payload[k])
            for k in (
                "sequence",
                "source_sample",
                "source_review_ns",
                "health_review_ns",
                "issued_ns",
                "expires_ns",
            )
        )
        or not _hash(payload["health_sha256"])
    ):
        raise ValueError("finite bounded proof observations and clock required")
    issued, expires = payload["issued_ns"], payload["expires_ns"]
    if not issued <= now_ns < expires or not 0 < expires - issued <= MAX_PROOF_LIFETIME_NS:
        raise ValueError("supervision proof expired or clock moved backwards")
    for name in ("source_review_ns", "health_review_ns"):
        if not 0 <= issued - payload[name] <= MAX_OBSERVATION_AGE_NS:
            raise ValueError("supervision observation stale or in the future")
        if now_ns - payload[name] > MAX_OBSERVATION_AGE_NS:
            raise ValueError("supervision observation expired")


def sign_execution_proof(payload, *, pins, key, now_ns):
    """Internal SDK/daemon helper, supplementary to canonical authorization.

    A signer is not an evidence collector. Integration must derive the payload
    from actual readers, not accept it from an action envelope or model output.
    """
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("private 256-bit fixture key required")
    _validate_payload(payload, pins, now_ns)
    signature = hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()
    return _canonical({"payload": payload, "hmac_sha256": signature})


class ExecutionProofVerifier:
    """Permanent rejection latch; recreate only with new operator admission.

    Re-reading the identical still-fresh proof is allowed within its lifetime.
    Sequence rollback, changed bytes at one sequence, and source cursor rollback
    are terminal. A recovered producer cannot resume an old admitted sink.
    """

    def __init__(self, *, pins, key):
        if not isinstance(pins, ExecutionProofPins) or type(key) is not bytes or len(key) != 32:
            raise ValueError("explicit immutable pins and private fixture key required")
        self.pins, self._key = pins, key
        self._sequence = self._sample = self._clock = 0
        self._digest = None
        self.failed_reason = None

    def verify(self, raw, *, now_ns):
        if self.failed_reason is not None:
            raise ValueError("supervision permanently latched: " + self.failed_reason)
        try:
            if type(raw) is not bytes or not 0 < len(raw) <= MAX_PROOF_BYTES:
                raise ValueError("missing or oversized supervision proof")
            envelope = json.loads(raw, object_pairs_hook=_unique_object)
            if type(envelope) is not dict or set(envelope) != {"payload", "hmac_sha256"}:
                raise ValueError("closed authenticated supervision envelope required")
            payload = envelope["payload"]
            _validate_payload(payload, self.pins, now_ns)
            expected = hmac.new(self._key, _canonical(payload), hashlib.sha256).hexdigest()
            if not _hash(envelope["hmac_sha256"]) or not hmac.compare_digest(
                envelope["hmac_sha256"], expected
            ):
                raise ValueError("supervision authentication failed")
            digest = hashlib.sha256(_canonical(payload)).hexdigest()
            if (
                now_ns < self._clock
                or payload["sequence"] < self._sequence
                or payload["source_sample"] < self._sample
                or (payload["sequence"] == self._sequence and digest != self._digest)
            ):
                raise ValueError("supervision clock, sequence or source cursor rolled back")
            self._sequence, self._sample = payload["sequence"], payload["source_sample"]
            self._clock, self._digest = now_ns, digest
            return dict(payload)
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
            self.failed_reason = str(exc)[:300]
            raise ValueError("supervision permanently latched: " + self.failed_reason) from exc
