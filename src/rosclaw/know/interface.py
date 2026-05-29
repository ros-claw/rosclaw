"""KnowledgeInterface - Online query engine for Agent Runtime.

Resident module managed by Runtime lifecycle.
Zero LLM calls in hot path. Uses SeekDB + in-memory keyword matching.

Design decisions for v1.0:
- No sentence-transformer in hot path (loaded only if assets available)
- Keyword/regex matching for symptoms (sufficient for 10-50 patterns)
- Hard-coded curated patterns as fallback when no assets loaded
- All patterns loaded into RAM at initialize() (~500KB for 100 patterns)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from rosclaw.core.lifecycle import LifecycleMixin

logger = logging.getLogger("rosclaw.know.interface")


class KnowledgeInterface(LifecycleMixin):
    """
    Online knowledge query engine.

    Loads knowledge from SeekDB at startup and keeps it in RAM
    for fast query response (<5ms per query).
    """

    # Hard-coded safety patterns as baseline fallback.
    # These ship inline so the runtime can always serve baseline heuristics
    # even when no assets are loaded.
    _SAFETY_PATTERNS: dict[str, dict[str, Any]] = {
        "Torque_Overflow": {
            "symptom": "PID integral wind-up drives actuator into torque saturation",
            "domain": "Control_Locomotion",
            "fix": "Apply conditional integration: stop accumulating integral term when actuator output is saturated. Clamp tau_cmd with torch.clamp(tau, -tau_max, tau_max).",
            "anti_pattern": "Cranking up Kp/Ki to fix tracking error during saturation — deepens wind-up.",
            "keywords": ["torque", "overflow", "saturation", "wind-up", "windup", "anti-windup", "pid", "integral", "actuator"],
        },
        "Velocity_Divergence": {
            "symptom": "Commanded velocity diverges to ±∞ when integrator has no clamp",
            "domain": "Control_Locomotion",
            "fix": "Wrap every commanded velocity through torch.clamp(v_cmd, -v_max, v_max). Add integral-leak term (integ *= 0.99 per step) in steady state.",
            "anti_pattern": "Adding only soft-start ramp on user-side command — cannot stop internal feedback divergence.",
            "keywords": ["velocity", "diverg", "infinite", "explode", "saturation", "clamp", "limit"],
        },
        "Memory_Exhaustion": {
            "symptom": "Unbounded KV-cache growth during long-horizon LLM rollouts causes CUDA OOM",
            "domain": "Memory_Reasoning",
            "fix": "Cap per-layer KV tensor at fixed window N (256-512 tokens). Evict oldest key/value rows on each forward. Keep optional global-attention sink.",
            "anti_pattern": "Increasing --gpu-memory-utilization or moving to larger GPU — only buys one more batch.",
            "keywords": ["memory", "exhaustion", "oom", "out of memory", "cuda", "kv-cache", "kv cache", "sequence", "long horizon"],
        },
        "Numerical_Instability": {
            "symptom": "NaN/Inf in loss or weights after a step explodes the gradient",
            "domain": "Learning_Training",
            "fix": "Apply torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) before optimizer.step(). If NaN persists, halve learning rate.",
            "anti_pattern": "Catching NaN after it surfaces and zeroing it out — bad direction already corrupted optimizer moment buffers.",
            "keywords": ["nan", "inf", "numerical instability", "loss explod", "gradient explod", "gradient clip", "learning rate"],
        },
        "Oscillation_Divergence": {
            "symptom": "Open-loop plan tracks ground truth poorly when latency exceeds 50 ms",
            "domain": "Planning_Decision",
            "fix": "Replace open-loop planner with Model-Predictive Control loop: re-solve horizon-H optimization every dt using latest measurement.",
            "anti_pattern": "Compensating for tracking error by adding feed-forward terms tuned offline.",
            "keywords": ["oscillat", "diverg", "tracking", "drift", "latency", "open-loop", "open loop"],
        },
        "Communication_Timeout": {
            "symptom": "Network/RPC timeout cascades cause request storms after partial outage",
            "domain": "Systems_Compute",
            "fix": "Wrap network calls with exponential backoff (base 0.5s, factor 2, jitter ±30%). Cap retries at 5. Add circuit-breaker.",
            "anti_pattern": "Tight while True: retry() loops — turn transient blip into thundering herd.",
            "keywords": ["timeout", "timed out", "deadline exceeded", "retry", "rpc", "grpc"],
        },
        "Gradient_Explosion": {
            "symptom": "Gradient magnitude explodes during backprop, causing weight overflow",
            "domain": "Learning_Training",
            "fix": "Apply gradient clipping with clip_grad_norm_. Reduce learning rate by 10x. Check for log(<=0) or divide-by-zero in loss.",
            "anti_pattern": "Increasing batch size to 'absorb' large gradients — masks the root cause.",
            "keywords": ["gradient", "explod", "inf", "nan", "loss", "backprop", "weight", "overflow"],
        },
        "Compile_Error": {
            "symptom": "Python syntax or import error prevents code execution",
            "domain": "Systems_Compute",
            "fix": "Fix syntax/import path BEFORE calling Wiki. This is not a deadlock, it's a typo.",
            "anti_pattern": "Retrying execution without reading the traceback.",
            "keywords": ["syntaxerror", "indentationerror", "nameerror", "typeerror", "importerror", "compile"],
        },
    }

    # Task decomposition hints: high-level task -> ordered sub-task list.
    # v1.0: Curated patterns for common embodied-intelligence tasks.
    _TASK_DECOMPOSITIONS: dict[str, list[str]] = {
        "pick and place": ["navigate_to_object", "align_gripper", "grasp", "lift", "move_to_target", "release"],
        "sort objects": ["scan_workspace", "classify_object", "grasp", "move_to_bin", "release", "repeat"],
        "walk to point": ["balance", "plan_footsteps", "step", "step", "step", "arrive_check"],
        "assembly": ["grasp_part", "align_holes", "insert", "verify_fit", "release"],
        "inspect surface": ["position_sensor", "scan_line", "scan_line", "analyze_defects", "report"],
        "handover object": ["detect_human", "approach", "grasp", "extend_arm", "wait_signal", "release"],
        "open door": ["approach_door", "grasp_handle", "pull", "hold", "pass_through", "release"],
        "stack blocks": ["grasp_block", "lift", "move_above_target", "align", "lower", "release", "repeat"],
    }

    # Task -> required capability mapping for compositional reasoning.
    # Each task maps to the minimum set of capabilities a robot must have.
    _TASK_CAPABILITY_REQUIREMENTS: dict[str, list[str]] = {
        "pick and place": ["grasp", "pick_and_place"],
        "sort objects": ["grasp", "sort_objects"],
        "walk to point": ["locomotion", "balance"],
        "assembly": ["grasp", "assembly"],
        "inspect surface": ["locomotion", "inspect_surface"],
        "handover object": ["grasp", "handover_object"],
        "open door": ["grasp", "open_door"],
        "stack blocks": ["grasp", "stack_blocks"],
    }

    # Cross-domain analogies for common symptoms.
    _CROSS_DOMAIN_ANALOGIES: dict[str, list[dict[str, str]]] = {
        "Torque_Overflow": [
            {"source_domain": "Systems_Compute", "insight": "Same back-pressure principle as bounded-queue producer-consumer.", "action_suggestion": "Pause the integrator the way you'd pause a queue writer when downstream is full."},
            {"source_domain": "Learning_Training", "insight": "Clamp gradient analogue — gradient clipping prevents one outlier from blowing up a step.", "action_suggestion": "Reuse clip_grad_norm_ mental model: an upper bound that fires only when magnitude exceeds a known physical limit."},
        ],
        "Memory_Exhaustion": [
            {"source_domain": "Control_Locomotion", "insight": "Like an anti-windup clamp on an integrator: keep the size of accumulating state finite.", "action_suggestion": "Treat the KV-cache as the integral term of attention; bound it the same way a PID bounds the integrator."},
        ],
        "Velocity_Divergence": [
            {"source_domain": "Memory_Reasoning", "insight": "Same as sliding-window KV-cache: cap the magnitude of the running state, not just the input.", "action_suggestion": "Bound the integrator state itself (integ = clamp(integ, -I_MAX, I_MAX)), mirroring the KV sliding window."},
        ],
    }

    def __init__(
        self,
        robot_id: str = "rosclaw_default",
        event_bus: Any = None,
        seekdb_client: Any = None,
        assets_path: str = "data/knowledge_assets",
        similarity_floor: float = 0.5,
    ):
        super().__init__()
        self.robot_id = robot_id
        self.event_bus = event_bus
        self.seekdb = seekdb_client
        self.assets_path = Path(assets_path)
        self.similarity_floor = similarity_floor

        # In-memory caches (populated at initialize())
        self._capabilities: dict[str, list[str]] = {}  # robot_id -> [capability, ...]
        self._symptoms: list[dict[str, Any]] = []      # loaded symptom clusters
        self._patterns: dict[str, dict[str, Any]] = {}  # pattern_id -> pattern dict
        self._initialized = False

    # -- Lifecycle --

    def _do_initialize(self) -> None:
        """Load knowledge assets into RAM."""
        logger.info("[Know] Initializing KnowledgeInterface for %s", self.robot_id)

        # 1. Load from SeekDB knowledge_graph (if available)
        if self.seekdb is not None:
            try:
                self._load_from_seekdb()
            except Exception as exc:
                logger.warning("[Know] SeekDB load failed: %s. Using fallback.", exc)

        # 2. Load from bridge_index.json (if assets path exists)
        bridge_path = self.assets_path / "bridge_index.json"
        if bridge_path.exists():
            try:
                self._load_bridge_index(bridge_path)
            except Exception as exc:
                logger.warning("[Know] bridge_index load failed: %s", exc)

        # 3. Always register curated safety patterns as baseline
        self._register_curated_patterns()

        self._initialized = True
        logger.info(
            "[Know] Initialized: %d capabilities, %d symptoms, %d patterns",
            len(self._capabilities.get(self.robot_id, [])),
            len(self._symptoms),
            len(self._patterns),
        )

    def _do_start(self) -> None:
        logger.info("[Know] KnowledgeInterface started")

    def _do_stop(self) -> None:
        logger.info("[Know] KnowledgeInterface stopped")
        self._capabilities.clear()
        self._symptoms.clear()
        self._patterns.clear()
        self._initialized = False

    # -- Public API --

    def query_robot_capabilities(self, robot_id: str | None = None) -> list[str]:
        """Return structured capability list for a robot."""
        rid = robot_id or self.robot_id
        return list(self._capabilities.get(rid, []))

    def match_symptom(self, error_signature: str) -> dict[str, Any] | None:
        """Match an error log/signature against known symptoms.

        Returns the best-matching pattern dict, or None if no match above floor.
        """
        if not error_signature or not self._initialized:
            return None

        best_match: dict[str, Any] | None = None
        best_score = 0.0

        for pattern_id, pattern in self._patterns.items():
            score = self._score_match(error_signature, pattern)
            if score > best_score:
                best_score = score
                best_match = {
                    "pattern_id": pattern_id,
                    "symptom": pattern.get("symptom", ""),
                    "domain": pattern.get("domain", ""),
                    "fix": pattern.get("fix", ""),
                    "anti_pattern": pattern.get("anti_pattern", ""),
                    "similarity": round(score, 4),
                }

        # v1.0: return best match if any keyword hit at all
        if best_match and best_score > 0.05:
            return best_match
        return None

    def get_analogy(self, situation: str) -> dict[str, Any] | None:
        """Find cross-domain analogy for a situation.

        First matches the situation to a known symptom, then returns
        cross-domain analogies for that symptom.
        """
        match = self.match_symptom(situation)
        if match is None:
            return None

        pattern_id = match.get("pattern_id", "")
        analogies = self._CROSS_DOMAIN_ANALOGIES.get(pattern_id, [])
        if not analogies:
            return None

        return {
            "pattern_id": pattern_id,
            "symptom": match["symptom"],
            "analogies": analogies,
        }

    def get_safety_rule(self, safety_label: str) -> str:
        """Return hard-coded safety constraint for a known dangerous condition."""
        pattern = self._SAFETY_PATTERNS.get(safety_label)
        if pattern is None:
            return ""
        fix = pattern.get("fix", "")
        anti = pattern.get("anti_pattern", "")
        parts = [f"SAFETY: {safety_label}", f"Fix: {fix}"]
        if anti:
            parts.append(f"Avoid: {anti}")
        return "\n".join(parts)

    def robot_capability_query(self, skill_name: str) -> list[str]:
        """Return list of robot IDs that possess a given skill/capability.

        Queries the SeekDB knowledge_graph for all robots with the
        ``has_capability`` predicate matching ``skill_name``.
        Falls back to in-memory cache if SeekDB is unavailable.
        """
        if not skill_name:
            return []

        # Prefer SeekDB query when available
        if self.seekdb is not None:
            try:
                from rosclaw.know.graph import get_related_robots
                return get_related_robots(self.seekdb, skill_name)
            except Exception as exc:
                logger.warning("[Know] robot_capability_query SeekDB failed: %s", exc)

        # Fallback: scan in-memory capabilities cache
        results = []
        skill_lower = skill_name.lower()
        for robot_id, caps in self._capabilities.items():
            for cap in caps:
                if cap.lower() == skill_lower:
                    results.append(robot_id)
                    break
        return results

    def task_decomposition_hint(self, task: str) -> dict[str, Any] | None:
        """Decompose a high-level task into ordered sub-tasks.

        Uses keyword matching against curated task decomposition patterns.
        Returns the best-matching decomposition with confidence score.
        """
        if not task or not self._initialized:
            return None

        task_lower = task.lower()
        best_key: str | None = None
        best_score = 0.0

        for pattern, steps in self._TASK_DECOMPOSITIONS.items():
            score = self._score_task_match(task_lower, pattern)
            if score > best_score:
                best_score = score
                best_key = pattern

        if best_key and best_score > 0.3:
            return {
                "task": task,
                "matched_pattern": best_key,
                "steps": self._TASK_DECOMPOSITIONS[best_key],
                "step_count": len(self._TASK_DECOMPOSITIONS[best_key]),
                "confidence": round(best_score, 4),
            }
        return None

    def can_perform_task(self, robot_id: str, task: str) -> dict[str, Any] | None:
        """Check if a robot has all required capabilities for a task.

        Returns a dict with robot_id, task, can_perform, missing_caps,
        and matched_caps, or None if task is unknown.
        """
        if not robot_id or not task:
            return None

        # Find the best matching task pattern
        task_lower = task.lower()
        matched_task: str | None = None
        best_score = 0.0
        for pattern in self._TASK_CAPABILITY_REQUIREMENTS:
            score = self._score_task_match(task_lower, pattern)
            if score > best_score:
                best_score = score
                matched_task = pattern

        if not matched_task or best_score < 0.3:
            return None

        required = set(self._TASK_CAPABILITY_REQUIREMENTS[matched_task])
        robot_caps = set(self.query_robot_capabilities(robot_id))
        missing = required - robot_caps
        matched = required & robot_caps

        return {
            "robot_id": robot_id,
            "task": task,
            "matched_pattern": matched_task,
            "can_perform": len(missing) == 0,
            "required_capabilities": sorted(required),
            "matched_capabilities": sorted(matched),
            "missing_capabilities": sorted(missing),
            "confidence": round(best_score, 4),
        }

    def recommend_robot_for_task(self, task: str) -> list[dict[str, Any]]:
        """Recommend robots best suited for a given task.

        Returns a ranked list of {robot_id, score, matched_caps, missing_caps}
        sorted by match score descending.
        """
        if not task:
            return []

        # Find the best matching task pattern
        task_lower = task.lower()
        matched_task: str | None = None
        best_score = 0.0
        for pattern in self._TASK_CAPABILITY_REQUIREMENTS:
            score = self._score_task_match(task_lower, pattern)
            if score > best_score:
                best_score = score
                matched_task = pattern

        if not matched_task or best_score < 0.3:
            return []

        required = set(self._TASK_CAPABILITY_REQUIREMENTS[matched_task])

        # Gather all known robots from capabilities cache + SeekDB
        all_robots = set(self._capabilities.keys())
        if self.seekdb is not None:
            try:
                rows = self.seekdb.query("knowledge_graph",
                    filters={"predicate": "has_capability"}, limit=1000)
                for r in rows:
                    rid = r.get("subject", "")
                    if rid:
                        all_robots.add(rid)
            except Exception:
                pass

        recommendations = []
        for rid in all_robots:
            caps = set(self._capabilities.get(rid, []))
            # Also query SeekDB for this robot's capabilities
            if self.seekdb is not None and not caps:
                try:
                    rows = self.seekdb.query("knowledge_graph",
                        filters={"subject": rid, "predicate": "has_capability"}, limit=100)
                    caps = {r.get("object", "") for r in rows if r.get("object")}
                except Exception:
                    pass

            matched = required & caps
            missing = required - caps
            score = len(matched) / max(len(required), 1)
            recommendations.append({
                "robot_id": rid,
                "score": round(score, 4),
                "matched_capabilities": sorted(matched),
                "missing_capabilities": sorted(missing),
                "task_match_confidence": round(best_score, 4),
            })

        recommendations.sort(key=lambda x: (-x["score"], x["robot_id"]))
        return recommendations

    def _score_task_match(self, task_lower: str, pattern: str) -> float:
        """Score how well a task description matches a decomposition pattern."""
        pattern_lower = pattern.lower()

        # Exact match
        if task_lower == pattern_lower:
            return 1.0

        # Substring match
        if pattern_lower in task_lower or task_lower in pattern_lower:
            return 0.85

        # Keyword overlap
        task_words = set(re.findall(r"[a-z][a-z0-9_]+", task_lower))
        pattern_words = set(re.findall(r"[a-z][a-z0-9_]+", pattern_lower))
        if task_words and pattern_words:
            overlap = len(task_words & pattern_words)
            return min(1.0, overlap / max(len(pattern_words), 1) * 0.7 + 0.1)
        return 0.0

    # -- Internal loaders --

    def _load_from_seekdb(self) -> None:
        """Read knowledge_graph table from SeekDB."""
        # Query capabilities for this robot
        records = self.seekdb.query(
            "knowledge_graph",
            filters={"subject": self.robot_id, "predicate": "has_capability"},
            limit=100,
        )
        for rec in records:
            obj = rec.get("object", "")
            if obj:
                self._capabilities.setdefault(self.robot_id, []).append(obj)

        # Query all symptoms
        records = self.seekdb.query(
            "knowledge_graph",
            filters={"predicate": "has_symptom"},
            limit=1000,
        )
        for rec in records:
            subject = rec.get("subject", "")
            obj = rec.get("object", "")
            if subject and obj:
                self._symptoms.append({
                    "id": rec.get("id", ""),
                    "subject": subject,
                    "symptom": obj,
                    "confidence": rec.get("confidence", 1.0),
                })

    def _load_bridge_index(self, bridge_path: Path) -> None:
        """Load symptom clusters from bridge_index.json."""
        data = json.loads(bridge_path.read_text(encoding="utf-8"))
        clusters = data.get("symptom_clusters", {})
        for cluster_id, cluster in clusters.items():
            symptom = cluster.get("standard_name", cluster_id)
            domain = cluster.get("domain", "")
            keywords = cluster.get("matched_keywords", [])
            analogies = cluster.get("cross_domain_analogies", [])

            self._patterns[cluster_id] = {
                "symptom": symptom,
                "domain": domain,
                "fix": "",  # Would be loaded from code_patterns/*.md
                "anti_pattern": "",
                "keywords": keywords,
                "analogies": analogies,
            }

            # Also index as a symptom entry
            self._symptoms.append({
                "id": cluster_id,
                "subject": self.robot_id,
                "symptom": symptom,
                "confidence": 1.0,
            })

    def _register_curated_patterns(self) -> None:
        """Register hard-coded safety patterns as baseline."""
        for label, pattern in self._SAFETY_PATTERNS.items():
            self._patterns[label] = {
                "symptom": pattern["symptom"],
                "domain": pattern["domain"],
                "fix": pattern["fix"],
                "anti_pattern": pattern["anti_pattern"],
                "keywords": pattern["keywords"],
                "analogies": self._CROSS_DOMAIN_ANALOGIES.get(label, []),
            }

    def _score_match(self, error_sig: str, pattern: dict[str, Any]) -> float:
        """Score how well error_sig matches a pattern.

        v1.0: Keyword overlap + symptom text overlap.
        v1.1: Replace with sentence-transformer cosine similarity.
        """
        error_lower = error_sig.lower()
        keywords = pattern.get("keywords", [])
        symptom = pattern.get("symptom", "").lower()
        pattern_id = pattern.get("pattern_id", "")

        if not keywords and not symptom:
            return 0.0

        # Keyword match score — exact substring hits
        keyword_hits = sum(1 for kw in keywords if kw.lower() in error_lower)
        keyword_score = keyword_hits / max(len(keywords), 1)

        # Symptom text overlap score (word-level Jaccard)
        error_words = set(re.findall(r"[a-z][a-z0-9_]+", error_lower))
        symptom_words = set(re.findall(r"[a-z][a-z0-9_]+", symptom))
        if error_words and symptom_words:
            jaccard = len(error_words & symptom_words) / len(error_words | symptom_words)
        else:
            jaccard = 0.0

        # Pattern ID boost — if the pattern name itself appears in the error
        id_boost = 0.0
        if pattern_id and pattern_id.lower().replace("_", " ") in error_lower:
            id_boost = 0.3

        # Weighted combination — keyword-heavy for v1.0
        return min(1.0, 0.7 * keyword_score + 0.2 * jaccard + id_boost)


__all__ = ["KnowledgeInterface"]
