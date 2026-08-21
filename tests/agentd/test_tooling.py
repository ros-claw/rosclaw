"""PR-05 Tool/Capability Catalog tests (大纲 §7 + 补充文档 §10).

- ToolDescriptorV2 contract invariants (fail closed)
- Catalog guards: PHYSICAL_ACTION never executable, quarantine, timeout
- ToolResolver hard filters (each reason) + explainability + ranking cap
- Safety never enters model-facing scoring
- MCP adapter: real stdio discovery against a LIMO-like fixture server,
  fail-closed classification, fault-injection quarantine
- Evidence wrapper: untrusted marking, artifact spill, honest errors
- Exit condition: LIMO MCP observation usable, action not directly executable
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.agentd.runtime_sources import CatalogCapabilitySource
from rosclaw.agentd.tooling.artifact_result import ArtifactResultStore
from rosclaw.agentd.tooling.catalog import (
    ToolCatalog,
    ToolNotCallableError,
    ToolQuarantinedError,
)
from rosclaw.agentd.tooling.catalog_registry import CatalogToolRegistry
from rosclaw.agentd.tooling.descriptor import physical_action_descriptor
from rosclaw.agentd.tooling.evidence import ARTIFACT_SPILL_BYTES, wrap_observation
from rosclaw.agentd.tooling.mcp_adapter import (
    McpCapabilityAdapter,
    McpServerConfig,
    _normalize_result,
)
from rosclaw.agentd.tooling.resolver import FilterContext, ToolResolver
from rosclaw.agentd.tooling.result import ToolExecutionResult
from rosclaw.agentd.tooling.strict_schema import to_strict_tool
from rosclaw.contracts.agent.tool import (
    ExecutionClass,
    ToolDescriptorV2,
    ToolSideEffectClass,
)
from rosclaw.contracts.common import ValidationError

FIXTURE_SERVER = Path(__file__).parent / "fixtures" / "limo_mcp_server.py"


def _obs(tool_id: str = "test.observe", **overrides) -> ToolDescriptorV2:
    payload = {
        "tool_id": tool_id,
        "source": "test",
        "execution_class": ExecutionClass.OBSERVE,
        "verifier": "schema+timestamp",
    }
    payload.update(overrides)
    return ToolDescriptorV2(**payload)


class TestDescriptorInvariants:
    def test_physical_action_never_model_callable(self) -> None:
        with pytest.raises(ValidationError, match="never model_callable"):
            ToolDescriptorV2(
                tool_id="bad",
                source="test",
                execution_class=ExecutionClass.PHYSICAL_ACTION,
                requires_exact_action_grant=True,
                model_callable=True,
            )

    def test_grant_requirement_implies_not_callable(self) -> None:
        with pytest.raises(ValidationError, match="model_callable=False"):
            ToolDescriptorV2(
                tool_id="bad",
                source="test",
                execution_class=ExecutionClass.OBSERVE,
                requires_exact_action_grant=True,
            )

    def test_physical_action_must_require_grant(self) -> None:
        with pytest.raises(ValidationError, match="requires_exact_action_grant"):
            ToolDescriptorV2(
                tool_id="bad",
                source="test",
                execution_class=ExecutionClass.PHYSICAL_ACTION,
                model_callable=False,
                side_effect_class=ToolSideEffectClass.REVERSIBLE,
            )

    def test_observe_cannot_have_side_effects(self) -> None:
        with pytest.raises(ValidationError, match="side_effect_class=NONE"):
            ToolDescriptorV2(
                tool_id="bad",
                source="test",
                execution_class=ExecutionClass.OBSERVE,
                side_effect_class=ToolSideEffectClass.REVERSIBLE,
            )

    def test_helper_constructs_valid_physical_action(self) -> None:
        d = physical_action_descriptor("limo.speaker.play_tone", source="mcp:limo")
        assert d.execution_class is ExecutionClass.PHYSICAL_ACTION
        assert not d.model_callable and d.requires_exact_action_grant


class TestCatalogGuards:
    async def test_physical_action_not_executable(self) -> None:
        catalog = ToolCatalog()
        catalog.register(physical_action_descriptor("limo.speaker.play_tone", source="mcp:x"))
        with pytest.raises(ToolNotCallableError, match="PHYSICAL_ACTION"):
            await catalog.execute("limo.speaker.play_tone", {})

    async def test_quarantined_tool_fails_honestly(self) -> None:
        catalog = ToolCatalog()
        catalog.register(_obs(), lambda args: _never_called(args))
        catalog.quarantine_tool("test.observe", "health check failed")
        with pytest.raises(ToolQuarantinedError, match="health check failed"):
            await catalog.execute("test.observe", {})

    async def test_unknown_tool_rejected(self) -> None:
        catalog = ToolCatalog()
        with pytest.raises(ValidationError, match="not in catalog"):
            await catalog.execute("ghost", {})

    async def test_duplicate_register_rejected_replace_ok(self) -> None:
        catalog = ToolCatalog()
        catalog.register(_obs())
        with pytest.raises(ValidationError, match="already registered"):
            catalog.register(_obs())
        catalog.replace(_obs(description="v2"))
        assert catalog.get("test.observe").description == "v2"

    async def test_timeout_enforced(self) -> None:
        import asyncio

        async def slow(args):
            await asyncio.sleep(5)
            return "{}"

        catalog = ToolCatalog()
        catalog.register(_obs(timeout_ms=50), slow)
        with pytest.raises(asyncio.TimeoutError):
            await catalog.execute("test.observe", {})


async def _never_called(args):  # pragma: no cover - guard
    raise AssertionError("must not execute")


class TestResolverHardFilters:
    def _resolver_with(self, *descriptors: ToolDescriptorV2):
        catalog = ToolCatalog()
        for d in descriptors:
            catalog.register(d, lambda args: _never_called(args))
        return ToolResolver(catalog), catalog

    def test_each_hard_filter_reason(self) -> None:
        resolver, catalog = self._resolver_with(
            _obs("t.body", required_body_types=["agilex-limo"]),
            _obs("t.mode", supported_modes=["REAL"]),
            _obs("t.caps", required_capabilities=["lidar.online"]),
            _obs("t.fresh", freshness_ms=500),
            _obs("t.policy"),
            _obs("t.budget", cost_hint=1.0),
            _obs("t.noverifier", verifier=""),
            _obs("t.quar"),
            _obs("t.ok"),
        )
        catalog.quarantine_tool("t.quar", "dead")
        ctx = FilterContext(
            body_type="other-robot",
            mode="SIMULATION",
            online_capabilities=frozenset(),
            self_snapshot_fresh=False,
            policy_denied_tools=frozenset({"t.policy"}),
            budget_exceeded=True,
        )
        result = resolver.resolve(ctx)
        assert [d.tool_id for d in result.injected] == ["t.ok"]
        reasons = {d.tool_id: d.reasons for d in result.excluded}
        assert "body_incompatible" in reasons["t.body"][0]
        assert "mode_not_allowed" in reasons["t.mode"][0]
        assert "capability_offline" in reasons["t.caps"][0]
        assert "self_snapshot_stale" in reasons["t.fresh"][0]
        assert "policy_denied" in reasons["t.policy"][0]
        assert "budget_exceeded" in reasons["t.budget"][0]
        assert "no_verifier" in reasons["t.noverifier"][0]
        assert reasons["t.quar"][0].startswith("quarantined")

    def test_permission_filter_binds_only_when_configured(self) -> None:
        resolver, _ = self._resolver_with(_obs("t.perm", required_capabilities=["cam"]))
        open_ctx = FilterContext(online_capabilities=frozenset({"cam"}))  # no permission set configured → not binding
        assert resolver.resolve(open_ctx).injected[0].tool_id == "t.perm"
        strict_ctx = FilterContext(
            online_capabilities=frozenset({"cam"}),
            granted_permissions=frozenset({"lidar"}),
        )
        result = resolver.resolve(strict_ctx)
        assert not result.injected
        assert "permission_not_granted" in result.excluded[0].reasons[0]

    def test_physical_action_filtered_as_not_model_callable(self) -> None:
        resolver, _ = self._resolver_with(
            physical_action_descriptor("limo.speaker.play_tone", source="mcp:x")
        )
        result = resolver.resolve(FilterContext())
        assert not result.injected
        assert "not_model_callable" in result.excluded[0].reasons

    def test_safety_never_in_score(self) -> None:
        """A maximally-relevant, high-reliability tool still loses to filters."""
        resolver, catalog = self._resolver_with(
            _obs(
                "pick red cube",
                description="pick red cube grasp",
                reliability=1.0,
                typical_latency_ms=0,
            )
        )
        catalog.quarantine_tool("pick red cube", "unsafe source")
        ctx = FilterContext(task_hint="pick red cube")
        result = resolver.resolve(ctx)
        assert not result.injected, "score must never override quarantine"

    def test_injection_cap_and_ranking(self) -> None:
        descriptors = [
            _obs(f"t.{i}", reliability=i / 20.0, typical_latency_ms=100 + i)
            for i in range(20)
        ]
        resolver, _ = self._resolver_with(*descriptors)
        result = resolver.resolve(FilterContext())
        assert len(result.injected) == 12
        scores = [d.reliability for d in result.injected]
        assert scores == sorted(scores, reverse=True)
        # overflow explained, not silently dropped
        overflow = [d for d in result.excluded if d.reasons[0].startswith("injection_cap")]
        assert len(overflow) == 8

    def test_semantic_ranking_prefers_task_match(self) -> None:
        resolver, _ = self._resolver_with(
            _obs("limo.localization.get_pose", description="robot pose localization"),
            _obs("camera.list", description="list cameras"),
        )
        ctx = FilterContext(task_hint="定位 pose 在哪里")
        result = resolver.resolve(ctx)
        assert result.injected[0].tool_id == "limo.localization.get_pose"


class TestStrictSchema:
    def test_observe_converts(self) -> None:
        tool = to_strict_tool(
            _obs(input_schema={"type": "object", "properties": {"f": {"type": "string"}}})
        )
        assert tool.name == "test__observe"  # wire 名（点号 → __）
        assert tool.parameters["additionalProperties"] is False
        assert tool.parameters["required"] == ["f"]
        assert "evidence_class" in tool.description

    def test_physical_action_never_converts(self) -> None:
        with pytest.raises(ValidationError, match="never become a model tool"):
            to_strict_tool(physical_action_descriptor("x.act", source="mcp:x"))


class TestMcpClassification:
    def _adapter(self, **cfg) -> McpCapabilityAdapter:
        return McpCapabilityAdapter(
            McpServerConfig(name="limo", command="true", **cfg), ToolCatalog()
        )

    def test_action_verb_is_physical(self) -> None:
        # PR-N5E：动词启发式不再上线——未显式声明 → None（调用方
        # QUARANTINED_UNCLASSIFIED）；启发式只在 suggest_classification
        # 给 doctor 建议。
        from rosclaw.agentd.tooling.mcp_adapter import suggest_classification

        adapter = self._adapter()
        assert adapter.classify("limo.speaker.play_tone", None) is None
        assert adapter.classify("limo.base.move_to", None) is None
        assert suggest_classification("limo.speaker.play_tone", None) == "PHYSICAL_ACTION"
        assert suggest_classification("limo.base.move_to", None) == "PHYSICAL_ACTION"

    def test_readonly_annotation_is_observe(self) -> None:
        from mcp.types import ToolAnnotations

        from rosclaw.agentd.tooling.mcp_adapter import suggest_classification

        adapter = self._adapter()
        ann = ToolAnnotations(readOnlyHint=True)
        # N5E：第三方自声明注解不是绑定依据——未显式声明即 None。
        assert adapter.classify("limo.localization.get_pose", ann) is None
        assert adapter.classify("limo.arm.set_pose", ann) is None
        assert suggest_classification("limo.localization.get_pose", ann) == "OBSERVE"
        assert suggest_classification("limo.arm.set_pose", ann) == "PHYSICAL_ACTION"

    def test_ambiguous_fails_closed(self) -> None:
        # PR-N5E：含糊工具不再默认 PHYSICAL_ACTION 上线——未声明即
        # None（隔离到 QUARANTINED_UNCLASSIFIED，比"当物理动作上线"
        # 更诚实：不执行、不展示为可用）。
        adapter = self._adapter()
        assert adapter.classify("limo.misc.unknown", None) is None

    def test_config_overrides(self) -> None:
        adapter = self._adapter(
            observation_tools=("custom.status",), action_tools=("custom.reset",)
        )
        assert adapter.classify("custom.status", None) is ExecutionClass.OBSERVE
        assert adapter.classify("custom.reset", None) is ExecutionClass.PHYSICAL_ACTION

    def test_destructive_annotation_is_physical(self) -> None:
        from mcp.types import ToolAnnotations

        from rosclaw.agentd.tooling.mcp_adapter import suggest_classification

        adapter = self._adapter()
        ann = ToolAnnotations(readOnlyHint=True, destructiveHint=True)
        # N5E：注解不参与上线分类；destructive 建议仍为 PHYSICAL_ACTION。
        assert adapter.classify("x.y", ann) is None
        assert suggest_classification("x.y", ann) == "PHYSICAL_ACTION"

    def test_image_content_is_preserved_with_bounded_metadata(self) -> None:
        from mcp.types import CallToolResult, ImageContent, TextContent

        result = CallToolResult(
            content=[
                TextContent(type="text", text='{"camera":"color"}'),
                ImageContent(type="image", data="iVBORw0KGgo=", mimeType="image/png"),
            ]
        )
        normalized = _normalize_result("limo_capture_camera_frame", "mcp:limo", result)
        assert isinstance(normalized, ToolExecutionResult)
        assert normalized.images[0].mime_type == "image/png"
        assert normalized.images[0].data_base64 == "iVBORw0KGgo="
        assert '"bytes": 8' in normalized.text
        assert "iVBORw0KGgo=" not in normalized.text

    def test_invalid_or_oversized_image_is_not_forwarded(self) -> None:
        from mcp.types import CallToolResult, ImageContent

        result = CallToolResult(
            content=[ImageContent(type="image", data="not-base64", mimeType="image/png")]
        )
        normalized = _normalize_result("camera", "mcp:limo", result)
        assert isinstance(normalized, str)
        assert "invalid_base64" in normalized

    def test_mislabeled_image_payload_is_not_forwarded(self) -> None:
        from mcp.types import CallToolResult, ImageContent

        result = CallToolResult(
            content=[ImageContent(type="image", data="aGVsbG8=", mimeType="image/png")]
        )
        normalized = _normalize_result("camera", "mcp:limo", result)
        assert isinstance(normalized, str)
        assert "image_signature_mismatch" in normalized


class TestSignedRobotPackCapabilityContext:
    def test_core_tool_after_catalog_prefilter_window_is_retained(self) -> None:
        catalog = ToolCatalog()
        for index in range(30):
            catalog.register(_obs(f"limo_alpha_{index:02d}"))
        catalog.register(_obs("limo_validate_navigation_goal"))

        source = CatalogCapabilitySource(catalog)
        infos = source.list_capabilities("navigate", 12)
        by_name = {info.name: info for info in infos}

        assert by_name["limo_validate_navigation_goal"].priority == 90

    def test_camera_frame_is_prioritized_for_bounded_multimodal_context(self) -> None:
        catalog = ToolCatalog()
        catalog.register(_obs("limo_capture_camera_frame"))
        catalog.register(_obs("limo_get_camera_state"))
        for index in range(30):
            catalog.register(_obs(f"limo_misc_{index:02d}"))

        infos = CatalogCapabilitySource(catalog).list_capabilities("camera", 12)
        by_name = {info.name: info for info in infos}
        assert by_name["limo_capture_camera_frame"].priority == 82
        assert by_name["limo_get_camera_state"].priority == 81

    def test_exact_pack_schema_replaces_mcp_action_alias(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pack_root = tmp_path / "pack"
        capability_path = pack_root / "capabilities" / "tone.yaml"
        capability_path.parent.mkdir(parents=True)
        capability_path.write_text(
            """schema_version: rosclaw.capability.v1
id: limo.play_tone
title: Tone
input:
  type: object
  additionalProperties: false
  properties:
    schema_version: {const: limo.tone.v1}
    volume_percent: {type: integer}
  required: [schema_version, volume_percent]
""",
            encoding="utf-8",
        )
        instance = SimpleNamespace(pack=SimpleNamespace(ref="pack-ref"))
        record = SimpleNamespace(path=str(pack_root))
        manifest = SimpleNamespace(
            components=[
                SimpleNamespace(
                    kind="capability",
                    path="capabilities/tone.yaml",
                    ref="rosclaw://capability/limo.play_tone@3",
                )
            ],
            capabilities=[
                SimpleNamespace(
                    id="limo.play_tone",
                    title="Play bounded tone",
                    adapter_tools_any_of=["limo_request_tone"],
                )
            ],
        )
        from rosclaw.robot_pack import instance as instance_module
        from rosclaw.robot_pack import store as store_module

        monkeypatch.setattr(
            instance_module,
            "load_robot_instance",
            lambda body_id, home: (instance, tmp_path / "limo.yaml"),
        )
        monkeypatch.setattr(
            store_module.RobotPackStore,
            "resolve_installed",
            lambda self, ref: (record, manifest),
        )

        catalog = ToolCatalog()
        catalog.register(physical_action_descriptor("limo_request_tone", source="mcp:limo"))
        catalog.register(_obs("limo_get_audio_state", description="read audio"))
        source = CatalogCapabilitySource(catalog, home=tmp_path, body_id="limo")

        infos = source.list_capabilities("tone", 12)
        by_name = {info.name: info for info in infos}
        assert "limo.play_tone" in by_name
        assert "limo_request_tone" not in by_name
        assert by_name["limo.play_tone"].permission == "operator_only"
        assert "volume_percent" in by_name["limo.play_tone"].summary
        assert "MUST NOT be placed inside arguments" in by_name["limo.play_tone"].summary


class TestMcpDiscovery:
    async def test_discover_limo_fixture(self, tmp_path: Path) -> None:
        catalog = ToolCatalog()
        adapter = McpCapabilityAdapter(
            McpServerConfig(
                name="limo-ros-mcp",
                command=sys.executable,
                args=(str(FIXTURE_SERVER),),
                supported_modes=("SIMULATION", "SHADOW"),
                required_body_types=("agilex-limo",),
                # PR-N5E 严格绑定：显式声明分类（生产必需——未声明
                # 的进 QUARANTINED_UNCLASSIFIED，见下方断言）。
                observation_tools=("limo.localization.get_pose",),
                action_tools=("limo.speaker.play_tone",),
            ),
            catalog,
        )
        report = await adapter.discover()
        assert report.ok, report.error
        by_id = {d.tool_id: d for d in report.tools}
        pose = by_id["limo.localization.get_pose"]
        tone = by_id["limo.speaker.play_tone"]
        ambiguous = by_id["limo.misc.ambiguous"]
        # 退出条件 1：observation 可用（model-callable OBSERVE）。
        assert pose.execution_class is ExecutionClass.OBSERVE
        assert pose.model_callable and not pose.requires_exact_action_grant
        # 退出条件 2：action 不可被模型直接执行。
        assert tone.execution_class is ExecutionClass.PHYSICAL_ACTION
        assert not tone.model_callable and tone.requires_exact_action_grant
        # 无声明工具 fail closed：QUARANTINED_UNCLASSIFIED（注册为
        # 不可调用形态 + 隔离——比"当物理动作上线"更诚实）。
        assert ambiguous.execution_class is ExecutionClass.PHYSICAL_ACTION
        assert catalog.quarantine_reason("limo.misc.ambiguous") is not None

        # observation 通过 catalog 真正执行（真实 MCP stdio 调用）。
        output = await catalog.execute("limo.localization.get_pose", {"frame": "map"})
        assert "1.25" in output and "map" in output
        # action 即使显式执行也被 catalog 拒绝。
        with pytest.raises(ToolNotCallableError):
            await catalog.execute("limo.speaker.play_tone", {"frequency_hz": 440})

        # resolver 只注入 observation。
        resolver = ToolResolver(catalog)
        result = resolver.resolve(
            FilterContext(body_type="agilex-limo", mode="SIMULATION", task_hint="pose")
        )
        injected = {d.tool_id for d in result.injected}
        assert "limo.localization.get_pose" in injected
        assert "limo.speaker.play_tone" not in injected

    async def test_dead_server_quarantines_source(self) -> None:
        catalog = ToolCatalog()
        adapter = McpCapabilityAdapter(
            McpServerConfig(name="dead", command="/nonexistent/binary-xyz"),
            catalog,
        )
        report = await adapter.discover()
        assert not report.ok and report.error
        # pre-registered tools from an earlier session get quarantined too
        catalog.register(_obs("limo.old", source="mcp:dead"))
        catalog.quarantine_source("mcp:dead", f"discovery_failed: {report.error}")
        resolver = ToolResolver(catalog)
        result = resolver.resolve(FilterContext())
        assert not result.injected
        assert result.excluded[0].reasons[0].startswith("quarantined")


class TestEvidenceWrapper:
    def test_ok_envelope_marks_untrusted(self, tmp_path: Path) -> None:
        d = _obs("limo.localization.get_pose", source="mcp:limo")
        env = wrap_observation(d, '{"x": 1.0}', body_id="limo/01")
        text = env.render_for_model()
        assert "<untrusted_input" in text and "</untrusted_input>" in text
        assert "evidence_class: MEASURED" in text
        assert env.artifact_ref and env.artifact_ref.startswith("artifact://observation/")

    def test_large_output_spills_to_artifact_store(self, tmp_path: Path) -> None:
        store = ArtifactResultStore(tmp_path)
        d = _obs()
        big = "X" * (ARTIFACT_SPILL_BYTES + 100)
        env = wrap_observation(d, big, body_id="b", artifact_store=store)
        assert len(env.content) < len(big)
        assert env.artifact_ref is not None
        assert store.resolve(env.artifact_ref) == big
        assert store.resolve("artifact://observation/sha256:nonexistent") is None

    def test_error_envelope_is_honest(self) -> None:
        d = _obs()
        env = wrap_observation(d, '{"error": "boom"}', body_id="b", error="TimeoutError: boom")
        text = env.render_for_model()
        assert not env.ok
        assert "error: TimeoutError: boom" in text
        assert "do not fabricate" in text
        assert "<untrusted_input" not in text  # nothing trustworthy to wrap


class TestCatalogRegistry:
    async def test_registry_guards_and_envelope(self, tmp_path: Path) -> None:
        catalog = ToolCatalog()
        catalog.register(_obs("t.obs"), lambda args: _echo(args))
        catalog.register(physical_action_descriptor("t.act", source="mcp:x"))
        registry = CatalogToolRegistry(
            catalog, ToolResolver(catalog), artifact_store=ArtifactResultStore(tmp_path)
        )
        # strict_tools never surfaces the physical action
        names = [t.name for t in registry.strict_tools(["t.obs", "t.act"])]
        assert names == ["t__obs"]
        with pytest.raises(ToolNotCallableError):
            await registry.execute("t.act", {})
        env = registry.evidence_envelope("t.obs", "data", body_id="b")
        assert env.ok and env.content == "data"

    def test_resolve_tools_applies_filters(self, tmp_path: Path) -> None:
        catalog = ToolCatalog()
        catalog.register(_obs("t.sim", supported_modes=["SIMULATION"]))
        catalog.register(_obs("t.real", supported_modes=["REAL"]))
        registry = CatalogToolRegistry(catalog, ToolResolver(catalog))
        sim_names = {t.name for t in registry.resolve_tools(["t.sim", "t.real"], mode="SIMULATION")}
        assert sim_names == {"t__sim"}
        excluded = registry.excluded_reasons(["t.sim", "t.real"], mode="SIMULATION")
        assert "mode_not_allowed" in excluded["t.real"][0]


async def _echo(args):
    import json

    return json.dumps(args)


class TestServiceIntegration:
    """退出条件：LIMO MCP observation 可用，action 不可被模型直接执行 ——
    走 AgentService 全链路（真实 MCP stdio server + 真实 AgentLoop）。"""

    def _service(self, tmp_path: Path, script):
        import yaml as _yaml

        from rosclaw.agentd.config import load_agent_config
        from rosclaw.agentd.models.gateway import MockModelGateway
        from rosclaw.agentd.models.profiles import mock_profile
        from rosclaw.agentd.service import AgentService

        (tmp_path / "config.yaml").write_text(
            _yaml.safe_dump(
                {
                    "agent": {"enabled": True},
                    "mcp_servers": [
                        {
                            "name": "limo-ros-mcp",
                            "command": sys.executable,
                            "args": [str(FIXTURE_SERVER)],
                            "supported_modes": ["SIMULATION", "SHADOW"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        config = load_agent_config(tmp_path / "config.yaml")
        return AgentService(config, tmp_path, gateway=MockModelGateway(mock_profile(), script))

    @staticmethod
    def _script(req):
        import json as _json

        from rosclaw.contracts.agent.model_turn import ModelTurnResultV1

        if not hasattr(TestServiceIntegration._script, "n"):
            TestServiceIntegration._script.n = 0
        TestServiceIntegration._script.n += 1
        if TestServiceIntegration._script.n == 1:
            decision = {
                "next_intent": "OBSERVE",
                "summary": "读取 LIMO 位姿",
                "evidence_refs": [],
                "proposed_operation": {
                    "type": "observe",
                    "payload": {
                        "tool": "limo.localization.get_pose",
                        "arguments": {"frame": "map"},
                    },
                },
            }
        else:
            decision = {
                "next_intent": "ANSWER",
                "summary": "位姿 x=1.25（MEASURED 观测证据）",
                "evidence_refs": ["artifact://observation/x"],
            }
        decision.update(
            {
                "schema_version": "rosclaw.decision.v1",
                "decision_id": f"dec_{TestServiceIntegration._script.n}",
                "mission_id": req.mission_id,
                "context_id": req.context_id,
                "context_revision": req.context_revision,
            }
        )
        return ModelTurnResultV1(
            turn_id="t",
            provider="mock",
            model="m",
            content=f"```json\n{_json.dumps(decision)}\n```",
            assistant_message={"role": "assistant", "content": "x"},
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},  # type: ignore[arg-type]
        )
