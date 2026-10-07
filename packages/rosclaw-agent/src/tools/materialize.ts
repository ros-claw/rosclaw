// HP2-COMPAT: 工具定义原语（defineTool/Type/ToolDefinition）——工具层在 HP3 投影层（Codex MCP）落地前保持 Pi 形态；不新增会话装配引用。
/** 动态工具物化（PR-N5D，调整方案 §三.N5D）。
 *
 * CapabilitySnapshot → 当前回合的精确强类型工具：模型直接看到
 * ur5e__plan_cartesian_path(shape: "star5"|"circle")，而不是
 * rosclaw_compute(arguments: unknown) 猜参数。
 *
 * 规则（测试钉住）：
 * - direct → 精确工具（input_schema 的 provider-facing 深拷贝投影成为 parameters）；
 * - propose_only → propose_<slug>，走 admission 链（rosclaw_execute
 *   wire）——物理效应原始 executor 永不直接暴露；
 * - excluded 不产生工具（原因经 rosclaw inspect capability 可查）；
 * - wire 仍走内核验证链（pi.tools.execute），capability_id 钉住 +
 *   携带 snapshot digest（registry 变了 → CAPABILITY_SNAPSHOT_CHANGED，
 *   不静默换工具）。
 */

import { defineTool } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";

import type { BridgeToolContext } from "./bridge-tools.js";
import { displayLabelFor, summarizeToolResultText } from "../ui/tool-display.js";

export interface SnapshotActiveTool {
	tool_name: string;
	capability_id: string;
	exposure: "direct" | "propose_only" | "internal";
	effect_class: string;
	description: string;
	input_schema: Record<string, unknown>;
	output_schema: Record<string, unknown>;
}

export interface CapabilitySnapshot {
	schema_version: string;
	generation: number;
	digest: string;
	body_id: string;
	mode: string;
	active: SnapshotActiveTool[];
	excluded: Array<{ capability_id: string; reason: string }>;
}

/** direct 能力 → wire 入口（READ_ONLY 观测链 / 其余 compute 链）。 */
function wireEntryFor(entry: SnapshotActiveTool): string {
	if (entry.exposure === "propose_only") return "rosclaw_execute";
	return entry.effect_class === "READ_ONLY" ? "rosclaw_observe" : "rosclaw_compute";
}

/** Provider-only projection: JS cannot represent int64 bounds exactly.
 * Omit unsafe integer-valued bounds on locally declared integers; retain fractions.
 * Type-less unsafe bounds fail closed with PROVIDER_SCHEMA_UNSUPPORTED because
 * inherited type resolution is unsupported. Never clamp canonical domain or data.
 * Traverse schema positions only; enum/default/const values are data, not schemas.
 */
export function providerCompatibleSchema(schema: Record<string, unknown>): Record<string, unknown> {
	const clone = (value: unknown): unknown => {
		if (Array.isArray(value)) return value.map(clone);
		if (value !== null && typeof value === "object") {
			return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, clone(v)]));
		}
		return value;
	};
	const maps = new Set(["properties", "patternProperties", "$defs", "definitions", "dependentSchemas"]);
	const singles = new Set(["items", "additionalItems", "additionalProperties", "unevaluatedItems", "unevaluatedProperties", "contains", "propertyNames", "not", "if", "then", "else", "contentSchema"]);
	const arrays = new Set(["allOf", "anyOf", "oneOf", "prefixItems"]);
	const bounds = new Set(["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]);
	const project = (value: unknown): unknown => {
		if (value === null || typeof value !== "object" || Array.isArray(value)) return clone(value);
		const node = value as Record<string, unknown>;
		const integer = node.type === "integer" || (Array.isArray(node.type) && node.type.includes("integer"));
		return Object.fromEntries(Object.entries(node).flatMap(([key, child]) => {
			const unsafeBound = bounds.has(key) && typeof child === "number" && Number.isInteger(child) && !Number.isSafeInteger(child);
			// Type-less bounds may inherit integer through allOf/$ref at any depth.
			// Do not guess that context or advertise an unsafe numeric constraint.
			if (unsafeBound && node.type === undefined) {
				throw Object.assign(new Error(`PROVIDER_SCHEMA_UNSUPPORTED: ${key} has an unsafe integer-valued bound without a local type; inherited type projection is unsupported`), { code: "PROVIDER_SCHEMA_UNSUPPORTED" });
			}
			if (integer && unsafeBound) return [];
			if (maps.has(key) && child !== null && typeof child === "object" && !Array.isArray(child)) {
				return [[key, Object.fromEntries(Object.entries(child).map(([k, v]) => [k, project(v)]))]];
			}
			if (key === "dependencies" && child !== null && typeof child === "object" && !Array.isArray(child)) {
				return [[key, Object.fromEntries(Object.entries(child).map(([k, v]) => [k, Array.isArray(v) ? clone(v) : project(v)]))]];
			}
			if (arrays.has(key) && Array.isArray(child)) return [[key, child.map(project)]];
			if (singles.has(key)) return [[key, Array.isArray(child) ? child.map(project) : project(child)]];
			return [[key, clone(child)]];
		}));
	};
	return project(schema) as Record<string, unknown>;
}

export function materializeCapabilityTools(
	snapshot: CapabilitySnapshot,
	ctx: BridgeToolContext,
) {
	const tools = [];
	for (const entry of snapshot.active) {
		if (entry.exposure === "internal") continue;
		const wireName = wireEntryFor(entry);
		const capabilityId = entry.capability_id;
		const digest = snapshot.digest;
		tools.push(defineTool({
			name: entry.tool_name,
			// WP-7：用户面 label 剥治理前缀（propose_）与双下划线——
			// 工具 name（模型调用面）不变，只改渲染。
			label: displayLabelFor(entry.tool_name),
			description:
				`${entry.description}（capability: ${capabilityId}；` +
				`effect: ${entry.effect_class}）`,
			// 精确 input_schema 经 provider-facing 深拷贝投影成为 parameters——不再是
			// Record<string, unknown> 猜参数。
			parameters: providerCompatibleSchema(entry.input_schema) as never,
			async execute(_id, params, _signal, _onUpdate, _ctx2) {
				const state = ctx.active.current;
				if (!state.missionId) {
					return {
						content: [{ type: "text" as const, text: "REJECTED [NO_MISSION]: 未绑定 Mission" }],
						details: { ok: false, error_code: "NO_MISSION" },
						isError: true,
					};
				}
				// wire 形状与 executeVia 一致（envelope.ok + result.*）。
				const request = {
					schema_version: "rosclaw.pi_tool_request.v1",
					request_id: `ptr_cap_${Date.now()}`,
					pi_session_id: state.sessionId,
					mission_id: state.missionId,
					context_revision: state.contextRevision,
					body_hash: state.bodyHash ?? "",
					mode: state.mode,
					tool_name: wireName,
					arguments: {
						capability_id: capabilityId,
						arguments: params ?? {},
						snapshot_digest: digest,
					},
					requested_at: new Date().toISOString(),
					idempotency_key: `idem_${state.sessionId}_${Date.now()}`,
					actor: { engine: "pi", process_id: process.pid, uid: process.getuid?.() ?? 0 },
				};
				const response = await ctx.center.call("pi.tools.execute", { request }) as {
					ok?: boolean; code?: string; error?: string;
					result?: { ok?: boolean; summary?: string; error_code?: string };
				};
				const result = response.result ?? {};
				const ok = response.ok === true;
				return {
					content: [{
						type: "text" as const,
						text: ok
							? (result.summary ?? "ok")
							: `REJECTED [${result.error_code ?? response.code ?? "?"}]: ${String(response.error ?? result.summary ?? "")}`,
					}],
					details: { ok, error_code: result.error_code ?? null },
					isError: !ok,
				};
			},
			// WP-7：用户面折叠渲染——模型上下文保留完整文本，TUI
			// 只渲染单行摘要（JSON 不刷屏；REJECTED 明确"未执行"）。
			renderResult(result: { content?: Array<{ type: string; text?: string }> }) {
				const text = (result.content ?? [])
					.map((b) => (b.type === "text" ? String(b.text ?? "") : ""))
					.join("\n");
				return new Text(summarizeToolResultText(text), 1, 0);
			},
		}));
	}
	return tools;
}
