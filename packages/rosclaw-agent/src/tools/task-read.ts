// HP2-COMPAT: 工具定义原语（defineTool/Type/ToolDefinition）——工具层在 HP3 投影层（Codex MCP）落地前保持 Pi 形态；不新增会话装配引用。
/** 只读任务/交付物工具（0901 体验探讨 P0-3）。
 *
 * 0901 实证：用户问"你这是啥？"——模型没有只读能力，猜
 * task.list_artifacts/artifact.open 撞 EFFECT_UNRESOLVABLE，
 * 降级 Shell 被 bwrap 拒，最后把任务重跑一遍制造第二套
 * trace/artifact。这三个工具让 Native Agent 认识确定性链刚做的事：
 *
 * - rosclaw_task_inspect：最近任务 + TaskOutcome（验收/交付/交付物）；
 * - rosclaw_artifact_list：最近任务的交付物（含绝对路径）；
 * - rosclaw_artifact_resolve：artifact_id → path/kind/size/digest。
 *
 * 全部只读——不产生新 task/trace/artifact。
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";
import type { BridgeToolContext } from "./bridge-tools.js";

function text(value: unknown): string {
	return JSON.stringify(value, null, 1);
}

export function buildReadOnlyTaskTools(ctx: BridgeToolContext) {
	const taskInspect = defineTool({
		name: "rosclaw_task_inspect",
		label: "ROSClaw Task Inspect",
		description:
			"Read the latest task and its outcome (state, verification, " +
			"delivery, artifact refs with absolute paths). Read-only — use " +
			"this to explain what just happened instead of re-running anything.",
		parameters: Type.Object({}),
		async execute() {
			const state = ctx.active.current;
			const call = (m: string, p: Record<string, unknown>) =>
				ctx.center.call(m, p);
			const latest = await call("pi.kernel.latest", {
				mission_id: state.missionId ?? "",
				session_ref: state.sessionId ?? "",
			});
			const task = (latest.task ?? null) as Record<string, unknown> | null;
			if (!task) {
				return {
					content: [{ type: "text" as const, text: "当前会话还没有任务。" }],
					details: { ok: true, task_id: "" },
				};
			}
			const considered = await call("pi.coordinator.consider", {
				task_id: task.task_id,
			});
			const outcome = (considered.outcome ?? null) as Record<string, unknown> | null;
			return {
				content: [{ type: "text" as const, text: text({ task, outcome }) }],
				details: { ok: true, task_id: String(task.task_id ?? "") },
			};
		},
	});

	const artifactList = defineTool({
		name: "rosclaw_artifact_list",
		label: "ROSClaw Artifact List",
		description:
			"List deliverables of the latest task (artifact id, kind, media " +
			"type, size, absolute path, open command). Read-only.",
		parameters: Type.Object({
			task_id: Type.Optional(Type.String({ description: "task id（缺省=最近任务）" })),
		}),
		async execute(_id, params) {
			const state = ctx.active.current;
			const result = await ctx.center.call("pi.artifact.list", {
				mission_id: state.missionId ?? "",
				session_ref: state.sessionId ?? "",
				task_id: String(params.task_id ?? ""),
			});
			return {
				content: [{ type: "text" as const, text: text(result) }],
				details: { ok: true },
			};
		},
	});

	const artifactResolve = defineTool({
		name: "rosclaw_artifact_resolve",
		label: "ROSClaw Artifact Resolve",
		description:
			"Resolve an artifact id to its absolute path, kind, size and " +
			"digest. Read-only — use before pointing the user at a file.",
		parameters: Type.Object({
			artifact_id: Type.String({ description: "artifact id（art_...）" }),
		}),
		async execute(_id, params) {
			const state = ctx.active.current;
			const result = await ctx.center.call("pi.artifact.resolve", {
				mission_id: state.missionId ?? "",
				artifact_id: String(params.artifact_id),
			});
			return {
				content: [{ type: "text" as const, text: text(result) }],
				details: { ok: Boolean(result.ok) },
				isError: !result.ok,
			};
		},
	});

	return [taskInspect, artifactList, artifactResolve];
}
