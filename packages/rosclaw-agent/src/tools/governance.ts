// HP2-COMPAT: 工具定义原语（defineTool/Type/ToolDefinition）——工具层在 HP3 投影层（Codex MCP）落地前保持 Pi 形态；不新增会话装配引用。
/** 治理工具集（十五审 PR-RF-1，ADR-0011 无为而治）。
 *
 * 模型唯一的任务入口：task_submit 交目标合同（TaskSpec），其余都是
 * 对同一 owning execution 的观察/steer/回答/暂停/恢复/取消。
 * 没有 worker 选择器、没有 retry 工厂、没有 transcript 搬运。
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ToolDefinition } from "@earendil-works/pi-coding-agent";

import { executeVia, type BridgeToolContext } from "./bridge-tools.js";

export function buildGovernanceTools(ctx: BridgeToolContext): ToolDefinition[] {
	return [
		defineTool({
			name: "rosclaw_task_submit",
			label: "ROSClaw Task Submit",
			description:
				"Submit a task goal contract (TaskSpec) to the Task Control Plane. " +
				"The ExecutionRouter picks the execution domain — you never choose a worker. " +
				"One task = one owning execution: re-submitting the same goal attaches to it. " +
				"Declare effects honestly: tasks needing shell/file writes MUST set " +
				"effects='workspace_only' (otherwise the task compiles read-only and " +
				"BLOCKs). Declare runtime_requirements.python_packages for needed " +
				"packages — ROSClaw's Runtime Manager provisions them deterministically; " +
				"NEVER make 'install a package' a worker task, and NEVER hand the user " +
				"manual shell commands to finish a task.",
			parameters: Type.Object({
				goal: Type.String({ description: "自包含目标（用户意图，非微步骤剧本）" }),
				kind: Type.Optional(Type.String({ description: "任务类型提示（如 simulation.render）" })),
				required_capabilities: Type.Optional(Type.Array(Type.String())),
				effects: Type.Optional(Type.String({ description: "none | workspace_only | simulation_only | physical_*（要写文件/跑脚本必须 workspace_only）" })),
				inputs: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
				deliverables: Type.Optional(Type.Array(Type.Record(Type.String(), Type.Unknown()), { description: "交付物：{type: MIME, path: 相对路径}——path 自动成为验收检查" })),
				acceptance: Type.Optional(Type.Record(Type.String(), Type.Unknown(), { description: "结构化验收：{required_files:[...]} 或 {run:{argv:[...]}}——禁止 shell 字符串" })),
				runtime_requirements: Type.Optional(Type.Record(Type.String(), Type.Unknown(), { description: "运行依赖：{python_packages:['Pillow>=10']}——Runtime Manager 托管预置" })),
			}),
			async execute(_id, params, _signal, _onUpdate, toolCtx) {
				// 建议-0816 P0-4：Native 当前模型快照注入（Worker 继承同一
				// provider/model/thinking——无 secret，凭据不走 WorkOrder）。
				const model = (toolCtx as { model?: { provider: string; id: string } } | undefined)?.model;
				const thinking = (toolCtx as { thinkingLevel?: string } | undefined)?.thinkingLevel;
				const args = { ...(params as Record<string, unknown>) };
				if (model && !args.model_snapshot) {
					args.model_snapshot = {
						provider: model.provider,
						model: model.id,
						...(thinking ? { thinking } : {}),
					};
				}
				return await executeVia(ctx, "rosclaw_task_submit", args);
			},
		}),
		defineTool({
			name: "rosclaw_task_observe",
			label: "ROSClaw Task Observe",
			description:
				"Authoritative execution state + summary + verifier verdict for a task " +
				"(execution ledger — conversation history is stale). Never returns full transcripts.",
			parameters: Type.Object({
				execution_id: Type.String(),
			}),
			async execute(_id, params, _signal, _onUpdate, _ctx) {
				return await executeVia(ctx, "rosclaw_task_observe", params as Record<string, unknown>);
			},
		}),
		defineTool({
			name: "rosclaw_task_steer",
			label: "ROSClaw Task Steer",
			description: "Send a steering note to the SAME running execution session.",
			parameters: Type.Object({
				execution_id: Type.String(),
				message: Type.String(),
			}),
			async execute(_id, params, _signal, _onUpdate, _ctx) {
				return await executeVia(ctx, "rosclaw_task_steer", params as Record<string, unknown>);
			},
		}),
		defineTool({
			name: "rosclaw_task_answer",
			label: "ROSClaw Task Answer",
			description: "Answer an execution's question (INPUT_REQUIRED) — same session continues.",
			parameters: Type.Object({
				execution_id: Type.String(),
				answer: Type.String(),
			}),
			async execute(_id, params, _signal, _onUpdate, _ctx) {
				return await executeVia(ctx, "rosclaw_task_answer", params as Record<string, unknown>);
			},
		}),
		defineTool({
			name: "rosclaw_task_pause",
			label: "ROSClaw Task Pause",
			description: "Pause a running execution (control ACK — session preserved).",
			parameters: Type.Object({ execution_id: Type.String() }),
			async execute(_id, params, _signal, _onUpdate, _ctx) {
				return await executeVia(ctx, "rosclaw_task_pause", params as Record<string, unknown>);
			},
		}),
		defineTool({
			name: "rosclaw_task_resume",
			label: "ROSClaw Task Resume",
			description: "Resume a paused execution (same session).",
			parameters: Type.Object({ execution_id: Type.String() }),
			async execute(_id, params, _signal, _onUpdate, _ctx) {
				return await executeVia(ctx, "rosclaw_task_resume", params as Record<string, unknown>);
			},
		}),
		defineTool({
			name: "rosclaw_task_cancel",
			label: "ROSClaw Task Cancel",
			description: "Cancel an execution (audited control cancel).",
			parameters: Type.Object({ execution_id: Type.String() }),
			async execute(_id, params, _signal, _onUpdate, _ctx) {
				return await executeVia(ctx, "rosclaw_task_cancel", params as Record<string, unknown>);
			},
		}),
	];
}
