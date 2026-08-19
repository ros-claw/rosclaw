/** Process 工具（PR-H3，总纲 v2 §10.2/§11.4）——长进程 = Operation。
 *
 * process_start 立即返回 operation_id（不在 tool call 里死等）；
 * 输出/进度经 task_events 流可查（process_output 按 seq 读）；终态
 * 由 OperationWatcher 一次性 followUp 注入同一 session。
 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ToolDefinition } from "@earendil-works/pi-coding-agent";

import { executeVia, type BridgeToolContext } from "./bridge-tools.js";

export function buildProcessTools(ctx: BridgeToolContext): ToolDefinition[] {
	return [
		defineTool({
			name: "process_start",
			label: "Process Start",
			description:
				"Start a long-running process as an Operation (returns operation_id " +
				"immediately — progress/output stream via process_output; you are " +
				"notified once on completion). Use for builds, simulations, renders, " +
				"long tests. For quick commands use bash instead.",
			parameters: Type.Object({
				command: Type.String({ description: "要后台运行的 shell 命令" }),
			}),
			async execute(_id, params, _signal, _onUpdate, _toolCtx) {
				return await executeVia(ctx, "rosclaw_process_start", {
					command: String(params.command ?? ""),
				});
			},
		}),
		defineTool({
			name: "process_status",
			label: "Process Status",
			description: "Read an operation's authoritative state (RUNNING/SUCCEEDED/FAILED/CANCELLED + heartbeat).",
			parameters: Type.Object({
				operation_id: Type.String(),
			}),
			async execute(_id, params, _signal, _onUpdate, _toolCtx) {
				return await executeVia(ctx, "rosclaw_process_status", {
					operation_id: String(params.operation_id ?? ""),
				});
			},
		}),
		defineTool({
			name: "process_output",
			label: "Process Output",
			description: "Read an operation's stdout/stderr stream (newest last; bounded tail).",
			parameters: Type.Object({
				operation_id: Type.String(),
				tail: Type.Optional(Type.Number({ description: "尾部条数（默认 50）" })),
			}),
			async execute(_id, params, _signal, _onUpdate, _toolCtx) {
				return await executeVia(ctx, "rosclaw_process_output", {
					operation_id: String(params.operation_id ?? ""),
					tail: Number(params.tail ?? 50),
				});
			},
		}),
		defineTool({
			name: "process_stop",
			label: "Process Stop",
			description: "Stop a running operation (ledger-first cancel — audited).",
			parameters: Type.Object({
				operation_id: Type.String(),
			}),
			async execute(_id, params, _signal, _onUpdate, _toolCtx) {
				return await executeVia(ctx, "rosclaw_process_stop", {
					operation_id: String(params.operation_id ?? ""),
				});
			},
		}),
	];
}
