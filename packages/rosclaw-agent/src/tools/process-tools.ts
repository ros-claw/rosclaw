// HP2-COMPAT: 工具定义原语（defineTool/Type/ToolDefinition）——工具层在 HP3 投影层（Codex MCP）落地前保持 Pi 形态；不新增会话装配引用。
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
				"immediately). For finite jobs (builds, simulations, renders, long tests), " +
				"end your turn when waiting for termination: an unconsumed terminal " +
				"result triggers continuation while the owning task/revision remains active. " +
				"For long-lived services, there is NO automatic startup/ready notification. " +
				"Admission, RUNNING and stdout alone do not prove readiness. Use bounded " +
				"process_status/process_output and actual service observations to verify " +
				"readiness under the existing task deadline, then continue dependent " +
				"startup/test steps in the current turn; do not wait for service termination. " +
				"Do not sleep or poll without a bound. For quick commands use bash instead. " +
				"For an authorized isolated ROS fixture, discover the installed setup/overlay " +
				"and compatible interpreter; do not assume the ROS environment is inherited. " +
				"Invoke Bash explicitly for setup.bash (the managed wrapper may be sh). " +
				"If nounset is enabled, disable it only while sourcing ROS setup, exit on " +
				"source failure, then restore the prior option. Preserve the authorized " +
				"domain/namespace; this guidance grants no ROS graph or hardware authority.",
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
