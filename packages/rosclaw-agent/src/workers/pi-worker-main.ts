/** 内置 Pi headless Worker 入口（十审 W1，审计 §8）。
 *
 * `rosclaw-agent worker --headless --work-order <path>`：
 * - 从只读 WorkOrder envelope 读任务（goal/instructions/profile/模型快照）；
 * - 与主 Agent 共用同一 agentDir/auth.json/models.json（不配置第二份 key）；
 * - stdout 输出 JSONL WorkerEvent（attempt_started/model_started/
 *   tool_started/tool_finished/usage/attempt_finished/attempt_failed）；
 * - SIGTERM/SIGINT → session.abort()（进程组 kill 由父 supervisor 兜底）。
 *
 * 移植自 Pi 上游 examples/extensions/subagent（MIT）的事件解析/usage
 * 聚合思想——见 THIRD_PARTY_NOTICES.md。不发现 ~/.pi agents、不执行
 * PATH 上的 pi、不加载项目资源。
 */

import { readFileSync } from "node:fs";
import { writeSync } from "node:fs";

import { profileFor } from "./profiles.js";
import { createSharedModelRuntime } from "../runtime/model-runtime.js";

export interface WorkerEnvelope {
	work_order_id: string;
	attempt_id: string;
	profile: string;
	goal: string;
	instructions: string;
	cwd: string;
	budget: { wall_time_sec: number; model_tokens: number };
	model?: { provider: string; model: string; thinking?: string };
	/** W3：工件目录（bash log、patch、渲染产物）。 */
	artifacts_dir?: string;
}

interface Usage {
	input: number;
	output: number;
	cost: number;
	turns: number;
}

let _seq = 0;
function emit(workOrderId: string, attemptId: string, kind: string, payload: Record<string, unknown>) {
	_seq += 1;
	const line = JSON.stringify({
		schema_version: "rosclaw.worker_event.v1",
		work_order_id: workOrderId,
		attempt_id: attemptId,
		seq: _seq,
		kind,
		...payload,
		emitted_at: new Date().toISOString(),
	});
	// 直接写 fd 1——console.log 可能被上游库劫持/缓冲。
	writeSync(1, `${line}\n`);
}

function finalTextOf(messages: Array<Record<string, unknown>>): string {
	for (let i = messages.length - 1; i >= 0; i -= 1) {
		const msg = messages[i] as { role?: string; content?: unknown };
		if (msg.role !== "assistant") continue;
		const parts = (msg.content ?? []) as Array<{ type?: string; text?: string }>;
		for (const part of parts) {
			if (part.type === "text" && part.text) return part.text;
		}
	}
	return "";
}

export async function runHeadlessWorker(argv: string[]): Promise<number> {
	let orderPath = "";
	for (let i = 0; i < argv.length; i += 1) {
		if (argv[i] === "--work-order" && argv[i + 1]) {
			orderPath = argv[i + 1];
			i += 1;
		}
	}
	if (!orderPath) {
		writeSync(2, "worker --headless requires --work-order <path>\n");
		return 2;
	}
	const envelope = JSON.parse(readFileSync(orderPath, "utf-8")) as WorkerEnvelope;
	const wo = envelope.work_order_id;
	const att = envelope.attempt_id || "att_0";
	const profile = profileFor(envelope.profile);
	const rosclawHome = process.env.ROSCLAW_HOME ?? `${process.env.HOME}/.rosclaw`;
	const agentDir = `${rosclawHome}/agent`;

	emit(wo, att, "attempt_started", {
		profile: profile.name,
		worker: "worker:rosclaw:pi",
	});

	try {
		const {
			createAgentSessionServices,
			createAgentSessionFromServices,
			SessionManager,
			SettingsManager,
		} = await import("@earendil-works/pi-coding-agent");
		// 与主 Agent 同一 ModelRuntime 配置（auth.json/models.json）。
		const modelRuntime = await createSharedModelRuntime(agentDir, "developer");
		const snapshot = envelope.model;
		const model = snapshot
			? modelRuntime.getModel(snapshot.provider, snapshot.model)
			: undefined;
		if (snapshot && !model) {
			emit(wo, att, "attempt_failed", {
				error_code: "MODEL_UNAVAILABLE",
				message: `snapshot model ${snapshot.provider}/${snapshot.model} 在本机 ModelRuntime 不可用`,
			});
			return 1;
		}
		const settingsManager = SettingsManager.create(envelope.cwd, agentDir);
		settingsManager.setQuietStartup(true);
		settingsManager.setHideThinkingBlock(true);
		const services = await createAgentSessionServices({
			cwd: envelope.cwd,
			agentDir,
			settingsManager,
			modelRuntime,
			resourceLoaderOptions: {
				noExtensions: true,
				noSkills: true,
				noPromptTemplates: true,
				noThemes: true,
				noContextFiles: true,
				// profile 系统提示（W1 起定义但此前未注入——W3 修复）。
				systemPrompt: profile.systemPrompt,
			},
		});
		// 十审 W3：全部 profile 使用 Workbench 约束工具（custom 同名覆盖
		// Pi 内建）——scout/analyst 是只读子集，developer/sim-builder 增加
		// write/edit/bash。路径必须在 workspace 内、bash argv 白名单、
		// env 不含凭据（防 read auth.json / curl 外泄）。
		const { buildWorkbenchTools } = await import("./workbench.js");
		const workbenchTools = buildWorkbenchTools({
			root: envelope.cwd,
			bashLogPath: `${envelope.artifacts_dir ?? `${envelope.cwd}/.rosclaw-work`}/bash-log.txt`,
			emitProgress: (message) => emit(wo, att, "tool_progress", { message }),
		}).filter((tool) => profile.tools.includes(tool.name));
		const { session } = await createAgentSessionFromServices({
			services,
			sessionManager: SessionManager.inMemory(envelope.cwd),
			tools: profile.tools,
			customTools: workbenchTools,
			...(model ? { model } : {}),
		});

		const usage: Usage = { input: 0, output: 0, cost: 0, turns: 0 };
		const messages: Array<Record<string, unknown>> = [];
		let stopReason = "";
		let errorMessage = "";
		const unsubscribe = session.subscribe((event) => {
			const e = event as unknown as {
				type: string;
				message?: { role?: string; usage?: Record<string, number>; stopReason?: string; errorMessage?: string };
				toolName?: string;
				toolCallId?: string;
				isError?: boolean;
			};
			if (event.type === "turn_start") {
				emit(wo, att, "model_started", { turn: usage.turns + 1 });
			} else if (event.type === "tool_execution_start") {
				emit(wo, att, "tool_started", { tool: e.toolName ?? "?" });
			} else if (event.type === "tool_execution_end") {
				emit(wo, att, "tool_finished", {
					tool: e.toolName ?? "?",
					is_error: e.isError === true,
				});
			} else if (event.type === "message_end" && e.message) {
				messages.push(e.message as Record<string, unknown>);
				if (e.message.role === "assistant") {
					usage.turns += 1;
					const u = e.message.usage ?? {};
					usage.input += u.input ?? 0;
					usage.output += u.output ?? 0;
					const cost = (u as { cost?: { total?: number } }).cost?.total ?? 0;
					usage.cost += cost;
					if (e.message.stopReason) stopReason = e.message.stopReason;
					if (e.message.errorMessage) errorMessage = e.message.errorMessage;
					emit(wo, att, "usage", {
						input_tokens: usage.input,
						output_tokens: usage.output,
						turns: usage.turns,
					});
				}
			}
		});

		// SIGTERM/SIGINT：abort 当前 turn（attempt_cancelled 由父 supervisor
		// 在进程真正退出后落账——这里尽力 abort，保证 2s 级退出）。
		const abort = () => {
			void session.abort().catch(() => undefined);
		};
		process.on("SIGTERM", abort);
		process.on("SIGINT", abort);

		const task =
			`WorkOrder goal: ${envelope.goal}\n\n` +
			`Instructions: ${envelope.instructions || envelope.goal}\n\n` +
			"Deliverable: a concise final report as plain text (facts verified " +
			"with your tools, with concrete paths/line numbers where relevant).";
		await session.prompt(task);
		unsubscribe();

		if (stopReason === "aborted") {
			emit(wo, att, "attempt_cancelled", { usage });
			return 130;
		}
		if (stopReason === "error" || errorMessage) {
			emit(wo, att, "attempt_failed", {
				error_code: "MODEL_ERROR",
				message: errorMessage || stopReason,
				usage,
			});
			return 1;
		}
		emit(wo, att, "attempt_finished", {
			report: finalTextOf(messages),
			usage,
			model: snapshot ? `${snapshot.provider}/${snapshot.model}` : undefined,
		});
		return 0;
	} catch (err) {
		emit(wo, att, "attempt_failed", {
			error_code: "WORKER_CRASH",
			message: `${(err as Error).name}: ${(err as Error).message}`,
		});
		return 1;
	}
}
