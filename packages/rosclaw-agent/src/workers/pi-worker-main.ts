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

import { appendFileSync, readFileSync } from "node:fs";
import { writeSync } from "node:fs";

import { buildSystemPrompt, profileFor } from "./profiles.js";
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
	/** 十一审 PR-A：期望工件（DoD 注入——不再是永远的 plain text report）。 */
	expected_artifacts?: string[];
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

/** 十一审 PR-B：独立 transcript（会话级证据，不落主对话）。 */
function makeTranscriptWriter(envelope: WorkerEnvelope) {
	const dir = envelope.artifacts_dir
		? `${envelope.artifacts_dir}/..`
		: `${envelope.cwd}/.rosclaw-work`;
	const path = `${dir}/transcript.jsonl`;
	return (record: Record<string, unknown>) => {
		try {
			appendFileSync(path, `${JSON.stringify({ ts: new Date().toISOString(), ...record })}\n`, "utf-8");
		} catch {
			// transcript 失败不阻塞工作
		}
	};
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
				// 十一审 PR-A：manifest 化系统提示（prompt 与真实工具面
				// 逐字一致；DoD 按 expected artifacts 动态注入）。
				systemPrompt: buildSystemPrompt(
					profile,
					envelope.cwd,
					envelope.expected_artifacts ?? [],
				),
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

		// 十一审 PR-A：tool-contract 自检——session 实际工具必须与
		// profile 声明完全一致，否则不启动（诚实 TOOL_CONTRACT_MISMATCH，
		// 不让模型在错误工具面下工作）。
		{
			const active = new Set(session.getActiveToolNames());
			const missing = profile.tools.filter((name) => !active.has(name));
			if (missing.length > 0) {
				emit(wo, att, "attempt_failed", {
					error_code: "TOOL_CONTRACT_MISMATCH",
					message: `profile 声明的工具未在 session 生效: ${missing.join(", ")}`,
				});
				return 1;
			}
			emit(wo, att, "tool_contract_ok", { tools: [...active].sort() });
		}

		const usage: Usage = { input: 0, output: 0, cost: 0, turns: 0 };
		const messages: Array<Record<string, unknown>> = [];
		let stopReason = "";
		let errorMessage = "";
		// 十一审 PR-A：liveness/activity/semantic progress 三层分离。
		// liveness 每 2s 一条（只证明进程活着，不进模型上下文、不冒充
		// 进度）；semantic seq 只由真实事件推进。
		let phase = "STARTING";
		let spanStartedAt = Date.now();
		let semanticSeq = 0;
		let providerTimedOut = false;
		const semantic = (kind: string, payload: Record<string, unknown>) => {
			semanticSeq += 1;
			emit(wo, att, kind, payload);
		};
		const livenessTimer = setInterval(() => {
			emit(wo, att, "liveness", {
				phase,
				span_age_ms: Date.now() - spanStartedAt,
				pid_alive: true,
				last_semantic_seq: semanticSeq,
			});
		}, 2000);
		livenessTimer.unref();
		// provider request timeout：单个模型 turn 超过阈值（默认 10 分钟）
		// 才视为 provider 失败——高 thinking 长推理不再被 60s 误杀。
		const providerTimeoutMs = Number(process.env.ROSCLAW_WORKER_TURN_TIMEOUT_MS ?? 600_000);
		const providerTimer = setInterval(() => {
			if (phase === "RUNNING_MODEL" && Date.now() - spanStartedAt > providerTimeoutMs) {
				providerTimedOut = true;
				void session.abort().catch(() => undefined);
			}
		}, 2000);
		providerTimer.unref();
		const transcript = makeTranscriptWriter(envelope);
		const unsubscribe = session.subscribe((event) => {
			const e = event as unknown as {
				type: string;
				message?: { role?: string; usage?: Record<string, number>; stopReason?: string; errorMessage?: string };
				toolName?: string;
				toolCallId?: string;
				isError?: boolean;
			};
			if (event.type === "turn_start") {
				phase = "RUNNING_MODEL";
				spanStartedAt = Date.now();
				semantic("model_started", { turn: usage.turns + 1 });
			} else if (event.type === "tool_execution_start") {
				phase = "RUNNING_TOOL";
				spanStartedAt = Date.now();
				semantic("tool_started", { tool: e.toolName ?? "?" });
			} else if (event.type === "tool_execution_end") {
				phase = "RUNNING_MODEL";
				spanStartedAt = Date.now();
				semantic("tool_finished", {
					tool: e.toolName ?? "?",
					is_error: e.isError === true,
				});
				transcript({ role: "tool", tool: e.toolName ?? "?", is_error: e.isError === true });
			} else if (event.type === "message_end" && e.message) {
				messages.push(e.message as Record<string, unknown>);
				{
					const parts = ((e.message as { content?: unknown }).content ?? []) as Array<{ type?: string; text?: string }>;
					const text = parts.filter((b) => b.type === "text").map((b) => b.text ?? "").join("");
					transcript({
						role: e.message.role,
						text: text.length > 4000 ? `${text.slice(0, 4000)}…[truncated]` : text,
					});
				}
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

		// 十审 W4：stdin steer 通道——supervisor 写入单行 JSON
		// {type:"steer", text} 即转向运行中的 Worker（custom 消息，
		// 不冒充用户输入）。
		process.stdin.setEncoding("utf-8");
		let stdinBuf = "";
		process.stdin.on("data", (chunk: string) => {
			stdinBuf += chunk;
			const lines = stdinBuf.split("\n");
			stdinBuf = lines.pop() ?? "";
			for (const line of lines) {
				if (!line.trim()) continue;
				try {
					const msg = JSON.parse(line) as { type?: string; text?: string };
					if (msg.type === "steer" && msg.text) {
						void session
							.sendCustomMessage(
								{
									role: "custom",
									customType: "rosclaw.worker.steer",
									content: `Supervisor steer（追加约束，权威来自 agentd WorkOrder）：${msg.text}`,
									display: false,
									details: { source: "rosclaw_update_work" },
									timestamp: Date.now(),
								} as never,
								{ deliverAs: "steer" },
							)
							.then(() => emit(wo, att, "steer_ack", { text: msg.text?.slice(0, 200) }))
							.catch(() => undefined);
					}
				} catch {
					// malformed line 忽略
				}
			}
		});

		const task =
			`WorkOrder goal: ${envelope.goal}\n\n` +
			`Instructions: ${envelope.instructions || envelope.goal}\n\n` +
			"Deliverable: a concise final report as plain text (facts verified " +
			"with your tools, with concrete paths/line numbers where relevant).";
		await session.prompt(task);
		unsubscribe();
		clearInterval(livenessTimer);
		clearInterval(providerTimer);

		if (providerTimedOut) {
			emit(wo, att, "attempt_failed", {
				error_code: "PROVIDER_TIMEOUT",
				message: `单个模型请求超过 ${Math.round(providerTimeoutMs / 1000)}s`,
				usage,
			});
			return 1;
		}
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
