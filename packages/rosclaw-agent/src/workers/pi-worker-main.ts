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
import { finalTextOfMessages, normalizeAssistantContent } from "./content-normalize.js";
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
	/** 十二审 PR-12.3：持久 session 目录（本 attempt 的 Pi 原生会话）。 */
	session_dir?: string;
	/** 十二审 PR-12.3：resume——上一 attempt 的 session 文件（恢复同一
	 *  Pi 会话：工具历史与上下文保留）。 */
	resume_session_file?: string;
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

/** 十二审 PR-12.2：事件桥脱敏（与 Python redact 同规则）。 */
function redactText(text: string): string {
	return text
		.replace(/sk-[A-Za-z0-9]{12,}/g, "sk-***REDACTED***")
		.replace(/(api[_-]?key|secret|password|token)(\s*[:=]\s*)['"]?[\w\-]{8,}/gi, "$1$2***REDACTED***");
}

/** 工具参数预览（相对路径/截断/脱敏——不展示敏感全文）。 */
function argsPreview(tool: string, args: unknown, cwd: string): string {
	try {
		const a = (args ?? {}) as Record<string, unknown>;
		let raw: string;
		if (tool === "bash") raw = Array.isArray(a.argv) ? (a.argv as string[]).join(" ") : JSON.stringify(a);
		else if (typeof a.path === "string") {
			raw = a.path.startsWith(cwd) ? a.path.slice(cwd.length + 1) : a.path;
			if (a.pattern) raw += ` /${String(a.pattern)}/`;
		} else raw = JSON.stringify(a);
		return redactText(raw.slice(0, 120));
	} catch {
		return "";
	}
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

// 十二审 HOTFIX-12.1：统一经 content-normalize（任意 provider shape）。
const finalTextOf = finalTextOfMessages;

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
		// 十一审 PR-E：WAITING_INPUT——提问事件 + stdin 等回答（answer
		// 由 /job answer 经 supervisor 写入）。
		let answerResolve: ((text: string) => void) | undefined;
		const waitersApi = {
			resolve: (text: string) => {
				answerResolve?.(text);
				answerResolve = undefined;
			},
		};
		(globalThis as Record<string, unknown>).__rosclawAnswerWaiter = waitersApi;
		const workbenchTools = buildWorkbenchTools({
			root: envelope.cwd,
			bashLogPath: `${envelope.artifacts_dir ?? `${envelope.cwd}/.rosclaw-work`}/bash-log.txt`,
			emitProgress: (message) => emit(wo, att, "tool_progress", { message }),
			askUser: (question) => {
				emit(wo, att, "waiting_input", { question: question.slice(0, 500) });
				return new Promise<string>((resolvePromise) => {
					answerResolve = resolvePromise;
				});
			},
		}).filter((tool) => profile.tools.includes(tool.name) || tool.name === "ask_user");
		// 十二审 PR-12.3：持久 SessionManager（落盘）——崩溃/重启后
		// 会话可恢复；resume 时 SessionManager.open 继续同一 Pi 会话
		// （retry ≠ resume：retry 是新 attempt，resume 恢复上下文）。
		const sessionManager = envelope.resume_session_file
			? SessionManager.open(envelope.resume_session_file, envelope.session_dir)
			: envelope.session_dir
				? SessionManager.create(envelope.cwd, envelope.session_dir)
				: SessionManager.inMemory(envelope.cwd);
		const { session } = await createAgentSessionFromServices({
			services,
			sessionManager,
			tools: profile.tools,
			customTools: workbenchTools,
			...(model ? { model } : {}),
		});
		if (envelope.resume_session_file) {
			emit(wo, att, "session_resumed", { from: envelope.resume_session_file });
		}
		// session 文件路径回告（supervisor 记录 checkpoint 用）。
		emit(wo, att, "session_persisted", {
			session_file: sessionManager.getSessionFile() ?? "",
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
		// 十二审 PR-12.2：delta 批量 + tool update 限频状态。
		let deltaBuf = "";
		let deltaLastFlush = 0;
		const toolUpdateAt = new Map<string, number>();
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
				// 十二审 PR-12.2：工具事件带脱敏参数预览（可回溯的真实
				// 活动——不是只有工具名）。
				semantic("tool_started", {
					tool: e.toolName ?? "?",
					args_preview: argsPreview(e.toolName ?? "", (event as { args?: unknown }).args, envelope.cwd),
				});
			} else if (event.type === "tool_execution_update") {
				// 限频 passthrough（每工具 500ms 一条上限）。
				const callId = (event as { toolCallId?: string }).toolCallId ?? "";
				const nowMs = Date.now();
				if (nowMs - (toolUpdateAt.get(callId) ?? 0) >= 500) {
					toolUpdateAt.set(callId, nowMs);
					semantic("tool_progress", { tool: e.toolName ?? "?", tool_call_id: callId });
				}
			} else if (event.type === "tool_execution_end") {
				phase = "RUNNING_MODEL";
				spanStartedAt = Date.now();
				// 结果输出预览（归一化 + 截断 + 脱敏）。
				const resultText = redactText(
					normalizeAssistantContent(
						((event as { result?: { content?: unknown } }).result ?? {}).content ?? "",
					).text,
				).slice(0, 400);
				semantic("tool_finished", {
					tool: e.toolName ?? "?",
					is_error: e.isError === true,
					output_preview: resultText,
				});
				transcript({ role: "tool", tool: e.toolName ?? "?", is_error: e.isError === true });
			} else if (event.type === "message_update") {
				// 十二审 PR-12.2：text delta 批量落盘（150ms/2KiB），不丢
				// 也不风暴。
				const ame = (event as { assistantMessageEvent?: { type?: string; delta?: string } })
					.assistantMessageEvent;
				if (ame?.type === "text_delta" && ame.delta) {
					deltaBuf += ame.delta;
					const nowMs = Date.now();
					if (deltaBuf.length >= 2048 || nowMs - deltaLastFlush >= 150) {
						emit(wo, att, "message_delta", {
							chars: deltaBuf.length,
							preview: redactText(deltaBuf.slice(-160)),
						});
						deltaBuf = "";
						deltaLastFlush = nowMs;
					}
				}
			} else if (event.type === "message_end" && e.message) {
				messages.push(e.message as Record<string, unknown>);
				{
					// 十二审 HOTFIX-12.1：content 可能是字符串/单对象/嵌套/
					// null——归一化后消费，绝不因 shape 崩 Worker。
					const normalized = normalizeAssistantContent(
						(e.message as { content?: unknown }).content,
					);
					if (normalized.unknownPartTypes.length > 0) {
						emit(wo, att, "adapter_note", {
							note: "unknown content parts ignored",
							part_types: normalized.unknownPartTypes,
						});
					}
					const text = normalized.text;
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
					if (msg.type === "answer" && msg.text !== undefined) {
						const waiter = (globalThis as Record<string, unknown>).__rosclawAnswerWaiter as
							| { resolve(text: string): void }
							| undefined;
						waiter?.resolve(String(msg.text));
						emit(wo, att, "answer_received", {});
					} else if (msg.type === "steer" && msg.text) {
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

		// 十一审 PR-A/§2.2：最终要求按 expected artifacts 动态生成——
		// developer 必须真实实现+测试，不是"plain text report"。
		const expected = envelope.expected_artifacts ?? [];
		const dod = expected.length
			? `\n\nRequired deliverables (verified by ROSClaw): ${expected.join(", ")}. ` +
				"Real file changes, exact test commands with exit codes, and all " +
				"requested artifacts. A design document alone is NOT completion."
			: "";
		const resumePrefix = envelope.resume_session_file
			? "这是同一任务的中断恢复（同一 Pi 会话——你的工具历史与上下文还在）。从上次中断处继续；不要从零开始。\n\n"
			: "";
		const task =
			resumePrefix +
			`WorkOrder goal: ${envelope.goal}\n\n` +
			`Instructions: ${envelope.instructions || envelope.goal}\n\n` +
			"Deliverable: a concise final report as plain text (facts verified " +
			"with your tools, with concrete paths/line numbers where relevant)." +
			dod;
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
		// 十二审 HOTFIX-12.1：错误分类——provider 消息 shape/协议问题
		// 是 ADAPTER_PROTOCOL_ERROR（同版本盲重试无意义），不是 MODEL_ERROR。
		const e = err as Error;
		const isProtocol =
			e instanceof TypeError
			&& /filter|map|forEach|reduce|is not a function|of undefined|of null/i.test(e.message);
		emit(wo, att, "attempt_failed", {
			error_code: isProtocol ? "ADAPTER_PROTOCOL_ERROR" : "WORKER_CRASH",
			message: `${e.name}: ${e.message}`,
		});
		return 1;
	}
}
