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

import { appendFileSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { writeSync } from "node:fs";

import { buildSystemPrompt, profileFor } from "./profiles.js";
import { finalTextOfMessages, normalizeAssistantContent, terminalStatusFromReport } from "./content-normalize.js";
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

/** 十一审 PR-B：独立 transcript（会话级证据，不落主对话）。
 *  十四审 PR-14.3：tseq 单调游标 + channel 分频（conversation/tools/
 *  files/artifacts/usage/control）——完整公开会话，不再只有预览。 */
let _tseq = 0;
function makeTranscript(envelope: WorkerEnvelope) {
	const dir = envelope.artifacts_dir
		? `${envelope.artifacts_dir}/..`
		: `${envelope.cwd}/.rosclaw-work`;
	// 目录必须先行——artifacts/../ 路径穿越要求父目录存在（否则
	// appendFileSync ENOENT，catch 静默吞掉 = transcript 整体丢失）。
	mkdirSync(dir, { recursive: true });
	const path = `${dir}/transcript.jsonl`;
	return (channel: string, record: Record<string, unknown>) => {
		_tseq += 1;
		try {
			appendFileSync(
				path,
				`${JSON.stringify({ tseq: _tseq, ts: new Date().toISOString(), channel, ...record })}\n`,
				"utf-8",
			);
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
			emitRecord: (kind, payload) => transcript("files", { kind, ...payload }),
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
		// 建议-0816 P0-1：模型 turn 默认无硬截止（长推理/编译合法）。
		// 超阈值只发 provider_slow 告警（显示用）；只有显式
		// ROSCLAW_WORKER_TURN_TIMEOUT_MS（操作员/benchmark 权威）才 abort。
		let providerTimedOut = false;
		const providerWarnMs = Number(process.env.ROSCLAW_WORKER_TURN_WARN_MS ?? 600_000);
		const providerAbortMs = process.env.ROSCLAW_WORKER_TURN_TIMEOUT_MS
			? Number(process.env.ROSCLAW_WORKER_TURN_TIMEOUT_MS)
			: 0;
		let providerWarned = false;
		const providerTimer = setInterval(() => {
			const ageMs = Date.now() - spanStartedAt;
			if (phase === "RUNNING_MODEL" && ageMs > providerWarnMs && !providerWarned) {
				providerWarned = true;
				emit(wo, att, "provider_slow", {
					span_age_ms: ageMs,
					note: "单个模型 turn 超过提醒阈值——仍在运行（不杀）",
				});
			}
			if (providerAbortMs > 0 && phase === "RUNNING_MODEL" && ageMs > providerAbortMs) {
				providerTimedOut = true;
				void session.abort().catch(() => undefined);
			}
		}, 2000);
		providerTimer.unref();
		const transcript = makeTranscript(envelope);
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
				const rawArgs = (event as { args?: unknown }).args;
				semantic("tool_started", {
					tool: e.toolName ?? "?",
					args_preview: argsPreview(e.toolName ?? "", rawArgs, envelope.cwd),
				});
				// 十四审 PR-14.3：tools channel——完整（脱敏、8KiB 上限）
				// 参数进 transcript，preview 只用于卡片摘要。
				let argsFull = "";
				try {
					argsFull = redactText(JSON.stringify(rawArgs ?? {})).slice(0, 8192);
				} catch {
					argsFull = "";
				}
				transcript("tools", {
					phase: "start",
					tool: e.toolName ?? "?",
					args: argsFull,
				});
			} else if (event.type === "tool_execution_update") {
				// 限频 passthrough（每工具 500ms 一条上限）——带 partial
				// 输出预览（十三审 §1.2：不再只留工具名）。
				const callId = (event as { toolCallId?: string }).toolCallId ?? "";
				const nowMs = Date.now();
				if (nowMs - (toolUpdateAt.get(callId) ?? 0) >= 500) {
					toolUpdateAt.set(callId, nowMs);
					const partial = (event as { partialResult?: { content?: unknown } }).partialResult;
					const preview = partial
						? redactText(normalizeAssistantContent(partial.content ?? "").text).slice(-200)
						: "";
					semantic("tool_progress", {
						tool: e.toolName ?? "?",
						tool_call_id: callId,
						message: preview,
					});
				}
			} else if (event.type === "tool_execution_end") {
				phase = "RUNNING_MODEL";
				spanStartedAt = Date.now();
				const fullOutput = redactText(
					normalizeAssistantContent(
						((event as { result?: { content?: unknown } }).result ?? {}).content ?? "",
					).text,
				);
				// 结果输出预览（卡片摘要）vs 完整输出（transcript 证据）。
				semantic("tool_finished", {
					tool: e.toolName ?? "?",
					is_error: e.isError === true,
					output_preview: fullOutput.slice(0, 400),
				});
				transcript("tools", {
					phase: "end",
					tool: e.toolName ?? "?",
					is_error: e.isError === true,
					output: fullOutput.slice(0, 51_200),
				});
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
					// 十四审 PR-14.3：完整公开全文（脱敏，200KiB 上限）——
					// 不再 4000 字截断；隐藏思维链永不进 transcript（pi 设置
					// hideThinkingBlock，content 里本就不含）。
					transcript("conversation", {
						role: e.message.role,
						text: redactText(text).slice(0, 204_800),
						stop_reason: e.message.stopReason ?? undefined,
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
					transcript("usage", {
						input: usage.input,
						output: usage.output,
						turns: usage.turns,
						cost: usage.cost,
					});
				}
			}
		});

		// 十四审 PR-14.1：SIGTERM/SIGINT 不再静默映射 exit 130。
		// prompt 进行中：记录 signal 原因并 abort（prompt 返回后按
		// SIGNAL_UNKNOWN 落 termination.json——supervisor 据此判
		// INTERRUPTED_RESUMABLE 而非 FAILED）；空闲（PAUSED 等待中）：
		// 直接落 AGENTD_SHUTDOWN 并退出（信号即关闭意图）。
		let inPrompt = false;
		let abortReason: "pause" | "cancel" | "signal" | null = null;
		const abort = () => {
			if (inPrompt) {
				abortReason = abortReason ?? "signal";
				void session.abort().catch(() => undefined);
			} else {
				writeTermination("AGENTD_SHUTDOWN", "signal while idle", 143);
				process.exit(143);
			}
		};
		process.on("SIGTERM", abort);
		process.on("SIGINT", abort);

		// termination.json——终态原因唯一权威（总纲 §3.4）。原子写
		// （tmp+rename）；exit code 只是 Unix 表象，不得当语义。
		const workDir = envelope.artifacts_dir
			? `${envelope.artifacts_dir}/..`
			: `${envelope.cwd}/.rosclaw-work`;
		mkdirSync(workDir, { recursive: true });
		mkdirSync(envelope.artifacts_dir ?? `${envelope.cwd}/.rosclaw-work`, {
			recursive: true,
		});
		function writeTermination(cause: string, detail: string, exitCode: number) {
			try {
				const payload = JSON.stringify({
					schema_version: "rosclaw.worker_termination.v1",
					cause,
					detail: redactText(detail).slice(0, 500),
					exit_code: exitCode,
					session_file: sessionManager.getSessionFile() ?? "",
					at: new Date().toISOString(),
				});
				writeFileSync(`${workDir}/termination.json.tmp`, payload, "utf-8");
				renameSync(
					`${workDir}/termination.json.tmp`,
					`${workDir}/termination.json`,
				);
			} catch {
				// termination 落盘失败不阻塞退出（supervisor 按 SIGNAL_UNKNOWN 兜底）
			}
			emit(wo, att, "termination", { cause, exit_code: exitCode });
		}

		// 十四审 PR-14.1：WorkerSessionHost 控制状态机（总纲 §3.3）。
		// RUNNING → PAUSE_REQUESTED →（模型循环真实停止后）PAUSED →
		// resume → RUNNING（同一 session 继续）。只有 cancel 产生
		// CANCELLED；pause 后进程必须存活等待控制消息。
		type HostState = "RUNNING" | "PAUSE_REQUESTED" | "PAUSED";
		let hostState: HostState = "RUNNING";
		const pendingPauseIds: string[] = [];
		let resumeWaiter: (() => void) | null = null;
		const ack = (controlId: string, state: string) => {
			emit(wo, att, "control.ack", {
				control_id: controlId,
				state,
				session_id: sessionManager.getSessionFile() ?? "",
			});
			transcript("control", { control_id: controlId, state });
		};
		const requestPause = (controlId: string) => {
			if (hostState === "PAUSED") {
				ack(controlId, "PAUSED");
				return;
			}
			if (hostState === "PAUSE_REQUESTED") {
				pendingPauseIds.push(controlId);
				ack(controlId, "PAUSE_REQUESTED");
				return;
			}
			hostState = "PAUSE_REQUESTED";
			ack(controlId, "PAUSE_REQUESTED");
			if (inPrompt) {
				pendingPauseIds.push(controlId);
				abortReason = "pause";
				void session.abort().catch(() => undefined);
			} else {
				hostState = "PAUSED";
				ack(controlId, "PAUSED");
			}
		};
		const requestResume = (controlId: string) => {
			if (hostState === "PAUSE_REQUESTED") {
				// pause 尚未生效即 resume——撤销暂停：pause 请求方不能
				// 永远等 PAUSED（统一 ACK RUNNING 收尾）。
				hostState = "RUNNING";
				abortReason = null;
				for (const cid of pendingPauseIds.splice(0)) ack(cid, "RUNNING");
				ack(controlId, "RUNNING");
				return;
			}
			if (hostState !== "PAUSED") {
				ack(controlId, "RUNNING");
				return;
			}
			hostState = "RUNNING";
			ack(controlId, "RUNNING");
			resumeWaiter?.();
		};
		const requestCancel = (controlId: string) => {
			abortReason = "cancel";
			if (inPrompt) {
				ack(controlId, "CANCELLED");
				void session.abort().catch(() => undefined);
			} else {
				// PAUSED/空闲：无 prompt 可 abort——直接按用户取消终止。
				ack(controlId, "CANCELLED");
				emit(wo, att, "attempt_cancelled", { usage });
				writeTermination("USER_CANCELLED", "cancel while paused", 130);
				process.exit(130);
			}
		};

		// stdin 控制通道——supervisor 写入单行 JSON。新协议
		// control.request（带 control_id，必须 ACK）；legacy pause/extend
		// 映射进同一状态机（extend = 预算追加 + resume）。
		process.stdin.setEncoding("utf-8");
		let stdinBuf = "";
		process.stdin.on("data", (chunk: string) => {
			stdinBuf += chunk;
			const lines = stdinBuf.split("\n");
			stdinBuf = lines.pop() ?? "";
			for (const line of lines) {
				if (!line.trim()) continue;
				try {
					const msg = JSON.parse(line) as {
						type?: string;
						text?: string;
						control_id?: string;
						action?: string;
					};
					if (msg.type === "control.request" && msg.action) {
						const cid = msg.control_id || `ctl_${Date.now()}`;
						if (msg.action === "pause") requestPause(cid);
						else if (msg.action === "resume") requestResume(cid);
						else if (msg.action === "cancel") requestCancel(cid);
					} else if (msg.type === "pause") {
						// legacy（无 control_id）——同一状态机。
						requestPause(`ctl_legacy_${Date.now()}`);
					} else if (msg.type === "extend") {
						// 预算追加——resume 同一会话（若未暂停则空操作 ACK）。
						emit(wo, att, "extended", { add_tokens: (msg as { add_tokens?: number }).add_tokens ?? 0 });
						requestResume(`ctl_extend_${Date.now()}`);
					} else if (msg.type === "answer" && msg.text !== undefined) {
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
		const firstTask =
			resumePrefix +
			`WorkOrder goal: ${envelope.goal}\n\n` +
			`Instructions: ${envelope.instructions || envelope.goal}\n\n` +
			"Deliverable: a concise final report as plain text (facts verified " +
			"with your tools, with concrete paths/line numbers where relevant)." +
			dod;
		// WorkerSessionHost 主循环（总纲 §3.3）：prompt 返回不是进程终点；
		// aborted+pause → PAUSED 等待控制；只有完成/失败/取消/信号才退出。
		let nextPrompt = firstTask;
		let exitCode: number | null = null;
		while (exitCode === null) {
			if (hostState === "PAUSED") {
				// 进程存活等待 resume/cancel（控制通道与 transcript 保持）。
				await new Promise<void>((resolve) => {
					resumeWaiter = () => {
						resumeWaiter = null;
						resolve();
					};
				});
				nextPrompt =
					"控制通道已恢复。继续之前的任务——从当前状态接着做，不要从零开始。";
			}
			abortReason = null;
			stopReason = "";
			errorMessage = "";
			inPrompt = true;
			try {
				await session.prompt(nextPrompt);
			} catch (promptErr) {
				errorMessage = `${(promptErr as Error).name}: ${(promptErr as Error).message}`;
			}
			inPrompt = false;
			// 自审修复：session 文件在首个条目写入后才存在——每个 prompt
			// 边界回告真实路径（resume 才能拿到有效 checkpoint）。
			emit(wo, att, "session_persisted", {
				session_file: sessionManager.getSessionFile() ?? "",
			});

			if (providerTimedOut) {
				emit(wo, att, "attempt_failed", {
					error_code: "PROVIDER_TIMEOUT",
					message: `单个模型请求超过显式授权阈值 ${Math.round(providerAbortMs / 1000)}s`,
					usage,
				});
				writeTermination("PROVIDER_TRANSIENT", "provider turn timeout (explicit limit)", 1);
				exitCode = 1;
				break;
			}
			if (stopReason === "aborted") {
				if (abortReason === "pause") {
					// 模型循环真实停止——现在才可 ACK PAUSED（进程存活）。
					hostState = "PAUSED";
					for (const cid of pendingPauseIds.splice(0)) ack(cid, "PAUSED");
					emit(wo, att, "paused", {});
					continue;
				}
				if (abortReason === "cancel") {
					// 只有用户取消产生 CANCELLED。
					emit(wo, att, "attempt_cancelled", { usage });
					writeTermination("USER_CANCELLED", "cancel requested", 130);
					exitCode = 130;
					break;
				}
				// 信号/未知 abort：不是取消也不是失败——中断可恢复。
				emit(wo, att, "attempt_cancelled", { usage, reason: "signal" });
				writeTermination("SIGNAL_UNKNOWN", "aborted without control request", 130);
				exitCode = 130;
				break;
			}
			if (stopReason === "error" || errorMessage) {
				const transient =
					/timeout|timed out|429|5\d\d|ECONNRESET|ETIMEDOUT|ENOTFOUND|rate.?limit|overloaded/i.test(
						errorMessage,
					);
				emit(wo, att, "attempt_failed", {
					error_code: "MODEL_ERROR",
					message: errorMessage || stopReason,
					usage,
				});
				writeTermination(
					transient ? "PROVIDER_TRANSIENT" : "PROVIDER_FATAL",
					errorMessage || stopReason,
					1,
				);
				exitCode = 1;
				break;
			}
			const finalReport = finalTextOf(messages);
			// 十六审 A3：终态协议解析——模型报告末尾的
			// `TERMINAL STATUS: BLOCKED` 是 harness 协议标记，转成
			// termination.json 的结构化 cause（ROSClaw 侧不做文本推断）。
			const terminalStatus = terminalStatusFromReport(finalReport);
			emit(wo, att, "attempt_finished", {
				report: finalReport,
				status: terminalStatus,
				usage,
				model: snapshot ? `${snapshot.provider}/${snapshot.model}` : undefined,
			});			// 十四审 PR-14.3：artifacts channel——产物清单带 sha256（部分
			// 成果产品化的账本证据：trace/CSV/帧/日志都可审计）。
			try {
				const { createHash } = await import("node:crypto");
				const { readdirSync, statSync } = await import("node:fs");
				const dir = envelope.artifacts_dir ?? `${envelope.cwd}/.rosclaw-work`;
				const files = readdirSync(dir)
					.filter((name) => {
						try {
							return statSync(`${dir}/${name}`).isFile();
						} catch {
							return false;
						}
					})
					.map((name) => {
						const content = readFileSync(`${dir}/${name}`);
						return {
							name,
							bytes: content.length,
							sha256: createHash("sha256").update(content).digest("hex"),
						};
					});
				transcript("artifacts", { files });
			} catch {
				// artifacts 清点失败不阻塞完成
			}
			if (terminalStatus === "BLOCKED") {
				// Worker 诚实报告缺能力/缺输入——结构化 BLOCKED 终态
				// （退出码 0：进程正常结束，状态由 termination.json 携带）。
				writeTermination("BLOCKED", finalReport.slice(-300), 0);
			} else {
				writeTermination("COMPLETED", "", 0);
			}
			exitCode = 0;
		}
		unsubscribe();
		clearInterval(livenessTimer);
		clearInterval(providerTimer);
		return exitCode;
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
		// 十四审：崩溃也要落 termination.json（WORKER_CRASH）——try 内的
		// writeTermination 此处不可见，内联原子写兜底。
		try {
			const dir = envelope.artifacts_dir
				? `${envelope.artifacts_dir}/..`
				: `${envelope.cwd}/.rosclaw-work`;
			writeFileSync(
				`${dir}/termination.json.tmp`,
				JSON.stringify({
					schema_version: "rosclaw.worker_termination.v1",
					cause: "WORKER_CRASH",
					detail: `${e.name}: ${e.message}`.slice(0, 500),
					exit_code: 1,
					session_file: "",
					at: new Date().toISOString(),
				}),
				"utf-8",
			);
			renameSync(`${dir}/termination.json.tmp`, `${dir}/termination.json`);
		} catch {
			// 落盘失败不阻塞退出
		}
		return 1;
	}
}
