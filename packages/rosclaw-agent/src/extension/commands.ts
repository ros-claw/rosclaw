// HP2-COMPAT: Pi 扩展宿主类型（ExtensionContext/Factory）——扩展运行於 Pi 扩展宿主内，HP3 前保持；不新增会话装配引用。
/** ROSClaw TUI 命令注册（NA-FIX-6，二次审计 P0-8 + 规格 §7.2）。
 *
 * 每个命令都有真实 handler + 权限路径 + 测试；/estop 走 dedicated
 * operatord 通道（模型/agentd/Pi session 全卡死也可触发）。
 */

import type { ExtensionCommandContext } from "@earendil-works/pi-coding-agent";
import { defaultOperatorSocket, operatorCall } from "../bridge/operatord-client.js";
import type { ActiveSessionContext } from "../session/active-context.js";
import type { ProductStateCenter } from "../session/state-center.js";
import type { LocaleManager } from "../i18n/locale.js";
import { t as i18nT } from "../i18n/index.js";

export interface CommandDeps {
	rosclawHome: string;
	active: ActiveSessionContext;
	/** PR-SIX-1：唯一状态中心——/status 与 rosclaw_status/Header/Footer
	 *  同一份快照（不再各自为政）。 */
	center: ProductStateCenter;
	/** PR-SIX-5：UI/回答语言策略（/language 读写并持久化）。 */
	locale: LocaleManager;
	registeredToolNames: () => string[];
}

type Handler = (args: string, ctx: ExtensionCommandContext) => Promise<void>;

export function buildCommandHandlers(deps: CommandDeps): Record<string, { description: string; handler: Handler }> {
	const notify = (ctx: ExtensionCommandContext, message: string, type?: "info" | "warning" | "error") =>
		ctx.ui.notify(message, type);

	return {
		status: {
			description: "运行时与具身状态（agentd/mission/body/mode）",
			handler: async (_args, ctx) => {
				try {
					const report = await deps.center.statusReport();
					const snap = report.snapshot;
					const mission = report.mission;
					notify(
						ctx,
						`agentd=${report.agentd || "?"} profile=${report.authorization_profile ?? ""}` +
							(mission
								? ` mission=${String(mission.mission_id)} [${String(mission.mode)}] ${String(mission.state)}`
								: " (未绑定 mission)") +
							` · context=${snap.context_state} r${snap.context_revision}` +
							` · lease=${snap.lease_state} · operator=${snap.operator}` +
							` · action=${snap.action_readiness.state} · seq=${snap.snapshot_seq}`,
						"info",
					);
				} catch (err) {
					notify(ctx, `agentd=UNREACHABLE（${(err as Error).message}）——不编造状态`, "error");
				}
			},
		},
		mission: {
			description: "当前 Mission 目标/状态/模式",
			handler: async (_args, ctx) => {
				const state = deps.active.current;
				notify(
					ctx,
					state.missionId
						? `Mission ${state.missionId} · mode=${state.mode} · revision=${state.contextRevision}`
						: "未绑定 Mission（/new 或 --mission 开始）",
					"info",
				);
			},
		},
		body: {
			description: "EffectiveBody hash/校准/问题",
			handler: async (_args, ctx) => {
				const missionId = deps.active.current.missionId;
				if (!missionId) {
					notify(ctx, "未绑定 Mission", "warning");
					return;
				}
				const response = await deps.center.call("pi.context", {
					mission_id: missionId,
				});
				const body = ((response.context as Record<string, unknown>)?.body ?? {}) as Record<string, unknown>;
				notify(
					ctx,
					`body=${String(body.body_id ?? "?")} hash=${String(body.effective_body_hash ?? "").slice(0, 16)}\n${String(body.summary ?? "")}`,
					"info",
				);
			},
		},
		tools: {
			description: "实际注册的工具（不是 prompt 愿望清单）",
			handler: async (_args, ctx) => {
				notify(ctx, `已注册工具：\n${deps.registeredToolNames().join("\n")}`, "info");
			},
		},
		approvals: {
			description: "待决授权卡（operatord 通道）",
			handler: async (_args, ctx) => {
				const listed = (await operatorCall(
					defaultOperatorSocket(deps.rosclawHome),
					"approvals.list",
					{ mission_id: deps.active.current.missionId },
				)) as { approvals?: Array<Record<string, unknown>> };
				const entries = listed.approvals ?? [];
				notify(
					ctx,
					entries.length === 0
						? "没有待决授权"
						: entries
								.map(
									(e) =>
										`${String(e.request_id)} [${String(e.risk_tier)}] ${String(e.title)} hash=${String(e.display_hash)}`,
								)
								.join("\n"),
					"info",
				);
			},
		},
		revoke: {
			description: "撤销 grant：/revoke <grant_id>（经 operatord）",
			handler: async (args, ctx) => {
				const grantId = args.trim();
				if (!grantId) {
					notify(ctx, "用法：/revoke <grant_id>", "warning");
					return;
				}
				const result = (await operatorCall(
					defaultOperatorSocket(deps.rosclawHome),
					"grants.revoke",
					{ grant_id: grantId },
				)) as { ok: boolean; error?: string };
				notify(
					ctx,
					result.ok ? `grant ${grantId} 已撤销` : `撤销被拒：${result.error ?? "unknown"}`,
					result.ok ? "info" : "error",
				);
			},
		},
		task: {
			description: "当前任务清单与阶段（/task）",
			handler: async (_args, ctx) => {
				const missionId = deps.active.current.missionId;
				if (!missionId) {
					notify(ctx, "未绑定 Mission", "warning");
					return;
				}
				const result = await deps.center.call("pi.task.list", { mission_id: missionId });
				const tasks = (result.tasks ?? []) as Array<Record<string, unknown>>;
				if (!tasks.length) {
					notify(ctx, "当前无任务", "info");
					return;
				}
				notify(
					ctx,
					tasks
						.map((t) => `${t.task_id} [${t.state}] ${t.goal}${t.error ? ` — ${String(t.error).slice(0, 80)}` : ""}`)
						.join("\n"),
					"info",
				);
			},
		},
		trace: {
			description: "任务全审计链（/trace <task_id>）",
			handler: async (args, ctx) => {
				const taskId = args.trim();
				if (!taskId) {
					notify(ctx, "用法：/trace <task_id>（/task 查看清单）", "warning");
					return;
				}
				const result = await deps.center.call("pi.task.trace", { task_id: taskId });
				if (!result.ok) {
					notify(ctx, `trace: ${String(result.error ?? "")}`, "error");
					return;
				}
				const tr = (result.trace ?? {}) as Record<string, never | Record<string, unknown>>;
				const task = (tr.task ?? {}) as Record<string, unknown>;
				const approval = (tr.approval ?? {}) as Record<string, unknown>;
				const txn = (tr.txn ?? {}) as Record<string, unknown>;
				notify(
					ctx,
					`task ${taskId} [${task.state}]\n` +
					`plan: ${task.plan_id || "-"}\n` +
					`approval: ${approval.request_id || "-"} [${approval.status ?? "-"}] by ${approval.decided_by ?? "-"}\n` +
					`txn: ${txn.txn_id || "-"} [${txn.state ?? "-"}] receipt: ${txn.receipt_id || "-"}`,
					"info",
				);
			},
		},
		context: {
			description: "具身检查点摘要（权威存储重建，非 LLM 摘要）",
			handler: async (_args, ctx) => {
				const missionId = deps.active.current.missionId;
				if (!missionId) {
					notify(ctx, "未绑定 Mission", "warning");
					return;
				}
				const result = await deps.center.call("pi.context.checkpoint", { mission_id: missionId });
				if (!result.ok) {
					notify(ctx, `checkpoint: ${String(result.error ?? "")}`, "error");
					return;
				}
				const cp = (result.checkpoint ?? {}) as Record<string, unknown>;
				const nonterminal = (cp.nonterminal_tasks ?? []) as Array<Record<string, unknown>>;
				const pending = (cp.pending_approvals ?? []) as string[];
				notify(
					ctx,
					`mission ${String(cp.mission_id)} [${cp.mode}] body=${cp.body_id} sim_policy=${cp.sim_policy}\n` +
					`非终态任务 ${nonterminal.length} · 待批准 ${pending.length} · ` +
					`最近回执 ${((cp.recent_receipt_refs ?? []) as string[]).filter(Boolean).join(", ") || "无"}`,
					"info",
				);
			},
		},
		why: {
			description: "解释最近一次任务/策略结果（/why）",
			handler: async (_args, ctx) => {
				const missionId = deps.active.current.missionId;
				if (!missionId) {
					notify(ctx, "未绑定 Mission", "warning");
					return;
				}
				const result = await deps.center.call("pi.task.list", { mission_id: missionId });
				const tasks = (result.tasks ?? []) as Array<Record<string, unknown>>;
				if (!tasks.length) {
					notify(ctx, "当前无任务记录", "info");
					return;
				}
				const latest = tasks[0];
				notify(
					ctx,
					`最近任务 ${latest.task_id} [${latest.state}]：${latest.error || "无错误"}（/trace ${latest.task_id} 看全链）`,
					"info",
				);
			},
		},
		tokens: {
			description: "Token/延迟用量分解（/tokens）",
			handler: async (_args, ctx) => {
				const loc = deps.locale.effective;
				const missionId = deps.active.current.missionId;
				if (!missionId) {
					notify(ctx, i18nT("tokens.no_mission", loc), "warning");
					return;
				}
				try {
					const result = await deps.center.call("pi.usage", { mission_id: missionId });
					if (!result.ok) {
						notify(ctx, `usage: ${String(result.error ?? "")}`, "error");
						return;
					}
					const u = (result.usage ?? {}) as {
						model_turns?: number; prompt_tokens?: number;
						completion_tokens?: number; total_tokens?: number;
						cost_microunits?: number; wall_span_ms?: number | null;
						provider_latency_ms?: { p50?: number | null; p95?: number | null };
						tool_calls?: { proposed?: number; completed?: number };
					};
					const lat = u.provider_latency_ms ?? {};
					const tools = u.tool_calls ?? {};
					notify(
						ctx,
						`${i18nT("tokens.title", loc)}:
` +
						`模型请求 ${u.model_turns ?? 0} · tokens in/out/total ` +
						`${u.prompt_tokens ?? 0}/${u.completion_tokens ?? 0}/${u.total_tokens ?? 0} · ` +
						`成本 ${(u.cost_microunits ?? 0) / 1e6} 元
` +
						`provider 延迟 p50/p95 ${lat.p50 ?? "-"}/${lat.p95 ?? "-"}ms · ` +
						`端到端跨度 ${u.wall_span_ms ?? "-"}ms
` +
						`工具调用 proposed/completed ${tools.proposed ?? 0}/${tools.completed ?? 0}`,
						"info",
					);
				} catch (err) {
					notify(ctx, `agentd=UNREACHABLE（${(err as Error).message}）`, "error");
				}
			},
		},
		doctor: {
			description: "诊断摘要 + 任务就绪检查：/doctor [task <目标>]",
			handler: async (args, ctx) => {
				const loc = deps.locale.effective;
				const [sub, ...rest] = args.trim().split(/\s+/).filter(Boolean);
				if (sub === "task") {
					if (!rest.length) {
						notify(ctx, "用法：/doctor task <目标>（如 /doctor task 画五角星）", "warning");
						return;
					}
					try {
						const result = await deps.center.call("pi.doctor.task", { goal: rest.join(" ") });
						const remediation = (result.remediation ?? null) as { command?: string } | null;
						notify(
							ctx,
							result.state === "READY"
								? `${i18nT("doctor.task_ready", loc)}: ${((result.required ?? []) as string[]).join(" + ")}`
								: `${i18nT("doctor.task_missing", loc)}: ${((result.missing ?? []) as string[]).join(", ")}` +
									(remediation?.command
										? `\n${i18nT("doctor.remediation", loc)}: ${remediation.command}`
										: ""),
							result.state === "READY" ? "info" : "warning",
						);
					} catch (err) {
						notify(ctx, `agentd=UNREACHABLE（${(err as Error).message}）`, "error");
					}
					return;
				}
				const status = await deps.center.call("pi.status", {});
				notify(
					ctx,
					`agentd=${String(status.agentd ?? "?")} profile=${String(status.authorization_profile ?? "")}`,
					"info",
				);
			},
		},
		estop: {
			description: "紧急停止（独立 operatord 通道，不经模型/agentd）",
			handler: async (_args, ctx) => {
				try {
					const result = (await operatorCall(
						defaultOperatorSocket(deps.rosclawHome),
						"estop",
						{ reason: "operator /estop from ROSClaw Native Agent" },
					)) as { ok: boolean; error?: string };
					notify(
						ctx,
						result.ok
							? "E-STOP 已请求 rosclawd 执行（只减权限）"
							: `E-STOP 未执行：${result.error ?? "unknown"}`,
						result.ok ? "error" : "warning",
					);
				} catch (err) {
					notify(ctx, `E-STOP 通道不可用：${(err as Error).message}（未假装已停止）`, "error");
				}
			},
		},
		cancel: {
			description: "取消当前任务/回合（/cancel [task_id]）",
			handler: async (args, ctx) => {
				// 八审 §4 P0-9：/cancel 必须取消真实 task，不只是 LLM 回合。
				const taskId = args.trim();
				if (taskId) {
					try {
						const result = await deps.center.call("pi.task.cancel", { task_id: taskId });
						notify(
							ctx,
							result.ok
								? `任务 ${taskId}：${String(result.state)}${result.changed ? "" : "（已是终态）"}`
								: `取消失败：${String(result.error ?? "")}`,
							result.ok ? "info" : "error",
						);
					} catch (err) {
						notify(ctx, `取消失败：${(err as Error).message}`, "error");
					}
					return;
				}
				ctx.abort();
				notify(ctx, "已请求取消当前回合（/cancel <task_id> 可取消具体任务）", "info");
			},
		},
		evidence: {
			description: "最近的执行回执摘要",
			handler: async (_args, ctx) => {
				const missionId = deps.active.current.missionId;
				if (!missionId) {
					notify(ctx, "未绑定 Mission", "warning");
					return;
				}
				const result = await deps.center.call("pi.tools.execute", {
					request: {
						schema_version: "rosclaw.pi_tool_request.v1",
						request_id: `ptr_ev_${Date.now()}`,
						pi_session_id: deps.active.current.sessionId,
						mission_id: missionId,
						context_revision: deps.active.current.contextRevision,
						tool_name: "rosclaw_verify",
						arguments: {},
						requested_at: new Date().toISOString(),
						idempotency_key: `idem_ev_${Date.now()}`,
						actor: { engine: "pi-command" },
					},
				});
				const r = (result.result ?? {}) as { summary?: string };
				notify(ctx, (r.summary ?? "无回执").slice(0, 400), "info");
			},
		},
		memory: {
			description: "Memory/Practice/How 查询指引",
			handler: async (_args, ctx) => {
				notify(ctx, "用自然语言提问即可——模型会经 rosclaw_memory_query 带证据查询。", "info");
			},
		},
		safety: {
			description: "SIM 审批策略：/safety sim auto|ask-every-time",
			handler: async (args, ctx) => {
				const arg = args.trim();
				if (arg === "sim auto" || arg === "sim ask-every-time") {
					const policy = arg === "sim auto" ? "auto" : "ask";
					const result = await deps.center.call("pi.safety.set", { sim_policy: policy });
					notify(
						ctx,
						result.ok
							? `SIM 审批策略已更新：${policy === "auto" ? "安全仿真自动执行" : "每次人工确认"}`
							: `更新失败：${String(result.error ?? "")}`,
						result.ok ? "info" : "error",
					);
					return;
				}
				const current = await deps.center.call("pi.safety.get", {});
				notify(
					ctx,
					`SIM 审批策略：${String(current.sim_policy ?? "auto")}（auto=安全仿真自动执行 / ask=每次人工确认）。REAL 永远人工确认。`,
					"info",
				);
			},
		},
		"operator-init": {
			description: "初始化并启动本机 Operator（仅 SIMULATION developer）",
			handler: async (_args, ctx) => {
				const loc = deps.locale.effective;
				try {
					const status = await deps.center.call("pi.operator.status", {});
					if (status.running) {
						notify(ctx, i18nT("operator.bootstrap_done", loc), "info");
						return;
					}
					const result = await deps.center.call("pi.operator.bootstrap", {
						mission_id: deps.active.current.missionId ?? "",
					});
					notify(
						ctx,
						result.ok
							? i18nT("operator.bootstrap_done", loc)
							: `${i18nT("operator.bootstrap_failed", loc)}: ${String(result.error ?? "")}`,
						result.ok ? "info" : "error",
					);
					await deps.center.probeOperator(true);
				} catch (err) {
					notify(ctx, `${i18nT("operator.bootstrap_failed", loc)}: ${(err as Error).message}`, "error");
				}
			},
		},
		effort: {
			// PR-N9：/effort auto|low|medium|high——真实切换 reasoning
			// effort（Pi thinking level 同映射），持久化在 settings。
			description: "推理强度 auto|low|medium|high",
			handler: async (args, ctx) => {
				const value = args.trim().toLowerCase();
				const allowed = new Set(["auto", "low", "medium", "high"]);
				if (!allowed.has(value)) {
					notify(ctx, "用法：/effort auto|low|medium|high", "warning");
					return;
				}
				(ctx as unknown as { setThinkingLevel(l: string): void })
					.setThinkingLevel(value);
				notify(ctx, `推理强度已设为 ${value}`, "info");
			},
		},
		sessions: {
			// PR-N9：会话面——打开会话选择器（与 rosclaw resume 同入口）。
			description: "浏览/切换会话",
			handler: async (_args, ctx) => {
				const c = ctx as unknown as {
					ui: { notify(m: string, k?: "info" | "warning" | "error"): void };
					newSession?(options?: { parentSession?: string }): Promise<void>;
					switchSession?(path: string): Promise<void>;
					sessionManager: { listAll(dir: string): Promise<unknown[]> };
				};
				notify(ctx, "会话列表见 rosclaw sessions；/resume <id|前缀|标题> 切换", "info");
			},
		},
		resume: {
			description: "恢复会话（id/前缀/标题）",
			handler: async (args, ctx) => {
				const query = args.trim();
				if (!query) {
					notify(ctx, "用法：/resume <id|前缀|标题>", "warning");
					return;
				}
				const c = ctx as unknown as {
					sessionManager: { listAll(dir: string): Promise<Array<{ id: string; path: string; name?: string; firstMessage: string }>> };
					switchSession?(path: string): Promise<void>;
				};
				const sessionDir = `${deps.rosclawHome}/agent/sessions`;
				const sessions = await c.sessionManager.listAll(sessionDir);
				const hit = sessions.find((s) => s.id === query)
					?? (sessions.filter((s) => s.id.startsWith(query)).length === 1
						? sessions.find((s) => s.id.startsWith(query))
						: undefined)
					?? (sessions.filter(
						(s) => (s.name ?? "").includes(query) || s.firstMessage.includes(query),
					).length === 1
						? sessions.find(
							(s) => (s.name ?? "").includes(query) || s.firstMessage.includes(query),
						)
						: undefined);
				if (!hit) {
					notify(ctx, `会话 ${query} 不唯一或不存在——rosclaw sessions 查看全部`, "error");
					return;
				}
				if (!c.switchSession) {
					notify(ctx, "当前运行模式不支持会话内切换——用 rosclaw resume", "warning");
					return;
				}
				await c.switchSession(hit.path);
				notify(ctx, `已切换到会话 ${hit.id}`, "info");
			},
		},
		language: {
			description: "界面/回答语言：/language [中文|English|auto|lock 中文|lock English]",
			handler: async (args, ctx) => {
				const lm = deps.locale;
				const arg = args.trim();
				if (!arg) {
					notify(
						ctx,
						`语言策略：UI=${lm.current.ui_locale}（生效 ${lm.effective}）· ` +
						`回答=${lm.current.reply_language}。用法：/language 中文|English|auto|lock 中文`,
						"info",
					);
					return;
				}
				if (arg === "auto") {
					lm.setUiLocale("auto");
				} else if (arg === "中文" || arg === "zh-CN") {
					lm.setUiLocale("zh-CN");
				} else if (arg === "English" || arg === "en-US" || arg === "英文") {
					lm.setUiLocale("en-US");
				} else if (arg.startsWith("lock ")) {
					const lang = arg.slice(5).trim();
					if (lang === "中文" || lang === "zh-CN") {
						lm.setReplyLanguage("zh-CN");
					} else if (lang === "English" || lang === "en-US" || lang === "英文") {
						lm.setReplyLanguage("en-US");
					} else if (lang === "auto" || lang === "跟随") {
						lm.setReplyLanguage("follow-user");
					} else {
						notify(ctx, `未知语言：${lang}`, "warning");
						return;
					}
				} else {
					notify(ctx, `未知参数：${arg}（中文|English|auto|lock …）`, "warning");
					return;
				}
				notify(
					ctx,
					`已更新：UI=${lm.current.ui_locale}（生效 ${lm.effective}）· 回答=${lm.current.reply_language}`,
					"info",
				);
			},
		},
		robot: {
			description: "当前机器人：/robot [use <body_id>|repair [kit_id]]",
			handler: async (args, ctx) => {
				const loc = deps.locale.effective;
				const [sub, ...rest] = args.trim().split(/\s+/).filter(Boolean);
				try {
					if (sub === "use") {
						const bodyId = rest.join(" ");
						if (!bodyId) {
							notify(ctx, "用法：/robot use <body_id>（如 sim/ur5e）", "warning");
							return;
						}
						const result = await deps.center.call("pi.robot.use", { body_id: bodyId });
						notify(
							ctx,
							result.ok
								? result.changed
									? i18nT("robot.use_saved", loc)
									: `${i18nT("robot.current", loc)}: ${bodyId}`
								: `${i18nT("robot.use_refused", loc)}: ${String(result.error ?? "")}`,
							result.ok ? "info" : "error",
						);
						await deps.center.refreshRobotInfo(true);
						return;
					}
					if (sub === "repair") {
						const kitId = rest.join(" ");
						const result = await deps.center.call("pi.robot.repair", { kit_id: kitId });
						const kit = (result.robot_kit ?? {}) as { display_name?: string; state?: string };
						notify(
							ctx,
							result.ok
								? `${i18nT("robot.repair_done", loc)}: ${kit.display_name ?? kitId} [${kit.state ?? "?"}]`
								: `${i18nT("robot.repair_failed", loc)}: ${String(result.error ?? kit.state ?? "")}`,
							result.ok ? "info" : "error",
						);
						await deps.center.refreshRobotInfo(true);
						await deps.center.refreshCapabilities(true);
						return;
					}
					const status = await deps.center.call("pi.status", {});
					const kit = (status.robot_kit ?? {}) as {
						display_name?: string; state?: string; reason?: string;
						remediation?: { command?: string } | null;
					};
					const lines = [
						`${i18nT("robot.current", loc)}: ${String(status.body_display ?? status.body_id ?? "?")} [${String(kit.state ?? "?")}]`,
					];
					if (kit.state === "BROKEN") {
						lines.push(
							`${i18nT("robot.kit_broken", loc)}: ${kit.reason ?? ""}` +
							(kit.remediation?.command
								? ` — ${i18nT("robot.repair_hint", loc)}: ${kit.remediation.command}`
								: ""),
						);
					}
					notify(ctx, lines.join("\n"), kit.state === "BROKEN" ? "warning" : "info");
				} catch (err) {
					notify(ctx, `agentd=UNREACHABLE（${(err as Error).message}）`, "error");
				}
			},
		},
		robots: {
			description: "可用机器人套件清单",
			handler: async (_args, ctx) => {
				const loc = deps.locale.effective;
				try {
					const result = await deps.center.call("pi.robot.list", {});
					const kits = (result.kits ?? []) as Array<{
						display_name?: string; kit_id?: string; state?: string; active?: boolean;
					}>;
					if (!kits.length) {
						notify(ctx, i18nT("robot.none_available", loc), "warning");
						return;
					}
					const lines = kits.map((k) =>
						`${k.active ? "●" : "○"} ${k.display_name ?? k.kit_id} [${k.state ?? "?"}]`,
					);
					notify(ctx, lines.join("\n"), "info");
				} catch (err) {
					notify(ctx, `agentd=UNREACHABLE（${(err as Error).message}）`, "error");
				}
			},
		},
		capabilities: {
			description: "当前机器人能力清单（观测/计算/动作 + 被排除）",
			handler: async (_args, ctx) => {
				const loc = deps.locale.effective;
				const missionId = deps.active.current.missionId;
				if (!missionId) {
					notify(ctx, "未绑定 Mission", "warning");
					return;
				}
				try {
					const result = await deps.center.call("pi.capabilities", { mission_id: missionId });
					if (!result.ok) {
						notify(ctx, `capabilities: ${String(result.error ?? "")}`, "error");
						return;
					}
					const names = (list: unknown) =>
						((list ?? []) as Array<{ capability_id?: string }>)
							.map((c) => String(c.capability_id ?? "")).filter(Boolean);
					const excluded = ((result.excluded ?? []) as Array<{ capability_id?: string; reason?: string }>)
						.map((e) => `${e.capability_id}(${e.reason})`);
					notify(
						ctx,
						`${i18nT("capabilities.summary", loc)}:\n` +
						`观测: ${names(result.observation_capabilities).join(", ") || "-"}\n` +
						`计算: ${names(result.compute_capabilities).join(", ") || "-"}\n` +
						`动作: ${names(result.action_capabilities).join(", ") || "-"}` +
						(excluded.length
							? `\n${i18nT("capabilities.excluded", loc)}: ${excluded.join(", ")}`
							: ""),
						"info",
					);
				} catch (err) {
					notify(ctx, `agentd=UNREACHABLE（${(err as Error).message}）`, "error");
				}
			},
		},
		};
}
