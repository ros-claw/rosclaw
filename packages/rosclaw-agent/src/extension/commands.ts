// HP2-COMPAT: Pi 扩展宿主类型（ExtensionContext/Factory）——扩展运行於 Pi 扩展宿主内，HP3 前保持；不新增会话装配引用。
/** ROSClaw TUI 命令注册（NA-FIX-6，二次审计 P0-8 + 规格 §7.2）。
 *
 * 每个命令都有真实 handler + 权限路径 + 测试；/estop 走 dedicated
 * operatord 通道（模型/agentd/Pi session 全卡死也可触发）。
 */

import type { ExtensionAPI, ExtensionCommandContext, SessionInfo } from "@earendil-works/pi-coding-agent";
import { resolveSessionQuery } from "../harness/pi/pi-resolve.js";
import { defaultOperatorSocket, operatorCall } from "../bridge/operatord-client.js";
import type { ActiveSessionContext } from "../session/active-context.js";
import type { ProductStateCenter } from "../session/state-center.js";
import type { LocaleManager } from "../i18n/locale.js";
import { t as i18nT } from "../i18n/index.js";
import type { MirrorDiagnostics } from "./event-mirror.js";

export interface CommandDeps {
	rosclawHome: string;
	active: ActiveSessionContext;
	/** PR-SIX-1：唯一状态中心——/status 与 rosclaw_status/Header/Footer
	 *  同一份快照（不再各自为政）。 */
	center: ProductStateCenter;
	/** PR-SIX-5：UI/回答语言策略（/language 读写并持久化）。 */
	locale: LocaleManager;
	registeredToolNames: () => string[];
	/** PI puts thinking controls on ExtensionAPI, not the command context. */
	thinking?: Pick<ExtensionAPI, "setThinkingLevel" | "getThinkingLevel" | "getSettings">;
	listSessions?: () => Promise<SessionInfo[]>;
	/** BOUNDED-IDENTITY：实际 EventMirror 的同步诊断（pending 含在飞/
	 *  overflow_dropped/unconfirmed）。返回 null = 镜像诊断不可用——
	 *  /tokens 必须诚实报"同步状态未知"，绝不假装完整。 */
	mirrorDiagnostics?: () => MirrorDiagnostics | null;
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
						native_usage?: {
							known_message_count?: number; input_uncached?: number | string | null;
							cache_read?: number | string | null; cache_write?: number | string | null;
							input_including_cache?: number | string | null; output?: number | string | null;
							reasoning_subset_output?: number | string | null; total_tokens?: number | string | null;
							unknown_message_count?: number;
							reasoning_unknown_message_count?: number;
							billing_authority_unknown_message_count?: number;
							cost_usd_known_subtotal?: number | null;
							cost_usd_estimate?: number | null;
							cost_unknown_message_count?: number;
							identity_scope?: string;
						};
					};
					const lat = u.provider_latency_ms ?? {};
					const tools = u.tool_calls ?? {};
					const lines: string[] = [`${i18nT("tokens.title", loc)}:`];
					const n = u.native_usage;
					if (n) {
						// NATIVE-TOKENS：当前任务原生会话累计（PI message_end 口径，
						// mission 聚合——可跨多个 PI session）。USD 估计与人民币旧账
						// 分开——不做汇率换算；成本缺失是未知而不是免费；进行中/中止/
						// 部分的消息不算已确认零付费。超出 JS 精确整数范围的聚合计数
						// 以精确十进制字符串到达——原样呈现，不假装是安全数值。
						const num = (v: number | string | null | undefined): string =>
							v === null || v === undefined ? "未知" : String(v);
						const usd = n.cost_usd_estimate === null || n.cost_usd_estimate === undefined
							? "未知（有消息缺少成本数据——不按零计）"
							: `${n.cost_usd_estimate} USD`;
						const unconfirmed = (n.unknown_message_count ?? 0) > 0
							? ` · 缺失或部分用量尚未确认 ${n.unknown_message_count ?? 0} 条` +
								"（不是 live pending request proof）"
							: "";
						// NATIVE-TOKENS-TERMINAL：已上报成本下限与未知完成/计费
						// 权威分开陈述——部分记录的已上报金额是事实下限，绝不
						// 被未知记录清零，也不被说成完整账单。
						const subtotal = n.cost_usd_known_subtotal === null ||
							n.cost_usd_known_subtotal === undefined
							? ""
							: ` · 已上报成本小计 ${n.cost_usd_known_subtotal} USD` +
								"（已知下限，不含尚未确认部分）";
						const billingUnknown = (n.billing_authority_unknown_message_count ?? 0) > 0
							? ` · 缺少终端完成凭证 ${n.billing_authority_unknown_message_count} 条` +
								"（计费未知——不按零计，也不反向推断中止）"
							: "";
						const reasoningUnknown = (n.reasoning_unknown_message_count ?? 0) > 0
							? ` · reasoning 细分未知 ${n.reasoning_unknown_message_count} 条` +
								"（provider 未上报，按未知处理而非零）"
							: "";
						lines.push(
							`当前任务原生会话累计：已确认 ${n.known_message_count ?? 0} 条 · ` +
							`tokens 输入 ${num(n.input_uncached)} · 缓存读入/写入 ` +
							`${num(n.cache_read)}/${num(n.cache_write)} · 含缓存输入合计 ` +
							`${num(n.input_including_cache)} · 输出 ${num(n.output)} · ` +
							`总计 ${num(n.total_tokens)}${unconfirmed}${reasoningUnknown}`,
							`USD成本估计 ${usd}${subtotal}${billingUnknown}（与人民币旧账分开，不换算）`,
						);
					}
					lines.push(
						`其他模型请求（旧账）${u.model_turns ?? 0} · tokens in/out/total ` +
						`${u.prompt_tokens ?? 0}/${u.completion_tokens ?? 0}/${u.total_tokens ?? 0} · ` +
						`人民币成本 ${(u.cost_microunits ?? 0) / 1e6} 元`,
						`provider 延迟 p50/p95 ${lat.p50 ?? "-"}/${lat.p95 ?? "-"}ms · ` +
						`端到端跨度 ${u.wall_span_ms ?? "-"}ms`,
						`工具调用 proposed/completed ${tools.proposed ?? 0}/${tools.completed ?? 0}`,
					);
					// BOUNDED-IDENTITY：镜像同步完整性用用户可读语言呈现——
					// 待同步（pending，含在飞）与溢出丢弃（overflow loss）是
					// 同步完整性事实，绝不伪装成已确认完整/免单；诊断不可用
					// 时诚实报未知。只含有界计数，不含原始内容/表名。
					const mirrorDiag = deps.mirrorDiagnostics?.() ?? null;
					if (mirrorDiag === null) {
						lines.push(
							"用量镜像同步状态未知（镜像诊断不可用——上方仅为已确认部分，" +
							"完整性未知，不据此推断账单完整）",
						);
					} else if (mirrorDiag.unconfirmed) {
						const gaps: string[] = [];
						if (mirrorDiag.pending > 0) {
							// BOUNDED-IDENTITY：诚实重试语义——镜像没有后台
							// 定时器；待同步事件只随真实事件生命周期触发
							// （下一条助手 message_end / turn_end / 会话关闭
							// 时的 flush）重放，绝不承诺自动定时重试。
							gaps.push(
								`${mirrorDiag.pending} 条用量事件待同步（pending，含在飞批次）` +
								"——无后台定时自动重试；将在下一条助手消息、回合结束" +
								"或会话关闭时随事件生命周期重试同步",
							);
						}
						if (mirrorDiag.overflow_dropped > 0) {
							gaps.push(
								`累计 ${mirrorDiag.overflow_dropped} 条用量事件因缓冲溢出` +
								"（overflow）被丢弃——这部分用量已丢失（loss），上方汇总不完整",
							);
						}
						lines.push(
							`用量镜像同步未完成：${gaps.join("；")}。` +
							"上方“已确认”数字不包含这些未同步/丢失部分——不是完整账单。",
						);
					} else {
						lines.push("用量镜像已同步：无待同步事件（pending 0），无溢出丢失。");
					}
					notify(ctx, lines.join("\n"), "info");
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
			// auto 恢复配置默认；PI API 仅改变当前会话，不冒称修改全局设置。
			description: "推理强度 auto|low|medium|high",
			handler: async (args, ctx) => {
				const value = args.trim().toLowerCase();
				const allowed = new Set(["auto", "low", "medium", "high"]);
				if (!allowed.has(value)) {
					notify(ctx, "用法：/effort auto|low|medium|high", "warning");
					return;
				}
				const host = deps.thinking;
				if (!host) {
					notify(ctx, "当前宿主不提供推理强度接口，未变更", "warning");
					return;
				}
				const settings = host.getSettings();
				const modelDefault = ctx.model
					? settings.modelThinkingLevels?.[`${ctx.model.provider}/${ctx.model.id}`]
					: undefined;
				const level = value === "auto"
					? modelDefault ?? settings.defaultThinkingLevel ?? "medium"
					: value as "low" | "medium" | "high";
				host.setThinkingLevel(level);
				notify(ctx, `推理强度 ${value} → ${host.getThinkingLevel()}（当前会话）`, "info");
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
				notify(ctx, "会话列表见 rosclaw sessions；/switch <id|前缀|标题> 切换", "info");
			},
		},
		// WP-7：原 /resume 与 Pi 内置命令冲突（启动 [Extension issues]
		// 实证）——改名 /switch。
		switch: {
			description: "恢复会话（id/前缀/标题）",
			handler: async (args, ctx) => {
				const query = args.trim();
				if (!query) {
					notify(ctx, "用法：/switch <id|前缀|标题>", "warning");
					return;
				}
				if (!deps.listSessions) {
					notify(ctx, "当前宿主不提供会话列表——用 rosclaw resume", "warning");
					return;
				}
				try {
					const hit = resolveSessionQuery(query, await deps.listSessions());
					if (!hit.ok) {
						notify(ctx, `会话 ${query} 不唯一或不存在——rosclaw sessions 查看全部`, "error");
						return;
					}
					const result = await ctx.switchSession(hit.path);
					notify(ctx, result.cancelled ? "会话切换已取消" : `已切换到会话 ${hit.info.id}`, "info");
				} catch (error) {
					notify(ctx, `会话切换失败：${(error as Error).message}`, "error");
				}
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
