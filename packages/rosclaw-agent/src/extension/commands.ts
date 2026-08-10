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
		doctor: {
			description: "agentd/modeld/授权剖面诊断摘要",
			handler: async (_args, ctx) => {
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
			description: "取消当前回合",
			handler: async (_args, ctx) => {
				ctx.abort();
				notify(ctx, "已请求取消当前回合", "info");
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
	};
}
