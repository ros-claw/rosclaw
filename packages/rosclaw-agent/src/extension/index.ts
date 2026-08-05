/** ROSClaw 内联扩展（PNA-0）：品牌 + 安全基线。
 *
 * - header/footer/title/working 动画替换为 ROSClaw；
 * - `!` bash 功能级关闭（user_bash full replacement）；
 * - 会话生命周期观察埋点（PNA-1 挂 SessionBinding）。
 */

import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { bridgeCall } from "../bridge/bridge-client.js";
import {
	handleSessionStart,
	sessionIdOf,
	shouldCancelSwitch,
	shouldCancelTree,
	type LifecycleDeps,
} from "../session/lifecycle.js";
import { defaultOperatorSocket, operatorCall } from "../bridge/operatord-client.js";
import { ApprovalCardComponent } from "../ui/approval-card.js";
import { fetchEmbodiedContext, renderTrustedContext } from "./context-injection.js";

export interface RosclawExtensionOptions {
	profile: "developer" | "robot";
	version: string;
	/** PNA-2：v2 系统提示词（native_agent_v2.md 内容，构建期打包）。 */
	systemPrompt: string;
	/** 当前绑定的 Mission（PNA-1 SessionBinding 先行版本：启动时确定）。 */
	missionId?: string;
	piSessionId?: string;
	rosclawHome: string;
}

const WORKING_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

export function createRosclawExtension(options: RosclawExtensionOptions): ExtensionFactory {
	return (pi) => {
		// -- 品牌 ----------------------------------------------------------------
		pi.on("session_start", async (_event, ctx) => {
			if (!ctx.hasUI) return;
			ctx.ui.setTitle(`ROSClaw Native Agent`);
			ctx.ui.setHeader((_tui, _theme) => {
				return new Text(
					`ROSClaw Native Agent v${options.version} ` +
						`[engine=pi profile=${options.profile}]  ·  /help 查看命令`,
				);
			});
			ctx.ui.setWorkingIndicator({ frames: WORKING_FRAMES, intervalMs: 80 });
		});

		// -- `!` bash 功能级关闭（PNA-0 即生效；PNA-9 再做 profile 化 UI 拦截） -----
		pi.on("user_bash", async () => {
			return {
				result: {
					output: "bash execution is disabled by ROSClaw policy (engine=pi)",
					exitCode: 1,
					cancelled: false,
					truncated: false,
				},
			};
		});

		// -- 每轮注入最新具身上下文（PNA-2，规格 §14.2） ---------------------------
		pi.on("before_agent_start", async (_event, _ctx) => {
			if (!options.missionId) {
				return { systemPrompt: options.systemPrompt };
			}
			const fetched = await fetchEmbodiedContext(options.rosclawHome, options.missionId);
			return {
				systemPrompt: options.systemPrompt,
				message: {
					customType: "rosclaw.embodied_context",
					content: renderTrustedContext(fetched),
					display: false,
					details: { stale: fetched.stale, note: fetched.note },
				},
			};
		});

		// -- Worker 命令（PNA-4，规格 §19）：/workers /delegate ------------------
		pi.registerCommand("workers", {
			description: "列出 Worker 与当前 Mission 的 WorkOrder 状态",
			handler: async (_args, ctx) => {
				if (!options.missionId) {
					ctx.ui.notify("未绑定 Mission——/workers 不可用", "warning");
					return;
				}
				try {
					const status = await bridgeCall(options.rosclawHome, "pi.worker.status", {
						mission_id: options.missionId,
					});
					const orders = (status.orders ?? []) as Array<Record<string, unknown>>;
					if (orders.length === 0) {
						ctx.ui.notify("当前 Mission 没有 WorkOrder", "info");
						return;
					}
					ctx.ui.notify(
						orders
							.map(
								(o) =>
									`${String(o.work_order_id)}  ${String(o.assigned_to ?? "?")}  ${String(o.status)}  ${String(o.goal ?? "")}`,
							)
							.join("\n"),
						"info",
					);
				} catch (err) {
					ctx.ui.notify(`查询失败：${(err as Error).message}`, "error");
				}
			},
		});
		pi.registerCommand("delegate", {
			description: "显式委派：/delegate <worker|auto> <自包含目标>（不经模型）",
			handler: async (args, ctx) => {
				if (!options.missionId) {
					ctx.ui.notify("未绑定 Mission——/delegate 不可用", "warning");
					return;
				}
				const match = args.trim().match(/^(\S+)\s+([\s\S]+)$/);
				if (!match) {
					ctx.ui.notify("用法：/delegate <worker|auto> <goal>", "warning");
					return;
				}
				const [, workerId, goal] = match;
				ctx.ui.notify(`委派中（${workerId}）：${goal.slice(0, 60)}…`, "info");
				try {
					const response = await bridgeCall(options.rosclawHome, "pi.tools.execute", {
						request: {
							schema_version: "rosclaw.pi_tool_request.v1",
							request_id: `ptr_cmd_${Date.now()}`,
							pi_session_id: options.piSessionId ?? "",
							mission_id: options.missionId,
							context_revision: 0,
							tool_name: "rosclaw_delegate",
							arguments: { goal, worker_id: workerId },
							requested_at: new Date().toISOString(),
							idempotency_key: `idem_cmd_${Date.now()}`,
							actor: { engine: "pi-command" },
						},
					});
					const result = (response.result ?? {}) as { summary?: string; error_code?: string };
					if (response.ok) {
						ctx.ui.notify(`Worker 完成（已验证）：${(result.summary ?? "").slice(0, 200)}`, "info");
					} else {
						ctx.ui.notify(
							`委派失败 [${result.error_code ?? response.code ?? "?"}]：${result.summary ?? response.error ?? ""}`,
							"error",
						);
					}
				} catch (err) {
					ctx.ui.notify(`委派失败：${(err as Error).message}`, "error");
				}
			},
		});

		// -- Approval 卡片（PNA-5，规格 §20）：tool 触发 → 拉卡片 → 前台 Y/N →
		//    operatord 签名链。模型文本永远到不了这里（只有真实按键）。
		pi.on("tool_execution_start", async (event, ctx) => {
			if (event.toolName !== "rosclaw_request_action" || !ctx.hasUI) return;
			if (!options.missionId) return;
			const args = (event.args ?? {}) as Record<string, unknown>;
			// 等 agentd 建卡（bridge tool 先创建授权卡再等待决定）。
			let entry: { request_id: string; display_hash: string } | undefined;
			for (let attempt = 0; attempt < 10; attempt += 1) {
				await new Promise((resolve) => setTimeout(resolve, 500));
				try {
					const listed = (await operatorCall(
						defaultOperatorSocket(options.rosclawHome),
						"approvals.list",
						{ mission_id: options.missionId },
					)) as { ok: boolean; approvals?: Array<{ request_id: string; display_hash: string }> };
					entry = (listed.approvals ?? [])[0];
					if (entry) break;
				} catch {
					// operatord 未运行 → 下方诚实提示
				}
			}
			if (!entry) {
				ctx.ui.notify(
					"授权卡未出现（operatord 未运行？）——动作不会执行。启动：rosclaw operatord start",
					"error",
				);
				return;
			}
			const cardEntry = entry;
			try {
				await ctx.ui.custom<boolean>((_tui, _theme, _kb, done) => {
					return new ApprovalCardComponent(
						{
							requestId: cardEntry.request_id,
							title: String(args.expected_effect ?? args.capability_id ?? ""),
							summary: String(args.capability_id ?? ""),
							riskTier: String(args.risk_tier ?? "LOW"),
							mode: "ACTION",
							capability: String(args.capability_id ?? ""),
							parameters: (args.arguments ?? {}) as Record<string, unknown>,
							expiresAt: "",
							displayHash: cardEntry.display_hash,
						},
						(approve) => done(approve),
					);
				}, { overlay: true }).then(async (approve) => {
					const decided = (await operatorCall(
						defaultOperatorSocket(options.rosclawHome),
						"approvals.decide",
						{
							request_id: cardEntry.request_id,
							display_hash: cardEntry.display_hash,
							approve,
						},
					)) as { ok: boolean; error?: string };
					ctx.ui.notify(
						decided.ok
							? approve
								? "已批准（等待执行回执）"
								: "已拒绝"
							: `决定被拒：${decided.error ?? "unknown"}`,
						decided.ok ? "info" : "error",
					);
				});
			} catch (err) {
				ctx.ui.notify(`授权卡交互失败：${(err as Error).message}`, "error");
			}
		});

		// -- Session 生命周期映射（PNA-6，规格 §13） ------------------------------
		const lifecycle: LifecycleDeps = {
			rosclawHome: options.rosclawHome,
			getMissionId: () => options.missionId,
			setMissionId: (missionId) => {
				options.missionId = missionId;
			},
			notify: (message, type) => undefined,
		};
		pi.on("session_start", async (event, ctx) => {
			lifecycle.notify = (message, type) => ctx.ui.notify(message, type);
			try {
				await handleSessionStart(lifecycle, event.reason, sessionIdOf(ctx));
			} catch (err) {
				ctx.ui.notify(`session 绑定异常：${(err as Error).message}`, "error");
			}
		});
		pi.on("session_before_switch", async (event, ctx) => {
			lifecycle.notify = (message, type) => ctx.ui.notify(message, type);
			const veto = await shouldCancelSwitch(lifecycle, sessionIdOf(ctx));
			return veto ? { cancel: true } : undefined;
		});
		pi.on("session_before_tree", async (_event, ctx) => {
			lifecycle.notify = (message, type) => ctx.ui.notify(message, type);
			const veto = await shouldCancelTree(lifecycle);
			if (veto) {
				ctx.ui.notify(veto, "warning");
				return { cancel: true };
			}
			return undefined;
		});
		// fork：authority 结构性不复制（grant/permit 只在 agentd）；
		// 新 mission 绑定在 session_start(reason=fork) 完成。
		pi.on("session_before_fork", async () => {
			return undefined;
		});
	};
}
