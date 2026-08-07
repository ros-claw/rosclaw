/** ROSClaw 内联扩展（PNA-0）：品牌 + 安全基线。
 *
 * - header/footer/title/working 动画替换为 ROSClaw；
 * - `!` bash 功能级关闭（user_bash full replacement）；
 * - 会话生命周期观察埋点（PNA-1 挂 SessionBinding）。
 */

import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { readFileSync } from "node:fs";
import { bridgeCall } from "../bridge/bridge-client.js";
import { ActiveSessionContext } from "../session/active-context.js";
import type { AgentSessionCoordinator } from "../session/coordinator.js";
import {
	handleSessionStart,
	sessionIdOf,
	shouldCancelSwitch,
	shouldCancelTree,
	type LifecycleDeps,
} from "../session/lifecycle.js";
import { defaultOperatorSocket, operatorCall } from "../bridge/operatord-client.js";
import { ApprovalCardComponent } from "../ui/approval-card.js";
import { ProductUiState, renderHeader } from "../ui/product-state.js";
import { EventMirror } from "./event-mirror.js";
import { buildCommandHandlers } from "./commands.js";
import { guardInput } from "./input-guard.js";
import { fetchEmbodiedContext, renderTrustedContext } from "./context-injection.js";

export interface RosclawExtensionOptions {
	profile: "developer" | "robot";
	version: string;
	/** PNA-2：v2 系统提示词（native_agent_v2.md 内容，构建期打包）。 */
	systemPrompt: string;
	/** NA-FIX-2：动态 session 上下文（切换事务的唯一真实源）。 */
	active: ActiveSessionContext;
	/** P0-NA-12：唯一 session/mission/lease 事务协调器。 */
	coordinator: AgentSessionCoordinator;
	rosclawHome: string;
}

const WORKING_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

export function createRosclawExtension(options: RosclawExtensionOptions): ExtensionFactory {
	return (pi) => {
		// -- 品牌（P0-NA-16：header 只读权威快照——版本来自 launcher 传入的
		//    产品版本；Operator 状态来自真实 socket 探测；body/context 来自
		//    验证过的 envelope；bootstrap 未完成显示 LOADING，不乐观默认） --
		const uiState = new ProductUiState(
			options.active,
			defaultOperatorSocket(options.rosclawHome),
			options.version,
		);
		let modelDisplay = "";
		let refreshHeader: () => void = () => undefined;
		let probeTimer: ReturnType<typeof setInterval> | null = null;
		pi.on("session_start", async (_event, ctx) => {
			if (!ctx.hasUI) return;
			ctx.ui.setTitle(`ROSClaw Native Agent`);
			refreshHeader = () => {
				ctx.ui.setHeader((_tui, _theme) => new Text(renderHeader(uiState.snapshot(), modelDisplay)));
			};
			refreshHeader();
			// 真实探测 operatord——结果回来后再刷一次（OFFLINE/READY/
			// UNKNOWN 都是真实返回值，绝不硬编码 ready）。
			void uiState.probeOperator().then(() => refreshHeader());
			// 30s 周期复探（HOTFIX-3：timer 有明确生命周期——session
			// shutdown 时清理，不积累）。
			if (probeTimer !== null) clearInterval(probeTimer);
			probeTimer = setInterval(() => {
				void uiState.probeOperator().then(() => refreshHeader());
			}, 30_000);
			probeTimer.unref();
			ctx.ui.setWorkingIndicator({ frames: WORKING_FRAMES, intervalMs: 80 });
			// P1-TUI-01：中性本地化状态词——不伪造思维过程（"Thinking..."
			// 会让人以为在看模型推理；只有真实事件阶段才显示具体阶段）。
			ctx.ui.setWorkingMessage("正在处理…");
			ctx.ui.setHiddenThinkingLabel("正在处理…");
			// ROSClaw footer：模型简称 + 上下文占用 + Operator 状态。
			// 不显示上游费用缩写/scope 噪声（费用只在 provider 有可信
			// 价格时才值得显示——当前不显示）。
			ctx.ui.setFooter((_tui, theme, _footerData) => {
				const snap = uiState.snapshot();
				const model = modelDisplay || "未选模型";
				const parts = [model, snap.mode, `Operator ${snap.operatorState}`];
				return new Text(theme.fg("dim", parts.join(" · ")), 1, 0);
			});
		});

		// -- `!` bash 功能级关闭（PNA-0 即生效；PNA-9 再做 profile 化 UI 拦截） -----
		pi.on("user_bash", async () => {
			return {
				result: {
					output: "bash execution is disabled by ROSClaw policy",
					exitCode: 1,
					cancelled: false,
					truncated: false,
				},
			};
		});

		// -- 每轮注入最新具身上下文（PNA-2，规格 §14.2） ---------------------------
		pi.on("before_agent_start", async (_event, ctx) => {
			// header 模型名取真实当前 model（P0-NA-16：同一快照语义）。
			const current = ctx.model as { name?: string; id?: string } | undefined;
			const display = current ? String(current.name ?? current.id ?? "") : "";
			if (display && display !== modelDisplay) {
				modelDisplay = display;
				uiState.noteContextChanged();
				refreshHeader();
			}
			const missionId = options.active.current.missionId;
			if (!missionId) {
				return { systemPrompt: options.systemPrompt };
			}
			const fetched = await fetchEmbodiedContext(
				options.rosclawHome,
				missionId,
				options.active.current.sessionId,
			);
			if (!fetched.stale && fetched.envelope) {
				// P0-7：验证通过后写入精确 revision/body/mode。
				options.active.applyEnvelope(fetched.envelope, fetched.contextLeaseId);
				// P0-NA-16：fresh envelope 到达 → header 从 LOADING 转 FRESH。
				uiState.noteContextChanged();
				refreshHeader();
			} else {
				// HOTFIX-3（P0-4E）：context 拉取失败/过期 → 立即标记
				// STALE + 禁动作（不再有"revision 碰巧没变就能动作"）。
				options.active.markContextStale(fetched.note);
				uiState.noteContextChanged();
				refreshHeader();
			}
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

		// -- 全量 ROSClaw 命令（NA-FIX-6，P0-8：InputGuard 允许的必须真实注册） --
		for (const [name, spec] of Object.entries(
			buildCommandHandlers({
				rosclawHome: options.rosclawHome,
				active: options.active,
				registeredToolNames: () => [
					"rosclaw_status",
					"rosclaw_observe",
					"rosclaw_verify",
					"rosclaw_memory_query",
					"rosclaw_fail_safe",
					"rosclaw_delegate",
					"rosclaw_request_action",
				],
			}),
		)) {
			pi.registerCommand(name, spec);
		}

		// -- Worker 命令（PNA-4，规格 §19）：/workers /delegate ------------------
		pi.registerCommand("workers", {
			description: "列出 Worker 与当前 Mission 的 WorkOrder 状态",
			handler: async (_args, ctx) => {
				if (!options.active.current.missionId) {
					ctx.ui.notify("未绑定 Mission——/workers 不可用", "warning");
					return;
				}
				try {
					const status = await bridgeCall(options.rosclawHome, "pi.worker.status", {
						mission_id: options.active.current.missionId,
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
				if (!options.active.current.missionId) {
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
							pi_session_id: options.active.current.sessionId,
							mission_id: options.active.current.missionId,
							context_revision: options.active.current.contextRevision,
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

		// -- Approval 卡片（NA-FIX-5，P0-5 修复）：tool 返回精确 approval_id
		//    后才展卡——绝不取 pending 列表第一个。
		pi.on("tool_execution_update", async (event, ctx) => {
			if (event.toolName !== "rosclaw_request_action" || !ctx.hasUI) return;
			const details = (event.partialResult?.details ?? {}) as {
				phase?: string;
				approval_id?: string;
				display_hash?: string;
			};
			if (details.phase !== "AWAITING_OPERATOR" || !details.approval_id) return;
			const approvalId = details.approval_id;
			const displayHash = String(details.display_hash ?? "");
			// 明确告知等待态——tool 的 onUpdate partial 文本会被 TUI spinner
			// 覆盖；spinner 行持续重绘，是执行中唯一稳定可见的通道。
			ctx.ui.setWorkingMessage(`等待 Operator 决定（approval ${approvalId}）…默认拒绝`);
			ctx.ui.notify(`等待 Operator 决定（approval ${approvalId}）…默认拒绝`, "info");
			// P0-NA-14：经 approvals.get 精确拉卡（不扫 list）；拉取失败、
			// 字段缺失或 display_hash 不一致 → fail-closed，不显示可批准卡。
			let cardData: Record<string, unknown> | undefined;
			let cardError = "";
			try {
				const got = (await operatorCall(
					defaultOperatorSocket(options.rosclawHome),
					"approvals.get",
					{ request_id: approvalId },
				)) as { ok: boolean; approval?: Record<string, unknown>; error?: string };
				if (got.ok && got.approval) {
					cardData = got.approval;
				} else {
					cardError = String(got.error ?? "card not found");
				}
			} catch (err) {
				cardError = (err as Error).message;
			}
			// 完整性与 hash 绑定校验：卡片必须带齐 mode/risk/capability/
			// parameters/expires_at，且服务端 display_hash 与 tool 报告的一致。
			// （early-return 收窄——TS 不跨复合布尔收窄。）
			if (
				cardData === undefined
				|| typeof cardData.title !== "string"
				|| typeof cardData.mode !== "string" || cardData.mode === ""
				|| typeof cardData.risk_tier !== "string"
				|| typeof cardData.expires_at !== "string" || cardData.expires_at === ""
				|| typeof cardData.parameters !== "object" || cardData.parameters === null
				|| (displayHash !== "" && String(cardData.display_hash ?? "") !== displayHash)
			) {
				ctx.ui.notify(
					`授权卡不可用（${cardError || "字段缺失或 hash 不一致"}）——` +
					"动作未执行。为安全起见本卡不可在此批准；请重新发起请求。",
					"error",
				);
				return;
			}
			const card: Record<string, unknown> = cardData;
			try {
				await ctx.ui.custom<boolean>((_tui, _theme, _kb, done) => {
					return new ApprovalCardComponent(
						{
							requestId: approvalId,
							title: String(card.title ?? approvalId),
							summary: String(card.summary ?? ""),
							riskTier: String(card.risk_tier ?? ""),
							mode: String(card.mode ?? "ACTION"),
							capability: String(card.capability_id ?? ""),
							parameters: (card.parameters ?? {}) as Record<string, unknown>,
							expiresAt: String(card.expires_at ?? ""),
							displayHash,
							expectedEffect: String(card.expected_effect ?? ""),
							failureHandling: String(card.failure_handling ?? ""),
							bodyId: String(card.body_id ?? ""),
						},
						(approve) => done(approve),
					);
				}, { overlay: true }).then(async (approve) => {
					const decided = (await operatorCall(
						defaultOperatorSocket(options.rosclawHome),
						"approvals.decide",
						{
							request_id: approvalId,
							display_hash: displayHash,
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

		// -- 认知事件镜像（PNA-8，规格 §24.2）：hash-only，不双写全文 ----------
		// NA-FIX-2：mirror 动态读 active（切换后不再写旧 mission）。
		const mirror = new EventMirror(
			options.rosclawHome,
			options.active.current.sessionId,
			options.active.current.missionId ?? "",
		);
		const mirrorSession = options.active;
		if (mirror) {
			const activeMirror = mirror;
			pi.on("message_end", async (event) => {
				const message = event.message as { role?: string; content?: unknown };
				if (message.role !== "assistant") return undefined;
				// 只镜像 hash——全文权威在 Pi session。
				const text = JSON.stringify(message.content ?? "");
				activeMirror.retarget(mirrorSession.current.sessionId, mirrorSession.current.missionId ?? "");
				activeMirror.push("message_end", {
					text,
					model: String((event.message as { model?: string }).model ?? ""),
					usage: (event.message as { usage?: Record<string, unknown> }).usage,
				});
				await activeMirror.flush();
				return undefined;
			});
			pi.on("turn_end", async (event) => {
				activeMirror.retarget(mirrorSession.current.sessionId, mirrorSession.current.missionId ?? "");
				activeMirror.push("turn_end", {
					text: JSON.stringify((event.message as { content?: unknown }).content ?? ""),
				});
				await activeMirror.flush();
				return undefined;
			});
			pi.on("session_shutdown", async () => {
				// HOTFIX-3：timer 生命周期——shutdown 清理 operator probe
				// timer（不积累）+ mirror 落盘。
				if (probeTimer !== null) {
					clearInterval(probeTimer);
					probeTimer = null;
				}
				await activeMirror.flush();
				return undefined;
			});
		}

		// -- Session 生命周期（P0-NA-12：全部经唯一 coordinator 事务） --------
		const lifecycle: LifecycleDeps = {
			coordinator: options.coordinator,
			coordinatorMissionId: () => options.active.current.missionId,
			notify: (message, type) => undefined,
		};
		pi.on("session_start", async (event, ctx) => {
			options.coordinator.setNotify((message, type) => ctx.ui.notify(message, type));
			lifecycle.notify = (message, type) => ctx.ui.notify(message, type);
			try {
				await handleSessionStart(lifecycle, event.reason, sessionIdOf(ctx));
			} catch (err) {
				ctx.ui.notify(`session 绑定异常：${(err as Error).message}`, "error");
			}
		});
		pi.on("session_before_switch", async (event, ctx) => {
			lifecycle.notify = (message, type) => ctx.ui.notify(message, type);
			// 只读预检：target 文件头可解析——绑定动作在 session_start
			// （此时 target 已是活动 session，id 无歧义）。
			const veto = await shouldCancelSwitch(event.targetSessionFile, (file) => {
				try {
					const firstLine = readFileSync(file, "utf-8").split("\n", 1)[0] ?? "";
					const header = JSON.parse(firstLine) as { id?: string };
					return typeof header.id === "string" ? header.id : null;
				} catch {
					return null;
				}
			});
			if (veto) {
				ctx.ui.notify(veto, "warning");
				return { cancel: true };
			}
			return undefined;
		});
		pi.on("session_before_tree", async (_event, ctx) => {
			lifecycle.notify = (message, type) => ctx.ui.notify(message, type);
			const veto = await shouldCancelTree({
				rosclawHome: options.rosclawHome,
				missionId: options.active.current.missionId,
			});
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
