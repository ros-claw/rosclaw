// HP2-COMPAT: Pi 扩展宿主类型（ExtensionContext/Factory）——扩展运行於 Pi 扩展宿主内，HP3 前保持；不新增会话装配引用。
/** ROSClaw 内联扩展（PNA-0）：品牌 + 安全基线。
 *
 * - header/footer/title/working 动画替换为 ROSClaw；
 * - `!` bash 功能级关闭（user_bash full replacement）；
 * - 会话生命周期观察埋点（PNA-1 挂 SessionBinding）。
 */

import type { ExtensionContext, ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { readFileSync } from "node:fs";
import { spawn, spawnSync } from "node:child_process";
import { join } from "node:path";
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
import { ActionResultCardComponent, type ActionResultData } from "../ui/action-result-card.js";
import { kernelMessageRenderers } from "../ui/kernel-message-cards.js";
import {
	formatKitBrokenHint,
	formatKitRecoveredHint,
	renderFooter,
	renderHeader,
} from "../ui/product-state.js";
import type { ProductStateCenter } from "../session/state-center.js";
import { InputController } from "../native/input-controller.js";
import { OperationWatcher } from "../native/operation-watcher.js";
import { suppressModelTurn } from "../native/turn-disposition.js";
import { ProviderStallWatchdog } from "../native/provider-watchdog.js";
import {
	renderTerminalReply,
	type TerminalOutcome,
} from "../native/terminal-presenter.js";
import { classifyModelError, ProviderErrorGate } from "../native/model-errors.js";
import { _bwrapAvailable } from "../tools/workspace-pack.js";
import {
	renderArtifactList,
	renderOperationLogs,
	renderTaskActivity,
	type KernelEvent,
} from "../native/task-activity.js";
import type { LocaleManager } from "../i18n/locale.js";
import { t as i18nT } from "../i18n/index.js";
import { EventMirror } from "./event-mirror.js";
import { WorkspaceStore } from "../session/workspace.js";
import { buildCommandHandlers } from "./commands.js";
import { guardInput } from "./input-guard.js";
import { materializeCapabilityTools, type CapabilitySnapshot } from "../tools/materialize.js";
import { MODEL_TOOL_NAMES } from "../tools/surface.js";
import { fetchEmbodiedContext, renderTrustedContext } from "./context-injection.js";
import { registerCompactAnchor } from "./compact-anchor.js";
import { appendFileSync, mkdirSync } from "node:fs";
import { phaseWorkingMessage } from "./activity.js";
import { ROSCLAW_SHORTCUTS } from "./shortcuts.js";
import { StableIdDeduper } from "./dedup.js";
import { AutoNamer } from "../session/auto-name.js";
import { formatPolicyAutoNotice } from "../ui/tool-display.js";
import { classifyNotice, NotificationLevelFilter } from "../ui/levels.js";

export interface RosclawExtensionOptions {
	profile: "developer" | "robot";
	version: string;
	/** PNA-2：v2 系统提示词（native_agent_v2.md 内容，构建期打包）。 */
	systemPrompt: string;
	/** NA-FIX-2：动态 session 上下文（切换事务的唯一真实源）。 */
	active: ActiveSessionContext;
	/** P0-NA-12：唯一 session/mission/lease 事务协调器。 */
	coordinator: AgentSessionCoordinator;
	/** PR-SIX-1：唯一产品状态中心（Header/Footer/status/tool 同源）。 */
	center: ProductStateCenter;
	/** PR-SIX-5：UI/回答语言策略。 */
	locale: LocaleManager;
	rosclawHome: string;
	/** WP-P0-3：恢复启动——session_start 展示 Resume Report。 */
	resumed?: boolean;
	/** 会话命名（WP-P0-2 标题产品化）：写入 Pi session_info。 */
	sessionManager?: import("@earendil-works/pi-coding-agent").SessionManager;
	/** 十一审 PR-D：Workspace 一等状态。 */
	workspaceStore?: import("../session/workspace.js").WorkspaceStore;
	workspaceAutoBound?: boolean;
	/** PR-N1：唯一工作区事实源（session 创建前解析并冻结）。 */
	taskContext: import("../native/active-task-context.js").ActiveTaskContext;
	/** PR-N5D：创建后回填的 session 引用（物化工具激活用）。 */
	lateSession?: { session?: { setActiveToolsByName(names: string[]): void } };
	/** 0902 R1-c（§5.3）：OS 隔离探测（测试可注入）——默认消费
	 *  doctor 落盘的 os-isolation.json，无记录时回落 bwrap 存在性。 */
	osIsolationProbe?: () => { isolationReady: boolean };
}

const WORKING_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/** 0902 R1-c 默认 OS 隔离探测：doctor 落盘的 os-isolation.json 为
 *  单源（§5.3"setup/doctor 一次探测"）；无记录时回落 bwrap 存在性。 */
function makeDefaultOsIsolationProbe(rosclawHome: string): () => { isolationReady: boolean } {
	return () => {
		try {
			const raw = readFileSync(
				join(rosclawHome, "agent", "os-isolation.json"), "utf-8",
			);
			const data = JSON.parse(raw) as { isolation_ready?: unknown };
			if (typeof data.isolation_ready === "boolean") {
				return { isolationReady: data.isolation_ready };
			}
		} catch {
			// 无记录/坏记录 → 回落存在性探测。
		}
		return { isolationReady: _bwrapAvailable() };
	};
}

export function createRosclawExtension(options: RosclawExtensionOptions): ExtensionFactory {
	const defaultOsIsolationProbe = makeDefaultOsIsolationProbe(options.rosclawHome);
	return (pi) => {
		// -- 品牌 + 单一状态源（六审 PR-SIX-1：Header/Footer 在同一次
		//    refreshChrome 里用同一个 KernelSnapshotV1 重绘——不允许顶部
		//    Kimi K3/OFFLINE 与底部未选模型/UNKNOWN 长期共存） --
		const center = options.center;
		const locale = options.locale;
		// PR-N5D：动态工具物化——CapabilitySnapshot（按当前 body/mode/
		// health 过滤）→ 精确强类型工具；digest 变化才重物化（不静默
		// 换工具：执行期 digest 失配由 bridge 以
		// CAPABILITY_SNAPSHOT_CHANGED 拒绝，这里在下一回合前刷新）。
		let materializedDigest = "";
		const refreshCapabilityTools = async (): Promise<void> => {
			const missionId = options.active.current.missionId;
			if (!missionId) return;
			const res = await center.call("pi.capability.snapshot", {
				mission_id: missionId,
			}) as { ok?: boolean; snapshot?: CapabilitySnapshot } | undefined;
			if (!res?.ok || !res.snapshot) return;
			const snap = res.snapshot;
			if (snap.digest === materializedDigest) return;
			const tools = materializeCapabilityTools(snap, {
				center, active: options.active, rosclawHome: options.rosclawHome,
			});
			for (const tool of tools) pi.registerTool(tool);
			materializedDigest = snap.digest;
			options.lateSession?.session?.setActiveToolsByName([
				...MODEL_TOOL_NAMES,
				...tools.map((tool) => tool.name),
			]);
		};
		// WP-P0-3：恢复对账报告（恢复了什么/重新验证了什么/哪些权限
		// 失效）——一次性展示，不重复。
		let resumeReportShown = !options.resumed;
		let refreshChrome: () => void = () => undefined;
		let probeTimer: ReturnType<typeof setInterval> | null = null;
		// 十审 W2：Worker 完成推送——custom message 注入（不冒充用户
		// 输入），投递账本持久化（重启/compact 不重复、不丢失）。
		type LatestCtx = {
			isIdle(): boolean;
			hasUI: boolean;
			/** pi ExtensionContext 自带——编程取消停滞的模型回合。 */
			abort?: () => void;
			ui: {
				notify(t: string, k: "info" | "warning" | "error"): void;
				setWidget(key: string, lines: string[] | undefined): void;
			};
		};
		let latestCtx: LatestCtx | undefined;
		// PR-H3：OperationWatcher——operation 终态一次性 followUp（同一
		// session；progress/heartbeat 绝不进模型上下文）。
		// P1-B2：progress 经 setWidget 按 operation_id 原位更新（单活动区）。
		const operationWatcher = new OperationWatcher({
			call: (method, params) => center.call(method, params),
			sink: () =>
				latestCtx
					? {
							api: pi,
							isIdle: latestCtx.isIdle(),
							notify: latestCtx.hasUI
								? (text: string) => latestCtx?.ui.notify(text, "info")
								: undefined,
							setWidget: latestCtx.hasUI
								? (key: string, lines: string[] | undefined) =>
										latestCtx?.ui.setWidget(key, lines)
								: undefined,
						}
					: undefined,
		});
		// 十一审 PR-D：Workspace——header 快照 + /workspace 命令（命令层
		// 直接处理，不进模型）。
		const workspaceStore = options.workspaceStore ?? new WorkspaceStore(options.rosclawHome);
		// PR-N1：header 显示真实 workspaceRoot（来源诚实标注）——
		// 不再用持久化 Project 名谎报（显示 Project rosclaw 而工具
		// 在别处工作的 split-brain 已消灭）。
		const ctxLabel = options.taskContext.workspaceSource === "default"
			? options.taskContext.workspaceRoot
			: options.taskContext.workspaceRoot.split("/").filter(Boolean).pop();
		(center.noteWorkspace?.bind(center) as ((d?: string) => void) | undefined)?.(
			ctxLabel,
		);
		pi.registerCommand("debug", {
			description: "切换调试信息层（治理/审计机制细节默认隐藏）",
			handler: async (_args, ctx) => {
				levelFilter.toggle();
				ctx.ui.notify(
					levelFilter.visible("debug")
						? "调试信息层已开启（approval/grant/lease 等机制细节可见）"
						: "调试信息层已关闭",
					"info",
				);
			},
		});
		pi.registerCommand("workspace", {
			description: "项目 workspace：/workspace show | use <path> | recent",
			handler: async (args, ctx) => {
				const sub = args.trim();
				if (!sub || sub === "show") {
					const current = workspaceStore.current;
					notifyLeveled(ctx, 
						current ? `当前 Project：${current}` : "未绑定 Project（从 git 仓库内启动自动绑定，或 /workspace use <path>）",
						"info",
					);
					return;
				}
				if (sub === "recent") {
					const recent = workspaceStore.recent;
					notifyLeveled(ctx, 
						recent.length ? `最近的 workspace：\n${recent.join("\n")}` : "（无最近记录）",
						"info",
					);
					return;
				}
				const useMatch = sub.match(/^use\s+(.+)$/);
				if (useMatch) {
					try {
						const bound = workspaceStore.bind(useMatch[1]);
						(center.noteWorkspace?.bind(center) as ((d?: string) => void) | undefined)?.(
							workspaceStore.displayName(),
						);
						notifyLeveled(ctx, `已绑定 Project：${bound}`, "info");
					} catch (err) {
						notifyLeveled(ctx, `绑定失败：${(err as Error).message}`, "error");
					}
					return;
				}
				notifyLeveled(ctx, "用法：/workspace show | use <path> | recent", "warning");
			},
		});
		pi.on("session_start", async (_event, ctx) => {
			latestCtx = ctx;
			operationWatcher.start();
			// PR-N5D：无静态 allowlist——启动即把激活面钉回
			// MODEL_TOOL_NAMES（+已物化名，digest 变化后重钉）。
			options.lateSession?.session?.setActiveToolsByName([
				...MODEL_TOOL_NAMES,
			]);
			if (options.workspaceAutoBound && workspaceStore.current) {
				notifyLeveled(ctx, `已自动绑定 Project：${workspaceStore.current}（/workspace show 查看）`, "info");
			}
			// 0902 R1-c（§5.3）：无 OS 沙箱 → 会话开始一次性提示（任务
			// 开始前），而不是 shell 执行到一半才甩卡。探测单源 =
			// doctor 落盘的 os-isolation.json（无记录回落 bwrap 存在性）。
			if (ctx.hasUI) {
				try {
					const probe = options.osIsolationProbe ?? defaultOsIsolationProbe;
					if (!probe().isolationReady) {
						notifyLeveled(ctx,
							"本机无 OS 沙箱（bwrap 不可用）——shell 类操作将在会话内"
							+ "弹确认卡降级运行；rosclaw doctor 查看结论与修复建议",
							"warning",
						);
					}
				} catch {
					// 探测失败不阻塞会话启动（doctor 可重跑）。
				}
			}
		});
		pi.on("session_shutdown", async () => {
			operationWatcher.stop();
		});
		pi.on("session_start", async (_event, ctx) => {
			if (!ctx.hasUI) return;
			ctx.ui.setTitle(`ROSClaw Native Agent`);
			if (!resumeReportShown) {
				resumeReportShown = true;
				try {
					const sessionId = options.active.current.sessionId;
					const result = await center.call("pi.session.resume_report", {
						pi_session_id: sessionId,
					});
					const report = (result.report ?? {}) as {
						verdict?: string; lines?: string[];
					};
					if (result.ok && report.lines?.length) {
						notifyLeveled(ctx, 
							`已恢复（${report.verdict ?? "?"}）\n${report.lines.join("\n")}`,
							report.verdict === "RESUMED" ? "info" : "warning",
						);
					}
				} catch {
					// 报告失败不阻塞会话——恢复本身已由 coordinator 完成。
				}
			}
			refreshChrome = () => {
				const snap = center.snapshot(); // 一次读取，Header/Footer 共享
				const loc = locale.effective;
				ctx.ui.setHeader((_tui, _theme) => new Text(renderHeader(snap, loc)));
				ctx.ui.setFooter((_tui, theme, _footerData) => {
					return new Text(theme.fg("dim", renderFooter(snap, loc)), 1, 0);
				});
			};
			refreshChrome();
			// 统一订阅：任何状态变化（context/lease/operator/model/kernel/
			// locale）触发同一次 chrome 重绘。
			center.subscribe(() => refreshChrome());
			locale.subscribe(() => refreshChrome());
			// R0-6：启动事务——bridge ping 有限重试（内核行为不耗
			// token）；完成前 chrome 显示"正在准备"，不是假 Blocked。
			void center.bootstrap();
			// 真实探测 operatord——结果回来后经 subscribe 统一重绘
			// （OFFLINE/READY/UNKNOWN 都是真实返回值，绝不硬编码 ready）。
			void center.probeOperator();
			// 六审 §7：Operator 未就绪时的非模态提示 + Ctrl+O/命令一键
			// 初始化——模态 overlay 会劫持按键（TUI 矩阵/perf 实测回归），
			// widget 只展示不抢输入。
			const runBootstrap = async (cmdCtx: typeof ctx) => {
				try {
					const status = await center.call("pi.operator.status", {});
					if (status.running) {
						cmdCtx.ui.notify(i18nT("operator.bootstrap_done", locale.effective), "info");
						return;
					}
					const result = await center.call("pi.operator.bootstrap", {
						mission_id: options.active.current.missionId ?? "",
					});
					cmdCtx.ui.notify(
						result.ok
							? i18nT("operator.bootstrap_done", locale.effective)
							: `${i18nT("operator.bootstrap_failed", locale.effective)}: ${String(result.error ?? "")}`,
						result.ok ? "info" : "error",
					);
					await center.probeOperator(true);
				} catch {
					// 失败诚实保持 OFFLINE——不伪造 READY。
				}
			};
			// 幂等：目标状态未变时不发 UDS 调用、不重绘（idle CPU 红线）。
			let bootstrapWidgetState: "hidden" | "new" | "stopped" = "hidden";
			const updateBootstrapWidget = async () => {
				if (!ctx.hasUI) return;
				if (options.profile !== "developer") return;
				if (options.active.current.mode !== "SIMULATION") return;
				// 七审 §2.5：auto SIM 不需要 operator——不提示初始化。
				if (center.isSimAutoPolicy) {
					if (bootstrapWidgetState !== "hidden") {
						bootstrapWidgetState = "hidden";
						ctx.ui.setWidget("rosclaw-operator", undefined);
					}
					return;
				}
				if (center.snapshot().operator !== "OFFLINE") {
					if (bootstrapWidgetState !== "hidden") {
						bootstrapWidgetState = "hidden";
						ctx.ui.setWidget("rosclaw-operator", undefined);
					}
					return;
				}
				// 已展示且 operator 仍 OFFLINE——enrollment/running 只能经
				// 我们发起的 bootstrap 改变（它会显式复探）——跳过重复查询。
				if (bootstrapWidgetState !== "hidden") return;
				try {
					const status = await center.call("pi.operator.status", {});
					if (status.running) {
						return;
					}
					const loc = locale.effective;
					bootstrapWidgetState = status.enrolled ? "stopped" : "new";
					ctx.ui.setWidget("rosclaw-operator", [
						`${i18nT("operator.bootstrap_title", loc)} — ${i18nT(
							status.enrolled
								? "operator.bootstrap_state_stopped"
								: "operator.bootstrap_state_new",
							loc,
						)}`,
						i18nT("operator.bootstrap_offer", loc),
					]);
				} catch {
					// 探测失败保持 OFFLINE 展示。
				}
			};
			// operator 探测结果变化 → 统一刷新 widget（subscribe 链）。
			center.subscribe(() => {
				void updateBootstrapWidget();
			});
			// 七审 PR-SEVEN-5：Robot Kit BROKEN → 用户输入前给一键修复
			// （变化驱动——同一 BROKEN 状态只提示一次）。
			let kitHintState: string | null = null;
			center.subscribe(() => {
				if (!ctx.hasUI) return;
				const kit = center.snapshot().robot_kit;
				const state = kit?.state ?? null;
				if (state === kitHintState) return;
				const prev = kitHintState;
				kitHintState = state;
				const loc = locale.effective;
				// 八审 P0-9 修复并入（此前 #296 合并时 index.ts 接线
				// 丢失——helpers 在但未调用，行为测试首跑抓到）：
				// 空 reason 不渲染悬空冒号；READY 恢复正面清除。
				if (state === "BROKEN") {
					notifyLeveled(ctx, formatKitBrokenHint(kit ?? {}, loc), "warning");
				} else if (state === "READY" && prev === "BROKEN") {
					notifyLeveled(ctx, 
						formatKitRecoveredHint(
							center.snapshot().body_display ?? "", loc,
						),
						"info",
					);
				}
			});
			setTimeout(() => {
				void center.probeOperator(true);
			}, 800);
			pi.registerShortcut(ROSCLAW_SHORTCUTS.operatorBootstrap, {
				description: i18nT("operator.bootstrap_title", locale.effective),
				handler: async (shortcutCtx) => {
					await runBootstrap(shortcutCtx as typeof ctx);
				},
			});
			// 周期复探（HOTFIX-3：timer 有明确生命周期——session
			// shutdown 时清理，不积累）。七审合并级联实测：30s 周期在
			// 共享 runner 上有可观概率落进 5s idle 测量窗口（三个 UDS
			// 探测 + 重绘 ≈ 0.2s CPU）——全部对齐 60s（状态变化本来就
			// 走 subscribe/force 通道，周期探测只是兜底）。
			if (probeTimer !== null) clearInterval(probeTimer);
			probeTimer = setInterval(() => {
				void center.probeOperator();
				void center.refreshCapabilities();
				void center.refreshRobotInfo();
			}, 60_000);
			void center.refreshCapabilities(true);
			void center.refreshRobotInfo(true);
			probeTimer.unref();
			ctx.ui.setWorkingIndicator({ frames: WORKING_FRAMES, intervalMs: 80 });
			// P1-TUI-01：中性本地化状态词——不伪造思维过程（"Thinking..."
			// 会让人以为在看模型推理；只有真实事件阶段才显示具体阶段）。
			ctx.ui.setWorkingMessage(i18nT("working.default", locale.effective));
			ctx.ui.setHiddenThinkingLabel(i18nT("working.default", locale.effective));
		});

		// -- 九审 §1.4/§1.5（P0-INPUT-LOSS 热修）：自然语言绝不 handled。
		//    先显示、先落账、先分配 Turn ID，然后才路由/执行——Pi 的
		//    handled 语义是"扩展即时处理的小命令"，自然语言任务经
		//    handled 会从模型/消息链/session JSONL 消失（幽灵执行）。
		//    本 handler 只做两件无状态小事：自动命名 + 永远 continue。
		pi.on("input", async (event, _ctx) => {
			const text = String((event as { text?: string }).text ?? "").trim();
			if (!text || text.startsWith("/")) return { action: "continue" as const };
			// WP-7：命名权交给 AutoNamer——首个"驱动工具活动"的输入
			// 才命名（真实任务的确定性信号；闲聊会话不谎报任务名）。
			autoNamer.noteInput(text);
			// 九审 §6.1/NINE-2：UserTurn 落账（先落账再进模型）——任务
			// 因果链（caused_by_turn_id）的来源。落账失败不阻塞输入
			// （Pi JSONL 仍是消息账本；UserTurn 是任务因果投影）。
			try {
				const sessionId = options.active.current.sessionId;
				if (sessionId) {
					await center.call("pi.turn.record", {
						pi_session_id: sessionId,
						text,
						source: "interactive",
					});
				}
			} catch {
				// 落账失败不阻塞输入。
			}
			// P0-C（0824 总纲 §6.1）：输入先落会话（persist——不立即
			// 创建 Task；hello/解释/只读查询 tasks=0）。持久化失败不
			// 投递（handled + 通知重发=无幽灵执行，HP1 语义不变）。
			const persisted = await inputController.persist(text);
			if (!persisted) return { action: "handled" as const };
			// R0-1.5：输入路由自动执行——已知 recipe 由内核直接执行
			// （零模型调用）；watcher 跟踪进度/终态（确定性呈现）。
			// P0-1/P0-2（0827 审计·双控制者根治）：Input Arbiter——
			// 一条输入只有一个 Owner。TASK_ROUTER 认领后 suppress 模型
			// 回合（handled，不再 continue）——同一指令绝不进 Pi Agent。
			const autoTask = (persisted as { auto_task?: { task_id?: string } })
				.auto_task;
			if (suppressModelTurn(persisted)) {
				// 0901 P0-4（硬 Gate A）：解释性追问 → EXPLAIN_HANDLER
				// 只读确定性回答（从 TaskOutcome 直接呈现——零模型
				// 回合、零新 task/trace/artifact、零仿真）。
				const explain = (
					persisted as {
						explain?: {
							task_id?: string;
							goal?: string;
							state?: string;
							outcome?: Record<string, unknown>;
							artifacts?: Array<Record<string, unknown>>;
						};
					}
				).explain;
				if (explain) {
					const goal = String(explain.goal ?? "").slice(0, 80);
					const refs = (explain.outcome?.artifact_refs ??
						explain.artifacts ??
						[]) as Array<Record<string, unknown>>;
					const lines = refs
						.map((r) => {
							const kind = String(r.kind ?? r.media_type ?? "");
							const path = String(r.path ?? "");
							return `  · ${kind}：${path}`;
						})
						.filter((l) => !l.endsWith("："));
					// 进行中的任务没有终态 outcome——诚实"还在执行"，
					// 不把 UNKNOWN 渲染成"未达成"。
					const inFlight = !explain.outcome;
					const body = inFlight
						? `刚才的任务（${String(explain.task_id ?? "").slice(0, 20)}…）：`
							+ `${goal}——还在执行中（/activity 查看进度）`
							+ (lines.length ? `\n已产出：\n${lines.join("\n")}` : "")
						: `刚才的任务（${String(explain.task_id ?? "").slice(0, 20)}…）：`
							+ `${goal}——状态 ${String(explain.state ?? "")}\n`
							+ renderTerminalReply(
								(explain.outcome ?? {}) as TerminalOutcome,
							)
							+ (lines.length ? `\n交付物：\n${lines.join("\n")}` : "");
					pi.sendMessage(
						{
							customType: "rosclaw.task_explain",
							content: body,
							display: true,
							details: { task_id: explain.task_id ?? "" },
						},
						{ triggerTurn: false },
					);
					return { action: "handled" as const };
				}
				if (autoTask?.task_id) {
					operationWatcher.trackTask(String(autoTask.task_id));
				}
				// 指令回声：handled 的输入不进 Pi 会话——确定性链认领的
				// 指令必须落在 session transcript（HP1 输入丢失防线的
				// 会话证据 + 用户可见），但绝不触发模型回合。
				pi.sendMessage(
					{
						customType: "rosclaw.user_directive",
						content: text,
						display: true,
						details: {
							task_id: autoTask?.task_id ?? "",
							owner: "TASK_ROUTER",
						},
					},
					{ triggerTurn: false },
				);
				const ctx2 = latestCtx;
				ctx2?.ui.notify(
					"任务已由确定性链自动开始执行（/activity 查看进度）",
					"info",
				);
				return { action: "handled" as const };
			}
			return { action: "continue" as const };
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
		pi.on("before_agent_start", async (event, ctx) => {
			// PR-N5D：回合开始前刷新物化工具面（digest 未变则零成本）。
			await refreshCapabilityTools();
			// header 模型名取真实当前 model（P0-NA-16：同一快照语义）。
			const current = ctx.model as { name?: string; id?: string } | undefined;
			const display = current ? String(current.name ?? current.id ?? "") : "";
			if (display) center.noteModel(display);
			const missionId = options.active.current.missionId;
			if (!missionId) {
				// PR-N2：用事件携带的 Pi 组装提示词（含可信项目上下文 +
				// 内置签名 Skill + cwd）——此前每轮整体替换为
				// options.systemPrompt，Pi 加载的 context/skills 被丢弃
				// （"恢复项目认知"的事故根因之一）。
				return { systemPrompt: event.systemPrompt ?? options.systemPrompt };
			}
			const fetched = await fetchEmbodiedContext(
				options.rosclawHome,
				missionId,
				options.active.current.sessionId,
				(_home, method, params) => center.call(method, params),
			);
			if (!fetched.stale && fetched.envelope) {
				// P0-7：验证通过后写入精确 revision/body/mode。
				options.active.applyEnvelope(fetched.envelope, fetched.contextLeaseId);
			} else {
				// HOTFIX-3（P0-4E）：context 拉取失败/过期 → 立即标记
				// STALE + 禁动作（不再有"revision 碰巧没变就能动作"）。
				options.active.markContextStale(fetched.note);
			}
			// chrome 刷新由 active.subscribe → center → refreshChrome 统一
			// 完成（applyEnvelope/markContextStale 都会触发）。
			return {
				systemPrompt: event.systemPrompt ?? options.systemPrompt,
				message: {
					customType: "rosclaw.embodied_context",
					content: renderTrustedContext(fetched),
					display: false,
					details: { stale: fetched.stale, note: fetched.note },
				},
			};
		});

		// P1-A4（0824 总纲）：任何 compaction 完成后从内核权威账本把
		// TaskRefs 锚回 LLM 上下文——compact 后 task/artifact refs 不丢。
		registerCompactAnchor(pi as never, {
			call: (method: string, params: unknown) =>
				center.call(method, params as Record<string, unknown>) as never,
			missionId: () => options.active.current.missionId,
			sessionRef: () => options.active.current.sessionId,
			log: (message: string) => {
				// 诊断先行：compact 锚决策日志（小文件，随 logs 轮转）。
				try {
					const dir = `${options.rosclawHome}/logs`;
					mkdirSync(dir, { recursive: true });
					appendFileSync(
						`${dir}/compact-anchor.log`,
						`${new Date().toISOString()} ${message}
`,
					);
				} catch { /* 诊断不阻塞 */ }
			},
		} as never);

		// -- 全量 ROSClaw 命令（NA-FIX-6，P0-8：InputGuard 允许的必须真实注册） --
		for (const [name, spec] of Object.entries(
			buildCommandHandlers({
				rosclawHome: options.rosclawHome,
				active: options.active,
				center,
				locale,
				registeredToolNames: () => [
					"rosclaw_status",
					"rosclaw_capabilities",
					"rosclaw_compute",
					"rosclaw_observe",
					"rosclaw_verify",
					"rosclaw_deliver",
					"rosclaw_memory_query",
					"rosclaw_fail_safe",
					"rosclaw_delegate",
					"rosclaw_check_work",
					"rosclaw_cancel_work",
					"rosclaw_list_work",
					"rosclaw_update_work",
					"rosclaw_retry_work",
					"rosclaw_read_work_events",
					"rosclaw_read_work_transcript",
					"rosclaw_list_work_artifacts",
					"rosclaw_read_work_failure",
					"rosclaw_extend_work",
					"rosclaw_request_action",
				],
			}),
		)) {
			pi.registerCommand(name, spec);
		}

		// -- PR-H2：Task 命令（/new /done /tasks /task）+ InputController ----
		const inputController = new InputController({
			call: (method, params) => center.call(method, params),
			missionId: () => options.active.current.missionId ?? "",
			sessionRef: () => options.active.current.sessionId ?? "",
			backendNativeId: () => options.active.current.sessionId ?? "",
			cwd: () => options.taskContext.workspaceRoot,
			bodyId: () => options.active.current.bodyId ?? "",
			notify: (text, kind) => {
				latestCtx?.ui.notify(text, kind);
			},
		});
		pi.registerCommand("newtask", {
			description: "开始新任务（当前任务保持可恢复）",
			handler: async (_args, ctx) => {
				inputController.forceNewNext = true;
				notifyLeveled(ctx, "下一条消息将开始新任务", "info");
			},
		});
		pi.registerCommand("done", {
			description: "接受当前任务（用户验收——之后的新目标是新任务）",
			handler: async (_args, ctx) => {
				const doneTaskId = await inputController.activeTaskId();
				if (!doneTaskId) {
					notifyLeveled(ctx, "当前没有活跃任务", "warning");
					return;
				}
				try {
					// PR-N0：/done = 用户接受（user_accepted_at）——此后
					// 新消息开新任务；未接受的 SUCCEEDED 被用户修正重开。
					const r = await center.call("pi.kernel.accept", {
						task_id: doneTaskId,
					});
					if (r.ok === false) {
						notifyLeveled(ctx, `不能接受：${String(r.error ?? '')}`, "warning");
						return;
					}
					notifyLeveled(ctx, 
						`任务已接受（${doneTaskId.slice(0, 14)}…）`,
						"info",
					);
				} catch (err) {
					notifyLeveled(ctx, `操作失败：${(err as Error).message}`, "error");
				}
			},
		});
		pi.registerCommand("tasks", {
			description: "列出当前 Mission 的任务",
			handler: async (_args, ctx) => {
				try {
					const result = await center.call("pi.kernel.list", {
						mission_id: options.active.current.missionId ?? "",
					});
					const tasks = (result.tasks ?? []) as Array<Record<string, unknown>>;
					if (!tasks.length) {
						notifyLeveled(ctx, "当前没有任务", "info");
						return;
					}
					notifyLeveled(ctx, 
						tasks
							.map(
								(t) =>
									`${String(t.state)}  r${String(t.active_revision)}  ${String(t.root_goal).slice(0, 40)}  [${String(t.task_id).slice(0, 14)}…]`,
							)
							.join("\n"),
						"info",
					);
				} catch (err) {
					notifyLeveled(ctx, `查询失败：${(err as Error).message}`, "error");
				}
			},
		});
		pi.registerCommand("taskinfo", {
			description: "当前任务详情（root task/revision/workspace/状态）",
			handler: async (_args, ctx) => {
				const infoTaskId = await inputController.latestTaskId();
				if (!infoTaskId) {
					notifyLeveled(ctx, "当前没有绑定任务", "warning");
					return;
				}
				try {
					const result = await center.call("pi.kernel.get", {
						task_id: infoTaskId,
					});
					const task = (result.task ?? {}) as Record<string, unknown>;
					notifyLeveled(ctx, 
						`任务 ${String(task.task_id).slice(0, 14)}…\n` +
							`状态: ${String(task.state)}\n` +
							`revision: ${String(task.active_revision)}\n` +
							`工作区: ${String(task.workspace_path)}\n` +
							`目标: ${String(task.root_goal).slice(0, 120)}`,
						"info",
					);
				} catch (err) {
					notifyLeveled(ctx, `查询失败：${(err as Error).message}`, "error");
				}
			},
		});

		// -- PR-H8：Task Activity/Logs/Artifacts（数据全部来自 TaskKernel
		//    事件流/产物账本，不经 LLM——假进度不可能） --------------------
		const fetchTaskEvents = async (): Promise<KernelEvent[]> => {
			const taskId = await inputController.latestTaskId();
			if (!taskId) return [];
			const result = await center.call("pi.kernel.events", {
				task_id: taskId,
				after_seq: 0,
			});
			return (result.events ?? []) as KernelEvent[];
		};
		pi.registerCommand("activity", {
			description: "当前任务活动（阶段时间线——来自任务账本，非模型总结）",
			handler: async (_args, ctx) => {
				if (!(await inputController.latestTaskId())) {
					notifyLeveled(ctx, "当前没有绑定任务", "warning");
					return;
				}
				try {
					const events = await fetchTaskEvents();
					notifyLeveled(ctx, renderTaskActivity(events).join("\n"), "info");
				} catch (err) {
					notifyLeveled(ctx, `查询失败：${(err as Error).message}`, "error");
				}
			},
		});
		pi.registerCommand("logs", {
			description: "当前任务后台进程输出（operation 日志尾部）",
			handler: async (_args, ctx) => {
				if (!(await inputController.latestTaskId())) {
					notifyLeveled(ctx, "当前没有绑定任务", "warning");
					return;
				}
				try {
					const events = await fetchTaskEvents();
					notifyLeveled(ctx, renderOperationLogs(events).join("\n"), "info");
				} catch (err) {
					notifyLeveled(ctx, `查询失败：${(err as Error).message}`, "error");
				}
			},
		});
		pi.registerCommand("artifacts", {
			description: "当前任务交付物列表（登记才算交付）",
			handler: async (_args, ctx) => {
				if (!(await inputController.latestTaskId())) {
					notifyLeveled(ctx, "当前没有绑定任务", "warning");
					return;
				}
				try {
					const result = await center.call("pi.kernel.artifacts", {
						task_id: await inputController.latestTaskId(),
					});
					const artifacts = (result.artifacts ?? []) as Array<Record<string, unknown>>;
					notifyLeveled(ctx, renderArtifactList(artifacts).join("\n"), "info");
				} catch (err) {
					notifyLeveled(ctx, `查询失败：${(err as Error).message}`, "error");
				}
			},
		});
		// Ctrl+T：Task Activity 常驻 widget 开关（turn_end 自动刷新——
		// 内容不变不重绘由 setWidget 侧保证）。
		let activityWidgetOn = false;
		const refreshActivityWidget = async (): Promise<void> => {
			if (!activityWidgetOn || !latestCtx?.hasUI) return;
			if (!(await inputController.latestTaskId())) {
				latestCtx.ui.setWidget("rosclaw-activity", ["（当前没有绑定任务）"]);
				return;
			}
			try {
				const events = await fetchTaskEvents();
				latestCtx.ui.setWidget("rosclaw-activity", renderTaskActivity(events));
			} catch {
				// 桥暂不可用——保留旧内容，下回合再刷。
			}
		};
		pi.registerShortcut(ROSCLAW_SHORTCUTS.taskActivity, {
			description: "打开/关闭任务活动视图",
			handler: async (ctx) => {
				activityWidgetOn = !activityWidgetOn;
				if (!activityWidgetOn) {
					ctx.ui.setWidget("rosclaw-activity", undefined);
					return;
				}
				await refreshActivityWidget();
			},
		});

		// -- PR-H9：F2 Task Panel（kernel 背板——WorkOrder 旧链已删） ----
		// 任务由对话驱动：面板只读（修复/取消走对话与 /done，不做第二控
		// 制面）。数据全部来自 TaskKernel（pi.kernel.list/events/
		// artifacts——与 /activity /artifacts 同一渲染器）。
		type CmdCtx = {
			ui: {
				notify(t: string, k: "info" | "warning" | "error"): void;
				custom: ExtensionContext["ui"]["custom"];
			};
		};
		const openTasksCenter = async (ctx: CmdCtx) => {
			if (!options.active.current.missionId) {
				notifyLeveled(ctx, "未绑定 Mission——Task Panel 不可用", "warning");
				return;
			}
			const missionId = options.active.current.missionId;
			const { TasksCenterComponent } = await import("../workers/tasks-center.js");
			await ctx.ui.custom<boolean>((_tui, _theme, _kb, done) => {
				return new TasksCenterComponent({
					fetchTasks: async () => {
						const r = await center.call("pi.kernel.list", { mission_id: missionId });
						return (r.tasks ?? []) as Array<Record<string, unknown>>;
					},
					fetchEvents: async (taskId) => {
						const r = await center.call("pi.kernel.events", { task_id: taskId, after_seq: 0 });
						return (r.events ?? []) as KernelEvent[];
					},
					fetchArtifacts: async (taskId) => {
						const r = await center.call("pi.kernel.artifacts", { task_id: taskId });
						return (r.artifacts ?? []) as Array<Record<string, unknown>>;
					},
					notify: (text, kind) => notifyLeveled(ctx, text, kind),
					onClose: () => done(true),
					// 0902 R3-d（§6.2）：按 o 打开交付物——图形环境
					// 探测（xdg-open 存在才注入；否则组件内诚实提示）。
					openArtifact: spawnSync("which", ["xdg-open"]).status === 0
						? (path: string) => {
							spawn("xdg-open", [path], { detached: true, stdio: "ignore" }).unref();
						}
						: undefined,
				});
			}, { overlay: true });
		};
		pi.registerShortcut(ROSCLAW_SHORTCUTS.tasksCenter, {
			description: "打开/关闭任务面板",
			handler: async (ctx) => {
				await openTasksCenter(ctx);
			},
		});
		try {
			pi.registerShortcut(ROSCLAW_SHORTCUTS.tasksCenterAlt, {
				description: "任务面板（第二绑定）",
				handler: async (ctx) => {
					await openTasksCenter(ctx);
				},
			});
		} catch {
			// 键位冲突则不注册第二绑定（F2 仍可用）。
		}

		// -- Approval 卡片（NA-FIX-5，P0-5 修复）：tool 返回精确 approval_id
		//    后才展卡——绝不取 pending 列表第一个。
		// 0902 R1-a：shell 降级授权卡（Approval Broker——确认卡只给
		// 允许一次/本任务允许/拒绝；批准后立即继续原操作）。在
		// AWAITING_OPERATOR 分支之前拦截（不同 phase 名，不同决定面）。
		pi.on("tool_execution_update", async (event, ctx) => {
			if (event.toolName !== "bash" || !ctx.hasUI) return;
			const details = (event.partialResult?.details ?? {}) as {
				phase?: string;
				request_id?: string;
			};
			if (details.phase !== "AWAITING_SHELL_APPROVAL" || !details.request_id) return;
			const requestId = details.request_id;
			ctx.ui.setWorkingMessage(`等待授权决定（${requestId}）…默认拒绝`);
			notifyLeveled(ctx, `等待授权决定（${requestId}）…默认拒绝`, "info");
			// 卡打开 = 用户在场——暂停停滞看门狗（journey 实证：确认卡
			// 等待被 45s 流式 idle 误判取消）。
			stallWatchdog.pauseForUser();
			let choice: string | undefined;
			try {
				choice = await ctx.ui.select(
					"本机无 OS 沙箱（bwrap 不可用）——当前命令需要在任务工作区内无沙箱执行（凭据/控制面对 shell 可达）",
					["允许一次", "本任务允许（当前 revision）", "拒绝"],
					{ timeout: 120000 },
				);
			} finally {
				stallWatchdog.resumeFromUser();
			}
			try {
				const decision = choice === "允许一次"
					? "allow_once"
					: choice?.startsWith("本任务允许")
						? "allow_task"
						: "deny";
				const decided = (await center.call("pi.shell_gate.decide", {
					request_id: requestId,
					decision,
				})) as { ok: boolean; error?: string };
				notifyLeveled(ctx,
					decided.ok
						? decision === "deny"
							? "已拒绝（操作未执行）"
							: "已批准——原操作继续（结果带 TOOL_LAYER_ONLY 标记）"
						: `决定被拒：${decided.error ?? "unknown"}`,
					decided.ok ? "info" : "error",
				);
			} catch (err) {
				notifyLeveled(ctx, `授权卡交互失败：${(err as Error).message}`, "error");
			}
		});
		pi.on("tool_execution_update", async (event, ctx) => {
			if (
				(event.toolName !== "rosclaw_request_action")
				|| !ctx.hasUI
			) return;
			const details = (event.partialResult?.details ?? {}) as {
				phase?: string;
				approval_id?: string;
				display_hash?: string;
			};
			// 七审 §2.5：POLICY_AUTO——安全 SIM 政策自动授权，只通知不弹卡。
			if (details.phase === "POLICY_AUTO") {
				// WP-7：SIM 用户面隐藏 POLICY_AUTO/approval/grant 治理
				// 术语（审计链在事件账本，用户给可理解的说明）。
				notifyLeveled(ctx, 
					formatPolicyAutoNotice({ approvalId: details.approval_id }),
					"info",
				);
				return;
			}
			if (details.phase !== "AWAITING_OPERATOR" || !details.approval_id) return;
			const approvalId = details.approval_id;
			const displayHash = String(details.display_hash ?? "");
			// 明确告知等待态——tool 的 onUpdate partial 文本会被 TUI spinner
			// 覆盖；spinner 行持续重绘，是执行中唯一稳定可见的通道。
			ctx.ui.setWorkingMessage(`等待 Operator 决定（approval ${approvalId}）…默认拒绝`);
			notifyLeveled(ctx, `等待 Operator 决定（approval ${approvalId}）…默认拒绝`, "info");
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
			// 六审 §4.4.6：完整性扩展到 ExactAction 绑定字段——capability_id/
			// mission_id/body_id/context_revision/context_hash/
			// action_intent_hash/exact expiry 缺任一即 fail closed。
			const exactRaw = cardData?.exact_action_json;
			let exactIntegrity = false;
			if (typeof exactRaw === "string" && exactRaw !== "") {
				try {
					const exact = JSON.parse(exactRaw) as Record<string, unknown>;
					exactIntegrity =
						typeof exact.capability_id === "string" && exact.capability_id !== ""
						&& exact.mission_id === options.active.current.missionId
						&& typeof exact.body_id === "string" && exact.body_id !== ""
						&& typeof exact.context_revision === "number"
						&& typeof exact.context_hash === "string" && exact.context_hash !== ""
						&& typeof exact.action_intent_hash === "string" && exact.action_intent_hash !== ""
						&& typeof exact.expires_at === "string" && exact.expires_at !== ""
						&& exact.expires_at === cardData?.expires_at;
				} catch {
					exactIntegrity = false;
				}
			}
			if (
				cardData === undefined
				|| typeof cardData.title !== "string"
				|| typeof cardData.mode !== "string" || cardData.mode === ""
				|| typeof cardData.risk_tier !== "string"
				|| typeof cardData.expires_at !== "string" || cardData.expires_at === ""
				|| typeof cardData.parameters !== "object" || cardData.parameters === null
				|| (displayHash !== "" && String(cardData.display_hash ?? "") !== displayHash)
				|| !exactIntegrity
			) {
				notifyLeveled(ctx, 
					`授权卡不可用（${cardError || "字段缺失或 hash 不一致"}）——` +
					"动作未执行。为安全起见本卡不可在此批准；请重新发起请求。",
					"error",
				);
				return;
			}
			const card: Record<string, unknown> = cardData;
			// 0902 复核 H2：动作批准卡（物理动作——比 shell 卡更需要
			// 思考时间）必须暂停停滞看门狗——与 shell 确认卡同规则
			//（journey 实证类：卡等待被 45s 流式 idle 误判取消）。
			stallWatchdog.pauseForUser();
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
					notifyLeveled(ctx, 
						decided.ok
							? approve
								? "已批准（等待执行回执）"
								: "已拒绝"
							: `决定被拒：${decided.error ?? "unknown"}`,
						decided.ok ? "info" : "error",
					);
				});
			} catch (err) {
				notifyLeveled(ctx, `授权卡交互失败：${(err as Error).message}`, "error");
			} finally {
				stallWatchdog.resumeFromUser();
			}
		});

		// -- 内核消息卡（0901 体验审计 P0-6）----------------------------
		// 内部 customType（user_directive/task_terminal/task_explain）
		// 以卡片呈现——协议标签（[rosclaw.xxx]）绝不上屏。
		for (const [customType, card] of Object.entries(kernelMessageRenderers)) {
			pi.registerMessageRenderer(customType, (message) => card(message));
		}

		// -- 权威 Action Result Card（五审 P0-5F）--------------------------
		// 动作终态只由 ROSClaw runtime 渲染（结构化 outcome → 不可变卡）；
		// 模型自然语言不得宣布/改写完成状态。条目持久化进 session（可审计）
		// 但不进 LLM 上下文（模型看不到也改不了）。
		pi.registerEntryRenderer<ActionResultData>(
			"rosclaw.action_result",
			(entry) => (entry.data ? new ActionResultCardComponent(entry.data) : undefined),
		);
		pi.registerEntryRenderer<{ claim: string; status: string }>(
			"rosclaw.action_conflict",
			(entry) =>
				entry.data
					? new Text(
						`⚠ 模型叙述与内核权威结果冲突（实际：${entry.data.status}）——` +
						`该叙述未被接受，以 ROSClaw 动作结果卡为准。`,
						1,
						0,
					)
					: undefined,
		);
		// P0-A：同一动作的权威结果卡只渲染一张（provider retry/
		// 事件重放不产生第二张卡）。
		const actionCardDeduper = new StableIdDeduper();
		// P0-D：完成通知每 session 只发一次（不重复报喜）。
		// 每个 outcome 只校验紧随其后的第一段助手叙述；turn 结束清除。
		let lastOutcome: (ActionResultData & { narrativeSeen?: boolean; conflictClaim?: string }) | null = null;
		// PR-N9：结构化活动区——工具开始/结束驱动活动区文案
		// （可审计事件，不是静态 Working… 也不是思维链）。
		pi.on("tool_execution_start", async (event, ctx) => {
			if (!ctx.hasUI) return;
			// WP-7：首个真实任务命名——工具活动是确定性信号。
			autoNamer.noteToolActivity();
			const autoName = autoNamer.name();
			if (autoName) {
				try {
					const sm = options.sessionManager;
					if (sm && !sm.getSessionName()) sm.appendSessionInfo(autoName);
				} catch {
					// 命名失败不阻塞执行。
				}
			}
			ctx.ui.setWorkingMessage(
				phaseWorkingMessage({
					currentTool: String(event.toolName ?? ""), operation: null,
				}),
			);
		});
		pi.on("tool_execution_end", async (event, _ctx) => {
			if (event.toolName === "process_start") {
				// PR-H3：登记模型启动的 operation（终态后 followUp 一次）。
				const text = JSON.stringify(event.result?.details ?? {}) + JSON.stringify(event.result?.content ?? []);
				const match = text.match(/op_[a-f0-9]+/);
				if (match) operationWatcher.track(match[0]);
				return;
			}
			if (event.toolName !== "rosclaw_request_action") return;
			const details = ((event.result?.details ?? {}) as Record<string, unknown>);
			const data: ActionResultData = {
				status: String(details.status ?? (event.isError ? "FAILED" : "UNKNOWN")),
				capabilityId: String(details.capability_id ?? ""),
				approvalId: details.approval_id ? String(details.approval_id) : undefined,
				grantId: details.grant_id ? String(details.grant_id) : undefined,
				txnId: details.txn_id ? String(details.txn_id) : undefined,
				actionId: details.action_id ? String(details.action_id) : undefined,
				receiptId: details.receipt_id ? String(details.receipt_id) : undefined,
				errorCode: details.error_code ? String(details.error_code) : undefined,
			};
			lastOutcome = data;
			// P0-A：稳定 ID（txn/action/approval/call 任一可用）upsert。
			if (actionCardDeduper.check(
				data.txnId ?? data.actionId ?? data.approvalId ?? "",
			)) {
				pi.appendEntry("rosclaw.action_result", data);
			}
		});
		// 冲突检测：outcome 非 COMPLETED 而助手叙述自称完成 → 可见冲突标记。
		// pi 事件模型：tool 执行属于"tool_call turn"——turn_end 先于下一轮
		// 的助手叙述到达，所以 outcome 要等叙述处理完才清除（不能在第一个
		// turn_end 就清）。appendEntry 也只能在 turn_end 落（message_end
		// 内消息 finalize 进行中，会话层会丢条目）。
		const COMPLETION_CLAIM =
			/(已执行|已完成|已确认|执行完毕|成功执行|successfully executed|has been executed|action completed)/i;
		// P0-7（0827 审计 §八）：provider 错误经闸门——同一错误码一张
		// 中文卡（重试的重复 message_end 不重复显示）；原始错误进
		// /activity 账本；PROVIDER_PAUSED 进 Header/readiness；模型
		// 切换或下次成功复位（恢复同一 turn，不重建任务）。
		const providerGate = new ProviderErrorGate();
		pi.on("model_select", async () => {
			providerGate.onModelSwitch();
			center.noteProviderOk();
			return undefined;
		});
		pi.on("message_end", async (event) => {
			// PR-H7（§8.4）：provider 错误分类——403 配额≠鉴权错误；
			// 稳定错误码 + 用户可理解说明 + 恢复动作（task 可继续）。
			const msg = event.message as { role?: string; stopReason?: string; errorMessage?: string };
			if (msg.role === "assistant" && (msg.stopReason === "error" || msg.errorMessage)) {
				const raw = String(msg.errorMessage ?? "");
				const classified = classifyModelError(raw);
				let hasActiveTask = false;
				try {
					hasActiveTask = Boolean(await inputController.activeTaskId());
				} catch {
					hasActiveTask = false;
				}
				const verdict = providerGate.onError(classified, {
					hasActiveTask, raw,
				});
				if (verdict.showCard) {
					latestCtx?.ui.notify(verdict.cardText, "error");
					center.noteProviderPaused(classified.code);
				}
				// 原始错误永远进账本（即使卡片被去重——/activity 可查）。
				if (verdict.activity) {
					pi.appendEntry("rosclaw.provider_error", verdict.activity);
				}
			} else if (msg.role === "assistant" && !msg.errorMessage) {
				providerGate.onSuccess();
				center.noteProviderOk();
			}
			if (!lastOutcome || lastOutcome.narrativeSeen) return undefined;
			const message = event.message as { role?: string; content?: unknown };
			if (message.role !== "assistant") return undefined;
			const raw = message.content;
			const text = Array.isArray(raw)
				? raw
					.map((b) => (typeof b === "object" && b !== null ? String((b as { text?: string }).text ?? "") : ""))
					.join(" ")
				: String(raw ?? "");
			if (!text.trim()) return undefined; // toolCall 消息——叙述还没到，不消费
			const outcome = lastOutcome;
			outcome.narrativeSeen = true;
			if (outcome.status !== "COMPLETED" && COMPLETION_CLAIM.test(text)) {
				outcome.conflictClaim = text.slice(0, 120);
			}
			return undefined;
		});
		// 0902 R1-b（审计 §7）：Provider 停滞分阶段看门狗——首 token
		// 10s 提示/30s 取消；流式 idle 15s 状态/45s 恢复。有字节流动的
		// 长任务不杀（流动即续期）。"Provider 是根因"不等于 ROSClaw
		// 没责任——静默 300s 是产品缺陷。
		const stallWatchdog = new ProviderStallWatchdog({
			notice: (t) => latestCtx?.ui.notify(t, "warning"),
			stallAbort: () => {
				// 0902 复核 M8：abort 是可选链——pi 实际 ctx 若无此方法，
				// 不得宣称"已取消"（假取消文案）；诚实提示手动中断。
				if (latestCtx && !latestCtx.isIdle()) {
					if (typeof latestCtx.abort === "function") {
						latestCtx.abort();
					} else if (latestCtx.hasUI) {
						latestCtx.ui.notify(
							"Provider 停滞但当前 pi 版本不支持编程取消——请按 Esc 手动中断",
							"warning",
						);
					}
				}
			},
		});
		pi.on("turn_start", async () => {
			stallWatchdog.turnStarted();
		});
		pi.on("message_update", async () => {
			stallWatchdog.contentProgress();
		});
		pi.on("message_end", async (event) => {
			// 只数 assistant 消息——pi 在 turn_start 后立刻为用户 prompt
			// 自身发 message_start/message_end（agent-loop.js runAgentLoop）。
			// 若不按 role 过滤，stall 腿的 prompt message_end 会把看门狗
			// 提前推进流式阶段：首 token 计时被清，"响应迟滞"提示永不
			// 触发（journey pytest-2340/2341 实证）。
			const role = (event as { message?: { role?: unknown } }).message?.role;
			if (role === "assistant") stallWatchdog.contentProgress();
		});
		// 工具执行（含授权卡等待）是活跃证据——等卡的回合不是
		// 停滞（journey 实证：委派腿的确认卡等待被误判停滞取消）。
		pi.on("tool_execution_start", async () => {
			stallWatchdog.contentProgress();
		});
		pi.on("tool_execution_update", async () => {
			stallWatchdog.contentProgress();
		});
		pi.on("tool_execution_end", async () => {
			stallWatchdog.contentProgress();
		});
		pi.on("agent_end", async () => {
			stallWatchdog.turnEnded();
		});
		pi.on("turn_end", async () => {
			stallWatchdog.turnEnded();
		});
		pi.on("turn_end", async () => {
			// 大道至简 R0-2b：删除 turn_end 自动 consider→finish——
			// Kernel 不得在用户回合结束时自动宣布"任务完成：验收 PASS"
			//（用户目标完成与否由看过全过程的 Pi 判断；任务终态由
			// 模型显式 rosclaw_task_finish 驱动，Kernel 只核验客观
			// 执行事实）。0901 P0-5 的 consider→finish 自动终态通道
			// 一并关闭。
			// PR-H8：Task Activity widget 回合后自动刷新（开启时）。
			await refreshActivityWidget();
			if (lastOutcome?.conflictClaim) {
				pi.appendEntry("rosclaw.action_conflict", {
					claim: lastOutcome.conflictClaim,
					status: lastOutcome.status,
				});
			}
			// 只有叙述已处理（或 conflict 已标记）才清除——否则留给下一个
			// pi-turn 的助手消息（tool 执行后的最终回答在新 turn）。
			if (lastOutcome?.narrativeSeen) lastOutcome = null;
			return undefined;
		});

		// -- 认知事件镜像（PNA-8，规格 §24.2）：hash-only，不双写全文 ----------
		// NA-FIX-2：mirror 动态读 active（切换后不再写旧 mission）。
		// WP-7：会话自动命名（见 input/tool_execution_start 接线）。
		const autoNamer = new AutoNamer();
		// P0-H：三层信息密度——debug 层（治理/审计机制细节）默认
		// 隐藏，/debug 切换；conversation/activity 永远可见。
		const levelFilter = new NotificationLevelFilter();
		const notifyLeveled = (
			ctx: { ui: { notify(t: string, k?: "info" | "warning" | "error"): void } },
			text: string,
			kind?: "info" | "warning" | "error",
		): void => {
			if (levelFilter.visible(classifyNotice(text))) {
				ctx.ui.notify(text, kind);
			}
		};
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
			options.coordinator.setNotify((message, type) => notifyLeveled(ctx, message, type));
			lifecycle.notify = (message, type) => notifyLeveled(ctx, message, type);
			try {
				await handleSessionStart(lifecycle, event.reason, sessionIdOf(ctx));
			} catch (err) {
				notifyLeveled(ctx, `session 绑定异常：${(err as Error).message}`, "error");
			}
		});
		pi.on("session_before_switch", async (event, ctx) => {
			lifecycle.notify = (message, type) => notifyLeveled(ctx, message, type);
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
				notifyLeveled(ctx, veto, "warning");
				return { cancel: true };
			}
			return undefined;
		});
		pi.on("session_before_tree", async (_event, ctx) => {
			lifecycle.notify = (message, type) => notifyLeveled(ctx, message, type);
			const veto = await shouldCancelTree({
				rosclawHome: options.rosclawHome,
				missionId: options.active.current.missionId,
			});
			if (veto) {
				notifyLeveled(ctx, veto, "warning");
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
