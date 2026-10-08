/** Pi runtime 装配（PNA-0）：createAgentSessionRuntime + InteractiveMode。
 *
 * 安全基线（审计 §5）：noExtensions/noSkills/noPromptTemplates/noThemes/
 * noContextFiles 全关——项目 .pi、AGENTS.md、~/.agents/skills 一律不加载；
 * ROSClaw 内联扩展经 extensionFactories 注入（不受 noExtensions 影响）。
 */

import {
	createAgentSessionFromServices,
	createReadToolDefinition,
	createAgentSessionRuntime,
	createAgentSessionServices,
	ModelRuntime,
	SessionManager,
	SettingsManager,
	type AgentSessionRuntime,
} from "@earendil-works/pi-coding-agent";
import { resourcePolicy, trustFilterContextFiles } from "../../extension/resource-policy.js";
import { verifyBundledSkills } from "../../extension/bundled-skills.js";
import { createSharedModelRuntime } from "./pi-model-runtime.js";
import { filterModelTools, MODEL_TOOL_NAMES } from "../../tools/surface.js";
import { buildWorkspacePackTools } from "../../tools/workspace-pack.js";
import { buildProcessTools } from "../../tools/process-tools.js";
import { buildProductPackTools } from "../../tools/product-pack.js";
import { buildEmbodimentExecTools } from "../../tools/embodiment-exec.js";
import { ActiveSessionContext } from "../../session/active-context.js";
import { AgentSessionCoordinator } from "../../session/coordinator.js";
import { SessionLeaseManager } from "../../session/lease-manager.js";
import { ProductStateCenter } from "../../session/state-center.js";
import { LocaleManager } from "../../i18n/locale.js";
import { defaultOperatorSocket } from "../../bridge/operatord-client.js";
import { appendFileSync, mkdirSync, readFileSync } from "node:fs";
import { observeCompactionStream } from "./compaction-stream.js";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { getSupportedThinkingLevels, type Model, type Api } from "@earendil-works/pi-ai";
import type { ThinkingLevel } from "@earendil-works/pi-agent-core";
import { credentialStoreFor } from "../../credentials/store.js";

export interface ExplicitModelSelection {
	modelRuntime: ModelRuntime;
	model: Model<Api>;
}

/** Offline authority: no provider stream, login, refresh or credential writes. */
export async function resolveExplicitModel(
	agentDir: string, profile: "developer" | "robot", provider: string, modelId: string,
): Promise<ExplicitModelSelection> {
	const store = credentialStoreFor(profile, agentDir);
	const modelRuntime = await ModelRuntime.create({
		credentials: {
			read: store.read.bind(store), list: store.list.bind(store),
			modify: async () => { throw new Error("MODEL_OVERRIDE_AUTH_WRITE_FORBIDDEN"); },
			delete: async () => { throw new Error("MODEL_OVERRIDE_AUTH_WRITE_FORBIDDEN"); },
		} as never,
		authPath: `${agentDir}/auth.json`,
		modelsPath: profile === "robot" ? null : `${agentDir}/models.json`,
		allowModelNetwork: false,
	});
	if (!modelRuntime.getProvider(provider)) throw new Error("UNKNOWN_MODEL_PROVIDER");
	const model = modelRuntime.getPhysicalModel(provider, modelId);
	if (!model) throw new Error("UNKNOWN_MODEL_TARGET");
	if (!modelRuntime.getAvailableSnapshot().some(m => m.provider === provider && m.id === modelId)) {
		throw new Error("MODEL_AUTH_UNAVAILABLE: configure target credentials separately; no automatic login/refresh");
	}
	return { modelRuntime, model };
}

export function requireSupportedThinking(selection: ExplicitModelSelection, thinking: string): ThinkingLevel {
	if (!getSupportedThinkingLevels(selection.model).includes(thinking as ThinkingLevel)) {
		throw new Error(`MODEL_THINKING_UNSUPPORTED: ${thinking}; no effort fallback`);
	}
	return thinking as ThinkingLevel;
}

import { createRosclawExtension } from "../../extension/index.js";
import { createToolCallBudgetExtension, type ToolCallBudget } from "./tool-call-budget.js";
import {
	SessionWriterOwnership,
	canonicalSessionPath,
	isSessionInUse,
} from "./session-writer-ownership.js";
import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { buildBridgeTools } from "../../tools/bridge-tools.js";
import { buildRequestActionTool } from "../../tools/request-action.js";
import { buildCapabilitiesTool } from "../../tools/capabilities.js";
import { buildInspectTool } from "../../tools/inspect.js";

import { buildReadOnlyTaskTools } from "../../tools/task-read.js";
import { buildStatusTool } from "../../tools/status.js";

export interface RosclawRuntimeOptions {
	cwd: string;
	/** PR-N1：ActiveTaskContext（session 创建前解析并冻结——唯一
	 *  工作区事实源）。 */
	taskContext: import("../../native/active-task-context.js").ActiveTaskContext;
	rosclawHome: string;
	profile: "developer" | "robot";
	version: string;
	missionId?: string;
	/** NA-FIX-2：--resume/--continue 打开的既有 session（否则新建）。 */
	sessionManager?: import("@earendil-works/pi-coding-agent").SessionManager;
	/** WP-P0-3：本次启动是恢复——session_start 展示 Resume Report。 */
	resumed?: boolean;
	/** Validated before any fork/mission; only initial runtime uses this choice. */
	explicitModel?: ExplicitModelSelection;
	explicitThinking?: ThinkingLevel;
	overrideSourceId?: string;
	/** 十一审 PR-D：Workspace 一等状态。 */
	workspaceStore?: import("../../session/workspace.js").WorkspaceStore;
	workspaceAutoBound?: boolean;
	/** Optional restrictive tool budget; caller owns episode admission and persistence. */
	toolCallBudget?: ToolCallBudget;
	/** SESSION_WRITER：调用方持有的 writer owner（main/backend 的
	 *  resume 在 SessionManager.open 前已占有）；缺省时本 runtime
	 *  自建独立 owner——同 PID 的两个 runtime 是不同 writer。 */
	ownership?: SessionWriterOwnership;
}

/** session_before_switch 公共 veto/reservation seam：目标文件被其他
 *  活 owner（或 UNKNOWN 锁）持有时在 SDK SessionManager.open 之前
 *  取消切换。无冲突目标则在 open 之前**实际占有**（exclusive
 *  reservation）——check-only 预检不够：预检之后、SDK open 之前插入
 *  的竞争 writer 必须被拒绝。reservation 记入 `reservations` 集合：
 *  目标 open 失败时只释放该 target reservation，旧 runtime 与其旧
 *  排他 claim 保持存活；veto 路径不留下任何新 claim、不动他人锁。 */
export function createSessionWriterOwnershipExtension(
	ownership: SessionWriterOwnership,
	reservations?: Set<string>,
): ExtensionFactory {
	return (pi) => {
		(pi as unknown as {
			on(event: string, handler: (event: {
				reason?: string;
				targetSessionFile?: string;
			}) => { cancel: true } | undefined): void;
		}).on("session_before_switch", (event) => {
			if (!event.targetSessionFile) return undefined; // new session：无既有文件
			try {
				// 同文件自切换（target == 本 owner 已持有的当前文件）只是
				// 幂等复核，绝不记入 reservations——否则 open 失败的公共
				// wrapper 会把旧 runtime 的既有排他 claim 一起释放，泄漏
				// 给第二个 acquirer。只有真正新占有的 target 才是
				// reservation（open 失败时被 token-only 释放）。
				const alreadyHeld = ownership.has(event.targetSessionFile);
				const canonical = ownership.acquire(event.targetSessionFile);
				if (!alreadyHeld) reservations?.add(canonical);
				return undefined;
			} catch (err) {
				if (isSessionInUse(err)) return { cancel: true };
				throw err;
			}
		});
	};
}

/** native_agent_v2.md：构建期从 Python 源树拷入 dist/prompts（单一事实源）。 */
export function loadSystemPrompt(): string {
	const here = dirname(fileURLToPath(import.meta.url));
	const candidates = [
		join(here, "..", "..", "prompts", "native_agent_v2.md"),
		join(here, "..", "..", "..", "prompts", "native_agent_v2.md"),
	];
	for (const candidate of candidates) {
		try {
			return readFileSync(candidate, "utf-8");
		} catch {
			// next candidate
		}
	}
	throw new Error("native_agent_v2.md not found in dist/prompts (stale/incomplete build?)");
}

export interface RosclawRuntime {
	runtime: AgentSessionRuntime;
	active: ActiveSessionContext;
	/** P0-NA-12：唯一 session/mission/lease 事务协调器——main 的初始
	 * 绑定（--mission/--resume/--continue）与扩展的生命周期 hook 共用。 */
	coordinator: AgentSessionCoordinator;
	leaseManager: SessionLeaseManager;
	/** 本 runtime 的 session writer owner（正常退出/close 时 releaseAll）。 */
	ownership: SessionWriterOwnership;
}

export async function createRosclawRuntime(
	options: RosclawRuntimeOptions,
): Promise<RosclawRuntime> {
	const toolBudgetExtension = options.toolCallBudget === undefined
		? undefined : createToolCallBudgetExtension(options.toolCallBudget, options.taskContext.workspaceRoot);
	const active = new ActiveSessionContext({
		sessionId: "",
		missionId: options.missionId,
		contextRevision: 0,
		mode: "SIMULATION",
		profile: options.profile,
		contextState: "LOADING",
		leaseState: "NONE",
		actionsAllowed: false,
	});
	// P0-NA-12：coordinator 拥有 leaseManager；扩展 hook 与 main 初始
	// 绑定都经它，lease_token 绝不丢弃、heartbeat 唯一。
	const leaseManager = new SessionLeaseManager(options.rosclawHome);
	// HOTFIX-3：heartbeat 连续失败 → LEASE_LOST——ActiveSessionContext
	// 立即禁行动作（admission 的内核校验仍是最终权威）。
	leaseManager.onLeaseLost = () => {
		active.markLeaseLost();
	};
	const coordinator = new AgentSessionCoordinator({
		rosclawHome: options.rosclawHome,
		active,
		leaseManager,
		notify: () => undefined, // UI notify 在 hook 触发时注入
	});
	// PR-SIX-1：唯一产品状态中心——Header/Footer/status tool/context 全部
	// 从它读快照；任何变化经 subscribe 统一刷新 chrome。
	const center = new ProductStateCenter({
		rosclawHome: options.rosclawHome,
		active,
		operatorSocket: defaultOperatorSocket(options.rosclawHome),
		productVersion: options.version,
	});
	const agentDir = `${options.rosclawHome}/agent`;
	// PR-SIX-5：UI/回答语言策略（持久化；launcher 可经 ROSCLAW_UI_LOCALE
	// 覆盖）。
	const locale = new LocaleManager(agentDir);
	const loadedSettings = SettingsManager.create(options.cwd, agentDir);
	const settingsManager = options.explicitModel
		? SettingsManager.inMemory({ ...loadedSettings.getSettings(), retry: { enabled: false, maxRetries: 0 } })
		: loadedSettings;
	// P1-1：raw reasoning 默认不显示（live + resumed history 同策；
	// debug 可在 /settings 手动打开）。
	settingsManager.setHideThinkingBlock(true);
	// P0-NA-15：quiet startup——正常启动不显示 [Extensions] inline:rosclaw、
	// 上游 changelog/资源诊断（debug/doctor 另开）。
	settingsManager.setQuietStartup(true);
	// P0-8（patch-02）：内建命令前置拦截策略。ROBOT 全禁一批；
	// 所有 profile 都禁上游自更新通道（P0-NA-15：版本所有权属于
	// ROSClaw signed release，harness 不得自行更新）。
	{
		(globalThis as Record<string, unknown>).__rosclawBuiltinPolicy = {
			disabled: new Set(["/update", "/trust", "/share", "/import", "/reload"]),
		};
	}
	// 凭据后端按 profile：developer=加固文件（0600/原子写/fsync），
	// robot=env-only（写即拒）。十审 W1：与 Worker 共用同一构造逻辑。
	const modelRuntime = options.explicitModel?.modelRuntime ?? await createSharedModelRuntime(agentDir, options.profile);
	const systemPrompt = loadSystemPrompt();
	// PR-N5D：扩展工厂先于 session 创建注册——创建后回填引用，
	// 供物化工具激活（setActiveToolsByName）。
	const lateSession: { session?: { setActiveToolsByName(names: string[]): void } } = {};
	// SESSION_WRITER：本 runtime 的 writer owner（调用方传入 = resume
	// 已在 SDK open 前占有的同一 owner；缺省 = 独立新 owner）。
	const ownership = options.ownership ?? new SessionWriterOwnership();
	// SESSION_WRITER switch reservation 集合：session_before_switch 在
	// SDK open 之前占有的目标文件 canonical——open 成功即转为 session
	// 正常 claim（从集合移除）；open 失败只释放该 target reservation，
	// 旧 runtime 与其旧排他 claim 保持存活。
	const switchReservations = new Set<string>();
	// 预创建 seam：新 chat 在任何 SDK 初始 model/thinking append 之前就
	// 占有最终 session 文件；resume 路径对调用方已占有的文件幂等复核。
	// 同 PID 第二个独立 runtime owner 对同一文件在此即被拒绝。
	const initialSessionManager =
		options.sessionManager ??
		SessionManager.create(options.cwd, `${agentDir}/sessions`);
	const initialSessionFile = initialSessionManager.getSessionFile();
	if (!initialSessionFile) throw new Error("SESSION_WRITER_UNPERSISTED_SESSION_FILE");
	ownership.acquire(initialSessionFile);

	let runtime: AgentSessionRuntime;
	try {
	runtime = await createAgentSessionRuntime(
		async ({ cwd, sessionManager, sessionStartEvent }) => {
			// SESSION_WRITER replacement factory pre-create seam：初始
			// session 与 switch/new 的 incoming 文件都在
			// createAgentSessionFromServices 初始 model/thinking 写入
			// 之前完成占有（同 owner 幂等；他人活 owner/UNKNOWN → 抛
			// SESSION_IN_USE，切换失败且不留新 claim）。
			const incomingFile = sessionManager.getSessionFile();
			if (!incomingFile) throw new Error("SESSION_WRITER_UNPERSISTED_SESSION_FILE");
			const incomingCanonical = canonicalSessionPath(incomingFile);
			// session_before_switch 的 reservation 在此移交 open 路径；
			// 目标 open 失败时只释放该 reservation/新 claim，旧 claim 不动。
			const wasReserved = switchReservations.delete(incomingCanonical);
			const heldBefore = ownership.has(incomingFile);
			ownership.acquire(incomingFile);
			try {
			const services = await createAgentSessionServices({
				cwd,
				agentDir,
				settingsManager,
				modelRuntime,
				resourceLoaderOptions: {
					// PNA-9：profile 化资源策略（robot 全禁；developer 仅用户
					// 主题；项目 .pi/AGENTS.md/skills 一律不加载）。
					...(function () {
						// PR-N2（N 总纲 §PR-N2）：四通道拆分——可信只读
						// 上下文恢复（trustFilterContextFiles 按根+预算
						// 过滤）；内置签名 Skill 经 digest 校验后走
						// additionalSkillPaths；任意项目扩展/模板/可执行
						// 资源仍关。
						const policy = resourcePolicy(options.profile);
						const skillsDir = new URL("../../../skills/", import.meta.url).pathname;  // harness/pi/ 深一级
						const bundled = policy.skills === "bundled-signed"
							? verifyBundledSkills(skillsDir)
							: { verified: [], excluded: [], skillPaths: [] };
						return {
							noExtensions: true,
							// NATIVE-BASE-1：任意项目 skill 发现一律关闭；
							// developer 仅经 additionalSkillPaths 注入 digest
							// 校验过的内置签名 Skill（noSkills=true 仍允许显式
							// additionalSkillPaths——官方 PI104 语义）。
							noSkills: true,
							noPromptTemplates: true,
							noThemes: !policy.themes,
							noContextFiles: policy.contextFiles === "off",
							additionalSkillPaths: bundled.skillPaths,
							// NATIVE-BASE-2：原生可信系统基座
							// （native_agent_v2.md，构建期单一事实源）作为
							// ResourceLoader custom base——SDK 在其后追加
							// cwd/可信 AGENTS/签名 Skill，provider 首条 system
							// 恰好含一份 native base；扩展层不再整体替换
							// event.systemPrompt。
							systemPromptOverride: () => systemPrompt,
							agentsFilesOverride: (base: { agentsFiles: Array<{ path: string; content: string }> }) => {
								const allowed = [
									options.taskContext.workspaceRoot,
									options.taskContext.productRoot,
								];
								const filtered = trustFilterContextFiles(base.agentsFiles, {
									allowedRoots: allowed,
									maxTotalBytes: 64 * 1024,
								});
								return { agentsFiles: filtered.kept };
							},
						};
					})(),
					extensionFactories: [
						...(options.explicitModel ? [{ name: "rosclaw-model-selection-notice", factory: ((pi) => {
							pi.on("session_start", (_event, ctx) => {
								if (sessionManager !== initialSessionManager) return;
								const selected = ctx.model;
								ctx.ui.setStatus("model-selection", `${selected?.provider ?? "unknown"}/${selected?.id ?? "unknown"} · effort ${options.explicitThinking} · ${options.overrideSourceId ? `source ${options.overrideSourceId} → restored new` : "new"} session ${sessionManager.getSessionId()}`);
							});
						}) as ExtensionFactory }] : []),
						...(toolBudgetExtension ? [{ name: "rosclaw-tool-budget", factory: toolBudgetExtension }] : []),
						{
							name: "rosclaw-session-writer-ownership",
							factory: createSessionWriterOwnershipExtension(ownership, switchReservations),
						},
						{
							name: "rosclaw",
							factory: createRosclawExtension({
								profile: options.profile,
								version: options.version,
								systemPrompt,
								active,
								coordinator,
								center,
								locale,
								rosclawHome: options.rosclawHome,
								resumed: options.resumed === true,
								sessionManager,
								workspaceStore: options.workspaceStore,
								workspaceAutoBound: options.workspaceAutoBound === true,
								taskContext: options.taskContext,
								lateSession,
							}),
						},
					],
				},
			});
			active.patch({ sessionId: sessionManager.getSessionId() });
			// PR-H1（ADR-0012，总纲 v2）：Native Agent 自己干活——主会话
			// 直接拥有策略包装的工作工具（Workspace Pack：read/grep/find/ls
			// 内建 + bash/write/edit 同名策略覆盖）+ Embodiment Pack。
			// 普通任务不再委派第二个 Pi Session；task_submit/delegate/
			// work_* 退出模型面（root task 权威在 InputController——H2）。
			const budgetWrap = toolBudgetExtension?.wrapTools ?? ((tools: import("@earendil-works/pi-coding-agent").ToolDefinition<any, any>[]) => tools);
			const customTools = budgetWrap(filterModelTools([
				// Only configured read gets a custom SDK definition so its final
				// execute input crosses the same admission seam as write/edit/deliver.
				...(toolBudgetExtension?.hasPath("read") ? [createReadToolDefinition(cwd)] : []),
				// 策略包装的工作工具（GUARDED_MAIN_SESSION——第一层过滤，
				// 强隔离在 PR-H6）。
				...buildWorkspacePackTools({
					root: cwd,
					bashLogPath: `${options.rosclawHome}/logs/main-bash.log`,
					// P0-6：全模式 bash bwrap 强隔离（敏感路径遮蔽——
					// 凭据/控制 token/bridge socket 不经 shell 可达）。
					mode: () => active.current.mode,
					rosclawHome: options.rosclawHome,
					// R1-2a：任务沙箱 scratch 区是 write/edit/bash 的额外
					// 允许根（活跃任务 run dir——Pi 任务代码不写项目树）。
					extraRoots: () => {
						const runDir = active.current.taskRunDir;
						return runDir ? [join(runDir, "scratch")] : [];
					},
					// P0-C：bash/write/edit 执行前的原子 admission
					// （首个 effectful call 建 task——动机=session
					// 最新输入）。
					beforeEffect: async () => {
						// P0-C 本地一致性修复：admission RPC 的拒绝
						// 必须阻断 effect（此前忽略返回值，ok:false
						// 之后仍写盘）；并携带调用方请求上下文——服务
						// 端先校验 writer/context 身份再落账 Task/
						// revision/binding。
						const state = active.current;
						const admission = await center.call("pi.task.ensure_effect", {
							mission_id: state.missionId ?? "",
							session_ref: state.sessionId ?? "",
							backend_native_id: state.sessionId ?? "",
							cwd,
							mode: state.mode ?? "SIMULATION",
							request: {
								schema_version: "rosclaw.pi_tool_request.v1",
								request_id: `ptr_effect_${Date.now()}_${Math.floor(Math.random() * 1e6)}`,
								pi_session_id: state.sessionId ?? "",
								mission_id: state.missionId ?? "",
								context_revision: state.contextRevision,
								body_hash: state.bodyHash ?? "",
								mode: state.mode ?? "SIMULATION",
								tool_name: "rosclaw_workspace_effect",
								arguments: {},
								requested_at: new Date().toISOString(),
								idempotency_key: `idem_effect_${state.sessionId ?? ""}_${Date.now()}_${Math.floor(Math.random() * 1e6)}`,
								actor: { engine: "pi", process_id: process.pid, uid: process.getuid?.() ?? 0 },
							},
						});
						if (admission.ok !== true) {
							const code = String(admission.code ?? "EFFECT_ADMISSION_DENIED");
							const message = String(admission.error ?? "pi.task.ensure_effect rejected");
							throw new Error(`REJECTED [${code}]: ${message}`);
						}
					},
					// 大道至简 R1-2b：SIM 任务沙箱自动执行不弹卡——
					// R1-a 的降级确认卡 apparatus 整体退役（REAL/
					// SHADOW fail closed 不变）。
				}),
				// PR-H3：process 工具（长 Operation——立即返回 operation_id）。
				...buildProcessTools({
					rosclawHome: options.rosclawHome,
					active,
					center,
					// P0-C 本地一致性修复：首个 effectful process 的
					// admission 需要规范 session cwd（任务 workspace 单一
					// 事实源）——否则首个进程回落 private home/tasks。
					sessionCwd: cwd,
				}),
				// PR-H4：Product Pack（登记/收尾/阻塞——验收决定终态）。
				...buildProductPackTools({
					rosclawHome: options.rosclawHome,
					active,
					center,
					workspaceRoot: options.taskContext.workspaceRoot,
				}),
				// PR-H5/N5D：operation 控制（execute 已物化为精确工具）。
				...buildEmbodimentExecTools({
					rosclawHome: options.rosclawHome,
					active,
					center,
				}),
				buildStatusTool(center),
				// PR-N3：生态索引自检（inspect self/robot/capability/asset）。
				buildInspectTool({
					rosclawHome: options.rosclawHome,
					active,
					center,
				}),
				// PR-SIX-3：当前 body 的可信能力面（模型不再猜 ID）。
				buildCapabilitiesTool({
					rosclawHome: options.rosclawHome,
					active,
					center,
				}),
				// 大道至简 R0-2b：rosclaw_task（goal→recipe 确定性
				// 执行）已删除——Pi 用通用物理原语工具自己编排；
				// 固定流程只在 `rosclaw demo`（产品 CLI）。
				// 0901 P0-3：只读任务/交付物面（解释/查看已有结果——
				// 认识确定性链刚做的事，不重跑）。
				...buildReadOnlyTaskTools({
					rosclawHome: options.rosclawHome,
					active,
					center,
				}),
				// PNA-3/PNA-4/PNA-5：bridge 工具需要绑定 session/mission。
				...buildBridgeTools({
					rosclawHome: options.rosclawHome,
					active,
					center,
				}),
				// NA-FIX-4：request_action 必须真实注册（P0-4）。
				buildRequestActionTool({
					rosclawHome: options.rosclawHome,
					active,
					center,
				}),
			]));
			const initialOverride = options.explicitModel && sessionManager === initialSessionManager;
			const result = await createAgentSessionFromServices({
				services,
				sessionManager,
				sessionStartEvent,
				...(initialOverride ? { model: options.explicitModel!.model, thinkingLevel: options.explicitThinking } : {}),
				// PR-N5D：静态 allowlist 无法容纳物化工具名（snapshot
				// 在 mission 绑定后才可知）——不再传 tools allowlist；
				// 模型面由扩展在 session_start/before_agent_start 经
				// setActiveToolsByName 精确激活（MODEL_TOOL_NAMES +
				// 物化名），customTools 仍经 filterModelTools 过滤。
				customTools,
			});
			lateSession.session = result.session;
			// Public PI summary streams do not emit session token events.
			// Observe their real provider events without altering normal turns
			// or automatically aborting the summary/main task.
			result.session.agent.streamFunction = observeCompactionStream(
				result.session.agent.streamFunction, {
					enabled: () => result.session.isCompacting,
					owner: () => ({ session_id: result.session.sessionId, pid: process.pid,
						activity: "session_summary" }),
					record: record => {
						const dir = join(options.rosclawHome, "logs");
						mkdirSync(dir, { recursive: true });
						appendFileSync(join(dir, "compaction-stream.log"),
							`${JSON.stringify({ at: new Date().toISOString(), ...record })}\n`);
					},
				},
			);
			return {
				...result,
				services,
				diagnostics: services.diagnostics,
			};
			} catch (err) {
				// 目标 open 失败：只释放本 factory 新占的 claim 或
				// before_switch 移交的 target reservation——旧 runtime
				// 与其旧排他 claim 保持存活，绝不动他人 owner。
				if (!heldBefore || wasReserved) ownership.release(incomingFile);
				throw err;
			}
		},
		{
			cwd: options.cwd,
			agentDir,
			// 初始 session：--resume/--continue 用打开的既有 session；
			// 否则新建于默认 session 目录（上方已预创建并占有）。
			sessionManager: initialSessionManager,
		},
	);
	} catch (err) {
		// 构造期任何失败（含 replacement factory 抛出）——只释放自己
		// 拥有的 claim，绝不动他人 owner 的锁。
		ownership.releaseAll();
		throw err;
	}
	// SESSION_WRITER failed-switch reservation cleanup：公共 SDK switch
	// 顺序为 emitBeforeSwitch（本扩展在此实际占有 target reservation）
	// → SessionManager.open → assertSessionCwdExists → teardownCurrent
	// → createRuntime（replacement factory）。factory 自己的 catch 只能
	// 清理它已被调用之后的失败；SM.open / cwd 校验 / teardown 等
	// factory 尚未被调用前的失败会把 reservation 泄漏在
	// switchReservations 里。这里在公共 switchSession 上包一层：
	// 任何抛出路径释放仍挂起的 target reservation（token-only
	// release——只释放自己占有的 target，旧 runtime 的旧排他 claim
	// 与他人 owner 绝不动）。teardown 之后的失败同样只释放新 target：
	// SDK 已拆除旧 runtime，绝不伪造"旧 runtime 仍存活"。
	const originalSwitchSession = runtime.switchSession.bind(runtime);
	runtime.switchSession = (async (
		sessionPath: string,
		options?: Parameters<AgentSessionRuntime["switchSession"]>[1],
	) => {
		try {
			return await originalSwitchSession(sessionPath, options);
		} catch (err) {
			for (const canonical of [...switchReservations]) {
				switchReservations.delete(canonical);
				ownership.release(canonical);
			}
			throw err;
		}
	}) as AgentSessionRuntime["switchSession"];
	return { runtime, active, coordinator, leaseManager, ownership };
}
