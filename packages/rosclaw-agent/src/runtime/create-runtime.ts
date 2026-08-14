/** Pi runtime 装配（PNA-0）：createAgentSessionRuntime + InteractiveMode。
 *
 * 安全基线（审计 §5）：noExtensions/noSkills/noPromptTemplates/noThemes/
 * noContextFiles 全关——项目 .pi、AGENTS.md、~/.agents/skills 一律不加载；
 * ROSClaw 内联扩展经 extensionFactories 注入（不受 noExtensions 影响）。
 */

import {
	createAgentSessionFromServices,
	createAgentSessionRuntime,
	createAgentSessionServices,
	ModelRuntime,
	SessionManager,
	SettingsManager,
	type AgentSessionRuntime,
} from "@earendil-works/pi-coding-agent";
import { migrateProviders } from "../credentials/migration.js";
import { resourcePolicy } from "../extension/resource-policy.js";
import { createSharedModelRuntime } from "./model-runtime.js";
import { ActiveSessionContext } from "../session/active-context.js";
import { AgentSessionCoordinator } from "../session/coordinator.js";
import { SessionLeaseManager } from "../session/lease-manager.js";
import { ProductStateCenter } from "../session/state-center.js";
import { LocaleManager } from "../i18n/locale.js";
import { defaultOperatorSocket } from "../bridge/operatord-client.js";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { createRosclawExtension } from "../extension/index.js";
import { buildBridgeTools } from "../tools/bridge-tools.js";
import {
	buildDelegateTool,
	buildCheckWorkTool,
	buildCancelWorkTool,
	buildListWorkTool,
	buildUpdateWorkTool,
	buildRetryWorkTool,
	buildWorkDiagnosticTools,
} from "../tools/delegate.js";
import { buildRequestActionTool } from "../tools/request-action.js";
import { buildCapabilitiesTool } from "../tools/capabilities.js";
import { buildComputeTool } from "../tools/compute.js";
import { buildTaskTool } from "../tools/task.js";
import { buildStatusTool } from "../tools/status.js";

export interface RosclawRuntimeOptions {
	cwd: string;
	rosclawHome: string;
	profile: "developer" | "robot";
	version: string;
	missionId?: string;
	/** NA-FIX-2：--resume/--continue 打开的既有 session（否则新建）。 */
	sessionManager?: import("@earendil-works/pi-coding-agent").SessionManager;
	/** WP-P0-3：本次启动是恢复——session_start 展示 Resume Report。 */
	resumed?: boolean;
	/** 十一审 PR-D：Workspace 一等状态。 */
	workspaceStore?: import("../session/workspace.js").WorkspaceStore;
	workspaceAutoBound?: boolean;
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
}

export async function createRosclawRuntime(
	options: RosclawRuntimeOptions,
): Promise<RosclawRuntime> {
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
	// PNA-7（规格 §22.3）：legacy config.yaml → Pi settings 一次性迁移
	// （已有 defaultProvider/defaultModel 则不触碰）。
	migrateProviders(options.rosclawHome);
	const settingsManager = SettingsManager.create(options.cwd, agentDir);
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
	const modelRuntime = await createSharedModelRuntime(agentDir, options.profile);
	const systemPrompt = loadSystemPrompt();

	const runtime = await createAgentSessionRuntime(
		async ({ cwd, sessionManager, sessionStartEvent }) => {
			const services = await createAgentSessionServices({
				cwd,
				agentDir,
				settingsManager,
				modelRuntime,
				resourceLoaderOptions: {
					// PNA-9：profile 化资源策略（robot 全禁；developer 仅用户
					// 主题；项目 .pi/AGENTS.md/skills 一律不加载）。
					...(function () {
						const policy = resourcePolicy(options.profile);
						return {
							noExtensions: policy.noExtensions,
							noSkills: policy.noSkills,
							noPromptTemplates: policy.noPromptTemplates,
							noThemes: policy.noThemes,
							noContextFiles: policy.noContextFiles,
						};
					})(),
					extensionFactories: [
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
							}),
						},
					],
				},
			});
			active.patch({ sessionId: sessionManager.getSessionId() });
			// 具身主 Agent 不需要 coding 工具；ROSClaw 工具走 customTools。
			// 注意：noTools:"all" 会把 allowedToolNames 置空、连 customTools 一起
			// 过滤掉（模型将看不到任何工具）——必须用显式 allowlist。
			const customTools = [
				buildStatusTool(center),
				// PR-SIX-3：当前 body 的可信能力面（模型不再猜 ID）。
				buildCapabilitiesTool({
					rosclawHome: options.rosclawHome,
					active,
					center,
				}),
				// PR-SEVEN-2：COMPUTE 能力免审批调用。
				buildComputeTool({
					rosclawHome: options.rosclawHome,
					active,
					center,
				}),
				// PR-EIGHT-5：任务级入口（确定性编译器——模型只交
				// TaskSpec，不搬载荷、不逐点控制）。
				buildTaskTool({
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
				buildDelegateTool({
					rosclawHome: options.rosclawHome,
					active,
					center,
					workspace: () => options.workspaceStore?.current ?? null,
				}),
				// 十审 W0：异步 WorkOrder 协议（按精确 ID 查询/取消）。
				buildCheckWorkTool({
					rosclawHome: options.rosclawHome,
					active,
					center,
				}),
				buildCancelWorkTool({
					rosclawHome: options.rosclawHome,
					active,
					center,
				}),
				// 十审 W2：list/update 补齐五工具协议。
				buildListWorkTool({
					rosclawHome: options.rosclawHome,
					active,
					center,
				}),
				buildUpdateWorkTool({
					rosclawHome: options.rosclawHome,
					active,
					center,
				}),
				// 十审 W4：终态单 retry（lineage 保留）。
				buildRetryWorkTool({
					rosclawHome: options.rosclawHome,
					active,
					center,
				}),
				// 十三审 PR-13.5：只读 Worker 诊断（模型可自查失败原因）。
				...buildWorkDiagnosticTools({
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
			];
			const result = await createAgentSessionFromServices({
				services,
				sessionManager,
				sessionStartEvent,
				tools: customTools.map((tool) => tool.name),
				customTools,
			});
			return {
				...result,
				services,
				diagnostics: services.diagnostics,
			};
		},
		{
			cwd: options.cwd,
			agentDir,
			// 初始 session：--resume/--continue 用打开的既有 session；
			// 否则新建于默认 session 目录。
			sessionManager:
				options.sessionManager ??
				SessionManager.create(options.cwd, `${agentDir}/sessions`),
		},
	);
	return { runtime, active, coordinator, leaseManager };
}
