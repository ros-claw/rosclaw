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
import { credentialStoreFor } from "../credentials/store.js";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { createRosclawExtension } from "../extension/index.js";
import { buildBridgeTools } from "../tools/bridge-tools.js";
import { buildDelegateTool } from "../tools/delegate.js";
import { buildStatusTool } from "../tools/status.js";

export interface RosclawRuntimeOptions {
	cwd: string;
	rosclawHome: string;
	profile: "developer" | "robot";
	version: string;
	missionId?: string;
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

export async function createRosclawRuntime(
	options: RosclawRuntimeOptions,
): Promise<AgentSessionRuntime> {
	const agentDir = `${options.rosclawHome}/agent`;
	// PNA-7（规格 §22.3）：legacy config.yaml → Pi settings 一次性迁移
	// （已有 defaultProvider/defaultModel 则不触碰）。
	migrateProviders(options.rosclawHome);
	const settingsManager = SettingsManager.create(options.cwd, agentDir);
	// 凭据后端按 profile：developer=加固文件（0600/原子写/fsync），
	// robot=env-only（写即拒）。
	const modelRuntime = await ModelRuntime.create({
		credentials: credentialStoreFor(options.profile, agentDir) as never,
		authPath: `${agentDir}/auth.json`,
		modelsPath: null,
	});
	const systemPrompt = loadSystemPrompt();

	const runtime = await createAgentSessionRuntime(
		async ({ cwd, sessionManager, sessionStartEvent }) => {
			const services = await createAgentSessionServices({
				cwd,
				agentDir,
				settingsManager,
				modelRuntime,
				resourceLoaderOptions: {
					// 项目资源全关（审计 §5）；ROSClaw 扩展走内联工厂。
					noExtensions: true,
					noSkills: true,
					noPromptTemplates: true,
					noThemes: true,
					noContextFiles: true,
					extensionFactories: [
						{
							name: "rosclaw",
							factory: createRosclawExtension({
								profile: options.profile,
								version: options.version,
								systemPrompt,
								missionId: options.missionId,
								piSessionId: sessionManager.getSessionId(),
								rosclawHome: options.rosclawHome,
							}),
						},
					],
				},
			});
			const result = await createAgentSessionFromServices({
				services,
				sessionManager,
				sessionStartEvent,
				// 具身主 Agent 不需要 coding 工具；ROSClaw 工具走 customTools。
				noTools: "all",
				customTools: [
					buildStatusTool(options.rosclawHome),
					// PNA-3/PNA-4：bridge 工具需要绑定 session/mission 才有意义。
					...(options.missionId
						? [
								...buildBridgeTools({
									rosclawHome: options.rosclawHome,
									piSessionId: sessionManager.getSessionId(),
									missionId: options.missionId,
								}),
								buildDelegateTool({
									rosclawHome: options.rosclawHome,
									piSessionId: sessionManager.getSessionId(),
									missionId: options.missionId,
								}),
							]
						: []),
				],
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
			// 初始 session：新建于默认 session 目录（~/.rosclaw/agent/sessions）；
			// /new /resume /fork 由 InteractiveMode 经 runtime 切换。
			sessionManager: SessionManager.create(
				options.cwd,
				`${agentDir}/sessions`,
			),
		},
	);
	return runtime;
}
