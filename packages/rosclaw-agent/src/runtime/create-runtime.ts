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
	SessionManager,
	SettingsManager,
	type AgentSessionRuntime,
} from "@earendil-works/pi-coding-agent";
import { createRosclawExtension } from "../extension/index.js";
import { buildStatusTool } from "../tools/status.js";

export interface RosclawRuntimeOptions {
	cwd: string;
	rosclawHome: string;
	profile: "developer" | "robot";
	version: string;
}

export async function createRosclawRuntime(
	options: RosclawRuntimeOptions,
): Promise<AgentSessionRuntime> {
	const agentDir = `${options.rosclawHome}/agent`;
	const settingsManager = SettingsManager.create(options.cwd, agentDir);

	const runtime = await createAgentSessionRuntime(
		async ({ cwd, sessionManager, sessionStartEvent }) => {
			const services = await createAgentSessionServices({
				cwd,
				agentDir,
				settingsManager,
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
				customTools: [buildStatusTool(options.rosclawHome)],
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
