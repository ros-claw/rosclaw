/** 资源安全策略（PNA-9，规格 §10）。
 *
 * - robot：机器人/SHADOW/REAL——全部项目资源禁止，凭据 env-only；
 * - developer：桌面开发/SIM——允许用户级主题，项目资源仍需显式批准
 *   （默认不加载 .pi/extensions、AGENTS.md、~/.agents/skills）；
 * - worker：headless 原生 Worker——无 ROS 网络/控制 socket/机器人设备，
 *   仅 WorkOrder 指定目录的只读文件工具（本包内不启用任何文件工具，
 *   worker sandbox 的文件能力由 WorkerPack 侧实现并验证）。
 */

export type ResourceProfile = "robot" | "developer" | "worker";

export interface ResourcePolicy {
	noExtensions: boolean;
	noSkills: boolean;
	noPromptTemplates: boolean;
	noThemes: boolean;
	noContextFiles: boolean;
	credentialPolicy: "env-only" | "file-0600";
	allowBash: boolean;
	allowFileTools: boolean;
}

export function resourcePolicy(profile: ResourceProfile): ResourcePolicy {
	switch (profile) {
		case "robot":
			return {
				noExtensions: true,
				noSkills: true,
				noPromptTemplates: true,
				noThemes: true,
				noContextFiles: true,
				credentialPolicy: "env-only",
				allowBash: false,
				allowFileTools: false,
			};
		case "developer":
			return {
				noExtensions: true, // 项目扩展默认不加载（显式批准机制后续批次）
				noSkills: true,
				noPromptTemplates: true,
				noThemes: false, // 用户主题允许
				noContextFiles: true, // AGENTS.md 不注入（具身注入走 envelope）
				credentialPolicy: "file-0600",
				allowBash: false,
				allowFileTools: false,
			};
		case "worker":
			return {
				noExtensions: true,
				noSkills: true,
				noPromptTemplates: true,
				noThemes: true,
				noContextFiles: true,
				credentialPolicy: "env-only",
				allowBash: false,
				allowFileTools: false, // WorkerPack 侧按 WorkOrder 授权
			};
	}
}
