/** 资源安全策略（PR-N2 重构，N 总纲 §PR-N2）——四通道拆分。
 *
 * 旧五连布尔（noExtensions/noSkills/noPromptTemplates/noThemes/
 * noContextFiles）把"可信只读项目上下文"和"任意可执行扩展"一刀切
 * 全关——Native Agent 因此失去成熟 Harness 的项目认知入口。
 *
 * 拆分语义：
 * - contextFiles："off" | "trusted-readonly"——AGENTS.md/CLAUDE.md
 *   等只读上下文在信任根内允许（来源/路径/大小预算见
 *   trustFilterContextFiles）；
 * - skills："off" | "bundled-signed"——任意项目 Skill 不加载；仅
 *   ROSClaw 内置且 digest 校验通过的 Skill（bundled-skills.ts）；
 * - promptTemplates："off"——项目提示模板是文本注入面，默认关；
 * - extensions："off"——任意项目可执行扩展默认禁（后续显式批准
 *   机制另行开放）；
 * - executables："off"——项目 hooks/脚本类可执行资源默认禁。
 */

export type ResourceProfile = "robot" | "developer" | "worker";

export type ChannelPolicy = "off" | "trusted-readonly" | "bundled-signed";

export interface ResourcePolicy {
	contextFiles: ChannelPolicy;
	skills: ChannelPolicy;
	promptTemplates: "off";
	extensions: "off";
	executables: "off";
	/** 用户主题（纯展示，非项目面）。 */
	themes: boolean;
	credentialPolicy: "env-only" | "file-0600";
	allowBash: boolean;
	allowFileTools: boolean;
}

export function resourcePolicy(profile: ResourceProfile): ResourcePolicy {
	switch (profile) {
		case "robot":
			return {
				contextFiles: "off",
				skills: "off",
				promptTemplates: "off",
				extensions: "off",
				executables: "off",
				themes: false,
				credentialPolicy: "env-only",
				allowBash: false,
				allowFileTools: false,
			};
		case "developer":
			return {
				contextFiles: "trusted-readonly",
				skills: "bundled-signed",
				promptTemplates: "off",
				extensions: "off",
				executables: "off",
				themes: true,
				credentialPolicy: "file-0600",
				allowBash: false,
				allowFileTools: false,
			};
		case "worker":
			return {
				contextFiles: "off",
				skills: "off",
				promptTemplates: "off",
				extensions: "off",
				executables: "off",
				themes: false,
				credentialPolicy: "env-only",
				allowBash: false,
				allowFileTools: false,
			};
	}
}

/** 上下文文件信任过滤（来源+路径+大小预算；超预算/出根剔除并诊断）。 */
export function trustFilterContextFiles(
	files: Array<{ path: string; content: string }>,
	options: { allowedRoots: string[]; maxTotalBytes: number },
): { kept: Array<{ path: string; content: string }>; diagnostics: string[] } {
	const kept: Array<{ path: string; content: string }> = [];
	const diagnostics: string[] = [];
	let total = 0;
	for (const file of files) {
		const inRoot = options.allowedRoots.some(
			(root) => file.path === root || file.path.startsWith(root + "/"),
		);
		if (!inRoot) {
			diagnostics.push(`剔除（出信任根）：${file.path}`);
			continue;
		}
		const bytes = Buffer.byteLength(file.content, "utf-8");
		if (total + bytes > options.maxTotalBytes) {
			diagnostics.push(
				`剔除（超预算 ${options.maxTotalBytes}B）：${file.path}（${bytes}B）`,
			);
			continue;
		}
		total += bytes;
		kept.push(file);
	}
	return { kept, diagnostics };
}
