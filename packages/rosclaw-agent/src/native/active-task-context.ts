/** ActiveTaskContext（PR-N1，N 总纲 §5）——唯一工作区事实源。
 *
 * 在 session 创建前解析并冻结。runtime、SessionManager、Workspace
 * Pack、Process/Product 工具、bridge、artifact registry、verifier、
 * TUI header 全部从这里取路径——任务链禁止再读 process.cwd()。
 *
 * 启动解析顺序（§5.4）：
 * 1. 显式 --workspace；
 * 2. cwd 所在 git root；
 * 3. 有效持久化 workspace（provenance 诚实标注 restored）；
 * 4. ~/.rosclaw/workspaces/default/（真实创建——UI 不得谎称项目）。
 */

import { mkdirSync } from "node:fs";
import { resolve } from "node:path";

import { gitRootOf, WorkspaceStore } from "../session/workspace.js";

export type WorkspaceSource = "explicit" | "git" | "restored" | "default";

export interface ActiveTaskContext {
	/** session 创建后由 runtime 回填前的占位——冻结时可为空。 */
	readonly sessionId: string;
	readonly missionId?: string;
	readonly taskId?: string;
	readonly revision: number;
	/** ROSClaw 安装包/源码根（只读检索）。 */
	readonly productRoot: string;
	/** 用户当前项目（git root），可为空。 */
	readonly projectRoot?: string;
	/** 本次任务实际工作目录（Bash/write/read 的根）。 */
	readonly workspaceRoot: string;
	/** 交付物目录（当前 = workspaceRoot——产物即项目文件）。 */
	readonly artifactRoot: string;
	readonly rosclawHome: string;
	readonly mode: "SIMULATION" | "SHADOW" | "REAL";
	readonly robotId?: string;
	readonly simulatorId?: string;
	readonly workspaceSource: WorkspaceSource;
}

/** productRoot 探测：本包安装根（dist 上溯到包根再上一级=产品根）。 */
function detectProductRoot(): string {
	// dist/native/active-task-context.js → 包根 → 产品上溯。
	return resolve(import.meta.dirname, "..", "..", "..");
}

export function resolveTaskContext(options: {
	rosclawHome: string;
	cwd: string;
	mode: "SIMULATION" | "SHADOW" | "REAL";
	explicitWorkspace?: string;
	robotId?: string;
	simulatorId?: string;
}): ActiveTaskContext {
	let workspaceRoot: string;
	let workspaceSource: WorkspaceSource;
	let projectRoot: string | undefined;

	if (options.explicitWorkspace) {
		workspaceRoot = resolve(options.explicitWorkspace);
		workspaceSource = "explicit";
		projectRoot = gitRootOf(workspaceRoot) ?? workspaceRoot;
	} else {
		const repo = gitRootOf(options.cwd);
		if (repo) {
			workspaceRoot = repo;
			workspaceSource = "git";
			projectRoot = repo;
		} else {
			const persisted = new WorkspaceStore(options.rosclawHome).current;
			if (persisted) {
				workspaceRoot = resolve(persisted);
				workspaceSource = "restored";
				projectRoot = gitRootOf(workspaceRoot) ?? workspaceRoot;
			} else {
				workspaceRoot = resolve(options.rosclawHome, "workspaces", "default");
				workspaceSource = "default";
				mkdirSync(workspaceRoot, { recursive: true });
			}
		}
	}

	return Object.freeze<ActiveTaskContext>({
		sessionId: "",
		revision: 0,
		productRoot: detectProductRoot(),
		...(projectRoot ? { projectRoot } : {}),
		workspaceRoot,
		artifactRoot: workspaceRoot,
		rosclawHome: resolve(options.rosclawHome),
		mode: options.mode,
		...(options.robotId ? { robotId: options.robotId } : {}),
		...(options.simulatorId ? { simulatorId: options.simulatorId } : {}),
		workspaceSource,
	});
}
