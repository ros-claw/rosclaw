/** Workspace 一等状态（十一审 PR-D，总纲 §P0-5）。
 *
 * - 状态文件：$ROSCLAW_HOME/agent/workspace.json（current + recent）；
 * - 启动时：--workspace 显式指定 > cwd 所在 git repo 自动绑定 > 无绑定
 *   （header 显示 Project —）；
 * - WorkOrder 的 workspace 默认取绑定值——用户不再需要在自然语言里
 *   反复给路径；worker 不再去"猜" host path。
 */

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { basename, dirname, isAbsolute, resolve } from "node:path";

export interface WorkspaceState {
	current: string | null;
	recent: string[];
}

/** 向上找 git repo root（.git 目录或 worktree 的 .git 文件）。 */
export function gitRootOf(start: string): string | null {
	let dir = resolve(start);
	for (let depth = 0; depth < 32; depth += 1) {
		if (existsSync(`${dir}/.git`)) return dir;
		const parent = dirname(dir);
		if (parent === dir) return null;
		dir = parent;
	}
	return null;
}

export class WorkspaceStore {
	private readonly path: string;
	private state: WorkspaceState;

	constructor(rosclawHome: string) {
		this.path = `${rosclawHome}/agent/workspace.json`;
		this.state = this.load();
	}

	private load(): WorkspaceState {
		try {
			if (!existsSync(this.path)) return { current: null, recent: [] };
			const data = JSON.parse(readFileSync(this.path, "utf-8")) as WorkspaceState;
			return { current: data.current ?? null, recent: data.recent ?? [] };
		} catch {
			return { current: null, recent: [] };
		}
	}

	private persist(): void {
		try {
			mkdirSync(dirname(this.path), { recursive: true });
			const tmp = `${this.path}.tmp`;
			writeFileSync(tmp, JSON.stringify(this.state), { encoding: "utf-8", mode: 0o600 });
			renameSync(tmp, this.path);
		} catch {
			// 状态持久化失败不阻塞会话
		}
	}

	get current(): string | null {
		return this.state.current;
	}

	get recent(): string[] {
		return [...this.state.recent];
	}

	/** 绑定（/workspace use 或启动自动绑定）。返回规范化的绝对路径。 */
	bind(path: string): string {
		const abs = isAbsolute(path) ? resolve(path) : resolve(process.cwd(), path);
		if (!existsSync(abs)) {
			throw new Error(`路径不存在：${abs}`);
		}
		// git 目录自动归一到 repo root。
		const root = gitRootOf(abs) ?? abs;
		this.state.current = root;
		this.state.recent = [root, ...this.state.recent.filter((r) => r !== root)].slice(0, 10);
		this.persist();
		return root;
	}

	unbind(): void {
		this.state.current = null;
		this.persist();
	}

	displayName(): string {
		return this.state.current ? basename(this.state.current) : "—";
	}
}

/** 启动解析：显式 --workspace > cwd git 自动绑定 > 原状态 > null。 */
export function resolveStartupWorkspace(
	store: WorkspaceStore,
	explicit: string | undefined,
	cwd: string,
): { bound: string | null; auto: boolean } {
	if (explicit) {
		return { bound: store.bind(explicit), auto: false };
	}
	const repo = gitRootOf(cwd);
	if (repo) {
		return { bound: store.bind(repo), auto: true };
	}
	return { bound: store.current, auto: false };
}
