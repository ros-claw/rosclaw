/** Harness Runtime Descriptor + readiness 探测（十五审 PR-RF-2/RF-3，
 *  总纲 §7.1）。注册表是单一真相源——不存在的手工配置不假装 ready。 */

import { spawnSync } from "node:child_process";

export interface HarnessDescriptorInput {
	id: string;
	/** 启动命令（PATH 查找）。 */
	command: string;
	args?: string[];
	capabilities?: string[];
}

export interface HarnessDescriptor {
	id: string;
	command: string;
	args: string[];
	capabilities: string[];
	supports: {
		streaming: boolean;
		resume: boolean;
		steer: boolean;
	};
	/** 建议-0816 P0-8：隔离等级诚实命名——进程/环境层防护是
	 *  guarded_process，不是 sandboxed/container（无 bwrap/unshare）。 */
	isolation: {
		level: "guarded_process" | "sandboxed" | "container";
	};
	readiness: {
		runtime: "ready" | "not_installed";
	};
	ready: boolean;
}

/** readiness preflight（总纲 §8.1）：二进制存在才可注册 ready；
 *  缺失 → 诚实 not_installed（启动前失败，绝不创建执行再失败）。 */
export function probeHarness(input: HarnessDescriptorInput): HarnessDescriptor {
	const found = spawnSync("which", [input.command], { encoding: "utf-8" });
	const installed = found.status === 0 && Boolean(found.stdout.trim());
	return {
		id: input.id,
		command: input.command,
		args: input.args ?? [],
		capabilities: input.capabilities ?? [],
		supports: { streaming: true, resume: true, steer: true },
		// P0-8：当前只有进程/环境层防护（独立 HOME/TMP/env/白名单）
		// ——guarded_process，绝不宣称 sandboxed。
		isolation: { level: "guarded_process" },
		readiness: { runtime: installed ? "ready" : "not_installed" },
		ready: installed,
	};
}

/** 默认 Harness 目录（本机探测）：Claude Code/Pi/Codex（经 acp 入口）。 */
export function defaultHarnesses(): HarnessDescriptor[] {
	return [
		probeHarness({
			id: "claude-local",
			command: "claude-code-acp",
			capabilities: ["code.implement", "code.debug", "repo.analyze", "research"],
		}),
		probeHarness({
			id: "pi-acp",
			command: "pi-acp",
			capabilities: ["code.implement", "code.debug", "repo.analyze"],
		}),
		probeHarness({
			id: "codex-local",
			command: "codex",
			args: ["app-server"],
			capabilities: ["code.implement", "code.debug", "repo.analyze"],
		}),
	];
}
