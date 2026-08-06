/** Command registry (批次 C，§5.1)：命令先解析、按 owner 路由，永不发给模型。 */

import type { CommandSpec } from "../client/http.js";

export type CommandRoute =
	| { kind: "local"; name: string; args: string }
	| { kind: "remote"; spec: CommandSpec; args: string }
	| { kind: "unknown"; name: string; args: string }
	| { kind: "not_a_command" };

export interface LocalCommand {
	name: string;
	aliases: string[];
	description: string;
	argumentHint: string;
}

/** TUI 本地命令（LOCAL_UI owner）—— 不需要服务端。 */
export const LOCAL_COMMANDS: LocalCommand[] = [
	{ name: "help", aliases: ["h", "?"], description: "显示全部命令", argumentHint: "[command]" },
	{ name: "hotkeys", aliases: [], description: "显示快捷键", argumentHint: "" },
	{ name: "quit", aliases: ["q", "exit"], description: "退出 TUI（不停止已派发动作）", argumentHint: "" },
	{ name: "clear-screen", aliases: ["cls"], description: "清屏（不清 Mission）", argumentHint: "" },
	{ name: "approve", aliases: [], description: "批准授权请求", argumentHint: "<request_id>" },
	{ name: "deny", aliases: [], description: "拒绝授权请求", argumentHint: "<request_id>" },
	{ name: "missions", aliases: [], description: "列出 Mission", argumentHint: "" },
	{ name: "resume", aliases: [], description: "切换到其他 Mission（选择器）", argumentHint: "[mission_id]" },
	{ name: "estop", aliases: [], description: "紧急停止（直达 rosclawd，不经过模型）", argumentHint: "" },
];

export class CommandRegistry {
	private remote = new Map<string, CommandSpec>();

	loadRemote(specs: CommandSpec[]): void {
		this.remote.clear();
		for (const spec of specs) this.remote.set(spec.name, spec);
	}

	remoteSpecs(): CommandSpec[] {
		return [...this.remote.values()];
	}

	parse(input: string): CommandRoute {
		if (!input.startsWith("/")) return { kind: "not_a_command" };
		const [head, ...rest] = input.slice(1).split(/\s+/);
		const args = rest.join(" ");
		const name = head.toLowerCase();
		const local = LOCAL_COMMANDS.find((c) => c.name === name || c.aliases.includes(name));
		if (local) return { kind: "local", name: local.name, args };
		const remote = this.remote.get(name);
		if (remote) return { kind: "remote", spec: remote, args };
		return { kind: "unknown", name, args };
	}

	all(): Array<{ name: string; description: string; owner: string; disabled?: string; hint: string }> {
		const rows: Array<{
			name: string;
			description: string;
			owner: string;
			disabled?: string;
			hint: string;
		}> = LOCAL_COMMANDS.map((c) => ({
			name: c.name,
			description: c.description,
			owner: "LOCAL_UI",
			hint: c.argumentHint,
		}));
		for (const spec of this.remote.values()) {
			rows.push({
				name: spec.name,
				description: spec.description,
				owner: spec.owner,
				disabled: spec.disabled_reason || undefined,
				hint: spec.argument_hint,
			});
		}
		return rows.sort((a, b) => a.name.localeCompare(b.name));
	}
}
