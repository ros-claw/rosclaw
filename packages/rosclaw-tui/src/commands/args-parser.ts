/** Schema-driven command args parser（审计 P0-03）：TUI/ACP/Web 共享。 */

export interface PositionalSpec {
	name: string;
	type: "string" | "enum" | "rest";
	required?: boolean;
	enum?: string[];
}

export interface ArgsSchema {
	positional?: PositionalSpec[];
	flags?: Record<string, { type: string }>;
	interaction?: string;
	interaction_source?: string;
}

export interface ParseOutcome {
	ok: boolean;
	args: Record<string, unknown>;
	error?: string;
}

export function parseArgs(schema: ArgsSchema | undefined, input: string): ParseOutcome {
	const spec = schema ?? {};
	const positionals = spec.positional ?? [];
	const flags = spec.flags ?? {};
	const args: Record<string, unknown> = {};

	let rest = input.trim();
	// flags：前缀匹配 "--name" 或裸 "dry-run" 风格（与 /compact dry-run 一致）。
	for (const [flagName] of Object.entries(flags)) {
		for (const form of [`--${flagName}`, flagName]) {
			if (rest === form || rest.startsWith(form + " ")) {
				args[flagName.replace(/-/g, "_")] = true;
				rest = rest.slice(form.length).trim();
				break;
			}
		}
	}

	for (let i = 0; i < positionals.length; i += 1) {
		const pos = positionals[i];
		const isLast = i === positionals.length - 1;
		if (pos.type === "rest") {
			if (rest) args[pos.name] = rest;
			rest = "";
		} else {
			const [head, ...tail] = rest.split(/\s+/).filter(Boolean);
			if (head) args[pos.name] = head;
			rest = tail.join(" ");
		}
		if (args[pos.name] === undefined && pos.required) {
			return { ok: false, args: {}, error: `缺少参数 <${pos.name}>` };
		}
		if (pos.type === "enum" && args[pos.name] !== undefined) {
			const allowed = pos.enum ?? [];
			if (!allowed.includes(String(args[pos.name]))) {
				return {
					ok: false,
					args: {},
					error: `参数 ${pos.name} 必须是 ${allowed.join("|")} 之一`,
				};
			}
		}
		void isLast;
	}
	if (rest && positionals.length === 0) {
		return { ok: false, args: {}, error: "此命令不接受参数" };
	}
	return { ok: true, args };
}
