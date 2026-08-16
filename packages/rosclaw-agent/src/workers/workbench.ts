/** Developer Workbench 约束工具（十审 W3，审计 §9）。
 *
 * 本机 user namespace 被禁（unshare -rm / bwrap 不可用）——隔离在工具层
 * 实现，不假装有 OS 沙箱：
 *
 * 1. 路径约束：所有文件工具（read/grep/find/ls/write/edit）的目标必须
 *    realpath 在 workspace root 内——绝对路径、`..`、symlink 逃逸一律
 *    拒绝（写新文件时对最近存在的祖先做 realpath）。
 * 2. bash：argv/allow-policy（白名单二进制 + 危险参数模式拒绝 +
 *    逐参数路径检查）；网络命令（curl/wget/ssh/nc）、特权命令
 *    （sudo/su/chown/docker）、/dev 访问一律拒绝；
 *    env 白名单（**不含任何 API key**，HOME=workspace）；
 *    每条命令带超时/输出上限，全程日志写 artifacts/bash-log.txt。
 * 3. secret 隔离：env 不含凭据 + 路径约束挡住 ~/.rosclaw/agent/auth.json
 *    + 结果侧 verifier secret scan 兜底。
 *
 * 工具名与 Pi 内建同名（read/grep/find/ls/write/edit/bash）——custom
 * 工具在 AgentSession 注册表中覆盖同名内建（dist/core/agent-session.js
 * _refreshToolRegistry 后写覆盖先写，实证）。
 */

import { appendFileSync, existsSync, mkdirSync, realpathSync, statSync, writeFileSync, readFileSync, readdirSync } from "node:fs";
import { spawn } from "node:child_process";
import { dirname, isAbsolute, resolve, sep } from "node:path";

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ToolDefinition } from "@earendil-works/pi-coding-agent";

export interface WorkbenchOptions {
	/** workspace 根（必须是已存在的目录）。 */
	root: string;
	/** 十一审 PR-E：Worker 主动提问（WAITING_INPUT）——emit 等待事件 +
	 *  stdin 等回答。 */
	askUser?: (question: string) => Promise<string>;
	/** bash 日志（artifacts/bash-log.txt）。 */
	bashLogPath: string;
	/** 长命令保活：bash 运行期间周期性回调（喂 idle watchdog）。 */
	emitProgress?: (message: string) => void;
	/** 十四审 PR-14.3：结构化记录（file_change 等）→ files channel。 */
	emitRecord?: (kind: string, payload: Record<string, unknown>) => void;
	/** 单命令默认超时（ms）。 */
	defaultTimeoutMs?: number;
}

/** argv 白名单（首参数必须精确命中；不含任何网络/特权工具）。 */
const ALLOWED_BINARIES = new Set([
	"ls", "cat", "head", "tail", "grep", "find", "wc", "echo", "printf", "pwd",
	"mkdir", "touch", "cp", "mv", "rm", "diff", "sed", "awk", "sort", "uniq",
	"tr", "cut", "tee", "git", "python", "python3", "pytest",
	"node", "npm", "tsc", "make", "jq", "sha256sum", "file", "stat",
	"chmod", "basename", "dirname", "realpath", "true", "false", "test", "[",
	"sleep",
]);

/** 自审修复（argv 逃逸面）：白名单二进制的危险子命令/参数。
 *  xargs/npx 直接出列（可执行任意二进制/远程包）；git 禁网络子命令
 *  与任意配置注入；find 禁 -exec；npm 禁 install/publish；awk/sed
 *  禁命令执行原语。python -c 的网络面在工具层无法彻底封堵——
 *  如实记录为 tool_guard 残留风险（§8.2）。 */
const DENIED_SUBCOMMANDS: Record<string, (args: string[]) => string | null> = {
	git: (args) => {
		const sub = args.find((a) => !a.startsWith("-")) ?? "";
		if (["push", "fetch", "pull", "clone", "remote", "submodule", "send-email"].includes(sub)) {
			return `git ${sub} 涉及网络/远端——不允许`;
		}
		if (args.includes("-c") || args.some((a) => a.startsWith("--exec-path") || a.startsWith("-c "))) {
			return "git -c/--exec-path 可注入任意命令——不允许";
		}
		return null;
	},
	find: (args) =>
		args.some((a) => ["-exec", "-execdir", "-delete", "-ok"].includes(a))
			? "find -exec/-delete 不允许"
			: null,
	npm: (args) => {
		const sub = args.find((a) => !a.startsWith("-")) ?? "";
		return ["install", "i", "add", "publish", "link", "exec", "x"].includes(sub)
			? `npm ${sub} 涉及网络/任意执行——不允许`
			: null;
	},
	awk: (args) => (args.some((a) => /system\s*\(|getline.*\|/.test(a)) ? "awk system/getline-pipe 不允许" : null),
	sed: (args) => (args.some((a) => /^-[^-]*e|e\s*\/.+\/e?;/.test(a)) ? "sed -e 执行原语不允许" : null),
	xargs: () => "xargs 可执行任意二进制——不允许",
	npx: () => "npx 可拉取远程包——不允许",
};

/** 任何参数命中这些模式即拒绝（网络/特权/设备/宿主机敏感面）。 */
const DENIED_ARG_PATTERNS = [
	/^\/dev\//,
	/^\/proc\//,
	/^\/sys\//,
	/(^|\/)\.rosclaw\/agent\/auth\.json/,
	/^~\/\.rosclaw/,
	/DANGEROUSLY/,
];

const DENIED_BINARIES = new Set([
	"sudo", "su", "doas", "curl", "wget", "ssh", "scp", "sftp", "nc", "ncat",
	"netcat", "docker", "podman", "kubectl", "dd", "mount", "umount", "mkfs",
	"chown", "kill", "killall", "pkill", "systemctl", "service", "crontab",
	"pip", "pip3", "apt", "apt-get", "dpkg", "snap", "npm-publish",
]);

const MAX_OUTPUT_BYTES = 64 * 1024;

/** 路径必须解析在 root 内（symlink 感知）。返回解析后的绝对路径。 */
export function resolveInRoot(root: string, target: string): string {
	const base = resolve(root);
	const candidate = isAbsolute(target) ? resolve(target) : resolve(base, target);
	if (candidate !== base && !candidate.startsWith(base + sep)) {
		throw new Error(`path escapes workspace: ${target}`);
	}
	// symlink 检查：最近存在的祖先 realpath 必须在 root 内。
	let probe = candidate;
	while (!existsSync(probe)) {
		const parent = dirname(probe);
		if (parent === probe) break;
		probe = parent;
	}
	if (existsSync(probe)) {
		const real = realpathSync(probe);
		const realBase = realpathSync(base);
		if (real !== realBase && !real.startsWith(realBase + sep)) {
			throw new Error(`symlink escapes workspace: ${target}`);
		}
	}
	return candidate;
}

function argPathCheck(root: string, arg: string): void {
	for (const pattern of DENIED_ARG_PATTERNS) {
		if (pattern.test(arg)) throw new Error(`forbidden argument: ${arg}`);
	}
	// 以 / 开头的参数视作路径检查；相对路径含 .. 也检查。
	if (arg.startsWith("/") || arg.includes("..")) {
		// 允许以 - 开头的 flag。
		if (arg.startsWith("-")) return;
		resolveInRoot(root, arg);
	}
}

/** 建议-0816 P0-1：bash 默认无硬超时（长编译/仿真合法——stall 由
 *  supervisor 显示，用户随时可 cancel）。只有显式 timeout_sec
 *  （调用方/模型自选）或运营配置 defaultTimeoutMs 才装定时器；
 *  否则返回 null（无 SIGKILL 定时器）。 */
export function _bashTimeoutMs(
	params: { timeout_sec?: number },
	options: { defaultTimeoutMs?: number },
): number | null {
	if ((params.timeout_sec ?? 0) > 0) return Number(params.timeout_sec) * 1000;
	return options.defaultTimeoutMs ?? null;
}

export function buildWorkbenchTools(options: WorkbenchOptions): ToolDefinition[] {
	const root = resolve(options.root);
	const log = (line: string) => {
		try {
			mkdirSync(dirname(options.bashLogPath), { recursive: true });
			appendFileSync(options.bashLogPath, `${line}\n`, "utf-8");
		} catch {
			// 日志失败不阻塞工具结果（verifier 会看到缺口）。
		}
	};

	const denied = (err: unknown) => ({
		content: [{ type: "text" as const, text: `DENIED: ${(err as Error).message}` }],
		details: { error: "denied", reason: (err as Error).message } as Record<string, unknown>,
		isError: true,
	});

	const readTool = defineTool({
		name: "read",
		label: "read (workspace)",
		description: "Read a file inside the workspace (path-checked).",
		parameters: Type.Object({
			path: Type.String(),
			offset: Type.Optional(Type.Number()),
			limit: Type.Optional(Type.Number()),
		}),
		async execute(_id, params) {
			let p: string;
			try {
				p = resolveInRoot(root, String(params.path));
			} catch (err) {
				return denied(err);
			}
			const text = readFileSync(p, "utf-8");
			const lines = text.split("\n");
			const start = Math.max(0, (params.offset ?? 1) - 1);
			// 自审修复：默认上限 2000 行——整本大日志不进模型上下文。
			const effectiveLimit = params.limit ?? 2000;
			const slice = lines.slice(start, start + effectiveLimit);
			return {
				content: [{ type: "text" as const, text: slice.join("\n") }],
				details: { path: p, total_lines: lines.length },
			};
		},
	});

	const writeTool = defineTool({
		name: "write",
		label: "write (workspace)",
		description: "Create/overwrite a file inside the workspace (path-checked).",
		parameters: Type.Object({
			path: Type.String(),
			content: Type.String(),
		}),
		async execute(_id, params) {
			let p: string;
			try {
				p = resolveInRoot(root, String(params.path));
			} catch (err) {
				return denied(err);
			}
			mkdirSync(dirname(p), { recursive: true });
			writeFileSync(p, String(params.content), "utf-8");
			log(`write ${p} (${(params.content as string).length} bytes)`);
			options.emitRecord?.("file_change", {
				op: "write",
				path: p.startsWith(root) ? p.slice(root.length + 1) : p,
				bytes: (params.content as string).length,
			});
			return {
				content: [{ type: "text" as const, text: `wrote ${p}` }],
				details: { path: p },
			};
		},
	});

	const editTool = defineTool({
		name: "edit",
		label: "edit (workspace)",
		description: "Replace exact text in a workspace file (path-checked; single occurrence required).",
		parameters: Type.Object({
			path: Type.String(),
			old_text: Type.String(),
			new_text: Type.String(),
		}),
		async execute(_id, params) {
			let p: string;
			try {
				p = resolveInRoot(root, String(params.path));
			} catch (err) {
				return denied(err);
			}
			const oldText = String(params.old_text);
			const text = readFileSync(p, "utf-8");
			const occurrences = text.split(oldText).length - 1;
			if (occurrences === 0) {
				return {
					content: [{ type: "text" as const, text: `ERROR: old_text not found in ${p}` }],
					details: { path: p, error: "not_found" as string | null },
					isError: true,
				};
			}
			if (occurrences > 1) {
				return {
					content: [{ type: "text" as const, text: `ERROR: old_text occurs ${occurrences} times — be more specific` }],
					details: { path: p, error: "ambiguous" as string | null },
					isError: true,
				};
			}
			writeFileSync(p, text.replace(oldText, String(params.new_text)), "utf-8");
			log(`edit ${p}`);
			options.emitRecord?.("file_change", {
				op: "edit",
				path: p.startsWith(root) ? p.slice(root.length + 1) : p,
				old_bytes: oldText.length,
				new_bytes: String(params.new_text).length,
			});
			return {
				content: [{ type: "text" as const, text: `edited ${p}` }],
				details: { path: p, error: null as string | null },
			};
		},
	});

	const lsTool = defineTool({
		name: "ls",
		label: "ls (workspace)",
		description: "List a directory inside the workspace (path-checked).",
		parameters: Type.Object({ path: Type.Optional(Type.String()) }),
		async execute(_id, params) {
			let p: string;
			try {
				p = resolveInRoot(root, String(params.path ?? "."));
			} catch (err) {
				return denied(err);
			}
			const entries = readdirSync(p, { withFileTypes: true }).map(
				(e) => `${e.isDirectory() ? "d" : "-"} ${e.name}`,
			);
			return {
				content: [{ type: "text" as const, text: entries.join("\n") || "(empty)" }],
				details: { path: p, count: entries.length },
			};
		},
	});

	const findTool = defineTool({
		name: "find",
		label: "find (workspace)",
		description: "Find files by name substring inside the workspace (path-checked).",
		parameters: Type.Object({
			pattern: Type.String(),
			path: Type.Optional(Type.String()),
		}),
		async execute(_id, params) {
			let base: string;
			try {
				base = resolveInRoot(root, String(params.path ?? "."));
			} catch (err) {
				return denied(err);
			}
			const needle = String(params.pattern).toLowerCase();
			const hits: string[] = [];
			const walk = (dir: string, depth: number) => {
				if (depth > 8 || hits.length >= 200) return;
				for (const entry of readdirSync(dir, { withFileTypes: true })) {
					if (entry.name === "node_modules" || entry.name === ".git") continue;
					const full = `${dir}/${entry.name}`;
					if (entry.name.toLowerCase().includes(needle)) hits.push(full);
					if (entry.isDirectory()) walk(full, depth + 1);
				}
			};
			walk(base, 0);
			return {
				content: [{ type: "text" as const, text: hits.join("\n") || "(no matches)" }],
				details: { count: hits.length },
			};
		},
	});

	const grepTool = defineTool({
		name: "grep",
		label: "grep (workspace)",
		description: "Search file contents (literal substring) inside the workspace (path-checked).",
		parameters: Type.Object({
			pattern: Type.String(),
			path: Type.Optional(Type.String()),
			limit: Type.Optional(Type.Number()),
		}),
		async execute(_id, params) {
			let base: string;
			try {
				base = resolveInRoot(root, String(params.path ?? "."));
			} catch (err) {
				return denied(err);
			}
			const needle = String(params.pattern);
			const cap = params.limit ?? 100;
			const hits: string[] = [];
			const walk = (dir: string, depth: number) => {
				if (depth > 8 || hits.length >= cap) return;
				for (const entry of readdirSync(dir, { withFileTypes: true })) {
					if (hits.length >= cap) return;
					if (entry.name === "node_modules" || entry.name === ".git") continue;
					const full = `${dir}/${entry.name}`;
					if (entry.isDirectory()) {
						walk(full, depth + 1);
					} else {
						try {
							if (statSync(full).size > 2 * 1024 * 1024) continue;
							const lines = readFileSync(full, "utf-8").split("\n");
							lines.forEach((line, i) => {
								if (hits.length < cap && line.includes(needle)) {
									hits.push(`${full}:${i + 1}: ${line.slice(0, 200)}`);
								}
							});
						} catch {
							// 二进制/不可读跳过
						}
					}
				}
			};
			const st = statSync(base);
			if (st.isDirectory()) walk(base, 0);
			else {
				const lines = readFileSync(base, "utf-8").split("\n");
				lines.forEach((line, i) => {
					if (hits.length < cap && line.includes(needle)) hits.push(`${base}:${i + 1}: ${line.slice(0, 200)}`);
				});
			}
			return {
				content: [{ type: "text" as const, text: hits.join("\n") || "(no matches)" }],
				details: { count: hits.length },
			};
		},
	});

	const bashTool = defineTool({
		name: "bash",
		label: "bash (workspace, allowlisted)",
		description:
			"Run an allowlisted command in the workspace. argv only (no shell " +
			"metacharacters). No network, no privilege escalation, no paths " +
			"outside the workspace, no credentials in the environment.",
		parameters: Type.Object({
			argv: Type.Array(Type.String(), { description: "command argv (first element must be allowlisted)" }),
			timeout_sec: Type.Optional(Type.Number()),
		}),
		async execute(_id, params, signal) {
			const argv = (params.argv as string[]).map(String);
			const detailsOf = (fields: {
				error?: string | null;
				reason?: string | null;
				exit_code?: number | null;
			}) => ({
				error: fields.error ?? null,
				reason: fields.reason ?? null,
				exit_code: fields.exit_code ?? null,
				argv,
			});
			if (argv.length === 0) {
				return {
					content: [{ type: "text" as const, text: "ERROR: empty argv" }],
					details: detailsOf({ error: "empty_argv" }),
					isError: true,
				};
			}
			const bin = argv[0];
			try {
				if (DENIED_BINARIES.has(bin)) throw new Error(`binary not allowed: ${bin}`);
				if (!ALLOWED_BINARIES.has(bin)) throw new Error(`binary not in allowlist: ${bin}`);
				const subDeny = DENIED_SUBCOMMANDS[bin];
				if (subDeny) {
					const reason = subDeny(argv.slice(1));
					if (reason) throw new Error(reason);
				}
				for (const arg of argv.slice(1)) argPathCheck(root, arg);
			} catch (err) {
				log(`DENIED ${argv.join(" ")} (${(err as Error).message})`);
				return {
					content: [{ type: "text" as const, text: `DENIED: ${(err as Error).message}` }],
					details: detailsOf({ error: "denied", reason: (err as Error).message }),
					isError: true,
				};
			}
			const timeoutMs = _bashTimeoutMs(
				params as { timeout_sec?: number },
				{ defaultTimeoutMs: options.defaultTimeoutMs },
			);
			log(`$ ${argv.join(" ")}`);
			const result = await new Promise<{ code: number; output: string }>((resolvePromise) => {
				const proc = spawn(bin, argv.slice(1), {
					cwd: root,
					// 关键隔离：env 白名单——没有任何 API key；HOME=workspace
					// （~ 不再指向用户家目录）；不带 ROSCLAW_HOME。
					env: {
						PATH: process.env.PATH ?? "/usr/bin:/bin",
						HOME: root,
						LANG: process.env.LANG ?? "C.UTF-8",
						LC_ALL: process.env.LC_ALL ?? "",
						TZ: process.env.TZ ?? "UTC",
						ROSCLAW_WORKER_PROTOCOL: "workbench",
					},
					stdio: ["ignore", "pipe", "pipe"],
				});
				let output = "";
				let truncated = false;
				const collect = (chunk: Buffer) => {
					if (output.length < MAX_OUTPUT_BYTES) output += chunk.toString();
					else truncated = true;
				};
				proc.stdout.on("data", collect);
				proc.stderr.on("data", collect);
				const heartbeat = setInterval(() => {
					options.emitProgress?.(`bash 仍在运行：${argv.join(" ").slice(0, 80)}`);
				}, 5000);
				const timer = timeoutMs === null
					? null
					: setTimeout(() => {
						proc.kill("SIGKILL");
					}, timeoutMs);
				signal?.addEventListener("abort", () => proc.kill("SIGKILL"), { once: true });
				proc.on("close", (code) => {
					if (timer) clearTimeout(timer);
					clearInterval(heartbeat);
					if (truncated) output += "\n[output truncated]";
					resolvePromise({ code: code ?? -1, output });
				});
				proc.on("error", (err) => {
					if (timer) clearTimeout(timer);
					clearInterval(heartbeat);
					resolvePromise({ code: -1, output: `spawn error: ${err.message}` });
				});
			});
			const tail = result.output.length > 8000 ? `…${result.output.slice(-8000)}` : result.output;
			log(`${tail}\n(exit ${result.code})\n`);
			return {
				content: [
					{
						type: "text" as const,
						text: `exit ${result.code}\n${tail || "(no output)"}`,
					},
				],
				details: detailsOf({ exit_code: result.code }),
				isError: result.code !== 0,
			};
		},
	});

	const tools = [readTool, grepTool, findTool, lsTool, writeTool, editTool, bashTool];
	// 十一审 PR-E：WAITING_INPUT 真实状态——Worker 可以诚实提问，
	// 而不是猜或停滞。
	if (options.askUser) {
		const ask = options.askUser;
		tools.push(
			defineTool({
				name: "ask_user",
				label: "ask user (blocking)",
				description:
					"Ask the user a blocking clarifying question when a missing " +
					"constraint would materially change the outcome. The job enters " +
					"WAITING_INPUT until the user answers. Use sparingly.",
				parameters: Type.Object({
					question: Type.String(),
				}),
				async execute(_id, params) {
					const answer = await ask(String(params.question));
					return {
						content: [{ type: "text" as const, text: `用户回答：${answer}` }],
						details: { answered: true },
					};
				},
			}) as never,
		);
	}
	return tools;
}
