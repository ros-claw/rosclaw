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
	/** bash 日志（artifacts/bash-log.txt）。 */
	bashLogPath: string;
	/** 长命令保活：bash 运行期间周期性回调（喂 idle watchdog）。 */
	emitProgress?: (message: string) => void;
	/** 单命令默认超时（ms）。 */
	defaultTimeoutMs?: number;
}

/** argv 白名单（首参数必须精确命中；不含任何网络/特权工具）。 */
const ALLOWED_BINARIES = new Set([
	"ls", "cat", "head", "tail", "grep", "find", "wc", "echo", "printf", "pwd",
	"mkdir", "touch", "cp", "mv", "rm", "diff", "sed", "awk", "sort", "uniq",
	"tr", "cut", "xargs", "tee", "git", "python", "python3", "pytest",
	"node", "npm", "npx", "tsc", "make", "jq", "sha256sum", "file", "stat",
	"chmod", "basename", "dirname", "realpath", "true", "false", "test", "[",
	"sleep",
]);

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
			const slice = lines.slice(start, params.limit ? start + params.limit : undefined);
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
				for (const arg of argv.slice(1)) argPathCheck(root, arg);
			} catch (err) {
				log(`DENIED ${argv.join(" ")} (${(err as Error).message})`);
				return {
					content: [{ type: "text" as const, text: `DENIED: ${(err as Error).message}` }],
					details: detailsOf({ error: "denied", reason: (err as Error).message }),
					isError: true,
				};
			}
			const timeoutMs = Math.min(
				(params.timeout_sec ?? 0) > 0 ? Number(params.timeout_sec) * 1000 : (options.defaultTimeoutMs ?? 120_000),
				600_000,
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
				const timer = setTimeout(() => {
					proc.kill("SIGKILL");
				}, timeoutMs);
				signal?.addEventListener("abort", () => proc.kill("SIGKILL"), { once: true });
				proc.on("close", (code) => {
					clearTimeout(timer);
					clearInterval(heartbeat);
					if (truncated) output += "\n[output truncated]";
					resolvePromise({ code: code ?? -1, output });
				});
				proc.on("error", (err) => {
					clearTimeout(timer);
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

	return [readTool, grepTool, findTool, lsTool, writeTool, editTool, bashTool];
}
