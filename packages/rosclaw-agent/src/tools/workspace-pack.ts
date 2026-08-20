/** Workspace Pack 策略包装（PR-H1，总纲 v2 §10.2/§14.5）。
 *
 * 主会话的 bash/write/edit 是"用户自己的 Agent 在用户项目里工作"，
 * 不是 Worker 密封舱——但也不能裸奔：
 *
 * - bash：拒绝灾难性/系统修改命令（sudo/rm -rf 根/mkfs/dd 设备/
 *   systemctl/shutdown）；子进程 env 剥离 ROSCLAW 前缀（daemon 权威
 *   凭据不进 shell）与 ROS/RMW/CYCLONEDDS/FASTRTPS 前缀（SIM 下不
 *   接触真机写通道——诊断走 rosclaw_observe）；无默认 wall-clock
 *   kill（显式 timeout_sec 才装定时器）；输出截断 64KB。
 * - write/edit：路径解析在 cwd（项目根）内；拒绝 authority/token
 *   路径与系统路径。
 *
 * 诚实命名：GUARDED_MAIN_SESSION——这是第一层过滤；REAL/SHADOW 强
 * 隔离（bwrap/容器）在 PR-H6。模型可读的 prompt 必须与此面一致。
 */

import { execFileSync, spawn } from "node:child_process";
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ToolDefinition } from "@earendil-works/pi-coding-agent";

import { resolveInRoot, _bashTimeoutMs } from "../workers/workbench.js";

const MAX_OUTPUT_BYTES = 64 * 1024;

/** 灾难性/系统修改命令（主会话第一层）。 */
const DENIED_COMMAND = new Set([
	"sudo", "su", "doas", "mkfs", "mkfs.ext4", "fdisk", "parted",
	"shutdown", "reboot", "halt", "poweroff", "systemctl", "service",
	"crontab", "chown", "mount", "umount", "insmod", "modprobe",
]);

const DENIED_PATTERNS: RegExp[] = [
	/\brm\s+(-[a-zA-Z]*f[a-zA-Z]*\s+)?\/(\s|$)/, // rm / 或 rm -rf /
	/\bdd\s+.*of=\/dev\//, // dd 写设备
	/>\s*\/dev\/(sd|nvme|mapper)/, // 重定向写块设备
	/\b(eval|exec)\b.*\$\(/, // 明显二次展开执行
];

/** 子进程 env 剥离前缀：daemon 权威与真机通道不进 shell。 */
const ENV_DENY_PREFIXES = [
	"ROSCLAW_",
	"ROS_", "RMW_", "CYCLONEDDS", "FASTRTPS", "ROS_DOMAIN_ID",
];
/** PATH/HOME 等基础变量保留；模型凭据保留（主会话模型自己要用——
 *  但只保留变量引用指向的真实 env，不展开任何文件凭据）。 */
const ENV_KEEP_EXACT = new Set(["PATH", "HOME", "LANG", "LC_ALL", "TZ", "TERM", "TMPDIR"]);

function scrubEnv(source: NodeJS.ProcessEnv): Record<string, string> {
	const out: Record<string, string> = {};
	for (const [key, value] of Object.entries(source)) {
		if (value === undefined) continue;
		if (ENV_KEEP_EXACT.has(key)) {
			out[key] = value;
			continue;
		}
		if (ENV_DENY_PREFIXES.some((p) => key.startsWith(p))) continue;
		// 模型 provider key 透传（与 worker env 一致——变量值已在进程
		// 环境，shell 是主会话自己的工作进程）。
		if (/_API_KEY$|_API_TOKEN$/.test(key)) {
			out[key] = value;
		}
	}
	return out;
}

export interface WorkspacePackOptions {
	/** 项目根（write/edit 的作用域；bash 的 cwd）。 */
	root: string;
	/** 当前模式（PR-H6：REAL/SHADOW → bash 必须 bwrap 强隔离；
	 *  无 bwrap fail closed）。 */
	mode?: () => string;
	/** bwrap 路径探测（测试可注入）。 */
	bwrapPath?: () => string | null;
	/** bash 审计日志（可选）。 */
	bashLogPath?: string;
	/** 显式默认超时（运营配置；默认无定时器）。 */
	defaultBashTimeoutMs?: number;
}

export function buildWorkspacePackTools(options: WorkspacePackOptions): ToolDefinition[] {
	const root = options.root;
	const scrubbedEnv = scrubEnv(process.env);

	const denied = (reason: string) => ({
		content: [{ type: "text" as const, text: `DENIED: ${reason}` }],
		details: { error: "denied", reason } as Record<string, unknown>,
		isError: true,
	});

	const bashTool = defineTool({
		name: "bash",
		label: "bash (guarded)",
		description:
			"Run a shell command in the project workspace. Guarded: no sudo/system " +
			"modification, no device writes, daemon credentials and ROS/DDS channels " +
			"are stripped from the child environment. No default wall-clock kill.",
		parameters: Type.Object({
			command: Type.String({ description: "要执行的 shell 命令" }),
			timeout_sec: Type.Optional(Type.Number({ description: "显式超时（秒）——不填则无定时器" })),
		}),
		async execute(_id, params, signal) {
			const command = String(params.command ?? "").trim();
			if (!command) return denied("empty command");
			const argv0 = command.split(/\s+/)[0].replace(/^\(.*\)\s*/, "");
			const base = argv0.split("/").pop() ?? argv0;
			if (DENIED_COMMAND.has(base)) {
				return denied(`系统修改命令 ${base} 不在主会话授权内`);
			}
			for (const pattern of DENIED_PATTERNS) {
				if (pattern.test(command)) {
					return denied(`危险命令模式被拒：${command.slice(0, 80)}`);
				}
			}
			const timeoutMs = _bashTimeoutMs(
				{ timeout_sec: params.timeout_sec as number | undefined },
				{ defaultTimeoutMs: options.defaultBashTimeoutMs },
			);
			const started = Date.now();
			// PR-H6（§14.5）：REAL/SHADOW 模式的 shell 必须强隔离
			// （bwrap：宿主只读、workspace 可写、无网络、无 /dev 写、
			// env 已剥离）——无 bwrap fail closed，绝不裸跑。
			const mode = options.mode?.() ?? "SIMULATION";
			const sandboxed = mode === "REAL" || mode === "SHADOW";
			// options.bwrapPath 存在即以它为准（测试注入 null =
			// 强制不可用）；未注入才真实探测。
			const bwrap = options.bwrapPath
				? options.bwrapPath()
				: (_bwrapAvailable() ? "/usr/bin/bwrap" : null);
			if (sandboxed && !bwrap) {
				return denied(
					`${mode} 模式 shell 需要 bwrap 强隔离——本机无 bwrap，fail closed（不裸跑）`,
				);
			}
			const output = await new Promise<string>((resolvePromise) => {
				let spawnCmd = "sh";
				let spawnArgs = ["-c", command];
				if (sandboxed && bwrap) {
					spawnCmd = bwrap;
					spawnArgs = [
						"--ro-bind", "/", "/",
						"--bind", root, root, // workspace 可写（其余宿主只读）
						"--unshare-net",
						"--dev", "/dev", // 全新 devtmpfs——真设备不可见
						"--chdir", root,
						"sh", "-c", command,
					];
				}
				const child = spawn(spawnCmd, spawnArgs, {
					cwd: root,
					env: scrubbedEnv,
					signal: signal ?? undefined,
				});
				let buf = "";
				let timedOut = false;
				child.stdout?.on("data", (d) => {
					if (buf.length < MAX_OUTPUT_BYTES) buf += d.toString();
				});
				child.stderr?.on("data", (d) => {
					if (buf.length < MAX_OUTPUT_BYTES) buf += d.toString();
				});
				let timer: NodeJS.Timeout | null = null;
				if (timeoutMs !== null) {
					timer = setTimeout(() => {
						timedOut = true;
						child.kill("SIGKILL");
					}, timeoutMs);
				}
				child.on("close", (code) => {
					if (timer) clearTimeout(timer);
					const head = `exit=${code ?? "signal"} wall=${Date.now() - started}ms`
						+ (timedOut ? " TIMEOUT(explicit)" : "");
					resolvePromise(`${head}\n${buf.slice(0, MAX_OUTPUT_BYTES)}`);
				});
				child.on("error", (err) => {
					if (timer) clearTimeout(timer);
					resolvePromise(`spawn error: ${err.message}`);
				});
			});
			if (options.bashLogPath) {
				try {
					mkdirSync(dirname(options.bashLogPath), { recursive: true });
					appendFileSync(options.bashLogPath, `$ ${command}\n${output}\n`, "utf-8");
				} catch {
					// 日志失败不阻塞工具结果
				}
			}
			return {
				content: [{ type: "text" as const, text: output }],
				details: { command: command.slice(0, 200) },
			};
		},
	});

	const writeTool = defineTool({
		name: "write",
		label: "write (workspace)",
		description: "Create/overwrite a file inside the project workspace (path-checked).",
		parameters: Type.Object({
			path: Type.String(),
			content: Type.String(),
		}),
		async execute(_id, params) {
			try {
				const p = resolveInRoot(root, String(params.path));
				mkdirSync(dirname(p), { recursive: true });
				writeFileSync(p, String(params.content), "utf-8");
				return {
					content: [{ type: "text" as const, text: `wrote ${p} (${String(params.content).length} bytes)` }],
					details: { path: p },
				};
			} catch (err) {
				return denied((err as Error).message);
			}
		},
	});

	const editTool = defineTool({
		name: "edit",
		label: "edit (workspace)",
		description:
			"Replace exact text in a file inside the project workspace " +
			"(path-checked; oldText must match exactly once).",
		parameters: Type.Object({
			path: Type.String(),
			oldText: Type.String(),
			newText: Type.String(),
		}),
		async execute(_id, params) {
			try {
				const p = resolveInRoot(root, String(params.path));
				const text = readFileSync(p, "utf-8");
				const oldText = String(params.oldText);
				const occurrences = text.split(oldText).length - 1;
				if (occurrences !== 1) {
					return denied(`oldText 出现 ${occurrences} 次（必须恰好 1 次）`);
				}
				writeFileSync(p, text.replace(oldText, String(params.newText)), "utf-8");
				return {
					content: [{ type: "text" as const, text: `edited ${p}` }],
					details: { path: p },
				};
			} catch (err) {
				return denied((err as Error).message);
			}
		},
	});

	return [bashTool, writeTool, editTool];
}

/** bwrap 可用性（PR-H6 fail-closed 判定的真实探测）。 */
export function _bwrapAvailable(): boolean {
	try {
		// 真实探测（不是 --version）：userns 被禁的内核上
		// --version 正常但运行即失败（uid map Permission denied）。
		execFileSync("/usr/bin/bwrap", ["--ro-bind", "/", "/", "true"], { stdio: "ignore" });
		return true;
	} catch {
		return false;
	}
}
