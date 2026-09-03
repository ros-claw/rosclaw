// HP2-COMPAT: 工具定义原语（defineTool/Type/ToolDefinition）——工具层在 HP3 投影层（Codex MCP）落地前保持 Pi 形态；不新增会话装配引用。
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

/** 0902 R1-a：shell 降级的批准面（Approval Broker 桥接）。
 *  check/request/status 全部经 center.call 到 agentd 账本。 */
export interface ShellGate {
	/** standing grant 命中（task+revision+scope 绑定，任务活跃）。 */
	check(): Promise<boolean>;
	/** 登记授权请求 → request_id（PENDING）。 */
	request(): Promise<string>;
	/** 请求状态（PENDING/APPROVED_ONCE/APPROVED_TASK/DENIED）。 */
	status(requestId: string): Promise<string>;
}

/** 等待窗口（与 request-action 的 Operator 等待同语义：超时/中断
 *  = 拒绝语义）。 */
const SHELL_APPROVAL_TIMEOUT_MS = 10 * 60 * 1000;

/** 无沙箱降级的会话内授权流：request → onUpdate 卡 → 轮询决定。
 *  返回 true = 允许继续原操作（一次或本任务）。 */
export async function awaitShellApproval(
	gate: ShellGate,
	onUpdate: ((partial: {
		content: Array<{ type: "text"; text: string }>;
		details: Record<string, unknown>;
	}) => void) | undefined,
	signal: AbortSignal | undefined,
): Promise<boolean> {
	const requestId = await gate.request();
	onUpdate?.({
		content: [{
			type: "text" as const,
			text: `等待授权决定（${requestId}）…默认拒绝`,
		}],
		details: { phase: "AWAITING_SHELL_APPROVAL", request_id: requestId },
	});
	const deadline = Date.now() + SHELL_APPROVAL_TIMEOUT_MS;
	while (Date.now() < deadline) {
		if (signal?.aborted) return false; // 中断 = 拒绝语义
		const status = await gate.status(requestId);
		if (status === "APPROVED_ONCE" || status === "APPROVED_TASK") {
			return true;
		}
		if (status === "DENIED" || status === "UNKNOWN") return false;
		await new Promise((resolve) => setTimeout(resolve, 1500));
	}
	return false; // 超时 = 默认拒绝
}

export interface WorkspacePackOptions {
	/** 项目根（write/edit 的作用域；bash 的 cwd）。 */
	root: string;
	/** 当前模式（P0-6：全模式 bash 必须 bwrap 强隔离——无 bwrap
	 *  REAL/SHADOW fail closed；SIM 仅在操作者显式授权下降级
	 *  （TOOL_LAYER_ONLY 标记），否则同样 fail closed）。 */
	mode?: () => string;
	/** bwrap 路径探测（测试可注入）。 */
	bwrapPath?: () => string | null;
	/** bash 审计日志（可选）。 */
	bashLogPath?: string;
	/** 显式默认超时（运营配置；默认无定时器）。 */
	defaultBashTimeoutMs?: number;
	/** rosclaw home（P0-6：沙箱内遮蔽其 agent/agentd/run——凭据/
	 *  控制 token/bridge socket 不经 shell 可达，治理不可绕过）。 */
	rosclawHome?: string;
	/** 操作者显式授权的无沙箱降级（仅 SIM；bwrap 不可用的主机——
	 *  等价 ROSCLAW_ALLOW_UNSANDBOXED_SHELL=1）。 */
	allowUnsandboxedShell?: () => boolean;
	/** 0902 R1-a：无沙箱降级的 Runtime 批准面（会话内确认卡 →
	 *  task+revision+scope 绑定的 grant）。缺失 = fail closed。 */
	shellGate?: ShellGate;
	/** P0-C（0824 总纲 §6.2）：effectful 工具执行前的原子
	 *  admission（ensure_task_for_effect）——bash/write/edit
	 *  执行前触发。 */
	beforeEffect?: () => Promise<void>;
}

/** P0-6：沙箱内必须遮蔽的敏感路径（凭据/控制面/云凭据）。
 *  无条件遮蔽（不过滤存在性）——--tmpfs 对不存在路径同样成立
 *  （沙箱内呈现为空目录）：不向外泄漏"哪些路径存在"，会话中
 *  新建的凭据文件也被覆盖。 */
export function _sensitiveMasks(homeDir: string, rosclawHome?: string): string[] {
	const candidates = [
		`${homeDir}/.ssh`,
		`${homeDir}/.gnupg`,
		`${homeDir}/.aws`,
		`${homeDir}/.config/gh`,
		...(rosclawHome
			? [`${rosclawHome}/agent`, `${rosclawHome}/agentd`, `${rosclawHome}/run`]
			: []),
	];
	const args: string[] = [];
	for (const path of candidates) {
		args.push("--tmpfs", path);
	}
	return args;
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
		async execute(_id, params, signal, onUpdate) {
			const command = String(params.command ?? "").trim();
			if (!command) return denied("empty command");
			// P0-C：首个 effectful call 的原子 admission。
			await options.beforeEffect?.();
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
			// P0-6（0823 审计）：全模式 shell 必须 bwrap 强隔离——
			// SIM auto 下 Harness Shell 裸跑可绕过治理（读凭据/
			// 控制 token、直调 bridge socket、写项目源码树）。
			// 无 bwrap：REAL/SHADOW fail closed（H6 不变）；SIM 仅在
			// 操作者显式授权下降级（TOOL_LAYER_ONLY 诚实标记）。
			const mode = options.mode?.() ?? "SIMULATION";
			const strict = mode === "REAL" || mode === "SHADOW";
			// options.bwrapPath 存在即以它为准（测试注入 null =
			// 强制不可用）；未注入才真实探测。
			const bwrap = options.bwrapPath
				? options.bwrapPath()
				: (_bwrapAvailable() ? "/usr/bin/bwrap" : null);
			const sandboxed = bwrap !== null;
			let degradedMarker = "";
			if (!sandboxed) {
				if (strict) {
					return denied(
						`${mode} 模式 shell 需要 bwrap 强隔离——本机无可用 bwrap`
						+ "（user namespace 受限），fail closed（不裸跑）。",
					);
				}
				// 0902 R1-a（审计 §5.2）：SIM 降级走会话内批准——
				// 确认卡（允许一次/本任务允许/拒绝）→ grant 绑定
				// task+revision+scope → 立即继续原操作。删除全局
				// 环境变量授权的正式路径（0902 实证：用户已答"允许"
				// 仍被要求 export ROSCLAW_ALLOW_UNSANDBOXED_SHELL=1
				// 并重启——不可接受）。
				let granted = options.allowUnsandboxedShell?.() === true;
				if (!granted && options.shellGate) {
					granted = await options.shellGate.check();
				}
				if (!granted && options.shellGate) {
					granted = await awaitShellApproval(
						options.shellGate, onUpdate, signal,
					);
				}
				if (!granted) {
					return denied(
						`${mode} 模式 shell 需要 bwrap 强隔离——本机无可用 bwrap`
						+ "（user namespace 受限），fail closed（不裸跑）。"
						+ " SIM 下如确需无沙箱 shell：在确认卡选「允许一次」或"
						+ "「本任务允许」（结果带 TOOL_LAYER_ONLY 标记）",
					);
				}
				degradedMarker =
					"[TOOL_LAYER_ONLY: 本机无 OS 沙箱（bwrap 不可用）——"
					+ "操作者显式授权的降级运行，凭据/控制面在 shell 可达"
					+ "（风险已告知）]\n";
			}
			const output = await new Promise<string>((resolvePromise) => {
				let spawnCmd = "sh";
				let spawnArgs = ["-c", command];
				if (sandboxed && bwrap) {
					spawnCmd = bwrap;
					spawnArgs = [
						"--ro-bind", "/", "/",
						// P0-6：凭据/控制面遮蔽（顺序在 ro-bind 之后、
						// workspace rw 之前——socket/token/私钥不可读，
						// 治理链不可经 shell 绕过）。
						..._sensitiveMasks(
							process.env.HOME ?? "", options.rosclawHome,
						),
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
					resolvePromise(
						`${degradedMarker}${head}\n${buf.slice(0, MAX_OUTPUT_BYTES)}`,
					);
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
			// P0-C：首个 effectful call 的原子 admission。
			await options.beforeEffect?.();
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
			// P0-C：首个 effectful call 的原子 admission。
			await options.beforeEffect?.();
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
