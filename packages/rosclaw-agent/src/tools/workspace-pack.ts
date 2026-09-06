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
import { dirname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

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
	/** 大道至简 R1-2a：任务沙箱额外资根（活跃任务 run dir 的
	 *  scratch 区——Pi 写的任务代码/脚本放这里；write/edit/bash
	 *  的 cwd 允许落在 [root, ...extraRoots] 任一根内）。 */
	extraRoots?: () => string[];
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

	/** 大道至简 R1-2a：多根路径解析——[session workspace, 任务
	 *  scratch 区] 任一根内即合法（其余照旧拒绝）。 */
	const resolveAllowed = (target: string): string => {
		const roots = [root, ...(options.extraRoots?.() ?? [])];
		let lastErr: Error | null = null;
		for (const r of roots) {
			try {
				return resolveInRoot(r, target);
			} catch (err) {
				lastErr = err as Error;
			}
		}
		throw lastErr ?? new Error(`path escapes workspace: ${target}`);
	};

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
			cwd: Type.Optional(Type.String({
				description: "工作目录（限 session 工作区/任务 scratch 区内；缺省=session 工作区）",
			})),
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
			// R1-2a：cwd 参数（限允许的根内——任务 scratch 区的
			// 脚本可直接运行；越界直接拒绝，不进执行）。
			let effectiveCwd = root;
			if (params.cwd !== undefined && params.cwd !== null && String(params.cwd).trim()) {
				try {
					effectiveCwd = resolveAllowed(String(params.cwd));
				} catch (err) {
					return denied((err as Error).message);
				}
			}
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
				// 大道至简 R1-2b（2026-09-05 方案安全节）：SIM 任务沙箱
				// 代码自动执行，不弹批准卡——第一层过滤（灾难命令/危险
				// 模式/scrubbed env）+ 诚实标记照旧。R1-a 的降级确认卡
				// 退役（SIM 下卡片是噪音；REAL/SHADOW fail closed 不变）。
				degradedMarker =
					"[TOOL_LAYER_ONLY: 本机无 OS 沙箱（bwrap 不可用）——"
					+ "SIM 任务沙箱降级运行，凭据/控制面在 shell 可达"
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
						// workspace 与任务 scratch 区可写（其余宿主只读）。
						...[root, ...(options.extraRoots?.() ?? [])].flatMap(
							(r) => ["--bind", r, r],
						),
						"--unshare-net",
						"--dev", "/dev", // 全新 devtmpfs——真设备不可见
						"--chdir", effectiveCwd,
						"sh", "-c", command,
					];
				}
				const child = spawn(spawnCmd, spawnArgs, {
					cwd: effectiveCwd,
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
				const p = resolveAllowed(String(params.path));
				// 0903（§4.4）：产品核心源码只读——普通任务不得修改
				// 正在运行的产品（改产品走开发流程：clone+PR）。
				if (isProductSourcePath(p)) {
					return denied(
						"拒绝写入：目标是正在运行的 ROSClaw 产品核心源码"
						+ "（§4.4 普通任务不得修改产品核心）——改产品请走开发流程"
						+ "（克隆仓库+PR），不要在本会话内改运行中的代码",
					);
				}
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
				const p = resolveAllowed(String(params.path));
				if (isProductSourcePath(p)) {
					return denied(
						"拒绝编辑：目标是正在运行的 ROSClaw 产品核心源码（§4.4）"
						+ "——改产品请走开发流程（克隆仓库+PR）",
					);
				}
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

/** 运行中产品自身源码根（0903 体验实证 + §4.4：普通任务禁止修改
 *  正在运行的产品核心——模型在"画立方体"任务中途给规划器加形状）。
 *  本文件位于 <产品根>/packages/rosclaw-agent/(dist/)src/tools/，
 *  向上解析到产品根；守护 <根>/src/rosclaw 与 <根>/packages/
 *  rosclaw-agent/src。 */
export function productSourceRoots(): string[] {
	const here = dirname(fileURLToPath(import.meta.url));
	// src/tools 或 dist/src/tools → 产品根（rosclaw-agent 包的上一级
	// 的上一级：packages/rosclaw-agent → <根>）。
	let root = here;
	for (let i = 0; i < 5; i++) root = dirname(root);
	return [join(root, "src", "rosclaw"), join(root, "packages", "rosclaw-agent", "src")];
}

/** 路径是否落在运行中产品核心源码树。 */
export function isProductSourcePath(target: string): boolean {
	const real = resolve(target);
	return productSourceRoots().some(
		(r) => real === r || real.startsWith(r + sep),
	);
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
