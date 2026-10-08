#!/usr/bin/env node
// HP2-COMPAT: direct UI lifecycle ownership until helper exposes public stop.
/** rosclaw-agent 入口（PNA-0）：Pi InteractiveMode + ROSClaw 品牌。
 *
 * `rosclaw chat` 由 Python CLI 转调本入口。用户没有 engine 选择面
 * （ADR-0012A）：Pi 是唯一默认 Harness Backend，本进程不启动
 * Python AgentLoop。
 */

// PI_CODING_AGENT_DIR 必须在任何 pi 模块加载前设定（config.js 在
// import 期读取；ESM 静态 import 会被提升）——所有 pi 相关模块一律
// 动态 import。
// P0-NA-15：供应链边界——上游版本检查/自更新通道在 ROSClaw 产品里
// 一律关闭（host_managed：只有 ROSClaw signed release 能升级本产物，
// 内部 harness 不得自行更新）。同样必须在 pi 模块加载前设定。
// HP2-COMPAT: main owns InteractiveMode's public stop handle for error exit;
// the protected helper cannot return that handle after run rejects.
import { readFileSync } from "node:fs";
import { VERSION } from "./version.js";
// Type-only import（编译期擦除）——不会在 pi 模块加载前引入任何运行时依赖。
import type { ToolCallBudget } from "./harness/pi/tool-call-budget.js";

process.env.PI_SKIP_VERSION_CHECK = "1";
if (process.argv.includes("--continuation-target")) process.env.PI_OFFLINE = "1";

const rosclawHomeEnv = process.env.ROSCLAW_HOME ?? `${process.env.HOME}/.rosclaw`;
process.env.PI_CODING_AGENT_DIR ??= `${rosclawHomeEnv}/agent`;

// 0914 PR-1（审计 §3.6）：ROSCLAW_KIMI_API_KEY 迁移期旧别名——
// 内置 kimi-coding 期望 KIMI_API_KEY；旧用户只 export 了 ROSCLAW_
// KIMI_API_KEY 时进程内注入等价变量（只读、不落盘、不打印、
// 不覆盖已设的 KIMI_API_KEY 与 /login 的 auth.json——auth.json
// 在 Pi 解析顺序里本就优先于 env）。
if (!process.env.KIMI_API_KEY && process.env.ROSCLAW_KIMI_API_KEY) {
	process.env.KIMI_API_KEY = process.env.ROSCLAW_KIMI_API_KEY;
}

interface CliArgs {
	profile: "developer" | "robot";
	workspace?: string;
	initialMessage?: string;
	print: boolean;
	probe: boolean;
	deepProbe: boolean;
	missionId?: string;
	resumeSessionId?: string;
	resumeSessionPath?: string;
	browseSessions: boolean;
	continueLast: boolean;
	toolCallPolicyPath?: string;
}

// Stage A opt-in `--tool-call-policy JSON_FILE`：进入任何 runtime/session/
// provider 动态 import 或 auth 读取之前完成文件读取与 schema 语义校验——
// 拒绝路径绝不接触 provider/auth。校验语义与公共 schema
// （inputs/prebody_cli/tool_call_policy.schema.json）一致；保留字段名
// （__proto__ 等）作为 own data key 原样透传，不做重建赋值，不引入原型污染。
const TOOL_CALL_POLICY_KEYS = new Set([
	"allowedTools", "maxCalls", "maxTotalCalls", "exactCommands", "visibleBudget",
]);

function invalidToolCallPolicy(message: string): never {
	throw new Error(`INVALID_TOOL_CALL_POLICY: ${message}`);
}

function validatePolicyToolNames(value: unknown): string[] {
	if (!Array.isArray(value)) invalidToolCallPolicy("allowedTools must be an array");
	const names = value.map((entry) => {
		if (typeof entry !== "string" || entry.length === 0 || entry !== entry.trim()) {
			invalidToolCallPolicy("allowedTools entries must be non-empty trimmed strings");
		}
		return entry;
	});
	if (new Set(names).size !== names.length) {
		invalidToolCallPolicy("allowedTools entries must be unique");
	}
	return names;
}

function validatePolicyCount(value: unknown, label: string): void {
	// JSON 不产生 NaN/Infinity；bool/fraction/unsafe/negative 一并拒绝。
	if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
		invalidToolCallPolicy(`${label} must be a non-negative safe integer`);
	}
}

function validatePolicyCountMap(value: unknown, allowed: Set<string>): void {
	if (!value || typeof value !== "object" || Array.isArray(value)) {
		invalidToolCallPolicy("maxCalls must be an object");
	}
	for (const [name, count] of Object.entries(value)) {
		if (!allowed.has(name)) {
			invalidToolCallPolicy(`maxCalls key ${JSON.stringify(name)} not in allowedTools`);
		}
		validatePolicyCount(count, `maxCalls[${JSON.stringify(name)}]`);
	}
}

function validatePolicyExactCommands(value: unknown, allowed: Set<string>): void {
	if (!value || typeof value !== "object" || Array.isArray(value)) {
		invalidToolCallPolicy("exactCommands must be an object");
	}
	for (const [name, commands] of Object.entries(value)) {
		if (!allowed.has(name)) {
			invalidToolCallPolicy(`exactCommands key ${JSON.stringify(name)} not in allowedTools`);
		}
		if (!Array.isArray(commands) || commands.length === 0
			|| commands.some((command) => typeof command !== "string" || command.length === 0)
			|| new Set(commands).size !== commands.length) {
			invalidToolCallPolicy(`exactCommands[${JSON.stringify(name)}] must be a non-empty unique string array`);
		}
	}
}

/** 读取并校验 opt-in 策略文件；任何不合法都在此抛出（早失败）。 */
export function loadToolCallPolicyFile(path: string): ToolCallBudget {
	let text: string;
	try {
		text = readFileSync(path, "utf-8");
	} catch {
		invalidToolCallPolicy(`cannot read file: ${path}`);
	}
	let raw: unknown;
	try {
		raw = JSON.parse(text);
	} catch {
		invalidToolCallPolicy("malformed JSON");
	}
	if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
		invalidToolCallPolicy("root must be an object");
	}
	const doc = raw as Record<string, unknown>;
	for (const key of Object.keys(doc)) {
		if (!TOOL_CALL_POLICY_KEYS.has(key)) invalidToolCallPolicy(`unknown top-level key: ${key}`);
	}
	if (!("allowedTools" in doc)) invalidToolCallPolicy("missing allowedTools");
	const allowed = new Set(validatePolicyToolNames(doc.allowedTools));
	if (doc.maxCalls !== undefined) validatePolicyCountMap(doc.maxCalls, allowed);
	if (doc.maxTotalCalls !== undefined) validatePolicyCount(doc.maxTotalCalls, "maxTotalCalls");
	if (doc.exactCommands !== undefined) validatePolicyExactCommands(doc.exactCommands, allowed);
	if (doc.visibleBudget !== undefined && typeof doc.visibleBudget !== "boolean") {
		invalidToolCallPolicy("visibleBudget must be a boolean");
	}
	return raw as ToolCallBudget;
}

function parseArgs(argv: string[]): CliArgs {
	let profile: "developer" | "robot" = "developer";
	let initialMessage: string | undefined;
	let print = false;
	let probe = false;
	let deepProbe = false;
	let missionId: string | undefined;
	let resumeSessionId: string | undefined;
	let resumeSessionPath: string | undefined;
	let browseSessions = false;
	let continueLast = false;
	let workspace: string | undefined;
	let toolCallPolicyPath: string | undefined;
	for (let i = 0; i < argv.length; i += 1) {
		if (argv[i] === "--tool-call-policy") {
			// 缺参数立即拒绝（早失败，不进 runtime/auth）。
			if (argv[i + 1] === undefined) {
				throw new Error("INVALID_TOOL_CALL_POLICY: --tool-call-policy requires a JSON file path");
			}
			toolCallPolicyPath = argv[i + 1];
			i += 1;
		} else if (argv[i] === "--workspace" && argv[i + 1]) {
			workspace = argv[i + 1];
			i += 1;
		} else if (argv[i] === "--profile" && argv[i + 1]) {
			profile = argv[i + 1] === "robot" ? "robot" : "developer";
			i += 1;
		} else if (argv[i] === "--message" && argv[i + 1]) {
			initialMessage = argv[i + 1];
			i += 1;
		} else if (argv[i] === "--print") {
			print = true;
		} else if (argv[i] === "--probe") {
			// P1-A1：模型探测单源——setup/doctor 经此走 Pi ModelRuntime。
			probe = true;
		} else if (argv[i] === "--deep") {
			// R0-7：严格 tool call 探测（doctor --deep 专用——默认
			// 便宜探测不烧第二次模型请求）。
			deepProbe = true;
		} else if (argv[i] === "--mission" && argv[i + 1]) {
			missionId = argv[i + 1];
			i += 1;
		} else if (argv[i] === "--resume" && argv[i + 1]) {
			resumeSessionId = argv[i + 1];
			i += 1;
		} else if (argv[i] === "--resume-path" && argv[i + 1]) {
			// WP-P0-1：Python 侧已把 ID/前缀/标题解析成真实路径——
			// 不由用户输入拼路径。
			resumeSessionPath = argv[i + 1];
			i += 1;
		} else if (argv[i] === "--browse-sessions") {
			browseSessions = true;
		} else if (argv[i] === "--continue" || argv[i] === "-c") {
			continueLast = true;
		}
	}
	return {
		profile, initialMessage, print, probe, deepProbe, missionId, workspace,
		resumeSessionId, resumeSessionPath, browseSessions, continueLast,
		...(toolCallPolicyPath !== undefined ? { toolCallPolicyPath } : {}),
	};
}

async function main(): Promise<number> {
	// Stage A：opt-in 策略文件在任何动态 runtime/session/provider import
	// 与 auth 读取之前解析并校验——拒绝路径在最早的失败点退出。
	const {
		profile, initialMessage, print, probe, deepProbe, missionId, workspace,
		resumeSessionId, resumeSessionPath, browseSessions, continueLast,
		toolCallPolicyPath,
	} = parseArgs(process.argv.slice(2));
	const toolCallBudget = toolCallPolicyPath === undefined
		? undefined
		: loadToolCallPolicyFile(toolCallPolicyPath);
	// PR-HP2：Pi SDK 调用全部经 harness/pi/ 帮助函数——本文件不再
	// 直接引用 Pi 包。
	const {
		continueRecentPiSession, resolveContinuationTarget, listAllPiSessions, listPiSessions,
		openPiSession, runPiPrint,
	} = await import("./harness/pi/pi-sessions.js");
	const { SessionWriterOwnership, isSessionInUse } = await import(
		"./harness/pi/session-writer-ownership.js"
	);
	// SESSION_WRITER：本进程的 session writer owner——resume/continue 在
	// SessionManager.open 前占有目标文件；新 chat 在 runtime 预创建 seam
	// 占有最终文件。正常退出/失败退出只释放自己拥有的 claim。
	const ownership = new SessionWriterOwnership();
	const { createRosclawRuntime } = await import("./harness/pi/pi-runtime.js");
	const rosclawHome = rosclawHomeEnv;
	if (process.argv.includes("--continuation-target")) {
		const target = await resolveContinuationTarget(`${rosclawHome}/agent/sessions`);
		console.log(JSON.stringify(target ?? { status: "NO_RECORDED_SESSION" }));
		return target ? 0 : 2;
	}
	if (probe) {
		// P1-A1：headless 探测通道——输出单行 JSON（无 secret），
		// Python onboarding/doctor 解析；不建会话、不进 TUI。
		const { probePiModel } = await import("./harness/pi/pi-probe.js");
		const report = await probePiModel({
			agentDir: `${rosclawHome}/agent`,
			// 探测只读全局 settings（defaultProvider/Model）——project
			// settings 无关；probe 不新增进程 cwd 读取（N1 不变量）。
			cwd: workspace ?? rosclawHome,
			profile,
			deep: deepProbe,
		});
		console.log(JSON.stringify(report));
		// 便宜探测（默认）以 chat_ok 为收敛；deep 才要求 tool call。
		return report.reachable && report.chat_ok && (deepProbe ? report.tool_call_ok : true) ? 0 : 1;
	}
	// WP-P0-1（总纲 §5.1）：恢复路径全部经 Pi SessionManager 公开
	// API——不再有手写目录扫描/mtime 排序/文件名拼接（Pi 文件名可含
	// 时间前缀，拼接 id 是格式漂移风险）。
	const { WorkspaceStore } = await import("./session/workspace.js");
	const workspaceStore = new WorkspaceStore(rosclawHome);
	const startupCwd = process.cwd(); // 唯一启动解析输入
	let initialSession: import("./harness/pi/pi-sessions.js").SessionManager | undefined;
	const sessionDir = `${rosclawHome}/agent/sessions`;
	try {
	if (browseSessions) {
		const { browseSessions: openPicker } = await import("./harness/pi/pi-picker.js");
		const picked = await openPicker(
			(onProgress) => listPiSessions(workspace ?? startupCwd, sessionDir, onProgress),
			(onProgress) => listAllPiSessions(sessionDir, onProgress),
		);
		if (!picked) return 0;  // 用户取消——干净退出，不建会话
		initialSession = openPiSession(picked, sessionDir, ownership);
	} else if (resumeSessionPath) {
		initialSession = openPiSession(resumeSessionPath, sessionDir, ownership);
	} else if (resumeSessionId) {
		// 兼容路径：`chat --resume <id>`——精确 ID/唯一前缀经
		// listAll 解析（拒绝路径穿越由解析保证）。
		const { resolveSessionQuery } = await import("./harness/pi/pi-resolve.js");
		const sessions = await listAllPiSessions(sessionDir);
		const hit = resolveSessionQuery(resumeSessionId, sessions);
		if (!hit.ok) {
			console.error(
				hit.error === "AMBIGUOUS"
					? `会话不唯一（${hit.candidates.length} 个候选）——请用 rosclaw resume 打开选择器`
					: `会话 ${resumeSessionId} 不存在——rosclaw sessions 查看全部`,
			);
			return 2;
		}
		initialSession = openPiSession(hit.path, sessionDir, ownership);
	} else if (continueLast) {
		initialSession = await continueRecentPiSession(workspace ?? startupCwd, sessionDir, ownership);
		if (!initialSession) {
			console.error("没有可继续的已记录会话；请用 rosclaw chat 创建新会话");
			return 2;
		}
	}
	} catch (err) {
		// SESSION_WRITER：第二个同 session 进程在任何 SDK open/append/
		// provider/tool 副作用之前被拒绝——文件保持原样，非零退出。
		if (isSessionInUse(err)) {
			console.error((err as Error).message);
			ownership.releaseAll();
			return 2;
		}
		throw err;
	}
	// PR-N1：ActiveTaskContext 在 session 创建前解析并冻结——
	// runtime/工具/bridge/artifact/verifier/header 全从这里取路径。
	const { resolveTaskContext } = await import("./native/active-task-context.js");
	let taskContext = resolveTaskContext({
		rosclawHome,
		cwd: startupCwd,
		mode: "SIMULATION",
		explicitWorkspace: workspace,
		resumedWorkspace: initialSession?.getCwd(),
	});
	const isResume = Boolean(
		resumeSessionId || resumeSessionPath || browseSessions || continueLast,
	);
	// Persist only after a successful recorded-session selection. A failed
	// resume must not rewrite the current workspace, and continuation uses its
	// recorded cwd rather than implicit enclosing git inference.
	if (taskContext.workspaceSource === "explicit" || taskContext.workspaceSource === "git") {
		workspaceStore.bind(taskContext.workspaceRoot, { normalizeToGit: false });
	}
	const startupWs = { bound: workspaceStore.current, auto: taskContext.workspaceSource === "git" };
	// 十一审 PR-D：Workspace 一等状态——显式 --workspace > cwd git 自动
	// 绑定 > 既有绑定。
	const { runtime, coordinator, leaseManager, ownership: runtimeOwnership } = await createRosclawRuntime({
		cwd: taskContext.workspaceRoot,
		taskContext,
		rosclawHome,
		profile,
		version: VERSION,
		workspaceStore,
		workspaceAutoBound: startupWs.auto,
		ownership,
		...(toolCallBudget !== undefined ? { toolCallBudget } : {}),
		...(missionId ? { missionId } : {}),
		...(initialSession ? { sessionManager: initialSession } : {}),
		...(isResume ? { resumed: true } : {}),
	});
	// P0-NA-12：初始绑定统一经 coordinator——lease/heartbeat/fresh
	// context/原子状态替换是一个事务，lease_token 绝不丢弃。
	// PR-SIX-1：显式 --mission 也必须经 coordinator（attachInitialMission
	// 写回 leaseState=ACTIVE）——此前直接 leaseManager.bind，header 显示
	// Action LOCKED 而动作实际可执行（假锁）。
	const sessionId = runtime.session.sessionManager.getSessionId();
	try {
	if (missionId) {
		const outcome = await coordinator.attachInitialMission(sessionId, missionId);
		if (!outcome.ok) {
			console.error(`初始 Mission 接入失败：${outcome.reason}`);
			// The common confirmed teardown also covers startup binding failure.
			return 2;
		}
	} else if (resumeSessionId || resumeSessionPath || browseSessions || continueLast) {
		// 恢复路径：重接既有绑定（丢失/已归档 → coordinator 新建 SIM
		// 绑定并明确告知）——不再"只看到 header 就算恢复"。
		const outcome = await coordinator.resumeInitial(sessionId);
		if (!outcome.ok) {
			console.error(`恢复绑定失败：${outcome.reason}`);
			return 2;
		}
	}
		if (print) {
			// 非 TTY 单发模式（冒烟/脚本）。
			return await runPiPrint(runtime, {
				...(initialMessage ? { initialMessage } : {}),
			});
		}
		// Keep the consumer handle: the legacy helper loses it on run rejection.
		const { InteractiveMode } = await import("@earendil-works/pi-coding-agent");
		const mode = new InteractiveMode(runtime, {
			verbose: false,
			...(initialMessage ? { initialMessage } : {}),
		});
		try {
			await mode.run();
			return 0;
		} finally {
			// Local consumer termination only, never remote idle confirmation.
			mode.stop();
		}
	} finally {
		// UI/print completion is not proof that the SDK writer stopped.
		// Capture the current session (which may have changed through /resume).
		const session = runtime.session;
		try {
			await session.abort();
			if (session.isIdle !== true) throw new Error("SESSION_CLOSE_IDLE_UNCONFIRMED");
			await session.dispose();
			if (session.isIdle !== true) throw new Error("SESSION_CLOSE_IDLE_UNCONFIRMED");
			await runtime.dispose();
			if (session.isIdle !== true) throw new Error("SESSION_CLOSE_IDLE_UNCONFIRMED");
		} catch (err) {
			// Host shutdown terminates local extension consumers even on abort
			// failure. It cannot upgrade the sticky failed confirmation to success.
			try {
				// Best-effort local agent cancellation drains real terminal events;
				// it is NOT a successful session abort receipt or release authority.
				session.agent.abort();
				await session.waitForIdle();
				await runtime.dispose();
			} catch { /* Keep the first failure. */ }
			throw new Error(`MAIN_EXIT_TEARDOWN_UNCONFIRMED: ${(err as Error).message}`);
		}
		runtimeOwnership.releaseAll();
		await leaseManager.release();
	}
}

main().then(
	(code) => { process.exitCode = code; },
	(err) => {
		console.error(`rosclaw-agent failed: ${(err as Error).message}`);
		process.exitCode = 2;
	},
);
