#!/usr/bin/env node
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
import { VERSION } from "./version.js";

process.env.PI_SKIP_VERSION_CHECK = "1";

const rosclawHomeEnv = process.env.ROSCLAW_HOME ?? `${process.env.HOME}/.rosclaw`;
process.env.PI_CODING_AGENT_DIR ??= `${rosclawHomeEnv}/agent`;

interface CliArgs {
	profile: "developer" | "robot";
	workspace?: string;
	initialMessage?: string;
	print: boolean;
	missionId?: string;
	resumeSessionId?: string;
	resumeSessionPath?: string;
	browseSessions: boolean;
	continueLast: boolean;
}

function parseArgs(argv: string[]): CliArgs {
	let profile: "developer" | "robot" = "developer";
	let initialMessage: string | undefined;
	let print = false;
	let missionId: string | undefined;
	let resumeSessionId: string | undefined;
	let resumeSessionPath: string | undefined;
	let browseSessions = false;
	let continueLast = false;
	let workspace: string | undefined;
	for (let i = 0; i < argv.length; i += 1) {
		if (argv[i] === "--workspace" && argv[i + 1]) {
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
		profile, initialMessage, print, missionId, workspace,
		resumeSessionId, resumeSessionPath, browseSessions, continueLast,
	};
}

async function main(): Promise<number> {
	const { InteractiveMode, runPrintMode, SessionManager } = await import(
		"@earendil-works/pi-coding-agent"
	);
	const { createRosclawRuntime } = await import("./runtime/create-runtime.js");
	const {
		profile, initialMessage, print, missionId, workspace,
		resumeSessionId, resumeSessionPath, browseSessions, continueLast,
	} = parseArgs(process.argv.slice(2));
	const rosclawHome = rosclawHomeEnv;
	// WP-P0-1（总纲 §5.1）：恢复路径全部经 Pi SessionManager 公开
	// API——不再有手写目录扫描/mtime 排序/文件名拼接（Pi 文件名可含
	// 时间前缀，拼接 id 是格式漂移风险）。
	const { WorkspaceStore } = await import("./session/workspace.js");
	const workspaceStore = new WorkspaceStore(rosclawHome);
	// PR-N1：ActiveTaskContext 在 session 创建前解析并冻结——
	// runtime/工具/bridge/artifact/verifier/header 全从这里取路径。
	const { resolveTaskContext } = await import("./native/active-task-context.js");
	let taskContext = resolveTaskContext({
		rosclawHome,
		cwd: process.cwd(),  // 唯一允许的进程 cwd 读取（启动解析输入）
		mode: "SIMULATION",
		explicitWorkspace: workspace,
	});
	// 持久化绑定规则与旧 resolveStartupWorkspace 一致（explicit/git
	// 会 bind；restored/default 不覆盖既有绑定）。
	if (taskContext.workspaceSource === "explicit" || taskContext.workspaceSource === "git") {
		workspaceStore.bind(taskContext.workspaceRoot);
	}
	const startupWs = { bound: workspaceStore.current, auto: taskContext.workspaceSource === "git" };
	let initialSession: import("@earendil-works/pi-coding-agent").SessionManager | undefined;
	const sessionDir = `${rosclawHome}/agent/sessions`;
	if (browseSessions) {
		const { browseSessions: openPicker } = await import("./session/picker.js");
		const picked = await openPicker(
			(onProgress) => SessionManager.list(taskContext.workspaceRoot, sessionDir, onProgress),
			(onProgress) => SessionManager.listAll(sessionDir, onProgress),
		);
		if (!picked) return 0;  // 用户取消——干净退出，不建会话
		initialSession = SessionManager.open(picked, sessionDir);
	} else if (resumeSessionPath) {
		initialSession = SessionManager.open(resumeSessionPath, sessionDir);
	} else if (resumeSessionId) {
		// 兼容路径：`chat --resume <id>`——精确 ID/唯一前缀经
		// SessionManager.listAll 解析（拒绝路径穿越由解析保证）。
		const { resolveSessionQuery } = await import("./session/resolve.js");
		const sessions = await SessionManager.listAll(sessionDir);
		const hit = resolveSessionQuery(resumeSessionId, sessions);
		if (!hit.ok) {
			console.error(
				hit.error === "AMBIGUOUS"
					? `会话不唯一（${hit.candidates.length} 个候选）——请用 rosclaw resume 打开选择器`
					: `会话 ${resumeSessionId} 不存在——rosclaw sessions 查看全部`,
			);
			return 2;
		}
		initialSession = SessionManager.open(hit.path, sessionDir);
	} else if (continueLast) {
		initialSession = SessionManager.continueRecent(taskContext.workspaceRoot, sessionDir);
	}
	const isResume = Boolean(
		resumeSessionId || resumeSessionPath || browseSessions || continueLast,
	);
	// N4.2：resume 的 workspace 以会话记录为准（优先于 cwd 推导）。
	if (isResume && initialSession) {
		const resumedCwd = initialSession.getCwd();
		if (resumedCwd) {
			taskContext = resolveTaskContext({
				rosclawHome,
				cwd: process.cwd(),  // 唯一允许的进程 cwd 读取（启动解析输入）
				mode: "SIMULATION",
				explicitWorkspace: workspace,
				resumedWorkspace: resumedCwd,
			});
		}
	}
	// 十一审 PR-D：Workspace 一等状态——显式 --workspace > cwd git 自动
	// 绑定 > 既有绑定。
	const { runtime, coordinator, leaseManager } = await createRosclawRuntime({
		cwd: taskContext.workspaceRoot,
		taskContext,
		rosclawHome,
		profile,
		version: VERSION,
		workspaceStore,
		workspaceAutoBound: startupWs.auto,
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
	if (missionId) {
		const outcome = await coordinator.attachInitialMission(sessionId, missionId);
		if (!outcome.ok) {
			console.error(`初始 Mission 接入失败：${outcome.reason}`);
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
	try {
		if (print) {
			// 非 TTY 单发模式（冒烟/脚本）。
			return await runPrintMode(runtime, {
				mode: "text",
				...(initialMessage ? { initialMessage } : {}),
			});
		}
		const mode = new InteractiveMode(runtime, {
			verbose: false,
			...(initialMessage ? { initialMessage } : {}),
		});
		await mode.run();
		return 0;
	} finally {
		await leaseManager.release();
	}
}

main().then(
	(code) => process.exit(code),
	(err) => {
		console.error(`rosclaw-agent failed: ${(err as Error).message}`);
		process.exit(2);
	},
);
