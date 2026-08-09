#!/usr/bin/env node
/** rosclaw-agent 入口（PNA-0）：Pi InteractiveMode + ROSClaw 品牌。
 *
 * `rosclaw chat --engine pi` 由 Python CLI 转调本入口。
 * Pi 是唯一主认知循环（规格 §2.1）——本进程不启动 Python AgentLoop。
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
	initialMessage?: string;
	print: boolean;
	missionId?: string;
	resumeSessionId?: string;
	continueLast: boolean;
}

function parseArgs(argv: string[]): CliArgs {
	let profile: "developer" | "robot" = "developer";
	let initialMessage: string | undefined;
	let print = false;
	let missionId: string | undefined;
	let resumeSessionId: string | undefined;
	let continueLast = false;
	for (let i = 0; i < argv.length; i += 1) {
		if (argv[i] === "--profile" && argv[i + 1]) {
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
		} else if (argv[i] === "--continue" || argv[i] === "-c") {
			continueLast = true;
		}
	}
	return { profile, initialMessage, print, missionId, resumeSessionId, continueLast };
}

async function main(): Promise<number> {
	const { InteractiveMode, runPrintMode, SessionManager } = await import(
		"@earendil-works/pi-coding-agent"
	);
	const { createRosclawRuntime } = await import("./runtime/create-runtime.js");
	const { profile, initialMessage, print, missionId, resumeSessionId, continueLast } = parseArgs(
		process.argv.slice(2),
	);
	const rosclawHome = rosclawHomeEnv;
	// NA-FIX-2：--resume/--continue 走 SessionManager.open（不新建 session
	// 文件、不预建 Mission——由 coordinator 的 resumeInitial 事务接管绑定）。
	// P0-NA-12：session id 必须是纯标识符（拒绝 ../ 路径穿越与任意 JSONL）；
	// --continue 按文件 mtime 选最近 session，不按文件名排序。
	let initialSession: import("@earendil-works/pi-coding-agent").SessionManager | undefined;
	if (resumeSessionId || continueLast) {
		const sessionDir = `${rosclawHome}/agent/sessions`;
		const { readdirSync, statSync, existsSync } = await import("node:fs");
		let sessionFile = "";
		if (resumeSessionId) {
			if (!/^[A-Za-z0-9_-]+$/.test(resumeSessionId)) {
				console.error(`非法 session id：${resumeSessionId}（只允许字母数字-_）`);
				return 2;
			}
			const { join } = await import("node:path");
			sessionFile = join(sessionDir, `${resumeSessionId}.jsonl`);
			if (!existsSync(sessionFile)) {
				console.error(
					`session ${resumeSessionId} 不存在（${sessionDir} 下无此记录）——` +
					"未启动未绑定会话。用 `rosclaw chat` 开新会话或 `rosclaw chat --continue` 恢复最近会话。",
				);
				return 2;
			}
		} else {
			// --continue：mtime 最新且头可解析的 session 文件。
			const candidates = readdirSync(sessionDir)
				.filter((f) => f.endsWith(".jsonl"))
				.map((f) => ({ f, mtime: statSync(`${sessionDir}/${f}`).mtimeMs }))
				.sort((a, b) => b.mtime - a.mtime);
			sessionFile = candidates[0] ? `${sessionDir}/${candidates[0].f}` : "";
		}
		if (sessionFile) {
			initialSession = SessionManager.open(sessionFile, sessionDir);
		}
	}
	const { runtime, coordinator, leaseManager } = await createRosclawRuntime({
		cwd: process.cwd(),
		rosclawHome,
		profile,
		version: VERSION,
		...(missionId ? { missionId } : {}),
		...(initialSession ? { sessionManager: initialSession } : {}),
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
	} else if (resumeSessionId || continueLast) {
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
