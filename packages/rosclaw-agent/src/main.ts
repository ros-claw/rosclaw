#!/usr/bin/env node
/** rosclaw-agent 入口（PNA-0）：Pi InteractiveMode + ROSClaw 品牌。
 *
 * `rosclaw chat --engine pi` 由 Python CLI 转调本入口。
 * Pi 是唯一主认知循环（规格 §2.1）——本进程不启动 Python AgentLoop。
 */

// PI_CODING_AGENT_DIR 必须在任何 pi 模块加载前设定（config.js 在
// import 期读取；ESM 静态 import 会被提升）——所有 pi 相关模块一律
// 动态 import。
import { VERSION } from "./version.js";

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
	const { SessionLeaseManager } = await import("./session/lease-manager.js");
	const { profile, initialMessage, print, missionId, resumeSessionId, continueLast } = parseArgs(
		process.argv.slice(2),
	);
	const rosclawHome = rosclawHomeEnv;
	// NA-FIX-2：--resume/--continue 走 SessionManager.open（不新建 session
	// 文件、不预建 Mission——由扩展的 switch 事务接管绑定）。
	let initialSession: import("@earendil-works/pi-coding-agent").SessionManager | undefined;
	if (resumeSessionId || continueLast) {
		const sessionDir = `${rosclawHome}/agent/sessions`;
		const { readdirSync } = await import("node:fs");
		let sessionFile = "";
		if (resumeSessionId) {
			const { join } = await import("node:path");
			sessionFile = join(sessionDir, `${resumeSessionId}.jsonl`);
		} else {
			// --continue：最近的 session 文件。
			const files = readdirSync(sessionDir)
				.filter((f) => f.endsWith(".jsonl"))
				.sort()
				.reverse();
			sessionFile = files[0] ? `${sessionDir}/${files[0]}` : "";
		}
		if (sessionFile) {
			initialSession = SessionManager.open(sessionFile, sessionDir);
		}
	}
	const runtime = await createRosclawRuntime({
		cwd: process.cwd(),
		rosclawHome,
		profile,
		version: VERSION,
		...(missionId ? { missionId } : {}),
		...(initialSession ? { sessionManager: initialSession } : {}),
	});
	// NA-FIX-2：bind/heartbeat 由 SessionLeaseManager 统一管理
	// （切换事务复用同一管理点，lease_token 绝不丢弃）。
	const leaseManager = new SessionLeaseManager(rosclawHome);
	if (missionId) {
		await leaseManager.bind(runtime.session.sessionManager.getSessionId(), missionId);
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
