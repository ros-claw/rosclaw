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
}

function parseArgs(argv: string[]): CliArgs {
	let profile: "developer" | "robot" = "developer";
	let initialMessage: string | undefined;
	let print = false;
	let missionId: string | undefined;
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
		}
	}
	return { profile, initialMessage, print, missionId };
}

async function main(): Promise<number> {
	const { InteractiveMode, runPrintMode } = await import("@earendil-works/pi-coding-agent");
	const { createRosclawRuntime } = await import("./runtime/create-runtime.js");
	const { bindSession, releaseSession } = await import("./session/binding.js");
	const { profile, initialMessage, print, missionId } = parseArgs(process.argv.slice(2));
	const rosclawHome = rosclawHomeEnv;
	const runtime = await createRosclawRuntime({
		cwd: process.cwd(),
		rosclawHome,
		profile,
		version: VERSION,
		...(missionId ? { missionId } : {}),
	});
	// PNA-1：启动即绑定 Mission + 获取 writer lease（失败即退出——
	// 不猜绑定、不无 lease 运行，规格 §13.1/§12）。
	type SessionBinding = Awaited<ReturnType<typeof bindSession>>;
	let binding: SessionBinding | undefined;
	if (missionId) {
		binding = await bindSession(
			rosclawHome,
			runtime.session.sessionManager.getSessionId(),
			missionId,
		);
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
		if (binding) await releaseSession(rosclawHome, binding);
	}
}

main().then(
	(code) => process.exit(code),
	(err) => {
		console.error(`rosclaw-agent failed: ${(err as Error).message}`);
		process.exit(2);
	},
);
