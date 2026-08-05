#!/usr/bin/env node
/** rosclaw-agent 入口（PNA-0）：Pi InteractiveMode + ROSClaw 品牌。
 *
 * `rosclaw chat --engine pi` 由 Python CLI 转调本入口。
 * Pi 是唯一主认知循环（规格 §2.1）——本进程不启动 Python AgentLoop。
 */

import { InteractiveMode, runPrintMode } from "@earendil-works/pi-coding-agent";
import { createRosclawRuntime } from "./runtime/create-runtime.js";
import { bindSession, releaseSession, type SessionBinding } from "./session/binding.js";
import { VERSION } from "./version.js";

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
	const { profile, initialMessage, print, missionId } = parseArgs(process.argv.slice(2));
	const rosclawHome = process.env.ROSCLAW_HOME ?? `${process.env.HOME}/.rosclaw`;
	const runtime = await createRosclawRuntime({
		cwd: process.cwd(),
		rosclawHome,
		profile,
		version: VERSION,
		...(missionId ? { missionId } : {}),
	});
	// PNA-1：启动即绑定 Mission + 获取 writer lease（失败即退出——
	// 不猜绑定、不无 lease 运行，规格 §13.1/§12）。
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
