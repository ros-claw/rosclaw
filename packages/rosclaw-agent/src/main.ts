#!/usr/bin/env node
/** rosclaw-agent 入口（PNA-0）：Pi InteractiveMode + ROSClaw 品牌。
 *
 * `rosclaw chat --engine pi` 由 Python CLI 转调本入口。
 * Pi 是唯一主认知循环（规格 §2.1）——本进程不启动 Python AgentLoop。
 */

import { InteractiveMode, runPrintMode } from "@earendil-works/pi-coding-agent";
import { createRosclawRuntime } from "./runtime/create-runtime.js";
import { VERSION } from "./version.js";

interface CliArgs {
	profile: "developer" | "robot";
	initialMessage?: string;
	print: boolean;
}

function parseArgs(argv: string[]): CliArgs {
	let profile: "developer" | "robot" = "developer";
	let initialMessage: string | undefined;
	let print = false;
	for (let i = 0; i < argv.length; i += 1) {
		if (argv[i] === "--profile" && argv[i + 1]) {
			profile = argv[i + 1] === "robot" ? "robot" : "developer";
			i += 1;
		} else if (argv[i] === "--message" && argv[i + 1]) {
			initialMessage = argv[i + 1];
			i += 1;
		} else if (argv[i] === "--print") {
			print = true;
		}
	}
	return { profile, initialMessage, print };
}

async function main(): Promise<number> {
	const { profile, initialMessage, print } = parseArgs(process.argv.slice(2));
	const rosclawHome = process.env.ROSCLAW_HOME ?? `${process.env.HOME}/.rosclaw`;
	const runtime = await createRosclawRuntime({
		cwd: process.cwd(),
		rosclawHome,
		profile,
		version: VERSION,
	});
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
}

main().then(
	(code) => process.exit(code),
	(err) => {
		console.error(`rosclaw-agent failed: ${(err as Error).message}`);
		process.exit(2);
	},
);
