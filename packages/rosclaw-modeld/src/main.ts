#!/usr/bin/env node
/** rosclaw-modeld entry (批次 D)。
 *
 * 由 rosclaw-agentd 以子进程启动：
 *   ROSCLAW_MODELD_TOKEN=<random> rosclaw-modeld --socket <path> --home <dir>
 * token 只经进程环境传递；绝不作为命令行参数（ps 可见）。
 */

import { startModeld } from "./server.js";

function parseArgs(argv: string[]): Record<string, string> {
	const args: Record<string, string> = {};
	for (let i = 0; i < argv.length; i += 1) {
		if (argv[i].startsWith("--")) {
			args[argv[i].slice(2)] = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
		}
	}
	return args;
}

async function main(): Promise<void> {
	const [major, minor] = process.versions.node.split(".").map(Number);
	if (major < 22 || (major === 22 && minor < 19)) {
		console.error(`rosclaw-modeld 需要 Node >= 22.19.0（当前 ${process.versions.node}）`);
		process.exit(2);
	}
	const args = parseArgs(process.argv.slice(2));
	const token = process.env.ROSCLAW_MODELD_TOKEN;
	if (!token) {
		console.error("缺少 ROSCLAW_MODELD_TOKEN（应由 agentd 启动时注入）");
		process.exit(2);
	}
	const socketPath = args.socket;
	const homeDir = args.home;
	if (!socketPath || !homeDir) {
		console.error("用法: rosclaw-modeld --socket <path> --home <dir>");
		process.exit(2);
	}
	await startModeld({ socketPath, token, homeDir });
	console.error(`rosclaw-modeld listening on ${socketPath}`);
}

void main();
