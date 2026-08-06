#!/usr/bin/env node
/** rosclaw-tui entry (批次 C)。
 *
 * 用法：rosclaw-tui --url http://127.0.0.1:8787 --mission mis_x [--goal "新任务"]
 * Python CLI（rosclaw chat）负责启动/发现 AgentService 后 exec 本入口。
 */

import { RosclawTuiApp } from "./app.js";
import { AgentClient } from "./client/http.js";

const REQUIRED_NODE = "22.19.0";

function versionAtLeast(current: string, required: string): boolean {
	const cur = current.split(".").map(Number);
	const req = required.split(".").map(Number);
	for (let i = 0; i < 3; i += 1) {
		if ((cur[i] ?? 0) > (req[i] ?? 0)) return true;
		if ((cur[i] ?? 0) < (req[i] ?? 0)) return false;
	}
	return true;
}

function parseArgs(argv: string[]): Record<string, string> {
	const args: Record<string, string> = {};
	for (let i = 0; i < argv.length; i += 1) {
		if (argv[i].startsWith("--")) {
			const key = argv[i].slice(2);
			args[key] = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
		}
	}
	return args;
}

async function main(): Promise<void> {
	if (!versionAtLeast(process.versions.node, REQUIRED_NODE)) {
		console.error(
			`rosclaw-tui 需要 Node >= ${REQUIRED_NODE}（当前 ${process.versions.node}）。\n` +
				"请安装受支持的 Node 运行时；Python-only 安装可用 rosclaw chat --basic。",
		);
		process.exit(2);
	}
	const args = parseArgs(process.argv.slice(2));
	const baseUrl = (args.url ?? "http://127.0.0.1:8787").replace(/\/$/, "");
	const client = new AgentClient(baseUrl);

	let missionId = args.mission;
	try {
		if (!missionId && args.goal) {
			const created = await client.createMission(args.goal, args.mode);
			missionId = created.mission_id;
		}
		if (!missionId) {
			const missions = await client.listMissions();
			const active = missions.filter((m) => m.state !== "FAILED");
			if (active.length === 0) {
				console.error("没有可用 Mission。用 --goal \"任务目标\" 创建，或 rosclaw chat 引导创建。");
				process.exit(2);
			}
			missionId = active[active.length - 1].mission_id;
		}
	} catch (err) {
		console.error(
			`无法连接 AgentService（${baseUrl}）：${(err as Error).message}\n` +
				"请先启动 rosclaw-agentd（rosclaw chat 会自动启动），或用 --url 指定地址。",
		);
		process.exit(1);
	}

	const app = new RosclawTuiApp({ baseUrl, missionId });
	try {
		await app.start();
	} catch (err) {
		console.error((err as Error).message);
		process.exit(1);
	}
}

void main();
