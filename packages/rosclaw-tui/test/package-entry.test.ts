import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = join(here, "..", "..");

/** 审计 §1.1：package.json bin/start 与 tsc 输出结构必须一致——
 * 入口 smoke test 直接执行 npm 声明的路径，防止再次指向不存在的
 * dist/main.js。 */
test("package bin entry exists and is runnable", () => {
	const pkg = JSON.parse(readFileSync(join(pkgRoot, "package.json"), "utf8")) as {
		bin: Record<string, string>;
		scripts: Record<string, string>;
	};
	const entry = join(pkgRoot, pkg.bin["rosclaw-tui"]);
	assert.ok(existsSync(entry), `bin entry missing: ${entry}`);
	// --version 式的快速执行：不带参数启动应立即给出用法/连接错误而非
	// 模块找不到（非 TTY 下会报连接错误，这证明入口确实执行了）。
	let output = "";
	try {
		output = execFileSync(process.execPath, [entry], {
			encoding: "utf8",
			timeout: 15_000,
			stdio: ["ignore", "pipe", "pipe"],
		});
	} catch (err) {
		output = String((err as { stdout?: string; stderr?: string }).stdout ?? "") +
			String((err as { stdout?: string; stderr?: string }).stderr ?? "");
	}
	assert.ok(
		output.includes("AgentService") || output.includes("Mission") || output.includes("rosclaw"),
		`entry did not execute expected code path: ${output.slice(0, 200)}`,
	);
});

test("start script points at the same entry", () => {
	const pkg = JSON.parse(readFileSync(join(pkgRoot, "package.json"), "utf8")) as {
		bin: Record<string, string>;
		scripts: Record<string, string>;
	};
	assert.ok(pkg.scripts.start.includes(pkg.bin["rosclaw-tui"]));
});
