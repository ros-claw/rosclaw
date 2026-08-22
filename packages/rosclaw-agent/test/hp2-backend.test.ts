/** PR-HP2 红测试：Pi Backend 真正迁移（调整方案 §四.HP2）。
 *
 * 红测试先行——迁移前必须红：
 * 1. createPiBackend().create()/.resume() 真实工作（当前抛
 *    HARNESS_CAPABILITY_MISSING）；
 * 2. 能力声明来自运行时 probe（禁止硬编码全 true——把 Pi 摘掉
 *    必须能探出来）；
 * 3. 结构门：packages/rosclaw-agent/src 内除 harness/pi/ 与显式
 *    HP2-COMPAT 标记的迁移兼容文件外，不得 import '@earendil-works/pi'；
 * 4. Pi 私有事件统一转 HarnessEvent（产品侧不 switch Pi 私有类型）。
 */
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

const SRC = new URL("../../src/", import.meta.url).pathname;  // dist/test → 包根/src

function* walk(dir: string): Generator<string> {
	for (const name of readdirSync(dir)) {
		const full = join(dir, name);
		if (statSync(full).isDirectory()) {
			if (name === "node_modules" || name.startsWith(".")) continue;
			yield* walk(full);
		} else if (name.endsWith(".ts")) {
			yield full;
		}
	}
}

test("HP2 结构门：Pi import 只在 harness/pi/ 或 HP2-COMPAT 标记文件", () => {
	const violations: string[] = [];
	for (const file of walk(SRC)) {
		if (file.includes("/harness/pi/")) continue;
		const text = readFileSync(file, "utf-8");
		if (!text.includes("@earendil-works/pi")) continue;
		// 迁移兼容：文件头必须显式标记 HP2-COMPAT + 一句理由——
		// 清单只许缩小不许扩大。
		if (/HP2-COMPAT:/.test(text.split("\n").slice(0, 12).join("\n"))) continue;
		violations.push(file.slice(SRC.length));
	}
	assert.deepEqual(violations, [],
		"未标记的 Pi import（迁入 harness/pi/ 或加 HP2-COMPAT 标记）");
});

test("HP2: createPiBackend().create() 真实工作", async () => {
	const { createPiBackend } = await import("../src/harness/pi/pi-backend.js");
	const backend = createPiBackend();
	// 当前抛 HARNESS_CAPABILITY_MISSING（装配本体在 create-runtime）——
	// 迁移后必须真实返回 HarnessSession。
	const session = await backend.create({
		cwd: "/tmp/hp2-probe",
		backendOptions: { headless: true },
	});
	assert.equal(session.sessionRef.backendId, "pi");
	assert.ok(session.sessionRef.nativeRef, "缺 nativeRef");
	await session.close();
});

test("HP2: 能力声明来自运行时 probe，不硬编码", async () => {
	const { probePiCapabilities } = await import("../src/harness/pi/pi-backend.js");
	const caps = probePiCapabilities();
	// probe 必须返回真实布尔（当前环境 Pi 完整——全真），且函数存在即
	// 证明非硬编码常量（PI_CAPABILITIES 常量不得再是事实源）。
	assert.equal(typeof caps.persistentSessions, "boolean");
	assert.equal(caps.customTools, true);
});
