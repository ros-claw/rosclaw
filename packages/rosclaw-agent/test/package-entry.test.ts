import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";

import pkg from "../package.json" with { type: "json" };

test("package bin entry exists and dist is built", () => {
	const bin = pkg.bin["rosclaw-agent"];
	assert.ok(bin, "bin.rosclaw-agent must be declared");
	assert.ok(existsSync(join(import.meta.dirname, "..", "..", bin)), `${bin} must exist after build`);
});

test("pi dependencies are exactly pinned (no ^ ranges)", () => {
	// W08：pi 单源版本——deps/overrides 全部钉同一版本（0.83.0→0.85.1
	// 升级走 pi_upgrade_candidate.sh + 契约层验证后改这里）。
	const PI_PIN = "0.85.1";
	for (const [name, version] of Object.entries(pkg.dependencies)) {
		assert.ok(!version.startsWith("^") && !version.startsWith("~"), `${name} must be exact-pinned`);
		assert.equal(version, PI_PIN, `${name} must be ${PI_PIN}`);
	}
	for (const version of Object.values(pkg.overrides)) {
		assert.equal(version, PI_PIN);
	}
});

test("node engine floor matches Pi requirement", () => {
	assert.equal(pkg.engines.node, ">=22.19.0");
});
