import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";

import pkg from "../package.json" with { type: "json" };

test("package bin entry exists and dist is built", () => {
	const bin = pkg.bin["rosclaw-agent"];
	assert.ok(bin, "bin.rosclaw-agent must be declared");
	assert.ok(existsSync(join(import.meta.dirname, "..", "..", bin)), `${bin} must exist after build`);
});

test("pi dependencies are exactly pinned (no ^ ranges)", () => {
	// W08：deps/overrides 全部精确钉版；升级必须同步核对全部 PI 包。
	const PI_PIN = "1.0.4";
	for (const [name, version] of Object.entries(pkg.dependencies)) {
		assert.ok(!version.startsWith("^") && !version.startsWith("~"), `${name} must be exact-pinned`);
		assert.equal(version, PI_PIN, `${name} must be ${PI_PIN}`);
		const installed = JSON.parse(readFileSync(join(import.meta.dirname, "..", "..", "node_modules", name, "package.json"), "utf8"));
		assert.equal(installed.version, PI_PIN, `${name} installed runtime must match the pin`);
	}
	for (const version of Object.values(pkg.overrides)) {
		assert.equal(version, PI_PIN);
	}
});

test("node engine floor matches Pi requirement", () => {
	assert.equal(pkg.engines.node, ">=22.19.0");
});
