/**
 * NATIVE_BASE_G2_LOW_V2 regression (native-authored, SOURCE-only).
 *
 * Pins the two documented wirings in the BUILT pi-runtime:
 *  1. native_agent_v2.md is composed as the ResourceLoader custom system
 *     base (systemPromptOverride) so the SDK-assembled leading system
 *     contains the native base exactly once (extension no longer replaces
 *     event.systemPrompt wholesale).
 *  2. Arbitrary project skill discovery is closed (noSkills: true);
 *     developer skills enter only via explicit digest-verified
 *     additionalSkillPaths (bundled-signed, fail-closed).
 */
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

const here = dirname(fileURLToPath(import.meta.url));

function firstExisting(candidates: string[]): string {
	for (const c of candidates) {
		if (existsSync(c)) return c;
	}
	throw new Error(`none of the candidates exist: ${candidates.join(", ")}`);
}

const runtimePath = firstExisting([
	join(here, "..", "src", "harness", "pi", "pi-runtime.js"),
	join(here, "..", "harness", "pi", "pi-runtime.js"),
]);
const promptPath = firstExisting([
	join(here, "..", "prompts", "native_agent_v2.md"),
	join(here, "..", "..", "prompts", "native_agent_v2.md"),
]);

test("native system base exists and carries native identity/authority markers", () => {
	const prompt = readFileSync(promptPath, "utf-8");
	assert.ok(prompt.length > 200, "native base must be substantive");
	assert.match(prompt, /ROSClaw/, "native base must carry ROSClaw native identity");
});

test("built pi-runtime wires native base as ResourceLoader custom system base", () => {
	const src = readFileSync(runtimePath, "utf-8");
	assert.ok(
		src.includes("systemPromptOverride"),
		"resourceLoaderOptions must compose native base via systemPromptOverride",
	);
	// exactly one composition site — no duplicate base injection paths
	const occurrences = src.split("systemPromptOverride").length - 1;
	assert.equal(occurrences, 1, "native base composition must be defined exactly once");
});

test("built pi-runtime closes arbitrary project skill discovery (bundled-only)", () => {
	const src = readFileSync(runtimePath, "utf-8");
	assert.ok(/noSkills:\s*true/.test(src), "noSkills must be unconditionally true");
	assert.ok(
		!/noSkills:\s*policy\.skills/.test(src),
		"project skill discovery must not be re-enabled by profile",
	);
	assert.ok(
		src.includes("additionalSkillPaths"),
		"verified bundled-signed skills still enter via explicit additionalSkillPaths",
	);
});
