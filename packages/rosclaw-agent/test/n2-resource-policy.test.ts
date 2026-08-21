/** PR-N2 红测试：恢复 Pi 的可信上下文与 Skill（N 总纲 §PR-N2）。
 *
 * 红测试先行——修复前必须红：
 * 1. ResourcePolicy 拆分 context/skill/extension/executable 四通道
 *    （旧布尔五连删）；
 * 2. developer：可信 AGENTS.md/CLAUDE.md 允许；任意项目 Skill/扩展/
 *    提示模板仍关；ROSClaw 内置签名 Skill 允许；
 * 3. robot：全关（不变）；
 * 4. 内置 Skill manifest digest 校验——篡改即排除 + 诊断；
 * 5. 上下文信任过滤：出根/超限文件被剔除且带 path+size 诊断。
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";

import {
	resourcePolicy,
	trustFilterContextFiles,
} from "../src/extension/resource-policy.js";
import { verifyBundledSkills } from "../src/extension/bundled-skills.js";

test("N2: developer 拆分策略——可信上下文允许 + 项目可执行仍关", () => {
	const p = resourcePolicy("developer");
	assert.equal(p.contextFiles, "trusted-readonly");
	assert.equal(p.skills, "bundled-signed");
	assert.equal(p.extensions, "off");
	assert.equal(p.promptTemplates, "off");
	assert.equal(p.executables, "off");
});

test("N2: robot 全关不变", () => {
	const p = resourcePolicy("robot");
	assert.equal(p.contextFiles, "off");
	assert.equal(p.skills, "off");
	assert.equal(p.extensions, "off");
	assert.equal(p.promptTemplates, "off");
	assert.equal(p.executables, "off");
});

test("N2: 内置 Skill digest 校验——篡改即排除", () => {
	const root = mkdtempSync(join(tmpdir(), "n2-skill-"));
	const skillDir = join(root, "skills", "rosclaw-embodied");
	mkdirSync(skillDir, { recursive: true });
	const content = "---\nname: rosclaw-embodied\ndescription: 具身任务纪律\n---\n\n# 具身\n";
	writeFileSync(join(skillDir, "SKILL.md"), content, "utf-8");
	const digest = createHash("sha256").update(content).digest("hex");
	writeFileSync(
		join(root, "skills", "manifest.json"),
		JSON.stringify({ skills: { "rosclaw-embodied": digest } }),
		"utf-8",
	);
	const ok = verifyBundledSkills(join(root, "skills"));
	assert.deepEqual(ok.verified, ["rosclaw-embodied"]);
	assert.deepEqual(ok.excluded, []);

	// 篡改后必须排除 + 诊断。
	writeFileSync(join(skillDir, "SKILL.md"), content + "\ntampered\n", "utf-8");
	const bad = verifyBundledSkills(join(root, "skills"));
	assert.deepEqual(bad.verified, []);
	assert.equal(bad.excluded.length, 1);
	assert.match(bad.excluded[0].reason, /digest|hash|篡改|digest 不符/i);
});

test("N2: 上下文信任过滤——出根与超限剔除", () => {
	const root = mkdtempSync(join(tmpdir(), "n2-ctx-"));
	const inside = join(root, "CLAUDE.md");
	writeFileSync(inside, "x".repeat(100), "utf-8");
	const outside = join(tmpdir(), "outside-CLAUDE.md");
	writeFileSync(outside, "evil", "utf-8");
	const result = trustFilterContextFiles(
		[
			{ path: inside, content: "x".repeat(100) },
			{ path: outside, content: "evil" },
			{ path: join(root, "AGENTS.md"), content: "y".repeat(200 * 1024) },
		],
		{ allowedRoots: [root], maxTotalBytes: 64 * 1024 },
	);
	assert.deepEqual(result.kept.map((f) => f.path), [inside]);
	assert.equal(result.diagnostics.length, 2);
	assert.ok(result.diagnostics.some((d) => d.includes(outside)));
	assert.ok(result.diagnostics.some((d) => /预算|budget|超/.test(d)));
});
