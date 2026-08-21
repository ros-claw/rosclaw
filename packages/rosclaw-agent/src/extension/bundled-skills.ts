/** ROSClaw 内置签名 Skill 校验（PR-N2）。
 *
 * skills/manifest.json 记录每个 SKILL.md 的 sha256——加载前逐一
 * 校验；篡改/缺失 manifest 条目的 Skill 排除并给诊断（诚实降级，
 * 不静默加载）。
 */

import { createHash } from "node:crypto";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

export interface BundledSkillsResult {
	/** digest 校验通过的 skill 名。 */
	verified: string[];
	/** 被排除的 skill（含原因——诚实诊断）。 */
	excluded: Array<{ name: string; reason: string }>;
	/** 可传给 Pi additionalSkillPaths 的目录（仅含通过校验的）。 */
	skillPaths: string[];
}

export function verifyBundledSkills(skillsDir: string): BundledSkillsResult {
	const manifestPath = join(skillsDir, "manifest.json");
	const verified: string[] = [];
	const excluded: Array<{ name: string; reason: string }> = [];
	if (!existsSync(manifestPath)) {
		return {
			verified, skillPaths: [],
			excluded: [{ name: "*", reason: "manifest.json 缺失——全部排除（fail closed）" }],
		};
	}
	let manifest: { skills?: Record<string, string> };
	try {
		manifest = JSON.parse(readFileSync(manifestPath, "utf-8")) as typeof manifest;
	} catch (err) {
		return {
			verified, skillPaths: [],
			excluded: [{ name: "*", reason: `manifest.json 解析失败：${(err as Error).message}` }],
		};
	}
	const declared = manifest.skills ?? {};
	for (const entry of readdirSync(skillsDir, { withFileTypes: true })) {
		if (!entry.isDirectory()) continue;
		const skillFile = join(skillsDir, entry.name, "SKILL.md");
		if (!existsSync(skillFile)) continue;
		const expected = declared[entry.name];
		if (!expected) {
			excluded.push({ name: entry.name, reason: "不在 manifest——未声明的内置 Skill 不加载" });
			continue;
		}
		const actual = createHash("sha256")
			.update(readFileSync(skillFile, "utf-8"))
			.digest("hex");
		if (actual !== expected) {
			excluded.push({ name: entry.name, reason: `digest 不符（manifest ${expected.slice(0, 12)}… != 实际 ${actual.slice(0, 12)}…）——疑似篡改，排除` });
			continue;
		}
		verified.push(entry.name);
	}
	return {
		verified,
		excluded,
		skillPaths: verified.map((name) => join(skillsDir, name)),
	};
}
