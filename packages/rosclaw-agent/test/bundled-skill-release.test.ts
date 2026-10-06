/** Bundled-skill release regression.
 *
 * 维护性发布回归：manifest.json 是无签名 sha256 digest 清单（不是
 * 签名证书），加载/打包前必须逐一校验。覆盖：
 * 1. 实际打包内容——skills/ 源与 dist/skills/ 拷贝均通过 digest 校验，
 *    且 manifest 声明与实际文件 digest 一致；
 * 2. 负例：篡改 digest、未声明的 Skill 目录、manifest 缺失——
 *    全部 fail closed（排除 + 诊断，绝不静默加载）。
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import {
	appendFileSync,
	cpSync,
	existsSync,
	mkdtempSync,
	readFileSync,
	rmSync,
	writeFileSync,
	mkdirSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";

import { verifyBundledSkills } from "../src/extension/bundled-skills.js";

const pkgRoot = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

function sha256(path: string): string {
	return createHash("sha256").update(readFileSync(path, "utf-8")).digest("hex");
}

function declaredDigest(skillsDir: string, name: string): string {
	const manifest = JSON.parse(
		readFileSync(join(skillsDir, "manifest.json"), "utf-8"),
	) as { skills: Record<string, string> };
	return manifest.skills[name];
}

for (const rel of ["skills", "dist/skills"]) {
	test(`release: 实际打包内容 ${rel}/ digest 校验通过`, () => {
		const dir = join(pkgRoot, rel);
		const result = verifyBundledSkills(dir);
		assert.deepEqual(result.verified, ["rosclaw-embodied"]);
		assert.deepEqual(result.excluded, []);
		// manifest 声明即实际文件 digest（digest 清单，非签名）。
		assert.equal(
			declaredDigest(dir, "rosclaw-embodied"),
			sha256(join(dir, "rosclaw-embodied", "SKILL.md")),
		);
		assert.equal(result.skillPaths.length, 1);
	});
}

test("release: 篡改 SKILL.md —— digest 不符即排除", () => {
	const dir = join(mkdtempSync(join(tmpdir(), "bundle-release-")), "skills");
	try {
		cpSync(join(pkgRoot, "skills"), dir, { recursive: true });
		appendFileSync(join(dir, "rosclaw-embodied", "SKILL.md"), "\nTAMPER\n");
		const result = verifyBundledSkills(dir);
		assert.deepEqual(result.verified, []);
		assert.equal(result.skillPaths.length, 0);
		assert.ok(result.excluded.some((x) => x.name === "rosclaw-embodied" && /digest/.test(x.reason)));
	} finally {
		rmSync(dirname(dir), { recursive: true, force: true });
	}
});

test("release: 未声明的 Skill 目录 —— 不自动信任", () => {
	const dir = join(mkdtempSync(join(tmpdir(), "bundle-release-")), "skills");
	try {
		cpSync(join(pkgRoot, "skills"), dir, { recursive: true });
		mkdirSync(join(dir, "undeclared"));
		writeFileSync(join(dir, "undeclared", "SKILL.md"), "---\nname: undeclared\n---\n");
		const result = verifyBundledSkills(dir);
		assert.ok(!result.verified.includes("undeclared"));
		assert.ok(!result.skillPaths.some((p) => p.includes("undeclared")));
		assert.ok(result.excluded.some((x) => x.name === "undeclared"));
	} finally {
		rmSync(dirname(dir), { recursive: true, force: true });
	}
});

test("release: manifest 缺失 —— 全部排除 fail closed", () => {
	const dir = join(mkdtempSync(join(tmpdir(), "bundle-release-")), "skills");
	try {
		cpSync(join(pkgRoot, "skills"), dir, { recursive: true });
		rmSync(join(dir, "manifest.json"));
		const result = verifyBundledSkills(dir);
		assert.deepEqual(result.verified, []);
		assert.equal(result.skillPaths.length, 0);
		assert.ok(result.excluded.length > 0);
	} finally {
		rmSync(dirname(dir), { recursive: true, force: true });
	}
});

/** copy-assets 打包子进程 fixture：最小 pkgRoot 布局（scripts/skills/prompts）。 */
function makeCopyAssetsFixture(): { root: string; pkg: string } {
	const root = mkdtempSync(join(tmpdir(), "bundle-copy-edges-"));
	const pkg = join(root, "pkg");
	mkdirSync(join(pkg, "scripts"), { recursive: true });
	cpSync(join(pkgRoot, "scripts", "copy-assets.mjs"), join(pkg, "scripts", "copy-assets.mjs"));
	mkdirSync(join(pkg, "prompts"), { recursive: true });
	writeFileSync(join(pkg, "prompts", "native_agent_v2.md"), "prompt\n");
	cpSync(join(pkgRoot, "skills"), join(pkg, "skills"), { recursive: true });
	return { root, pkg };
}

function runCopyAssets(pkg: string) {
	return spawnSync(process.execPath, [join(pkg, "scripts", "copy-assets.mjs")], {
		encoding: "utf-8",
	});
}

test("release: 整个 skills 源目录缺失 —— 构建 fail closed", () => {
	const { root, pkg } = makeCopyAssetsFixture();
	try {
		rmSync(join(pkg, "skills"), { recursive: true, force: true });
		const run = runCopyAssets(pkg);
		assert.notEqual(run.status, 0);
		assert.match(run.stderr, /skills source directory missing/);
		assert.ok(!existsSync(join(pkg, "dist", "skills")));
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
});

test("release: dist/skills 是源的精确替换——历史 Skill 目录不残留", () => {
	const { root, pkg } = makeCopyAssetsFixture();
	try {
		// 首次发布：声明一个合成附加 Skill（fixture，非新产品信任）。
		const manifestPath = join(pkg, "skills", "manifest.json");
		const manifest = JSON.parse(readFileSync(manifestPath, "utf-8")) as {
			skills: Record<string, string>;
		};
		mkdirSync(join(pkg, "skills", "synthetic-extra"));
		writeFileSync(join(pkg, "skills", "synthetic-extra", "SKILL.md"), "---\nname: synthetic-extra\n---\n");
		manifest.skills["synthetic-extra"] = sha256(
			join(pkg, "skills", "synthetic-extra", "SKILL.md"),
		);
		writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
		assert.equal(runCopyAssets(pkg).status, 0);
		assert.ok(existsSync(join(pkg, "dist", "skills", "synthetic-extra", "SKILL.md")));

		// 下一次发布：源移除该 Skill（manifest 同步）——dist 必须精确替换。
		rmSync(join(pkg, "skills", "synthetic-extra"), { recursive: true, force: true });
		delete manifest.skills["synthetic-extra"];
		writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
		assert.equal(runCopyAssets(pkg).status, 0);
		assert.ok(!existsSync(join(pkg, "dist", "skills", "synthetic-extra")));
		const result = verifyBundledSkills(join(pkg, "dist", "skills"));
		assert.deepEqual(result.verified, ["rosclaw-embodied"]);
		assert.deepEqual(result.excluded, []);
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
});

test("release: 失败隔离——无效源不破坏已验证的 dist 目标", () => {
	const { root, pkg } = makeCopyAssetsFixture();
	try {
		assert.equal(runCopyAssets(pkg).status, 0);
		assert.deepEqual(verifyBundledSkills(join(pkg, "dist", "skills")).verified, [
			"rosclaw-embodied",
		]);
		// 篡改源后构建失败，既有有效 dist 目标保持完好。
		appendFileSync(join(pkg, "skills", "rosclaw-embodied", "SKILL.md"), "\nTAMPER\n");
		const run = runCopyAssets(pkg);
		assert.notEqual(run.status, 0);
		assert.match(run.stderr, /stale digest/);
		assert.deepEqual(verifyBundledSkills(join(pkg, "dist", "skills")).verified, [
			"rosclaw-embodied",
		]);
	} finally {
		rmSync(root, { recursive: true, force: true });
	}
});
