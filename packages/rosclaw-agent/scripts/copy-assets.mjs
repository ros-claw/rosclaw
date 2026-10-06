import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = join(here, "..");
const source = join(pkgRoot, "..", "..", "src", "rosclaw", "agentd", "context", "prompts", "native_agent_v2.md");
const fallback = join(pkgRoot, "prompts", "native_agent_v2.md");
const targetDir = join(pkgRoot, "dist", "prompts");
mkdirSync(targetDir, { recursive: true });
const from = existsSync(source) ? source : fallback;
if (!existsSync(from)) {
	console.error("native_agent_v2.md not found (python tree or package fallback)");
	process.exit(1);
}
copyFileSync(from, join(targetDir, "native_agent_v2.md"));
console.log(`copied prompt from ${from}`);

// PR-N2：内置签名 Skill 进 dist（dist/skills/）。
import { cpSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { createHash } from "node:crypto";
const skillsSource = join(pkgRoot, "skills");
const skillsTarget = join(pkgRoot, "dist", "skills");
if (!existsSync(skillsSource)) {
	// 整个 skills 源目录缺失也必须 fail closed——绝不静默发布
	// 未验证/陈旧的 dist skills（与 manifest 缺失同等处理）。
	console.error("skills source directory missing — refusing to publish unverified/stale dist skills");
	process.exit(1);
}
{
	// 发布门禁（fail closed）：manifest 是无签名 sha256 digest 清单。
	// 复制前先逐一校验——stale digest、未声明的 Skill 目录、manifest
	// 缺失/声明缺失一律构建失败，绝不自动信任被改动的内容。
	const manifestPath = join(skillsSource, "manifest.json");
	if (!existsSync(manifestPath)) {
		console.error("skills/manifest.json missing — refusing to bundle unverified skills");
		process.exit(1);
	}
	const declared = JSON.parse(readFileSync(manifestPath, "utf-8")).skills ?? {};
	const present = readdirSync(skillsSource, { withFileTypes: true })
		.filter((e) => e.isDirectory() && existsSync(join(skillsSource, e.name, "SKILL.md")))
		.map((e) => e.name);
	for (const name of present) {
		const expected = declared[name];
		if (!expected) {
			console.error(`undeclared bundled skill "${name}" — refusing to auto-trust`);
			process.exit(1);
		}
		const actual = createHash("sha256")
			.update(readFileSync(join(skillsSource, name, "SKILL.md"), "utf-8"))
			.digest("hex");
		if (actual !== expected) {
			console.error(`stale digest for "${name}" (manifest ${expected.slice(0, 12)}… != actual ${actual.slice(0, 12)}…)`);
			process.exit(1);
		}
	}
	for (const name of Object.keys(declared)) {
		if (!present.includes(name)) {
			console.error(`declared skill "${name}" missing from skills/`);
			process.exit(1);
		}
	}
	// 校验全部通过后才替换目标——dist/skills 是源的精确拷贝，
	// 先清空以移除历史遗留的 Skill 目录；校验失败时保留既有有效目标。
	rmSync(skillsTarget, { recursive: true, force: true });
	cpSync(skillsSource, skillsTarget, { recursive: true });
	console.log("copied skills (digest-verified)");
}

// PR-N5C：生成的 effect 表进 dist（单一 Effect Contract——由 Python
// Capability Registry 生成，见 effects.generated.json 头注）。
const effectsSource = join(pkgRoot, "src", "tools", "effects.generated.json");
const effectsTargetDir = join(pkgRoot, "dist", "src", "tools");
if (existsSync(effectsSource)) {
	mkdirSync(effectsTargetDir, { recursive: true });
	copyFileSync(effectsSource, join(effectsTargetDir, "effects.generated.json"));
	console.log("copied effects.generated.json");
}
