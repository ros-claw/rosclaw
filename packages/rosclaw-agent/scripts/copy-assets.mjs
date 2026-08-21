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
import { cpSync } from "node:fs";
const skillsSource = join(pkgRoot, "skills");
const skillsTarget = join(pkgRoot, "dist", "skills");
if (existsSync(skillsSource)) {
	cpSync(skillsSource, skillsTarget, { recursive: true });
	console.log("copied skills");
}
