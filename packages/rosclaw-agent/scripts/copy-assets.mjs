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
