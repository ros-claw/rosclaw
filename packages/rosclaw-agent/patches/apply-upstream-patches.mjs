/** 上游薄补丁应用器（NA-FIX-3）：patch-package 风格。
 *
 * 每个补丁：anchor（必须精确存在）→ replacement。锚点缺失即硬失败——
 * 静默不生效比构建失败更危险。
 */

import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const target = join(
	here, "..", "node_modules", "@earendil-works", "pi-coding-agent",
	"dist", "modes", "interactive", "interactive-mode.js",
);

const PATCHES = [
	{
		name: "patch-01: rosclaw resume command formatter",
		anchor:
			"    const args = [APP_NAME];\n" +
			"    if (!sessionManager.usesDefaultSessionDir()) {\n" +
			"        args.push(\"--session-dir\", quoteIfNeeded(sessionManager.getSessionDir()));\n" +
			"    }\n" +
			"    args.push(\"--session\", sessionManager.getSessionId());",
		replacement:
			"    // ROSCLAW-PATCH-01: AppIdentity resumeCommandFormatter——恢复必须\n" +
			"    // 经 ROSClaw runtime（kernel/binding/lease/policy），绝不引导外部 CLI。\n" +
			"    return `rosclaw chat --resume ${sessionManager.getSessionId()}`;\n" +
			"    const args = [APP_NAME];\n" +
			"    if (!sessionManager.usesDefaultSessionDir()) {\n" +
			"        args.push(\"--session-dir\", quoteIfNeeded(sessionManager.getSessionDir()));\n" +
			"    }\n" +
			"    args.push(\"--session\", sessionManager.getSessionId());",
	},
	{
		name: "patch-02: builtin command policy guard",
		anchor:
			"            text = text.trim();\n" +
			"            if (!text)\n" +
			"                return;\n" +
			"            // Handle commands",
		replacement:
			"            text = text.trim();\n" +
			"            if (!text)\n" +
			"                return;\n" +
			"            // ROSCLAW-PATCH-02: BuiltinCommandPolicy——内建 dispatch 前置拦截\n" +
			"            // （ROBOT profile 的 /trust /share /import /reload 语义由 ROSClaw 管理）。\n" +
			"            { const cmd = text.split(/\\s/, 1)[0].toLowerCase();\n" +
			"              if (globalThis.__rosclawBuiltinPolicy?.disabled?.has(cmd)) {\n" +
			"                  this.editor.setText(\"\");\n" +
			"                  this.chatContainer.addChild(new Text(theme.fg(\"warning\", `${cmd} is disabled by ROSClaw policy in this profile`), 1, 0));\n" +
			"                  return;\n" +
			"              } }\n" +
			"            // Handle commands",
	},
];

let source = readFileSync(target, "utf8");
for (const patch of PATCHES) {
	if (source.includes(patch.replacement)) {
		console.log(`[already applied] ${patch.name}`);
		continue;
	}
	if (!source.includes(patch.anchor)) {
		console.error(`PATCH ANCHOR MISSING: ${patch.name}`);
		console.error("上游已漂移——升级 Pi 后必须人工复核补丁锚点（见 patches/README.md）。");
		process.exit(1);
	}
	source = source.replace(patch.anchor, patch.replacement);
	console.log(`[applied] ${patch.name}`);
}
writeFileSync(target, source);
console.log("upstream patches applied");
