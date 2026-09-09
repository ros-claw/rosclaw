/** 上游薄补丁应用器（NA-FIX-3 + HOTFIX-4）：patch-package 风格。
 *
 * 每个补丁：anchor（必须精确存在）→ replacement。锚点缺失即硬失败——
 * 静默不生效比构建失败更危险。
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const T = (...parts) => join(here, "..", "node_modules", ...parts);

const PATCHES = [
	{
		// WP-P0-1：清理旧版 hint（'rosclaw chat --resume <id>'）——旧
		// patch 已应用的 node_modules 会残留旧 return（先 return 先生效）。
		// optionalAnchor：全新 npm ci 的 pristine 上游没有这行，跳过。
		target: T("@earendil-works", "pi-coding-agent", "dist", "modes", "interactive", "interactive-mode.js"),
		name: "patch-01b: retire pre-WP-P0-1 resume hint",
		optionalAnchor: true,
		anchor:
			"    return `rosclaw chat --resume ${sessionManager.getSessionId()}`;\n",
		replacement:
			"    // (retired by WP-P0-1: product resume hint below)\n",
	},
	{
		target: T("@earendil-works", "pi-coding-agent", "dist", "modes", "interactive", "interactive-mode.js"),
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
			"    // WP-P0-1：退出提示只给产品命令——不暴露内部 session id。\n" +
			"    const __name = sessionManager.getSessionName?.() || '';\n" +
			"    return `会话已保存${__name ? '：' + __name : ''}\\n继续：rosclaw continue\\n查看全部：rosclaw sessions`;\n" +
			"    // ROSCLAW-PATCH-01-APPLIED\n" +
			"    const args = [APP_NAME];\n" +
			"    if (!sessionManager.usesDefaultSessionDir()) {\n" +
			"        args.push(\"--session-dir\", quoteIfNeeded(sessionManager.getSessionDir()));\n" +
			"    }\n" +
			"    args.push(\"--session\", sessionManager.getSessionId());",
	},
	{
		target: T("@earendil-works", "pi-coding-agent", "dist", "modes", "interactive", "interactive-mode.js"),
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
	// -- 大道至简 W01（规格 §5.2）：patch-03/04 退役 ----------------------
	// TranscriptPolicy 全局删除补丁（写前剥离 + 绝不回放）违反 thinking
	// 模型的官方 continuation 契约（Kimi/Claude 都要求按原样保留历史
	// reasoning）。新政策：provider continuation 由上游按官方协议管理；
	// 会话存储是受控区域；泄漏面（UI/非协议字段/遥测）由 journey 的
	// _assert_reasoning_protocol 钉死。以下为退役-恢复条目：已打过
	// 补丁的 node_modules 恢复为上游原文；pristine 环境自然 no-op。
	{
		target: T("@earendil-works", "pi-coding-agent", "dist", "core", "session-manager.js"),
		name: "retire patch-03 (delete strip block)",
		retire: true,
		optionalAnchor: true,
		anchor:
			"        // ROSCLAW-PATCH-03: TranscriptPolicy——raw reasoning 不持久化。\n" +
			"        // thinking/reasoning 只用于内存瞬态进度；写 session 前剥离\n" +
			"        // （thinking blocks + provider reasoning 字段变体）。\n" +
			"        if (message && message.role === \"assistant\") {\n" +
			"            const clone = { ...message };\n" +
			"            delete clone.reasoning_content;\n" +
			"            delete clone.reasoning;\n" +
			"            delete clone.reasoning_text;\n" +
			"            delete clone.redacted_thinking;\n" +
			"            delete clone.thinking;\n" +
			"            if (Array.isArray(clone.content)) {\n" +
			"                clone.content = clone.content.filter((block) => {\n" +
			"                    const t = block && block.type;\n" +
			"                    return t !== \"thinking\" && t !== \"redacted_thinking\";\n" +
			"                });\n" +
			"            }\n" +
			"            message = clone;\n" +
			"        }\n",
		replacement:
			"",
	},
	{
		target: [
			T("@earendil-works", "pi-ai", "dist", "api", "openai-completions.js"),
			{ path: T("@earendil-works", "pi-coding-agent", "node_modules",
				"@earendil-works", "pi-ai", "dist", "api", "openai-completions.js"),
				optional: true },
		],
		name: "retire patch-04 (restore upstream thinking replay)",
		retire: true,
		optionalAnchor: true,
		anchor:
			"                    // ROSCLAW-PATCH-04: TranscriptPolicy——raw reasoning\n" +
			"                    // 绝不回放进 provider 请求（resume/历史消息同策）。\n" +
			"                    // thinking blocks 只作内存瞬态进度，不再外发。\n" +
			"                    if (false) {\n" +
			"                        let signature = nonEmptyThinkingBlocks[0].thinkingSignature;\n" +
			"                        if (model.provider === \"opencode-go\" && signature === \"reasoning\") {\n" +
			"                            signature = \"reasoning_content\";\n" +
			"                        }\n" +
			"                        if (signature && signature.length > 0) {\n" +
			"                            assistantMsg[signature] = nonEmptyThinkingBlocks.map((block) => block.thinking).join(\"\\n\");\n" +
			"                        }\n" +
			"                    }",
		replacement:
			"                    // Use the signature from the first thinking block if available (for llama.cpp server + gpt-oss)\n" +
			"                    let signature = nonEmptyThinkingBlocks[0].thinkingSignature;\n" +
			"                    if (model.provider === \"opencode-go\" && signature === \"reasoning\") {\n" +
			"                        signature = \"reasoning_content\";\n" +
			"                    }\n" +
			"                    if (signature && signature.length > 0) {\n" +
			"                        assistantMsg[signature] = nonEmptyThinkingBlocks.map((block) => block.thinking).join(\"\\n\");\n" +
			"                    }",
	},
];

const grouped = new Map();
for (const patch of PATCHES) {
	const targets = Array.isArray(patch.target) ? patch.target : [patch.target];
	for (const raw of targets) {
		const target = typeof raw === "string" ? raw : raw.path;
		const optional = typeof raw === "string" ? false : Boolean(raw.optional);
		if (optional && !existsSync(target)) {
			console.log(`[skip optional] ${target}（npm dedupe 后不存在，正常）`);
			continue;
		}
		if (!grouped.has(target)) grouped.set(target, []);
		grouped.get(target).push(patch);
	}
}

for (const [target, patches] of grouped) {
	let source = readFileSync(target, "utf8");
	for (const patch of patches) {
		// retire 条目：锚点（已打的补丁文本）在场才恢复——空 replacement
		// 或锚点即原文前缀时，"already applied" 恒真会跳过恢复（实证）。
		if (patch.retire) {
			if (!source.includes(patch.anchor)) {
				console.log(`[already retired] ${patch.name}`);
				continue;
			}
			source = source.replace(patch.anchor, patch.replacement);
			console.log(`[retired] ${patch.name}`);
			continue;
		}
		if (source.includes(patch.replacement)) {
			console.log(`[already applied] ${patch.name}`);
			continue;
		}
		if (!source.includes(patch.anchor)) {
			if (patch.optionalAnchor) {
				console.log(`[skip: anchor absent] ${patch.name}`);
				continue;
			}
			console.error(`PATCH ANCHOR MISSING: ${patch.name}`);
			console.error("上游已漂移——升级 Pi 后必须人工复核补丁锚点（见 patches/README.md）。");
			process.exit(1);
		}
		source = source.replace(patch.anchor, patch.replacement);
		console.log(`[applied] ${patch.name}`);
	}
	writeFileSync(target, source);
}
console.log("upstream patches applied");
