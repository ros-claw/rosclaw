/** 输入防护（PNA-9，规格 §11）。
 *
 * - 未知 slash 命令：不发模型，提示不存在（//text 显式转义发送普通文本）；
 * - ROBOT profile：/trust /share /import 一律拦截（内建 dispatch 先于
 *   input 事件，所以这里兜底的是非内建变形与直接文本注入）；
 * - 已知内建命令与 ROSClaw 命令放行。
 */

export interface InputGuardResult {
	action: "continue" | "handled" | "transform";
	text?: string;
	notice?: string;
}

const ROSCLAW_COMMANDS = new Set([
	"/workers",
	"/delegate",
	"/worker-jobs",
	"/mission",
	"/body",
	"/mode",
	"/status",
	"/tools",
	"/approvals",
	"/grants",
	"/revoke",
	"/evidence",
	"/memory",
	"/doctor",
	"/estop",
]);

// Pi 内建（审计 §4 全表）——放行进内建 dispatch。
const PI_BUILTINS = new Set([
	"/settings",
	"/model",
	"/scoped-models",
	"/export",
	"/import",
	"/share",
	"/copy",
	"/name",
	"/session",
	"/changelog",
	"/hotkeys",
	"/fork",
	"/clone",
	"/tree",
	"/trust",
	"/login",
	"/logout",
	"/new",
	"/compact",
	"/resume",
	"/reload",
	"/quit",
]);

// ROBOT profile 下禁止的内建（语义与 ROSClaw 信任模型冲突）。
const ROBOT_BLOCKED = new Set(["/trust", "/share", "/import", "/reload"]);

export function guardInput(raw: string, profile: "robot" | "developer"): InputGuardResult {
	const text = raw.trim();
	if (!text.startsWith("/")) return { action: "continue" };
	if (text.startsWith("//")) {
		// 显式转义：作为普通文本发送。
		return { action: "transform", text: text.slice(1) };
	}
	const name = text.split(/\s/, 1)[0].toLowerCase();
	if (profile === "robot" && ROBOT_BLOCKED.has(name)) {
		return {
			action: "handled",
			notice: `${name} 在 ROBOT profile 下禁用（授权与信任由 ROSClaw operatord/门禁管理）`,
		};
	}
	if (ROSCLAW_COMMANDS.has(name) || PI_BUILTINS.has(name)) {
		return { action: "continue" };
	}
	return {
		action: "handled",
		notice: `未知命令 ${name}（不会发送给模型）。发送普通文本请用 // 前缀或去掉开头的 /`,
	};
}
