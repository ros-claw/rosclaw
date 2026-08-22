/** Product chrome 渲染（三审 P0-NA-16 + 六审 PR-SIX-1/PR-SIX-5）：
 * Header/Footer 都从同一个 KernelSnapshotV1 渲染——纯函数，不读缓存；
 * 按生效 locale 走 i18n catalog（机器 code 保持英文）。
 *
 * 红线：
 * - 快照只能来自 ProductStateCenter（字段全部权威源）；
 * - 未完成的 bootstrap 显示 LOADING/UNKNOWN——绝不乐观默认；
 * - 模型不能决定或改写 UI 安全状态；
 * - Action 状态显示首要受阻原因（不是干巴巴的 LOCKED）。
 */

import { t, type EffectiveLocale } from "../i18n/index.js";
import type {
	ActionReadinessV1,
	KernelSnapshotV1,
} from "../session/state-center.js";

/** 七审 §6 PR-SEVEN-6：Unicode 分隔符降级——ROSCLAW_TUI_UNICODE=
 * auto|always|never；auto 按 LC_ALL/LC_CTYPE/LANG 探测（显式非 UTF-8
 * locale 才降级，未设置 locale 变量默认 Unicode）。非 UTF-8 终端用
 * ASCII 分隔符，避免 ·/— 渲染为 ?。 */
export function chromeSep(): string {
	const pref = (process.env.ROSCLAW_TUI_UNICODE ?? "auto").toLowerCase();
	if (pref === "never") return " | ";
	if (pref === "always") return " · ";
	const locales = [process.env.LC_ALL, process.env.LC_CTYPE, process.env.LANG]
		.filter((v): v is string => Boolean(v));
	if (locales.length > 0 && !locales.some((v) => /utf-?8/i.test(v))) {
		return " | ";
	}
	return " · ";
}

export function renderActionState(
	readiness: ActionReadinessV1,
	locale: EffectiveLocale = "zh-CN",
): string {
	if (readiness.state === "READY") {
		return `${t("chrome.action", locale)} ${t("action.ready", locale)}`;
	}
	const primary = readiness.reason_codes[0] ?? "UNKNOWN";
	const labelKey = `reason.${primary}`;
	const label = t(labelKey as never, locale);
	// 七审 §6 PR-SEVEN-6：普通层只显示本地化原因，不做 CODE（label）
	// 双重双语；机器 code 只在 /status 与 JSON 层暴露。
	const reason = label !== labelKey ? label : primary;
	return `${t("chrome.action", locale)} ${t("action.blocked", locale)} (${reason})`;
}

/** 推荐头部（P0-NA-16 规格 + P0-5F Action 状态 + 六审首要受阻原因 +
 *  PR-SIX-5 locale）：
 *   zh: ROSClaw 1.2.0 · 仿真 · Kimi K3
 *       任务 mis_... · 本体 sim/ur5e · 上下文 新鲜 r12 · 操作员 离线 · 动作 受阻（操作员离线）
 *   en: ROSClaw 1.2.0 · Simulation · Kimi K3
 *       Mission mis_... · Body sim/ur5e · Context Fresh r12 · Operator Offline · Action Blocked (OPERATOR_OFFLINE)
 */
export function renderHeader(
	state: KernelSnapshotV1,
	locale: EffectiveLocale = "zh-CN",
): string {
	const mode = t(`mode.${state.mode}` as never, locale);
	const sep = chromeSep();
	const line1Parts = [`ROSClaw ${state.product_version}`, mode];
	if (state.model) line1Parts.push(state.model);
	// 十一审 PR-D：Project 一等显示（无绑定显式占位；ASCII 终端降级）。
	const noWs = chromeSep() === " | " ? "-" : "—";
	line1Parts.push(`Project ${state.workspace ?? noWs}`);
	const line1 = line1Parts.join(sep);
	if (!state.mission_id) {
		return `${line1}\n${t("chrome.unbound", locale)}`;
	}
	const body = state.body_display ?? state.body_id ?? t("state.LOADING", locale);
	const contextState = t(`state.${state.context_state}` as never, locale);
	const context =
		state.context_state === "FRESH"
			? `${t("chrome.context", locale)} ${contextState} r${state.context_revision}`
			: `${t("chrome.context", locale)} ${contextState}`;
	const operator = renderOperator(state, locale);
	const kernel =
		state.kernel === "READY"
			? ""
			: `${sep}${t("chrome.kernel", locale)} ${t(`state.${state.kernel}` as never, locale)}`;
	const line2 =
		`${t("chrome.mission", locale)} ${state.mission_id.slice(0, 24)}${sep}` +
		`${t("chrome.body", locale)} ${body}${sep}${context}${sep}${operator}${kernel}${sep}` +
		renderActionState(state.action_readiness, locale);
	return `${line1}\n${line2}`;
}

/** 十四审 PR-14.7（§1.9）：纯 SIM 任务中 Operator 状态降为次要
 *  信息——"Operator Offline" 不应在 SIM 自动执行场景造成紧张或
 *  暗示需要人工审批；只有 REAL/涉及操作员的动作时才突出。 */
export function renderOperator(state: KernelSnapshotV1, locale: EffectiveLocale): string {
	const readiness = state.action_readiness as
		| { state?: string; reason_codes?: string[] }
		| undefined;
	const operatorInvolved =
		state.mode !== "SIMULATION"
		// 已初始化的 Operator 永远显示（用户明确启用——降级只针对
		// 未初始化时的 "Operator Offline" 紧张感）。
		|| state.operator === "READY"
		|| readiness?.state === "BLOCKED"
		|| (readiness?.reason_codes ?? []).some((c) => c.includes("OPERATOR"));
	if (!operatorInvolved) {
		return t("chrome.sim_auto", locale);
	}
	return `${t("chrome.operator", locale)} ${t(`state.${state.operator}` as never, locale)}`;
}

/** Footer：与 Header 同一快照——model/mode/operator 绝不分叉。 */
export function renderFooter(
	state: KernelSnapshotV1,
	locale: EffectiveLocale = "zh-CN",
): string {
	const model = state.model || t("chrome.no_model", locale);
	const parts = [
		model,
		t(`mode.${state.mode}` as never, locale),
		renderOperator(state, locale),
	];
	if (state.kernel !== "READY") {
		parts.push(`${t("chrome.kernel", locale)} ${t(`state.${state.kernel}` as never, locale)}`);
	}
	return parts.join(chromeSep());
}

/** 八审 §4 P0-9：kit BROKEN 提示格式——空 reason 不渲染悬空冒号；
 * 无 remediation 不渲染修复建议。 */
export function formatKitBrokenHint(
	kit: { state?: string; reason?: string; remediation?: { command?: string } | null },
	locale: EffectiveLocale = "zh-CN",
): string {
	const parts = [t("robot.kit_broken", locale)];
	if (kit.reason) parts.push(kit.reason);
	if (kit.remediation?.command) {
		parts.push(`${t("robot.repair_hint", locale)}: ${kit.remediation.command}`);
	}
	return parts.join(" — ");
}

/** kit 从 BROKEN 恢复 READY 的正面清除文案（旧 warning 不能悬留）。 */
export function formatKitRecoveredHint(
	displayName: string,
	locale: EffectiveLocale = "zh-CN",
): string {
	return locale === "en-US"
		? `Robot kit ready: ${displayName}`
		: `机器人套件就绪：${displayName}`;
}
