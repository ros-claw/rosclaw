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
	const reason = label !== labelKey ? `${primary}（${label}）` : primary;
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
	const line1Parts = [`ROSClaw ${state.product_version}`, mode];
	if (state.model) line1Parts.push(state.model);
	const line1 = line1Parts.join(" · ");
	if (!state.mission_id) {
		return `${line1}\n${t("chrome.unbound", locale)}`;
	}
	const body = state.body_id ?? t("state.LOADING", locale);
	const contextState = t(`state.${state.context_state}` as never, locale);
	const context =
		state.context_state === "FRESH"
			? `${t("chrome.context", locale)} ${contextState} r${state.context_revision}`
			: `${t("chrome.context", locale)} ${contextState}`;
	const operator = `${t("chrome.operator", locale)} ${t(`state.${state.operator}` as never, locale)}`;
	const kernel =
		state.kernel === "READY"
			? ""
			: ` · ${t("chrome.kernel", locale)} ${t(`state.${state.kernel}` as never, locale)}`;
	const line2 =
		`${t("chrome.mission", locale)} ${state.mission_id.slice(0, 24)} · ` +
		`${t("chrome.body", locale)} ${body} · ${context} · ${operator}${kernel} · ` +
		renderActionState(state.action_readiness, locale);
	return `${line1}\n${line2}`;
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
		`${t("chrome.operator", locale)} ${t(`state.${state.operator}` as never, locale)}`,
	];
	if (state.kernel !== "READY") {
		parts.push(`${t("chrome.kernel", locale)} ${t(`state.${state.kernel}` as never, locale)}`);
	}
	return parts.join(" · ");
}
