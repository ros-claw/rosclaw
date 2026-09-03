/** 工具结果用户面显示（WP-7，0823 审计 §四.WP-7）。
 *
 * 模型面与用户数分离：模型上下文保留完整 envelope/证据文本
 * （诚实性不降级），TUI 渲染折叠为单行摘要——0823 实测整段
 * plan/trace JSON 刷屏，propose_/grant 治理术语对用户无意义。
 */

/** 工具名 → 用户面显示名：剥 propose_ 治理前缀、__ → .。
 * （工具 name 不变——模型调用面不动；这只管渲染。） */
export function displayLabelFor(toolName: string): string {
	let label = toolName;
	if (label.startsWith("propose_")) label = label.slice("propose_".length);
	return label.replaceAll("__", ".");
}

/** envelope/结果文本 → 单行用户摘要（≤160 字符）。 */
export function summarizeToolResultText(raw: string): string {
	const text = raw.trim();
	// REJECTED：明确"未执行"——不得渲染任何完成字样（REJECTED vs
	// COMPLETED 同源纪律：拒绝就是拒绝）。
	if (text.startsWith("REJECTED")) {
		const match = text.match(/^REJECTED\s*\[([^\]]+)\]:?\s*(.*)$/s);
		if (match) {
			return `✗ 未执行 [${match[1]}] ${clip(match[2].replaceAll("\n", " "), 110)}`;
		}
		return `✗ 未执行 ${clip(text.replaceAll("\n", " "), 120)}`;
	}
	// envelope JSON：折叠为状态 + 摘要行。
	const parsed = tryParseJson(text);
	if (parsed && typeof parsed === "object") {
		const env = parsed as Record<string, unknown>;
		const status = String(env.status ?? "");
		const cap = String(env.capability_id ?? "");
		const value = (env.value ?? {}) as Record<string, unknown>;
		if (status === "SUCCEEDED") {
			const summary = typeof value.summary === "string" && value.summary
				? value.summary
				: scalarDigest(value);
			return `✓ ${cap}${summary ? ` — ${clip(summary, 110)}` : ""}`;
		}
		if (status) {
			const err = (env.error ?? {}) as Record<string, unknown>;
			const code = String(err.code ?? "ERROR");
			const message = String(err.message ?? "").replaceAll("\n", " ");
			return `✗ ${status} [${code}] ${cap} ${clip(message, 100)}`.trim();
		}
	}
	// 非 JSON（如 SIM 执行器自然语言回执）：去治理术语后截断。
	return clip(stripGovernanceTerms(text.replaceAll("\n", " ")), 160);
}

/** SIM auto 用户通知：隐藏 POLICY_AUTO/approval/grant 治理术语
 * （审计链仍在事件账本——用户面只给可理解的说明）。 */
export function formatPolicyAutoNotice(_details: { approvalId?: string }): string {
	return "安全仿真动作已自动放行执行（全程已记录审计，可用 /activity 查看）";
}

/** 治理术语清洗（用户面）：grant/propose/lease 等机制词汇替换为
 * 用户可理解的说明；事实部分（final_state/evidence_domain）保留。 */
function stripGovernanceTerms(text: string): string {
	return text
		.replace(/grant 已消费。?/g, "")
		.replace(/，?\s*grant[_ -]?[a-z0-9]*已?消费/gi, "")
		.replace(/\s{2,}/g, " ")
		.trim();
}

function scalarDigest(value: Record<string, unknown>): string {
	const parts: string[] = [];
	for (const [k, v] of Object.entries(value)) {
		if (parts.length >= 2) break;
		if (typeof v === "string" && v.length <= 60) parts.push(`${k}=${v}`);
		else if (typeof v === "number" || typeof v === "boolean") {
			parts.push(`${k}=${String(v)}`);
		}
	}
	return parts.join(" ");
}

function clip(text: string, max: number): string {
	return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function tryParseJson(text: string): unknown {
	if (!text.startsWith("{")) return undefined;
	try {
		return JSON.parse(text);
	} catch {
		return undefined;
	}
}


// ---------------------------------------------------------------------
// 0902 R3-c（§6.1）：状态 JSON → 单行摘要（纯函数，无 pi 依赖——
// HP2 结构门安全；pi-tui 渲染在 compact-result.ts）。
// ---------------------------------------------------------------------

/** 状态 JSON → 单行摘要（agentd/kernel/operator/mission 关键状态）。 */
export function summarizeStatusText(text: string): string {
	const parsed = tryParseJson(text.trim());
	if (parsed && typeof parsed === "object" && "agentd" in (parsed as object)) {
		const p = parsed as Record<string, unknown>;
		const mission = (p.mission ?? {}) as Record<string, unknown>;
		const parts = [
			`agentd=${String(p.agentd ?? "?")}`,
			`kernel=${String(p.kernel ?? "?")}`,
			`operator=${String(p.operator ?? "?")}`,
		];
		if (mission.state) parts.push(`mission=${String(mission.state)}`);
		if (p.action_readiness) parts.push(`actions=${String(p.action_readiness)}`);
		return `✓ 内核状态 ${parts.join(" · ")}`;
	}
	return summarizeToolResultText(text);
}
