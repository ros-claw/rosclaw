/** TerminalPresenter（0827 体验审计 P0-3）——Coordinator 是唯一终态
 *  发布者：任务终态的最终回复由 TaskOutcome 确定性生成，不经模型
 *  回合（0827 实证：模型先说"✅ 任务完成"，watcher 后报"交付
 *  MISSING"——两个发布者互相矛盾）。
 *
 * 输出原则：
 * - PASS + DELIVERED → 完成 + 交付物打开命令；
 * - 其余 → 诚实"未完全达成" + 验收/交付状态，绝不出现完成宣称。
 */

export interface TerminalOutcome {
	verification?: string;
	delivery?: string;
	lifecycle?: string;
	artifact_refs?: Array<{
		artifact_id?: string;
		open_command?: string;
	}>;
}

/** 任务终态的最终用户回复（确定性——同一 outcome 永远同一文本）。 */
export function renderTerminalReply(outcome: TerminalOutcome): string {
	const verification = String(outcome.verification ?? "UNKNOWN");
	const delivery = String(outcome.delivery ?? "UNKNOWN");
	const refs = outcome.artifact_refs ?? [];
	const opens = refs
		.map((r) => String(r.open_command ?? ""))
		.filter(Boolean);
	const passed = verification === "PASS" && delivery === "DELIVERED";
	if (passed) {
		const head = "✅ 任务完成：验收 PASS · 交付 DELIVERED";
		return opens.length
			? `${head}\n交付物：${opens.join(" · ")}`
			: head;
	}
	return (
		`⚠️ 任务未完全达成：验收 ${verification} · 交付 ${delivery}`
		+ "——如实说明限制，不宣称完整完成（/activity 查看账本）"
		+ (opens.length ? `\n已产出交付物：${opens.join(" · ")}` : "")
	);
}
