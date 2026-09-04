/** TerminalPresenter（0827 体验审计 P0-3）——Coordinator 是唯一终态
 *  发布者：任务终态的最终回复由 TaskOutcome 确定性生成，不经模型
 *  回合（0827 实证：模型先说"✅ 任务完成"，watcher 后报"交付
 *  MISSING"——两个发布者互相矛盾）。
 *
 * 输出原则：
 * - PASS + DELIVERED → 完成 + 交付物（文件名/绝对路径/打开命令）
 *   + 下一步；
 * - 其余 → 诚实"未完全达成" + 原因（验收失败逐条）+ 已产出 +
 *   下一步，绝不出现完成宣称。
 *
 * 0901 P0-6（体验审计 §十二 P0-6）：一行结论不是产品——用户必须
 * 看到 原因（为什么失败）、文件名、绝对路径、下一步（看完知道
 * 该干嘛）。
 */

export interface TerminalOutcome {
	verification?: string;
	delivery?: string;
	lifecycle?: string;
	/** P0-4：outputs/ 投影视图状态（DEGRADED = 投影失败但账本
	 *  交付有效——必须如实告知，不得静默）。 */
	workspace_projection?: string;
	/** P0-6：验收/交付失败原因（repair_directive.failures——
	 *  Coordinator 已持久化的权威失败清单）。 */
	repair_directive?: {
		failures?: string[];
	};
	artifact_refs?: Array<{
		artifact_id?: string;
		open_command?: string;
		/** P0-6：绝对路径 + 媒体类型 + 大小（交付面三要素）。 */
		path?: string;
		media_type?: string;
		size_bytes?: number;
	}>;
}

/** 交付物行：文件名（大小）+ 短打开命令——默认层不裸打内部
 *  绝对路径（0902 §6.1：普通用户不得看到 sim/traces 内部路径；
 *  0901 的可达性由 open/path/export 命令承接）。verbose（诊断层
 *  /activity）才附绝对路径。无标签前缀（0901 journey 实证：CJK
 *  换行会把"路径："从值中间撕开）。 */
function renderArtifactLine(ref: {
	artifact_id?: string;
	open_command?: string;
	path?: string;
	size_bytes?: number;
}, verbose: boolean): string {
	const path = String(ref.path ?? "");
	const name = path ? path.split("/").pop() ?? "" : "";
	const size = Number(ref.size_bytes ?? 0);
	const sizeText = size >= 1_048_576
		? `${(size / 1_048_576).toFixed(1)} MB`
		: size >= 1024
			? `${(size / 1024).toFixed(1)} KB`
			: size > 0
				? `${size} B`
				: "";
	const head = name
		? `• ${name}${sizeText ? `（${sizeText}）` : ""}`
		: `• ${String(ref.artifact_id ?? "artifact")}`;
	const pathLine = verbose && path ? `\n  ${path}` : "";
	const openLine = ref.open_command ? `\n  ${ref.open_command}` : "";
	return `${head}${pathLine}${openLine}`;
}

/** 任务终态的最终用户回复（确定性——同一 outcome 永远同一文本）。 */
export function renderTerminalReply(
	outcome: TerminalOutcome,
	options?: { verbose?: boolean },
): string {
	const verification = String(outcome.verification ?? "UNKNOWN");
	const delivery = String(outcome.delivery ?? "UNKNOWN");
	const refs = outcome.artifact_refs ?? [];
	const verbose = options?.verbose === true;
	const artifactLines = refs.map((ref) => renderArtifactLine(ref, verbose));
	const passed = (verification === "PASS" || verification === "PASS_NEAR_LIMIT")
		&& delivery === "DELIVERED";
	const degraded = outcome.workspace_projection === "DEGRADED"
		? "\n（工作区投影退化——交付物仍可用上面的 artifact open 命令打开）"
		: "";
	if (passed) {
		// P0-5：PASS_NEAR_LIMIT 如实标注（≥90% 阈值占用不显示普通
		// PASS——19.86mm/20mm 是"勉强通过"，不是"干净通过"）。
		const head = verification === "PASS_NEAR_LIMIT"
			? "✅ 任务完成：验收 PASS_NEAR_LIMIT（误差接近阈值上限，勉强通过） · 交付 DELIVERED"
			: "✅ 任务完成：验收 PASS · 交付 DELIVERED";
		if (!artifactLines.length) return head + degraded;
		return (
			`${head}\n交付物：\n${artifactLines.join("\n")}${degraded}\n`
			+ "下一步：用上面的「打开」命令查看交付物；/activity 查看完整账本"
		);
	}
	// 0903 体验实证：UNKNOWN·UNKNOWN 空卡（投诉被当新指令建了
	// 空 revision，consider 报 UNKNOWN/UNKNOWN 无原因）——双
	// UNKNOWN 且零产物时诚实说"尚未产生可验收结果"，不装成终态卡。
	if (verification === "UNKNOWN" && delivery === "UNKNOWN" && !refs.length) {
		return (
			"ℹ️ 任务尚未产生可验收结果（未执行或执行未完成）"
			+ "——说明你要的修正或新目标即可，我不会拿旧结果冒充。"
		);
	}
	// 失败/部分达成：原因逐条（用户必须看到"为什么"）+ 已产出 +
	// 下一步。
	const failures = outcome.repair_directive?.failures ?? [];
	const reasonLines = failures.length
		? `\n原因：\n${failures.map((f) => `• ${f}`).join("\n")}`
		: "";
	const producedLines = artifactLines.length
		? `\n已产出：\n${artifactLines.join("\n")}`
		: "";
	return (
		`⚠️ 任务未完全达成：验收 ${verification} · 交付 ${delivery}`
		+ "——如实说明限制，不宣称完整完成"
		+ reasonLines
		+ producedLines
		+ "\n下一步：/activity 查看完整账本；或直接说明如何修正（在同一任务内继续，不重新跑已完成的步骤）"
	);
}
