/** R0-8（0826 体验审计 §5.R0-8）：TUI 任务卡——默认层一张稳定
 * 任务卡，不是 raw JSON 日志窗口。
 *
 * 复用原则：模型上下文保留完整 payload（诚实性不降级，/activity
 * 可查）；TUI 渲染为结构化任务卡（节点一行一态 + 误差/阈值 +
 * 交付可打开命令）。失败诚实：✗ 失败节点、△ 缺失交付。
 */

import type { EffectiveLocale } from "../i18n/index.js";

type Payload = Record<string, unknown>;

const STAGE_ORDER: Array<{ ref: string; zh: string; en: string }> = [
	{ ref: "PlanRef", zh: "规划", en: "plan" },
	{ ref: "TraceRef", zh: "仿真", en: "simulate" },
	{ ref: "RenderRef", zh: "渲染", en: "render" },
	{ ref: "SceneRef", zh: "场景视频", en: "scene video" },
	{ ref: "VerificationRef", zh: "验证", en: "verify" },
];

const FAILED_NODE_STAGE: Record<string, string> = {
	make_path: "PlanRef",
	simulate: "TraceRef",
	render: "RenderRef",
	render_scene: "SceneRef",
	verify: "VerificationRef",
};

function asRecord(value: unknown): Payload {
	return value && typeof value === "object" ? (value as Payload) : {};
}

function fmtMm(meters: unknown): string {
	const value = Number(meters);
	if (!Number.isFinite(value)) return "?";
	return `${(value * 1000).toFixed(1)}mm`;
}

/** rosclaw_task payload → 任务卡（多行文本）。 */
export function renderTaskCard(payload: Payload, locale: EffectiveLocale = "zh-CN"): string {
	const zh = locale !== "en-US";
	const state = String(payload.state ?? "FAILED");
	const plan = asRecord(payload.plan);
	const refs = new Set(
		Array.isArray(plan.refs) ? (plan.refs as unknown[]).map(String) : [],
	);
	const failedNode = String(plan.failed_node ?? "");
	const failedRef = FAILED_NODE_STAGE[failedNode] ?? "";
	const stages = STAGE_ORDER.filter((stage) => {
		// 无场景节点的任务不显示场景行（诚实——不假装有场景要求）。
		if (stage.ref === "SceneRef" && !refs.has("SceneRef") && failedRef !== "SceneRef") {
			return false;
		}
		return true;
	}).map((stage) => {
		if (refs.has(stage.ref)) return `✓ ${zh ? stage.zh : stage.en}`;
		if (stage.ref === failedRef) return `✗ ${zh ? stage.zh : stage.en}`;
		return `△ ${zh ? stage.zh : stage.en}`;
	});
	const verification = asRecord(payload.verification);
	const maxError = fmtMm(verification.max_error_m);
	const threshold = fmtMm(verification.threshold_m);
	const frames = Number(verification.frames ?? 0);
	const minFrames = Number(verification.min_frames ?? 0);
	const title = state === "VERIFIED"
		? (zh ? "任务完成" : "Task completed")
		: (zh ? "任务未完成" : "Task not completed");
	const lines = [
		`${title} · ${String(payload.goal ?? "")}`,
		stages.join("   "),
		`${zh ? "误差" : "error"} max ${maxError} / ${zh ? "阈值" : "threshold"} ${threshold}`
		+ (minFrames ? ` · ${zh ? "帧" : "frames"} ${frames}/${minFrames}` : ""),
	];
	const failures = Array.isArray(payload.failures)
		? (payload.failures as unknown[]).map(String).filter(Boolean)
		: [];
	if (failures.length > 0) {
		lines.push(
			`${zh ? "问题" : "issues"}: ${failures.map((f) => clip(f, 90)).join("；")}`,
		);
	}
	const artifactRefs = Array.isArray(payload.artifact_refs)
		? (payload.artifact_refs as Payload[])
		: [];
	const openCommands = artifactRefs
		.map((ref) => String(ref.open_command ?? ""))
		.filter(Boolean);
	if (openCommands.length > 0) {
		lines.push(
			`${zh ? "交付" : "deliverables"}: ${openCommands.join(" · ")}`,
		);
	}
	const evidence = String(payload.evidence_level ?? "");
	if (evidence) {
		lines.push(
			zh
				? `证据等级 ${evidence}——仿真动力学自洽，不能证明真机执行效果`
				: `evidence ${evidence} — simulation only, not real-hardware proof`,
		);
	}
	return lines.join("\n");
}

/** task tool 的 renderResult：解析 payload JSON → 任务卡；非
 * JSON 回退单行摘要（不刷屏）。"" */
export function renderTaskToolResult(rawText: string): string {
	const text = rawText.trim();
	try {
		const parsed = JSON.parse(text) as Payload;
		if (parsed && typeof parsed === "object" && "state" in parsed) {
			return renderTaskCard(parsed);
		}
	} catch {
		// 非 JSON（REJECTED 文本等）——走单行折叠。
	}
	return clip(text.replaceAll("\n", " "), 200);
}

function clip(text: string, max: number): string {
	return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}
