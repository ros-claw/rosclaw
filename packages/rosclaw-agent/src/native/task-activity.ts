/** Task Activity 渲染（PR-H8，总纲 v2 §20）——TUI 任务进度视图。
 *
 * 数据全部来自 TaskKernel 事件流/产物账本，不经 LLM 总结（假进度
 * 不可能：模型说什么不影响这里显示什么）。
 *
 * - renderTaskActivity：阶段行（任务开始/修订/进程/产物/验收/终态）。
 *   operation.output 不进 Activity——逐行输出归 /logs，否则刷屏。
 * - renderOperationLogs：operation.output 尾部文本。
 * - renderArtifactList：产物名/路径/大小/哈希短码；空 → 诚实空态。
 */

export interface KernelEvent {
	seq: number;
	event_type: string;
	payload: Record<string, unknown>;
}

function short(id: unknown, head = 12): string {
	return String(id ?? "").slice(0, head);
}

function firstLine(text: unknown, max = 60): string {
	const line = String(text ?? "").split("\n")[0].trim();
	return line.length > max ? `${line.slice(0, max)}…` : line;
}

/** kernel 事件 → 中文阶段行（按 seq 顺序）。 */
export function renderTaskActivity(events: KernelEvent[]): string[] {
	const sorted = [...events].sort((a, b) => a.seq - b.seq);
	const lines: string[] = [];
	for (const event of sorted) {
		const p = event.payload;
		switch (event.event_type) {
			case "task.started":
				lines.push(`▶ 任务开始：${firstLine(p.goal)}`);
				break;
			case "task.revised":
				lines.push(
					`✎ 修订 r${String(p.revision ?? "?")}：${firstLine(p.delta)}`,
				);
				break;
			case "task.state_changed": {
				// 当前 Kernel 发 state/reason；兼容旧 from/to，不虚构前态。
				const state = p.state ?? p.to;
				if (String(state ?? "") !== "ACTIVE") {
					const transition = p.from != null && state != null
						? `${String(p.from)} → ${String(state)}`
						: String(state ?? "未知");
					const reason = p.reason ?? p.reason_code;
					lines.push(`… 状态：${transition}`
						+ (reason ? `（${firstLine(reason, 40)}）` : ""));
				}
				break;
			}
			case "task.terminal":
				lines.push(
					`■ 终态：${String(p.state ?? "?")}`
					+ (p.reason ? `（${firstLine(p.reason, 40)}）` : ""),
				);
				break;
			case "operation.started":
		case "operation.admitted": {
				const argv = Array.isArray(p.argv) ? p.argv.join(" ") : "";
				lines.push(
					`▸ 进程启动 [${short(p.operation_id)}]：${firstLine(argv)}`,
				);
				break;
			}
			case "operation.completed":
				lines.push(`✓ 进程完成 [${short(p.operation_id)}]`);
				break;
			case "operation.failed":
				lines.push(
					`✗ 进程失败 [${short(p.operation_id)}]`
					+ (p.failure_code ? `（${String(p.failure_code)}）` : "")
					+ (p.error ? `：${firstLine(p.error, 40)}` : ""),
				);
				break;
			case "artifact.created":
				lines.push(
					`◆ 产物：${String(p.path ?? "").split("/").pop() ?? ""}`
					+ `（${String(p.bytes ?? "?")} 字节）`,
				);
				break;
			case "verification.completed": {
				if (String(p.status) === "PASS") {
					const checks = Array.isArray(p.checks) ? p.checks.length
						: typeof p.checks === "number" && Number.isInteger(p.checks) && p.checks >= 0
							? p.checks : "未知";
					const scope = String(p.verification_scope ?? "UNKNOWN");
					const semantic = String(p.task_semantic_verification ?? "UNKNOWN");
					let label: string;
					if (scope === "artifact_integrity_only") {
						label = "已检查交付文件完整性；任务语义未验证";
					} else if (scope === "summary_nonempty_only") {
						label = "已记录回复；任务语义未验证";
					} else if ((["configured_acceptance", "configured_acceptance_and_deliverables",
						"declared_deliverables"].includes(scope) && semantic === "CONFIGURED_CHECKS_ONLY")
						|| (scope === "configured_criteria" && ["VERIFIED", "CONFIGURED_CHECKS_ONLY"].includes(semantic))) {
						label = "验收通过；仅限已配置检查/声明交付条件，不代表整体目标达成";
					} else {
						label = "校验记录 PASS；验收范围未知，未确认目标达成";
					}
					lines.push(`ℹ ${label}（${checks} 项检查；verification_scope=${scope}；`
						+ `task_semantic_verification=${semantic}）`);
				} else {
					const failures = Array.isArray(p.failures) ? p.failures : [];
					lines.push(
						`✗ 验收未过：${failures.map((f) => firstLine(f, 40)).join("；")}`,
					);
				}
				break;
			}
			default:
				// 未知事件类型不上 Activity（output 等噪音归 /logs）。
				break;
		}
	}
	if (!lines.length) lines.push("（还没有任务活动）");
	return lines;
}

/** operation.output 尾部渲染（/logs）。 */
export function renderOperationLogs(
	events: KernelEvent[],
	limit = 40,
): string[] {
	const sorted = [...events].sort((a, b) => a.seq - b.seq);
	const lines: string[] = [];
	let currentOp = "";
	for (const event of sorted) {
		const p = event.payload;
		if (event.event_type === "operation.started" || event.event_type === "operation.admitted") {
			currentOp = short(p.operation_id);
			const argv = Array.isArray(p.argv) ? p.argv.join(" ") : "";
			lines.push(`$ [${currentOp}] ${firstLine(argv, 80)}`);
		} else if (event.event_type === "operation.output") {
			for (const raw of String(p.text ?? "").split("\n")) {
				if (raw.trim()) lines.push(`  [${currentOp}] ${raw}`);
			}
		} else if (
			event.event_type === "operation.completed"
			|| event.event_type === "operation.failed"
		) {
			lines.push(
				`  [${short(p.operation_id)}] → ${String(p.state ?? "?")}`
				+ (p.failure_code ? `（${String(p.failure_code)}）` : ""),
			);
		}
	}
	if (!lines.length) lines.push("（还没有后台进程输出）");
	return lines.slice(-limit);
}

/** 产物账本渲染（/artifacts）。 */
export function renderArtifactList(
	artifacts: Array<Record<string, unknown>>,
): string[] {
	if (!artifacts.length) return ["（当前任务无产物登记）"];
	return artifacts.map((a) => {
		const path = String(a.path ?? "");
		const name = path.split("/").pop() ?? "";
		const size = Number(a.size_bytes ?? 0);
		const sha = String(a.sha256 ?? "").slice(0, 7);
		const media = String(a.media_type ?? "");
		// P0-H：OSC8 超链接——终端可点击打开交付物。
		const link = `\x1b]8;;file://${path}\x07${name}\x1b]8;;\x07`;
		return `◆ ${link}  ${size} 字节  sha:${sha}  ${media}`.trimEnd();
	});
}
