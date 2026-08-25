/** P1-A4（0824 总纲 P1-A）：/compact 保留 TaskRefs。
 *
 * Pi compaction（manual /compact、threshold、overflow）把会话历史压成
 * 摘要——task_id/root_goal/artifact refs 随被压消息丢失，模型 compact
 * 后"失忆"。本模块在任何 reason 的 compaction 完成后，从 TaskKernel
 * 权威账本（pi.kernel.latest + pi.kernel.artifacts——不是模型记忆）
 * 取最近 task 与产物 refs，经 sendMessage(deliverAs:"nextTurn") 把
 * TaskRefs 锚定回 LLM 上下文：
 *
 * - 锚消息进入会话历史 → 后续 compaction 的 summarizer 也会看到它，
 *   refs 跨多次 compact 传播；
 * - 无 task / 拉取失败 → 诚实 no-op（不编造 refs）；
 * - 同 task+revision 去重（连续 auto-compact 不刷屏）。
 */

interface KernelTask {
	task_id?: string;
	root_goal?: string;
	state?: string;
	active_revision?: number;
}

interface KernelArtifact {
	artifact_id?: string;
	path?: string;
	media_type?: string;
}

interface BridgeCaller {
	(method: string, params?: unknown): Promise<{ ok: boolean; task?: unknown; artifacts?: unknown[]; error?: string }>;
}

interface CompactAnchorDeps {
	call: BridgeCaller;
	/** 事件时取值（resume/switch 后绑定可能变化）。 */
	missionId: () => string;
	sessionRef: () => string;
	/** 诊断（诊断先行）：每次决策一行——skip 原因或 anchored。 */
	log?: (message: string) => void;
}

interface PiLike {
	on(event: "session_compact", handler: (event: { reason?: string }) => Promise<void>): void;
	sendMessage(
		message: { customType: string; content: string; display: boolean; details?: unknown },
		options?: { triggerTurn?: boolean; deliverAs?: "steer" | "followUp" | "nextTurn" },
	): void;
}

export function buildTaskAnchor(
	task: KernelTask,
	artifacts: KernelArtifact[],
	reason: string,
): string {
	const lines = [
		`[ROSClaw TaskRefs 锚——compaction(${reason}) 后由内核权威账本重建，非模型记忆]`,
		`task_id: ${task.task_id ?? ""}`,
		`state: ${task.state ?? ""} (revision ${task.active_revision ?? 0})`,
		`root_goal: ${task.root_goal ?? ""}`,
	];
	if (artifacts.length > 0) {
		lines.push("artifacts（已登记交付物——引用这些路径，不要凭记忆重造）:");
		for (const a of artifacts) {
			lines.push(`- ${a.artifact_id ?? ""} [${a.media_type ?? ""}] ${a.path ?? ""}`);
		}
	} else {
		lines.push("artifacts: （尚无登记产物）");
	}
	return lines.join("\n");
}

export function registerCompactAnchor(pi: PiLike, deps: CompactAnchorDeps): void {
	let lastKey = "";
	pi.on("session_compact", async (event) => {
		const latest = await deps.call("pi.kernel.latest", {
			mission_id: deps.missionId(),
			session_ref: deps.sessionRef(),
		}).catch(() => ({ ok: false }) as never);
		if (!latest.ok || !latest.task) {
			deps.log?.(
				`skip: no task (ok=${latest.ok} mission=${deps.missionId()} session=${deps.sessionRef()})`,
			);
			return; // 无 task/拉取失败——诚实 no-op
		}
		const task = latest.task as KernelTask;
		const key = `${task.task_id}:${task.active_revision}`;
		if (key === lastKey) {
			deps.log?.(`skip: dup anchor ${key}`);
			return; // 同 key 去重
		}
		const artifactResult = await deps.call("pi.kernel.artifacts", {
			task_id: task.task_id ?? "",
		}).catch(() => ({ ok: false }) as never);
		const artifacts = artifactResult.ok
			? ((artifactResult.artifacts ?? []) as KernelArtifact[])
			: [];
		lastKey = key;
		deps.log?.(`anchored: ${key} artifacts=${artifacts.length}`);
		pi.sendMessage(
			{
				customType: "rosclaw.task_anchor",
				content: buildTaskAnchor(task, artifacts, String(event.reason ?? "unknown")),
				display: false,
				details: { task_id: task.task_id, reason: event.reason },
			},
			{ deliverAs: "nextTurn" },
		);
	});
}
