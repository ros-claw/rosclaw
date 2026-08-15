/** Native Agent 模型可见工具面（十五审 PR-RF-1，ADR-0011）。
 *
 * 无为而治：模型只见治理工具（task_*）+ 只读摘要工具。底层 Worker
 * 操控（delegate/retry/resume/extend/check/list/update/cancel/
 * read_work_*）从模型面移除——它们仍是 bridge/命令层的 plumbing，
 * 但不再交给模型自由组合（裂变/横跳/猜因的根源）。
 */

/** 模型可见工具（唯一真相源）。 */
export const MODEL_TOOL_NAMES: readonly string[] = [
	// 治理：提交/观察/steer/回答/暂停/恢复/取消（同一 owning execution）
	"rosclaw_task_submit",
	"rosclaw_task_observe",
	"rosclaw_task_steer",
	"rosclaw_task_answer",
	"rosclaw_task_pause",
	"rosclaw_task_resume",
	"rosclaw_task_cancel",
	// 任务级确定性入口（八审保留）
	"rosclaw_task",
	// 物理动作提案（rosclawd 唯一准入——治理面的一部分）
	"rosclaw_request_action",
	// 只读摘要
	"rosclaw_status",
	"rosclaw_capabilities",
	"rosclaw_observe",
	"rosclaw_compute",
	"rosclaw_verify",
	"rosclaw_memory_query",
	"rosclaw_fail_safe",
];

/** 从模型面移除的底层 Worker 操控工具（plumbing 保留，不暴露）。 */
export const REMOVED_FROM_MODEL: readonly string[] = [
	"rosclaw_delegate",
	"rosclaw_retry_work",
	"rosclaw_resume_work",
	"rosclaw_extend_work",
	"rosclaw_check_work",
	"rosclaw_list_work",
	"rosclaw_update_work",
	"rosclaw_cancel_work",
	"rosclaw_answer_work",
	"rosclaw_read_work_events",
	"rosclaw_read_work_transcript",
	"rosclaw_list_work_artifacts",
	"rosclaw_read_work_failure",
];

/** 过滤装配好的工具数组——模型只见 MODEL_TOOL_NAMES。 */
export function filterModelTools<T extends { name: string }>(tools: T[]): T[] {
	const allowed = new Set(MODEL_TOOL_NAMES);
	return tools.filter((tool) => allowed.has(tool.name));
}
