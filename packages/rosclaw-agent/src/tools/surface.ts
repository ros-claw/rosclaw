/** Native Agent 模型可见工具面（PR-H1 重写，ADR-0012）。
 *
 * 总纲 v2：Native Agent 自己干活——主会话直接拥有策略包装的工作
 * 工具（Workspace Pack）+ 具身能力（Embodiment Pack）。普通任务
 * 不再委派第二个 Pi Session。
 *
 * 从模型面删除（§10.3）：
 * - rosclaw_task_submit 等 task_* 治理工具——root task 由
 *   InputController 创建（PR-H2），pause/resume/cancel 是产品控制，
 *   不消耗模型回合；
 * - delegate/work_* 全系——Worker 退出默认链（worker.enabled=false）。
 */

/** Workspace Pack：普通工作的基础能力（PR-H1 主会话直开）。 */
export const WORKSPACE_PACK: readonly string[] = [
	"read",
	"grep",
	"find",
	"ls",
	"edit",
	"write",
	"bash",
];

/** Embodiment Pack：具身/安全链（rosclawd 权威不变）。 */
export const EMBODIMENT_PACK: readonly string[] = [
	"rosclaw_status",
	"rosclaw_capabilities",
	"rosclaw_observe",
	"rosclaw_compute",
	"rosclaw_task",
	"rosclaw_verify",
	"rosclaw_request_action",
	"rosclaw_memory_query",
	"rosclaw_fail_safe",
];

/** 模型可见工具（唯一真相源）。 */
export const MODEL_TOOL_NAMES: readonly string[] = [
	...WORKSPACE_PACK,
	...EMBODIMENT_PACK,
];

/** 从模型面移除的工具（§10.3——plumbing 保留在 bridge/命令层）。 */
export const REMOVED_FROM_MODEL: readonly string[] = [
	// root task 创建/操控是 InputController/产品层的权威
	"rosclaw_task_submit",
	"rosclaw_task_observe",
	"rosclaw_task_steer",
	"rosclaw_task_answer",
	"rosclaw_task_pause",
	"rosclaw_task_resume",
	"rosclaw_task_cancel",
	// Worker 操控全系
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
