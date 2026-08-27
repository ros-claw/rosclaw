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
	// PR-H3：长进程 = Operation（立即返回/事件流/终态通知）
	"process_start",
	"process_status",
	"process_output",
	"process_stop",
];

/** Product Pack（P0-D，0824 总纲 §8.1）：模型面只剩幂等
 *  rosclaw_deliver（普通文件工具创建的交付物）——task_finish/
 *  task_blocked/artifact_register 退出模型面（Coordinator 自动
 *  收尾，capability 产物自动登记）。 */
export const PRODUCT_PACK: readonly string[] = [
	"rosclaw_deliver",
];

/** Embodiment Pack：具身/安全链（rosclawd 权威不变）。
 *
 * PR-N5D：rosclaw_compute / rosclaw_execute 退出默认模型面——能力以
 * CapabilitySnapshot 物化的精确强类型工具直接进入（materialize.ts）；
 * 两者仍是 bridge wire 上的验证链 plumbing（物化工具内部调用）。
 * R0-1.5：rosclaw_task 退出模型面——已知 recipe 由输入路由自动
 * 执行（TaskExecutionService 唯一生产链，零模型调用）；wire 层
 * adapter 保留兼容。 */
export const EMBODIMENT_PACK: readonly string[] = [
	"rosclaw_status",
	"rosclaw_capabilities",
	"rosclaw_observe",
	"rosclaw_verify",
	"rosclaw_request_action",
	"rosclaw_memory_query",
	"rosclaw_inspect",
	"rosclaw_fail_safe",
	// operation 停止（P0-D：轮询式 wait_operation 退出模型面——
	// Operation 事件流驱动，模型不轮询）
	"rosclaw_stop_operation",
];

/** 模型可见工具（唯一真相源）。 */
export const MODEL_TOOL_NAMES: readonly string[] = [
	...WORKSPACE_PACK,
	...EMBODIMENT_PACK,
	...PRODUCT_PACK,
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
	// R0-1.5（金丝雀实证 + 0826 审计 §6 删除清单）：任务级入口
	// 由输入路由自动执行（零模型调用）——模型面不再有
	// rosclaw_task；wire 层 adapter 保留（兼容既有 session）。
	"rosclaw_task",
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
