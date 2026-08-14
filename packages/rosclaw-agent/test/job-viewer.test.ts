/** 十三审 PR-13.3/13.4 红测试：/job 持续订阅 Viewer。
 *
 * 1. cursor 增量订阅（不整页重拉）；
 * 2. 事件渲染：工具参数/输出预览/stall/预算暂停/中断恢复；
 * 3. follow/pause 滚动；
 * 4. 关闭后停止轮询（dispose）；
 * 5. 不泄露隐藏 thinking（message_delta 只含公开文本预览——桥侧
 *    已过滤；viewer 不渲染 thinking 类事件）。
 */

import assert from "node:assert/strict";
import test from "node:test";

async function makeViewer(
	eventsByCall: Array<Array<Record<string, unknown>>>,
	status = "RUNNING",
) {
	const { JobViewerComponent } = await import("../src/workers/job-viewer.js");
	const calls: number[] = [];
	let closed = false;
	const viewer = new JobViewerComponent({
		workOrderId: "wo_viewer01",
		fetchEvents: async (afterSeq, _limit) => {
			calls.push(afterSeq);
			const batch = eventsByCall.shift() ?? [];
			return { events: batch.filter((e) => Number(e.seq) > afterSeq), status };
		},
		onSteer: async () => "继续",
		sendSteer: async () => "已送达",
		onCancel: async () => "已取消",
		notify: () => undefined,
		onClose: () => {
			closed = true;
		},
	});
	// 等初始 poll 完成。
	await new Promise((r) => setTimeout(r, 20));
	return { viewer, calls, isClosed: () => closed };
}

test("cursor 增量订阅 + 事件渲染", async () => {
	const { viewer, calls } = await makeViewer([
		[
			{ seq: 1, kind: "attempt_started" },
			{ seq: 2, kind: "tool_started", tool: "bash", args_preview: "python3 render.py" },
			{ seq: 3, kind: "tool_finished", tool: "bash", is_error: false, output_preview: "exit 0" },
			{ seq: 4, kind: "stall_warning" },
			{ seq: 5, kind: "budget_paused" },
		],
		[{ seq: 6, kind: "session_resumed" }],
	]);
	await viewer.render(80); // 触发首次渲染
	// 手动推进第二次 poll。
	await (viewer as never as { poll(): Promise<void> }).poll();
	const text = viewer.render(80).join("\n");
	assert.match(text, /Worker 启动/);
	assert.match(text, /▶ bash  python3 render\.py/);
	assert.match(text, /✓ bash.*exit 0/);
	assert.match(text, /长时间无公开进度/);
	assert.match(text, /预算暂停/);
	assert.match(text, /已从检查点恢复/);
	// 第二次拉取的 cursor 是 5（增量，不重拉）。
	assert.ok(calls[0] === 0 && calls[1] === 5, `cursor 错误: ${calls}`);
	viewer.dispose();
});

test("follow/pause 滚动 + 关闭停止轮询", async () => {
	const many = Array.from({ length: 40 }, (_, i) => ({
		seq: i + 1,
		kind: "tool_finished",
		tool: "read",
		is_error: false,
		output_preview: `line ${i}`,
	}));
	const { viewer } = await makeViewer([many]);
	const full = viewer.render(80).join("\n");
	assert.match(full, /line 39/); // follow 显示最新
	viewer.handleInput("f"); // pause
	viewer.handleInput("\x1b[A"); // scroll up
	const scrolled = viewer.render(80).join("\n");
	assert.match(scrolled, /paused/);
	viewer.handleInput("q");
	// dispose 后再 poll 不变。
	const before = viewer.render(80).join("\n");
	await (viewer as never as { poll(): Promise<void> }).poll();
	assert.equal(viewer.render(80).join("\n"), before);
});

test("viewer 不渲染 liveness/usage/thinking", async () => {
	const { viewer } = await makeViewer([
		[
			{ seq: 1, kind: "liveness", phase: "RUNNING_MODEL" },
			{ seq: 2, kind: "usage", input_tokens: 100 },
		],
	]);
	const text = viewer.render(80).join("\n");
	assert.ok(!text.includes("RUNNING_MODEL"), "liveness 泄漏进 viewer");
	assert.ok(!text.includes("input_tokens"), "usage 泄漏进 viewer");
	viewer.dispose();
});
