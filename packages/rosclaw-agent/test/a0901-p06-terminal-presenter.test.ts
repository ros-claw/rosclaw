/** 0901-P0-6 红测试：TerminalPresenter 产品化 + 内核消息卡渲染器。
 *
 * 0901 体验实证（用户原话：从用户角度真的不太好）：
 * 1. 终态回复只有一行结论——没有原因（为什么失败）、没有文件名、
 *    没有绝对路径、没有下一步（用户看完不知道该干嘛）；
 * 2. 内部 customType 标签（[rosclaw.user_directive]/[rosclaw.task_terminal]）
 *    原样漏到屏幕——内部协议细节不该出现在用户界面。
 */

import assert from "node:assert/strict";
import test from "node:test";

test("P0-6 PASS 呈现：文件名 + 打开命令 + 下一步（0902 R3-b：绝对路径进诊断层）", async () => {
	const { renderTerminalReply } = await import(
		"../src/native/terminal-presenter.js"
	);
	const reply = renderTerminalReply({
		verification: "PASS",
		delivery: "DELIVERED",
		artifact_refs: [
			{
				artifact_id: "art_g",
				path: "/home/u/proj/outputs/star-scene.gif",
				media_type: "image/gif",
				size_bytes: 1234567,
				open_command: "rosclaw artifact open art_g",
			},
		],
	});
	assert.match(reply, /任务完成/);
	// 文件名（basename）。
	assert.match(reply, /star-scene\.gif/);
	// 打开命令仍在。
	assert.match(reply, /rosclaw artifact open art_g/);
	// 0902 R3-b：默认层不裸打绝对路径（内部路径进 verbose 诊断层）；
	// 可达性由 open/path/export 命令承接。
	// 大道至简 R2：文件名带 OSC 8 可点击链接（escape 序列里嵌
	// 绝对路径但终端不可见）——断言剥掉 OSC 8 后的可见文本。
	const visible = reply.replace(/\x1b\]8;;[^\x07]*\x07/g, "");
	assert.doesNotMatch(visible, /\/home\/u\/proj\/outputs\/star-scene\.gif/);
	const verbose = renderTerminalReply({
		verification: "PASS",
		delivery: "DELIVERED",
		artifact_refs: [
			{
				artifact_id: "art_g",
				path: "/home/u/proj/outputs/star-scene.gif",
				media_type: "image/gif",
				size_bytes: 1234567,
				open_command: "rosclaw artifact open art_g",
			},
		],
	}, { verbose: true });
	assert.match(verbose, /\/home\/u\/proj\/outputs\/star-scene\.gif/);
	// 下一步指引。
	assert.match(reply, /下一步/);
});

test("P0-6 FAIL 呈现：原因逐条列出（repair_directive.failures）+ 下一步", async () => {
	const { renderTerminalReply } = await import(
		"../src/native/terminal-presenter.js"
	);
	const reply = renderTerminalReply({
		verification: "FAIL",
		delivery: "MISSING",
		artifact_refs: [],
		repair_directive: {
			failures: [
				"DELIVERABLE_MISSING: required 交付物 scene_video 未在产物账本",
				"TRACKING_ERROR: 最大跟踪误差 0.031m 超阈值 0.020m",
			],
		},
	});
	assert.match(reply, /未完全达成/);
	// 原因区：用户必须能看到"为什么"。
	assert.match(reply, /原因/);
	assert.match(reply, /scene_video/);
	assert.match(reply, /0\.031m/);
	// 下一步指引。
	assert.match(reply, /下一步/);
});

test("P0-6 内核消息卡：三个 customType 都有注册卡（不漏内部标签）", async () => {
	const { kernelMessageRenderers } = await import(
		"../src/ui/kernel-message-cards.js"
	);
	const types = Object.keys(kernelMessageRenderers);
	assert.ok(types.includes("rosclaw.user_directive"));
	assert.ok(types.includes("rosclaw.task_terminal"));
	assert.ok(types.includes("rosclaw.task_explain"));
	for (const t of types) {
		const component = kernelMessageRenderers[t](
			{ customType: t, content: "任务已完成：验收 PASS" },
		);
		assert.ok(component, `${t} 应有渲染卡`);
		const lines = component.render(80);
		const text = lines.join("\n");
		// 内容呈现。
		assert.match(text, /任务已完成/);
		// 内部 customType 标签绝不上屏。
		assert.doesNotMatch(text, /\[?rosclaw\.(user_directive|task_terminal|task_explain)\]?/);
	}
});

test("P0-6 卡片渲染永不超宽（PTY 实证：超宽 = pi 进程崩溃退出）", async () => {
	const { kernelMessageRenderers } = await import(
		"../src/ui/kernel-message-cards.js"
	);
	const { visibleWidth } = await import("@earendil-works/pi-tui");
	// 0901 journey 实证：长绝对路径（>80 列）让卡片行超宽 →
	// pi uncaughtException 直接退出（Rendered line exceeds terminal
	// width）。卡片必须 wrap 到终端宽度内。
	const longPath =
		"/tmp/pytest-of-nvidia/pytest-2286/test_full_journey_pty0/prefix/" +
		"current/home/agentd/outputs/star-scene.gif";
	for (const [t, render] of Object.entries(kernelMessageRenderers)) {
		const component = render({
			customType: t,
			content: `✅ 任务完成：验收 PASS\n交付物：\n  路径：${longPath}\n  打开：rosclaw artifact open art_x`,
		});
		assert.ok(component);
		for (const w of [80, 120, 40]) {
			for (const line of component.render(w)) {
				assert.ok(
					visibleWidth(line) <= w,
					`${t} 在宽度 ${w} 超宽：${visibleWidth(line)}——${line}`,
				);
			}
		}
		// wrap 后路径信息仍在（不丢字符）。
		const text = component.render(80).join("\n");
		assert.ok(text.includes("star-scene.gif"), `${t} 丢了文件名`);
	}
});

test("P0-6 user_directive 卡标注确定性链接管", async () => {
	const { kernelMessageRenderers } = await import(
		"../src/ui/kernel-message-cards.js"
	);
	const component = kernelMessageRenderers["rosclaw.user_directive"]({
		customType: "rosclaw.user_directive",
		content: "画一个五角星",
	});
	assert.ok(component);
	const text = component.render(80).join("\n");
	assert.match(text, /画一个五角星/);
	assert.match(text, /确定性链/);
});
