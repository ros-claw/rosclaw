/** 0902 R3-b 红测试（§6.1 三层界面）：终态卡默认层不显示内部
 *  绝对路径——文件名 + 大小 + 短打开命令；完整路径进诊断层
 *  （verbose）/ `rosclaw artifact path`。
 *
 * 0902 实证：任务卡给普通用户看 /home/ubuntu/.rosclaw/sim/traces/
 * trace_xxx/trace.json 这种内部路径——内部复杂度成了用户负担。
 * 0901 P0-6 的"必须看到绝对路径"由 open/path/export 命令承接
 * （可达性不丢），默认层不再裸打路径。
 *
 * 闭环断言：
 * 1. 默认：文件名 + 大小 + rosclaw open 短命令在；绝对路径不在；
 * 2. verbose：绝对路径仍在（诊断层保留）；
 * 3. 失败卡的"原因逐条"不受影响（0901 P0-6 红线保留）。
 */

import assert from "node:assert/strict";
import test from "node:test";

const REF = {
	artifact_id: "art_g",
	path: "/home/u/.rosclaw/sim/traces/trace_abc/trace_abc-scene.gif",
	media_type: "image/gif",
	size_bytes: 1234567,
	open_command: "rosclaw open art_g",
};

test("R3-b 默认层：文件名+大小+短命令在，内部绝对路径不在", async () => {
	const { renderTerminalReply } = await import(
		"../src/native/terminal-presenter.js"
	);
	const reply = renderTerminalReply({
		verification: "PASS",
		delivery: "DELIVERED",
		artifact_refs: [REF],
	});
	assert.match(reply, /trace_abc-scene\.gif/); // 文件名
	assert.match(reply, /1\.2 MB/); // 大小
	assert.match(reply, /rosclaw open art_g/); // 短打开命令
	// 大道至简 R2：剥 OSC 8 后断言可见文本（链接 escape 里嵌
	// 的绝对路径终端不可见——可见面仍无内部路径）。
	const visible = reply.replace(/\x1b\]8;;[^\x07]*\x07/g, "");
	assert.doesNotMatch(visible, /\/home\/u\/\.rosclaw\/sim\/traces/, // 内部路径不进默认层
		"默认层仍裸打内部绝对路径");
});

test("R3-b 诊断层（verbose）：绝对路径保留", async () => {
	const { renderTerminalReply } = await import(
		"../src/native/terminal-presenter.js"
	);
	const reply = renderTerminalReply({
		verification: "PASS",
		delivery: "DELIVERED",
		artifact_refs: [REF],
	}, { verbose: true });
	assert.match(reply, /\/home\/u\/\.rosclaw\/sim\/traces\/trace_abc\/trace_abc-scene\.gif/);
});

test("R3-b 失败卡原因逐条不受影响（0901 P0-6 红线）", async () => {
	const { renderTerminalReply } = await import(
		"../src/native/terminal-presenter.js"
	);
	const reply = renderTerminalReply({
		verification: "FAIL",
		delivery: "MISSING",
		artifact_refs: [],
		repair_directive: { failures: ["TRACKING_ERROR: 超阈值"] },
	});
	assert.match(reply, /原因/);
	assert.match(reply, /TRACKING_ERROR/);
});
