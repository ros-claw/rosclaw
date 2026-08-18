/** 十六审 A3 红测试：worker 终态协议解析（TERMINAL STATUS 行 →
 * 结构化 cause）。解析是 harness 自有协议的确定性行为——缺标记
 * COMPLETED，多标记取最后一行，大小写/空白容忍。
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { terminalStatusFromReport } from "../src/workers/content-normalize.js";

test("terminalStatusFromReport: 缺标记 = COMPLETED（有报告即完成）", () => {
	assert.equal(terminalStatusFromReport("答案是 42。"), "COMPLETED");
	assert.equal(terminalStatusFromReport(""), "COMPLETED");
});

test("terminalStatusFromReport: 末尾 BLOCKED 标记 → BLOCKED", () => {
	const report =
		"我只有 read/grep/find/ls，无法执行 pip install。\n" +
		"缺失能力：process.exec / network。\n\n" +
		"TERMINAL STATUS: BLOCKED";
	assert.equal(terminalStatusFromReport(report), "BLOCKED");
});

test("terminalStatusFromReport: 显式 COMPLETED 标记", () => {
	assert.equal(
		terminalStatusFromReport("已完成。\nTERMINAL STATUS: COMPLETED"),
		"COMPLETED",
	);
});

test("terminalStatusFromReport: 取最后一行（中途引用不误判）", () => {
	const report =
		"TERMINAL STATUS: BLOCKED 是协议示例。\n" +
		"实际已完成全部工作。\n" +
		"TERMINAL STATUS: COMPLETED";
	assert.equal(terminalStatusFromReport(report), "COMPLETED");
});
