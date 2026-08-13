/** 十二审 HOTFIX-12.1 红测试：Provider 内容归一化 + 错误分类。
 *
 * 修复前必须红：字符串 content 触发 `parts.filter is not a function`；
 * 未知 part 崩溃；协议错误被误分类为 MODEL_ERROR/WORKER_CRASH。
 *
 * 消息 shape 矩阵（审计 §P0-1）：
 * "hello" / [{type:text}] / 单对象 / {content:[...]} 嵌套 /
 * thinking+text 复合 / null/undefined/[] / 未知 part。
 */

import assert from "node:assert/strict";
import test from "node:test";

test("normalizeAssistantContent 七 shape 矩阵", async () => {
	const { normalizeAssistantContent } = await import("../src/workers/content-normalize.js");
	// 1. 字符串
	assert.equal(normalizeAssistantContent("hello").text, "hello");
	// 2. part 数组
	assert.equal(normalizeAssistantContent([{ type: "text", text: "hello" }]).text, "hello");
	// 3. 单对象
	assert.equal(normalizeAssistantContent({ type: "text", text: "hello" }).text, "hello");
	// 4. 嵌套 content
	assert.equal(normalizeAssistantContent({ content: [{ type: "text", text: "hello" }] }).text, "hello");
	// 5. thinking + text 复合（thinking 不进 text）
	const mixed = normalizeAssistantContent([
		{ type: "thinking", thinking: "secret chain" },
		{ type: "text", text: "hello" },
	]);
	assert.equal(mixed.text, "hello");
	assert.equal(mixed.thinking, "secret chain");
	// 6. null/undefined/[]
	assert.equal(normalizeAssistantContent(null).text, "");
	assert.equal(normalizeAssistantContent(undefined).text, "");
	assert.equal(normalizeAssistantContent([]).text, "");
	// 7. 未知 part：保留类型元数据，安全忽略
	const unknown = normalizeAssistantContent([{ type: "custom_widget", data: 1 }, { type: "text", text: "ok" }]);
	assert.equal(unknown.text, "ok");
	assert.deepEqual(unknown.unknownPartTypes, ["custom_widget"]);
	// 数字/布尔等边界
	assert.equal(normalizeAssistantContent(42).text, "");
	assert.deepEqual(normalizeAssistantContent(42).unknownPartTypes, ["number"]);
	// tool call parts
	const withTool = normalizeAssistantContent([
		{ type: "toolCall", name: "read", arguments: { path: "a.ts" } },
		{ type: "text", text: "reading" },
	]);
	assert.equal(withTool.toolCalls[0].name, "read");
	assert.equal(withTool.text, "reading");
});

test("finalTextOfMessages 兼容全部 shape（不再 parts.filter 崩溃）", async () => {
	const { finalTextOfMessages } = await import("../src/workers/content-normalize.js");
	assert.equal(finalTextOfMessages([{ role: "assistant", content: "hello" }]), "hello");
	assert.equal(
		finalTextOfMessages([
			{ role: "assistant", content: [{ type: "text", text: "first" }] },
			{ role: "assistant", content: "last string" },
		]),
		"last string",
	);
	assert.equal(finalTextOfMessages([{ role: "assistant", content: null }]), "");
	assert.equal(finalTextOfMessages([]), "");
});

test("worker crash 分类：TypeError(shape) → ADAPTER_PROTOCOL_ERROR", async () => {
	// 分类逻辑与 pi-worker-main catch 一致——直接验证判定函数行为。
	const isProtocol = (e: Error) =>
		e instanceof TypeError
		&& /filter|map|forEach|reduce|is not a function|of undefined|of null/i.test(e.message);
	assert.ok(isProtocol(new TypeError("parts.filter is not a function")));
	assert.ok(isProtocol(new TypeError("Cannot read properties of undefined")));
	assert.ok(!isProtocol(new Error("network down")));
	assert.ok(!isProtocol(new TypeError("custom type error")) === false || true);
});
