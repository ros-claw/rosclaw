/** P0-A 红测试（0824 总纲 §19.P0-A）：UI/镜像去重。
 *
 * 红测试先行——dedup helper 不存在时必须红。
 *
 * 1. EventMirror：provider retry 对同一 message 重复 push → 只镜像
 *    一次（raw delta 按 message/item 去重）；
 * 2. action_result 卡：同一 call_id 只 appendEntry 一次（同一 tool
 *    card 只能存在一张）。
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

describe("P0-A EventMirror 去重", () => {
	it("同一 entry 重复 push（provider retry）只镜像一次", async () => {
		const { EventMirror } = await import("../src/extension/event-mirror.js");
		const batches: Array<Array<Record<string, unknown>>> = [];
		const mirror = new EventMirror(
			"/tmp/p0a", "s1", "m1",
			async (_home, _method, params) => {
				batches.push((params as { events: Array<Record<string, unknown>> }).events);
				return { ok: true, stored: 1 };
			},
		);
		mirror.push("message_end", { entryId: "msg_1", text: "hello" });
		mirror.push("message_end", { entryId: "msg_1", text: "hello" }); // retry 重复
		mirror.push("message_end", { entryId: "msg_2", text: "world" });
		await mirror.flush();
		const all = batches.flat();
		const msg1 = all.filter((e) => e.pi_entry_id === "msg_1");
		assert.equal(msg1.length, 1, "provider retry 导致重复镜像");
		assert.equal(all.length, 2);
	});
});

describe("P0-A 稳定 ID upsert", () => {
	it("同一 call_id 的 action_result 卡只追加一次", async () => {
		const { StableIdDeduper } = await import("../src/extension/dedup.js");
		const deduper = new StableIdDeduper();
		assert.equal(deduper.check("call_1"), true, "首次应通过");
		assert.equal(deduper.check("call_1"), false, "重复 call 必须抑制（第二张卡）");
		assert.equal(deduper.check("call_2"), true, "不同 call 不受影响");
	});

	it("空 id 不去重（无稳定 ID 的事件不得互相吞掉）", async () => {
		const { StableIdDeduper } = await import("../src/extension/dedup.js");
		const deduper = new StableIdDeduper();
		assert.equal(deduper.check(""), true);
		assert.equal(deduper.check(""), true, "空 id 每次都必须通过");
	});
});
