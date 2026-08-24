/** 稳定 ID 去重（P0-A，0824 总纲 §19.P0-A）。
 *
 * 同一逻辑事件（provider retry/断流重连/重放注入）在 UI 只出现
 * 一次——同一 tool card 只能存在一张。空 id 不去重（无稳定 ID
 * 的事件不得互相吞掉）。
 */

export class StableIdDeduper {
	private readonly seen = new Set<string>();

	/** 首次返回 true；重复 id 返回 false（调用方应抑制渲染/追加）。 */
	check(id: string): boolean {
		if (!id) return true;
		if (this.seen.has(id)) return false;
		this.seen.add(id);
		return true;
	}
}
