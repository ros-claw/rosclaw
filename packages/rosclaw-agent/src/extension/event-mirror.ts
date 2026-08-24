/** 认知事件镜像（PNA-8，规格 §24.2）：只镜像 hash/元数据到 agentd。
 *
 * 绝不镜像 assistant 全文（防双写不一致）；mirror 可关联认知与物理
 * 证据链（pi_entry_id + content_hash + model + usage）。
 */

import { createHash, randomUUID } from "node:crypto";
import { bridgeCall } from "../bridge/bridge-client.js";

export interface MirrorEvent {
	pi_session_id: string;
	mission_id: string;
	event_type: string;
	pi_entry_id?: string;
	content_hash?: string;
	model?: string;
	usage?: Record<string, unknown>;
	occurred_at: string;
}

export function contentHash(text: string): string {
	return `sha256:${createHash("sha256").update(text, "utf8").digest("hex")}`;
}

export class EventMirror {
	private queue: MirrorEvent[] = [];
	private flushing = false;
	// P0-A：provider retry/重连重放对同一 message 重复 push——
	// 按 (event_type, entryId|content_hash) 去重，只镜像一次。
	private readonly mirrored = new Set<string>();

	constructor(
		private readonly rosclawHome: string,
		private piSessionId: string,
		private missionId: string,
		private readonly call: typeof bridgeCall = bridgeCall,
	) {}

	/** NA-FIX-2：切换事务后重定向（不再写旧 session/mission）。 */
	retarget(piSessionId: string, missionId: string): void {
		this.piSessionId = piSessionId;
		this.missionId = missionId;
	}

	push(eventType: string, options: { entryId?: string; text?: string; model?: string; usage?: Record<string, unknown> } = {}): void {
		// P0-A：稳定键去重（entryId 优先；无 entryId 用内容 hash）。
		const dedupKey = `${eventType}:${options.entryId || (options.text !== undefined ? contentHash(options.text) : "")}`;
		if (dedupKey !== `${eventType}:` && this.mirrored.has(dedupKey)) return;
		if (dedupKey !== `${eventType}:`) this.mirrored.add(dedupKey);
		this.queue.push({
			mirror_id: `mir_${randomUUID().slice(0, 12)}`,
			pi_session_id: this.piSessionId,
			mission_id: this.missionId,
			event_type: eventType,
			pi_entry_id: options.entryId ?? "",
			content_hash: options.text !== undefined ? contentHash(options.text) : "",
			model: options.model ?? "",
			usage: options.usage ?? {},
			occurred_at: new Date().toISOString(),
		} as MirrorEvent & { mirror_id: string });
	}

	async flush(): Promise<number> {
		if (this.flushing || this.queue.length === 0) return 0;
		this.flushing = true;
		try {
			const batch = this.queue.splice(0, this.queue.length);
			const response = await this.call(this.rosclawHome, "pi.events.batch", {
				events: batch,
			});
			if (!response.ok) {
				// 失败放回队列（bounded：最多保留 1000 条防爆内存）。
				this.queue = [...batch, ...this.queue].slice(0, 1000);
				return 0;
			}
			return Number(response.stored ?? 0);
		} catch {
			return 0;
		} finally {
			this.flushing = false;
		}
	}

	get pending(): number {
		return this.queue.length;
	}
}
