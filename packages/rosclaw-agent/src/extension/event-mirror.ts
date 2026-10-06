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
	/** 受保护 server 边界：pi.events.batch 每次请求原子接受 ≤256 条。
	 * 绝不把整个 pending 队列塞进一个请求——257+ 会被永久拒绝。 */
	private static readonly MAX_BATCH_EVENTS = 256;
	/** 有界 pending 上限：失败重放最多保留 1000 条（含在飞批次）防爆内存。
	 * 显式 overflow 策略：超出时丢弃**最新**尾部并计数（overflowDropped）
	 * ——本队列是有界缓冲，绝不声称无限不丢。 */
	private static readonly MAX_PENDING = 1000;
	private queue: MirrorEvent[] = [];
	private flushing = false;
	private overflowDropped = 0;
	// P0-A：provider retry/重连重放对同一 message 重复 push——
	// 只按稳定 provider 身份 (event_type, entryId) 去重；无身份的
	// 历史消息不按内容 hash 合并（NATIVE-TOKENS）。键→载荷指纹：
	// 同身份同载荷=幂等重放（静默一次）；同身份不同载荷=typed
	// conflict 抛错——绝不在 RPC 之前静默丢弃。
	private readonly mirrored = new Map<string, string>();

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
		// P0-A：稳定键去重——只有 provider 真身份（responseId/entryId）
		// 才能去重 provider retry/重连重放。
		// NATIVE-TOKENS：无稳定身份时**绝不**按内容 hash 去重——两条
		// 内容相同的历史真实消息是两笔独立付费用量；空 pi_entry_id
		// 的历史身份是 UNKNOWN，不能假装稳定。
		let dedupKey: string | null = null;
		if (options.entryId) dedupKey = `${eventType}:entry:${options.entryId}`;
		if (dedupKey !== null) {
			const fingerprint = contentHash(JSON.stringify({
				text: options.text ?? null,
				model: options.model ?? "",
				usage: options.usage ?? {},
			}));
			const seen = this.mirrored.get(dedupKey);
			if (seen !== undefined) {
				if (seen === fingerprint) return; // 同身份同载荷：幂等重放。
				// 同稳定身份不同载荷：冲突必须可观测（typed error），
				// 不能静默吞掉——provider 身份相同而内容不同意味着
				// 数据损坏或身份伪造，交给调用方显式处理。
				const conflict = new Error(
					`MIRROR_CONFLICT: stable identity ${dedupKey} replayed with a different payload`,
				);
				(conflict as Error & { code?: string }).code = "MIRROR_CONFLICT";
				throw conflict;
			}
			this.mirrored.set(dedupKey, fingerprint);
		}
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
		let stored = 0;
		try {
			// BACKLOG-FIX：有界 drain——每个请求只取 ≤256 条批次，
			// 循环直到队列清空或首次失败。flush 期间并发 push 的事件
			// 由同一循环的后续批次拾起（splice 是同步原子的），
			// 不重复、不丢、不改 payload、不动稳定身份。
			while (this.queue.length > 0) {
				const batch = this.queue.splice(0, EventMirror.MAX_BATCH_EVENTS);
				try {
					const response = await this.call(this.rosclawHome, "pi.events.batch", {
						events: batch,
					});
					if (!response.ok) {
						this.requeueFailedBatch(batch);
						return stored;
					}
					stored += Number(response.stored ?? 0);
				} catch {
					// bridge 抛错（瞬时故障/commit 后回执丢失）保留 bounded
					// pending：已 splice 的批次回到队首，后续 flush 重放。
					// commit-before-reply-loss 的重放靠 server 端同 mirror_id
					// 同载荷幂等去重——durable-idempotent，不产生重复行；
					// 稳定身份在确认落库前仍走幂等路径。
					this.requeueFailedBatch(batch);
					return stored;
				}
			}
			return stored;
		} finally {
			this.flushing = false;
		}
	}

	/** 失败批次回到队首（顺序保留：失败批次 + 后续 suffix），
	 * bounded 保留 ≤1000 条；超出丢弃最新尾部并计数（可观测）。 */
	private requeueFailedBatch(batch: MirrorEvent[]): void {
		const merged = [...batch, ...this.queue];
		if (merged.length > EventMirror.MAX_PENDING) {
			this.overflowDropped += merged.length - EventMirror.MAX_PENDING;
		}
		this.queue = merged.slice(0, EventMirror.MAX_PENDING);
	}

	get pending(): number {
		return this.queue.length;
	}

	/** 有界 overflow 策略下被丢弃的最新尾部条数（0 = 无丢弃）。 */
	get droppedOverflow(): number {
		return this.overflowDropped;
	}
}
