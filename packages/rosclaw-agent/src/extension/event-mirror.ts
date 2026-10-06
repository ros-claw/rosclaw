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
		const batch = this.queue.splice(0, this.queue.length);
		try {
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
			// bridge 抛错（瞬时故障）同样保留 bounded pending 队列——
			// 已 splice 的批次不能丢，后续 flush 可成功重放；稳定身份
			// 在成功落库前不算确认（重放同载荷仍走幂等路径）。
			this.queue = [...batch, ...this.queue].slice(0, 1000);
			return 0;
		} finally {
			this.flushing = false;
		}
	}

	get pending(): number {
		return this.queue.length;
	}
}
