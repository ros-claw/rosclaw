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

/** BOUNDED-IDENTITY：稳定身份载荷指纹的确定性序列化。
 * 对象键序（含嵌套对象）不影响语义相等——递归排序键后序列化；
 * 数组保持顺序（数组是有序语义）；undefined 值的键与 JSON 一样省略。
 * 真正变化的值仍产生不同指纹 → typed MIRROR_CONFLICT。 */
function canonicalJson(value: unknown): string {
	if (value === null || typeof value !== "object") {
		return JSON.stringify(value) ?? "null";
	}
	if (Array.isArray(value)) {
		return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
	}
	const record = value as Record<string, unknown>;
	const parts: string[] = [];
	for (const key of Object.keys(record).sort()) {
		if (record[key] === undefined) continue;
		parts.push(`${JSON.stringify(key)}:${canonicalJson(record[key])}`);
	}
	return `{${parts.join(",")}}`;
}

/** BOUNDED-IDENTITY：用户/调用方可观测的镜像同步诊断。
 * 只含有界计数/元数据——绝不携带原始消息/错误/thinking/token 内容。 */
export interface MirrorDiagnostics {
	/** 待同步条数（含在飞批次）。 */
	pending: number;
	/** 常驻稳定身份注册表大小（有界 ≤4096）。 */
	stable_identity_count: number;
	/** 有界容量策略下被拒绝/丢弃的累计条数（lifetime，0 = 无丢失）。 */
	overflow_dropped: number;
	/** true = 存在未确认交付（pending/在飞/溢出丢失）——绝不伪装成已同步完成。 */
	unconfirmed: boolean;
}

export class EventMirror {
	/** 受保护 server 边界：pi.events.batch 每次请求原子接受 ≤256 条。
	 * 绝不把整个 pending 队列塞进一个请求——257+ 会被永久拒绝。 */
	private static readonly MAX_BATCH_EVENTS = 256;
	/** BOUNDED-IDENTITY：单次 flush() 最多 4 个 RPC——连续生产下
	 * flush 必须有限返回；未确认的精确 suffix 留给后续 flush 调用。 */
	private static readonly MAX_FLUSH_RPCS = 4;
	/** 有界 pending 上限：最多保留 1000 条（含在飞批次）防爆内存。
	 * 显式 overflow 策略：容量已满时 typed 拒绝新 push 并计数
	 * （overflowDropped）——本队列是有界缓冲，绝不声称无限不丢，
	 * 被拒绝的载荷绝不静默变成已确认入账。 */
	private static readonly MAX_PENDING = 1000;
	/** 常驻稳定身份注册表上限 4096。超出时淘汰最旧条目——这不是
	 * 盲淘汰：server 端 durable stable-entry/v1 协议按
	 * (session, mission, event_type, pi_entry_id) 幂等去重，
	 * 淘汰后的重放由 durable 层兜底，绝不重复计费。 */
	private static readonly MAX_STABLE_IDENTITY = 4096;
	/** 新生产镜像显式声明的 durable 稳定身份协议（wire opt-in）。
	 * 缺省该字段的 legacy 载荷保持原 mirror_id 准入行为。 */
	static readonly STABLE_IDENTITY_PROTOCOL = "stable-entry/v1";
	private queue: MirrorEvent[] = [];
	private flushing = false;
	private inFlight = 0;
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
		// BOUNDED-IDENTITY：稳定键含 (session, mission, event_type, entryId)
		// ——retarget 后的不同 session/mission 作用域绝不错误去重，
		// 与 durable stable-entry/v1 server 键一致。
		let dedupKey: string | null = null;
		let fingerprint = "";
		if (options.entryId) {
			dedupKey = `${this.piSessionId}\u0000${this.missionId}\u0000${eventType}\u0000entry\u0000${options.entryId}`;
		}
		if (dedupKey !== null) {
			// 语义指纹：usage 对象键序（含嵌套）无关——canonicalJson
			// 递归排序键；真正变化的值仍是 typed conflict。
			fingerprint = contentHash(canonicalJson({
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
		}
		// BOUNDED-IDENTITY：pending（含在飞批次）容量 1000 是硬边界——
		// 满时 typed 拒绝并计 overflow，绝不静默接受后丢失。注意顺序：
		// 容量检查在注册 dedup 条目**之前**——被拒绝的事件不留下身份
		// 记录，容量释放后同身份可以真实重试（绝不静默抑制重放）。
		if (this.queue.length + this.inFlight >= EventMirror.MAX_PENDING) {
			this.overflowDropped += 1;
			const overflow = new Error(
				`MIRROR_OVERFLOW: pending mirror capacity ${EventMirror.MAX_PENDING} ` +
					"(including in-flight) reached; event rejected by bounded backpressure " +
					"policy and counted as overflow loss",
			);
			(overflow as Error & { code?: string }).code = "MIRROR_OVERFLOW";
			throw overflow;
		}
		if (dedupKey !== null) {
			// 有界注册表：满时淘汰最旧条目。durable server 端
			// stable-entry/v1 幂等准入兜底——淘汰后重放绝不重复计数。
			if (this.mirrored.size >= EventMirror.MAX_STABLE_IDENTITY) {
				const oldest = this.mirrored.keys().next();
				if (!oldest.done) this.mirrored.delete(oldest.value);
			}
			this.mirrored.set(dedupKey, fingerprint);
		}
		this.queue.push({
			mirror_id: `mir_${randomUUID().slice(0, 12)}`,
			identity_protocol: EventMirror.STABLE_IDENTITY_PROTOCOL,
			pi_session_id: this.piSessionId,
			mission_id: this.missionId,
			event_type: eventType,
			pi_entry_id: options.entryId ?? "",
			content_hash: options.text !== undefined ? contentHash(options.text) : "",
			model: options.model ?? "",
			usage: options.usage ?? {},
			occurred_at: new Date().toISOString(),
		} as MirrorEvent & { mirror_id: string; identity_protocol: string });
	}

	async flush(): Promise<number> {
		if (this.flushing || this.queue.length === 0) return 0;
		this.flushing = true;
		let stored = 0;
		let rpcs = 0;
		try {
			// BACKLOG-FIX：有界 drain——每个请求只取 ≤256 条批次。
			// BOUNDED-IDENTITY：单次 flush 最多 MAX_FLUSH_RPCS(4) 个 RPC——
			// 连续生产（flush 期间并发 push）下 flush 必须有限返回；
			// 未送出的精确 suffix 留在队列，由后续 flush 调用拾起，
			// 不重复、不丢、不改 payload、不动稳定身份。
			while (this.queue.length > 0 && rpcs < EventMirror.MAX_FLUSH_RPCS) {
				rpcs += 1;
				const batch = this.queue.splice(0, EventMirror.MAX_BATCH_EVENTS);
				this.inFlight += batch.length;
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
				} finally {
					this.inFlight -= batch.length;
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
		return this.queue.length + this.inFlight;
	}

	/** 有界 overflow 策略下被拒绝/丢弃的累计条数（0 = 无丢失）。 */
	get droppedOverflow(): number {
		return this.overflowDropped;
	}

	/** BOUNDED-IDENTITY：用户可观测同步诊断——pending（含在飞）/
	 * 常驻稳定身份数/溢出丢失累计/未确认标志。有界计数，无原始内容。 */
	diagnostics(): MirrorDiagnostics {
		const pending = this.queue.length + this.inFlight;
		return {
			pending,
			stable_identity_count: this.mirrored.size,
			overflow_dropped: this.overflowDropped,
			unconfirmed: pending > 0 || this.overflowDropped > 0,
		};
	}
}
