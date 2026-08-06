/** SSE client with Last-Event-ID resume (批次 B/C §5.3).
 *
 * - Reconnects with backoff, preserving the last seen sequence.
 * - Sequence gaps are surfaced via onGap so the consumer can stop
 *   optimistic rendering and re-pull the authoritative snapshot.
 * - Duplicate events are dropped by event_id.
 */

export interface AgentEvent {
	event_id: string;
	sequence: number;
	mission_id: string;
	turn_id?: string | null;
	type: string;
	visibility: string;
	payload: Record<string, unknown>;
	timestamp: string;
}

export interface SseOptions {
	signal?: AbortSignal;
	onGap?: (expected: number, got: number) => void;
	onReconnect?: (attempt: number) => void;
	maxAttempts?: number;
	/** R4/P1-1：resume 后从 transcript 的 latest_sequence 续接（exactly-once）。 */
	afterSequence?: number;
	/** P1-4：ephemeral control token。 */
	controlToken?: string;
}

/** 有界去重窗（P1-1：长会话不允许无界 seen Set）。 */
class BoundedDedup {
	private readonly queue: string[] = [];
	private readonly set = new Set<string>();

	constructor(private readonly capacity = 2048) {}

	has(id: string): boolean {
		return this.set.has(id);
	}

	add(id: string): void {
		if (this.set.has(id)) return;
		this.set.add(id);
		this.queue.push(id);
		if (this.queue.length > this.capacity) {
			const evicted = this.queue.shift();
			if (evicted !== undefined) this.set.delete(evicted);
		}
	}
}

export async function* streamEvents(
	baseUrl: string,
	missionId: string,
	options: SseOptions = {},
): AsyncGenerator<AgentEvent> {
	const seen = new BoundedDedup();
	let lastSeq = Math.max(0, options.afterSequence ?? 0);
	let attempt = 0;
	let emptyStreak = 0;
	const maxAttempts = options.maxAttempts ?? 100;

	while (attempt < maxAttempts && emptyStreak < 3) {
		if (options.signal?.aborted) return;
		const headers: Record<string, string> = { accept: "text/event-stream" };
		if (options.controlToken) headers["x-rosclaw-token"] = options.controlToken;
		if (lastSeq > 0) headers["last-event-id"] = String(lastSeq);
		let res: Response;
		try {
			res = await fetch(`${baseUrl}/v2/missions/${missionId}/events`, { headers, signal: options.signal });
		} catch {
			attempt += 1;
			options.onReconnect?.(attempt);
			await sleep(Math.min(200 * attempt, 3000), options.signal);
			continue;
		}
		if (!res.ok || !res.body) {
			attempt += 1;
			options.onReconnect?.(attempt);
			await sleep(Math.min(200 * attempt, 3000), options.signal);
			continue;
		}
		attempt = 0;
		let yielded = 0;
		const reader = res.body.getReader();
		const decoder = new TextDecoder();
		let buffer = "";
		try {
			for (;;) {
				const { done, value } = await reader.read();
				if (done) break;
				buffer += decoder.decode(value, { stream: true });
				let idx: number;
				while ((idx = buffer.indexOf("\n\n")) !== -1) {
					const frame = buffer.slice(0, idx);
					buffer = buffer.slice(idx + 2);
					const event = parseFrame(frame);
					if (!event) continue;
					if (seen.has(event.event_id)) continue; // 去重
					seen.add(event.event_id);
					if (lastSeq > 0 && event.sequence > lastSeq + 1) {
						options.onGap?.(lastSeq + 1, event.sequence);
					}
					lastSeq = Math.max(lastSeq, event.sequence);
					yielded += 1;
					yield event;
				}
			}
		} catch {
			// connection dropped → reconnect with Last-Event-ID
		}
		// 连接正常结束但没有新事件：对 bounded replay（follow=false）这是
		// 终止信号；连续空重连即退出，避免无限轮询。
		emptyStreak = yielded === 0 ? emptyStreak + 1 : 0;
		attempt += 1;
		options.onReconnect?.(attempt);
		await sleep(Math.min(200 * attempt, 3000), options.signal);
	}
}

function parseFrame(frame: string): AgentEvent | null {
	let data = "";
	for (const line of frame.split("\n")) {
		if (line.startsWith("data: ")) data += line.slice(6);
	}
	if (!data) return null;
	try {
		return JSON.parse(data) as AgentEvent;
	} catch {
		return null;
	}
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
	return new Promise((resolve) => {
		const timer = setTimeout(resolve, ms);
		signal?.addEventListener("abort", () => {
			clearTimeout(timer);
			resolve();
		}, { once: true });
	});
}
