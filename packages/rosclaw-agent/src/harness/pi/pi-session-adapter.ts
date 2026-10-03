/** PiHarnessSession（PR-HP2）——AgentSession → HarnessSession 适配器。
 *
 * Pi 私有事件统一映射为 HarnessEvent；产品侧不 switch Pi 私有类型。
 * nativeRef = Pi 原生 session id（只在 binding 里，不进产品 UI）。
 */

import type { AgentSession } from "@earendil-works/pi-coding-agent";

import type {
	HarnessEvent,
	HarnessInput,
	HarnessSession,
	HarnessSessionRef,
} from "../port.js";
import { PI_BACKEND_ID } from "./pi-backend.js";

export class PiHarnessSession implements HarnessSession {
	readonly sessionRef: HarnessSessionRef;
	readonly cwd: string;
	private readonly _session: AgentSession;
	private _closed = false;
	private readonly _eventWaiters = new Set<() => void>();

	constructor(session: AgentSession, cwd: string) {
		this._session = session;
		this.cwd = cwd;
		this.sessionRef = {
			backendId: PI_BACKEND_ID,
			nativeRef: session.sessionId,
		};
	}

	async prompt(input: HarnessInput): Promise<void> {
		await this._session.prompt(input.text);
	}

	async steer(input: HarnessInput): Promise<void> {
		await this._session.steer(input.text);
	}

	async followUp(input: HarnessInput): Promise<void> {
		await this._session.followUp(input.text);
	}

	async *events(): AsyncIterable<HarnessEvent> {
		// 简单拉模式：subscribe 收集到队列，调用方按节奏消费。
		const queue: HarnessEvent[] = [];
		let notify: (() => void) | undefined;
		const unsubscribe = this._session.subscribe((event) => {
			const mapped = mapPiEvent(event as { type?: string } & Record<string, unknown>);
			if (mapped) {
				queue.push(mapped);
				notify?.();
			}
		});
		try {
			while (!this._closed) {
				if (!queue.length) {
					await new Promise<void>((resolve) => {
						const wake = () => {
							clearTimeout(timer);
							this._eventWaiters.delete(wake);
							resolve();
						};
						const timer = setTimeout(wake, 250);
						notify = wake;
						this._eventWaiters.add(wake);
					});
					notify = undefined;
					continue;
				}
				const next = queue.shift();
				if (next) yield next;
			}
		} finally {
			unsubscribe();
		}
	}

	async compact(instruction?: string): Promise<{ ok: boolean; detail?: string }> {
		try {
			await this._session.compact(instruction);
			return { ok: true };
		} catch (err) {
			return { ok: false, detail: (err as Error).message };
		}
	}

	async setModel(model: { provider: string; model: string }): Promise<void> {
		const found = this._session.modelRuntime.getModel(model.provider, model.model);
		if (!found) {
			throw new Error(`MODEL_NOT_FOUND: ${model.provider}/${model.model}`);
		}
		await this._session.setModel(found);
	}

	async setThinking(level: string): Promise<void> {
		(this._session as unknown as { setThinkingLevel?(l: string): void })
			.setThinkingLevel?.(level);
	}

	async cancelTurn(): Promise<void> {
		await this._session.abort();
	}

	async waitUntilIdle(): Promise<void> {
		while (!this._session.isIdle) {
			await new Promise((resolve) => setTimeout(resolve, 100));
		}
	}

	async close(): Promise<void> {
		if (this._closed) return;
		this._closed = true;
		for (const wake of this._eventWaiters) wake();
		await this._session.abort();
		this._session.dispose();
	}
}

/** Pi 私有事件 → HarnessEvent（唯一映射点）。 */
export function mapPiEvent(event: { type?: string } & Record<string, unknown>): HarnessEvent | undefined {
	const turnId = String(event.turnId ?? "");
	const callId = String(event.toolCallId ?? event.callId ?? event.id ?? "");
	switch (event.type) {
		case "turn_start":
			return { type: "turn.started", turnId };
		case "message_update": {
			const assistantEvent = event.assistantMessageEvent as { type?: string; delta?: string } | undefined;
			if (assistantEvent?.type === "text_delta") {
				return { type: "assistant.delta", turnId, text: String(assistantEvent.delta ?? "") };
			}
			return undefined;
		}
		case "message_end": {
			const message = event.message as { role?: string; stopReason?: string; errorMessage?: string } | undefined;
			if (message?.role !== "assistant") return undefined;
			if (message.stopReason === "aborted") return { type: "turn.cancelled", turnId };
			if (message.stopReason === "error") return {
				type: "turn.failed", turnId,
				error: { code: "PROVIDER_UNAVAILABLE", message: message.errorMessage ?? "Model request failed", retryable: true },
			};
			return { type: "assistant.completed", turnId, messageId: String(event.messageId ?? "") };
		}
		case "agent_end":
			return { type: "session.idle" };
		case "tool_execution_start":
			return { type: "tool.started", callId, tool: String(event.toolName ?? ""), args: event.args };
		case "tool_execution_update":
			return { type: "tool.updated", callId, update: event };
		case "tool_execution_end":
			return event.isError
				? {
						type: "tool.failed", callId,
						error: { code: "PROVIDER_UNAVAILABLE", message: String(event.result ?? ""), retryable: false },
					}
				: { type: "tool.completed", callId, result: event.result };
		case "compaction_start":
		case "auto_compaction_start":
			return { type: "compaction.started" };
		case "compaction_end":
		case "auto_compaction_end":
			if (event.aborted === true) return { type: "compaction.cancelled" };
			if (event.errorMessage || !event.result) return {
				type: "compaction.failed",
				error: { code: "PROVIDER_UNAVAILABLE", message: String(event.errorMessage ?? "Compaction returned no summary"), retryable: true },
			};
			return { type: "compaction.completed" };
		default:
			return undefined;
	}
}
