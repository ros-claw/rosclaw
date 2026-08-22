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
		let done = false;
		const unsubscribe = this._session.subscribe((event) => {
			const mapped = mapPiEvent(event as { type?: string } & Record<string, unknown>);
			if (mapped) {
				queue.push(mapped);
				notify?.();
			}
		});
		try {
			while (!done) {
				if (!queue.length) {
					await new Promise<void>((resolve) => {
						notify = resolve;
						// 无事件时的兜底轮询（adapter 边界，避免永久挂起）。
						setTimeout(resolve, 250);
					});
					notify = undefined;
					continue;
				}
				const next = queue.shift();
				if (next) yield next;
				if (next?.type === "session.idle") done = false; // 保持流开着
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
		// ModelRegistry 在运行时装配——经 session 的 modelRegistry 解析。
		const registry = (this._session as unknown as {
			modelRegistry?: { find(provider: string, id: string): unknown };
		}).modelRegistry;
		const found = registry?.find(model.provider, model.model);
		if (!found) {
			throw new Error(`MODEL_NOT_FOUND: ${model.provider}/${model.model}`);
		}
		await this._session.setModel(found as never);
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
		this._session.dispose();
	}
}

/** Pi 私有事件 → HarnessEvent（唯一映射点）。 */
function mapPiEvent(event: { type?: string } & Record<string, unknown>): HarnessEvent | undefined {
	const turnId = String(event.turnId ?? "");
	const callId = String(event.callId ?? event.id ?? "");
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
		case "message_end":
			return { type: "assistant.completed", turnId, messageId: String(event.messageId ?? "") };
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
		case "auto_compaction_start":
			return { type: "compaction.started" };
		case "auto_compaction_end":
			return { type: "compaction.completed" };
		default:
			return undefined;
	}
}
