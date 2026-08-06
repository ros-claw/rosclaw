/** ModelTurnRequest → pi-ai stream → modeld SSE 事件（批次 D §7.2/§7.3）。
 *
 * modeld 只做协议转换与转发：不解析 DecisionV1、不选工具、不维护状态。
 */

import type { Api, Context, Message, Model, Provider, Tool } from "@earendil-works/pi-ai";

type AnyModel = Model<Api>;
type AnyProvider = Provider<Api>;
import { redact } from "./redact.js";

export interface ModeldTurnRequest {
	provider: string;
	model: string;
	system_prompt?: string;
	messages: Array<Record<string, unknown>>;
	tools?: Array<{ name: string; description: string; parameters: Record<string, unknown> }>;
	max_tokens?: number;
	reasoning_effort?: string;
}

export type ModeldEvent =
	| { type: "text.delta"; text: string }
	| { type: "tool_call"; call_id: string; name: string; arguments: Record<string, unknown> }
	| { type: "usage"; input: number; output: number; total: number }
	| { type: "done"; stop_reason: string; assistant_message: Record<string, unknown> }
	| { type: "error"; kind: string; message: string };

/** OpenAI chat 消息 → pi-ai Message（含 tool_calls / tool result 回填）。 */
export function toPiMessages(raw: Array<Record<string, unknown>>): Message[] {
	const out: Message[] = [];
	const now = Date.now();
	for (const msg of raw) {
		const role = String(msg.role ?? "");
		if (role === "user" || role === "system") {
			out.push({ role: "user", content: String(msg.content ?? ""), timestamp: now });
		} else if (role === "assistant") {
			const content: Array<
				{ type: "text"; text: string } | { type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> }
			> = [];
			if (msg.content) content.push({ type: "text", text: String(msg.content) });
			for (const call of (msg.tool_calls as Array<Record<string, unknown>>) ?? []) {
				const fn = (call.function ?? {}) as Record<string, unknown>;
				let args: Record<string, unknown> = {};
				try {
					args = JSON.parse(String(fn.arguments ?? "{}")) as Record<string, unknown>;
				} catch {
					args = {};
				}
				content.push({
					type: "toolCall",
					id: String(call.id ?? ""),
					name: String(fn.name ?? ""),
					arguments: args,
				});
			}
			out.push({
				role: "assistant",
				content,
				api: "openai-completions",
				provider: "unknown",
				model: "unknown",
				usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: ZERO_COST },
				stopReason: "stop",
				timestamp: now,
			});
		} else if (role === "tool") {
			out.push({
				role: "toolResult",
				toolCallId: String(msg.tool_call_id ?? ""),
				toolName: String(msg.name ?? "tool"),
				content: [{ type: "text", text: String(msg.content ?? "") }],
				isError: false,
				timestamp: now,
			});
		}
	}
	return out;
}

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 };

function toPiTools(raw: ModeldTurnRequest["tools"]): Tool[] {
	return (raw ?? []).map((t) => ({
		name: t.name,
		description: t.description,
		parameters: t.parameters as Tool["parameters"],
	}));
}

export async function* streamTurn(
	provider: AnyProvider,
	model: AnyModel,
	request: ModeldTurnRequest,
	apiKey: string | undefined,
	signal: AbortSignal,
): AsyncGenerator<ModeldEvent> {
	const context: Context = {
		systemPrompt: request.system_prompt,
		messages: toPiMessages(request.messages),
		tools: toPiTools(request.tools),
	};
	try {
		const stream = provider.stream(model, context, {
			apiKey,
			maxTokens: request.max_tokens,
			signal,
		});
		for await (const event of stream) {
			if (event.type === "text_delta") {
				yield { type: "text.delta", text: event.delta };
			} else if (event.type === "toolcall_end") {
				yield {
					type: "tool_call",
					call_id: event.toolCall.id,
					name: event.toolCall.name,
					arguments: event.toolCall.arguments,
				};
			} else if (event.type === "done") {
				const message = event.message;
				yield {
					type: "usage",
					input: message.usage.input,
					output: message.usage.output,
					total: message.usage.totalTokens,
				};
				yield {
					type: "done",
					stop_reason: event.reason,
					assistant_message: {
						role: "assistant",
						content: message.content
							.filter((c) => c.type === "text")
							.map((c) => (c as { text: string }).text)
							.join(""),
						tool_calls: message.content
							.filter((c) => c.type === "toolCall")
							.map((c) => {
								const call = c as { id: string; name: string; arguments: Record<string, unknown> };
								return {
									id: call.id,
									type: "function",
									function: { name: call.name, arguments: JSON.stringify(call.arguments) },
								};
							}),
					},
				};
			} else if (event.type === "error") {
				yield {
					type: "error",
					kind: event.reason,
					message: redact(event.error.errorMessage ?? "provider error", apiKey ? [apiKey] : []),
				};
			}
		}
	} catch (err) {
		yield {
			type: "error",
			kind: "transport",
			message: redact((err as Error).message, apiKey ? [apiKey] : []),
		};
	}
}
