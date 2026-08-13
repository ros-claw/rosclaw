/** Provider 内容归一化（十二审 HOTFIX-12.1）。
 *
 * 根因：不同 Provider/兼容层的 message.content 可能是字符串、part 数组、
 * 单对象、嵌套对象或 null——直接 `.filter()` 即 `parts.filter is not a
 * function`（崩溃且被误分类为 MODEL_ERROR）。
 *
 * 所有 transcript/final text/事件桥接只消费归一化结构；未知 part 保留
 * 类型元数据并安全忽略，绝不让单条消息崩掉整个 WorkOrder。
 */

export interface NormalizedMessage {
	/** 公开文本（text parts 拼接）。 */
	text: string;
	/** 隐藏推理块——收集但绝不进 UI/transcript（审计 §3.1：不展示 CoT）。 */
	thinking: string;
	/** 工具调用（name + arguments）。 */
	toolCalls: Array<{ name: string; arguments: unknown }>;
	/** 未知 part 的类型元数据（安全忽略但留痕）。 */
	unknownPartTypes: string[];
}

interface RawPart {
	type?: string;
	text?: string;
	thinking?: string;
	name?: string;
	arguments?: unknown;
	input?: unknown;
	[id: string]: unknown;
}

function normalizePart(part: RawPart, out: NormalizedMessage): void {
	const type = part?.type;
	if (type === "text") {
		out.text += String(part.text ?? "");
	} else if (type === "thinking" || type === "reasoning" || type === "redacted_thinking") {
		out.thinking += String(part.thinking ?? part.text ?? "");
	} else if (type === "toolCall" || type === "tool_call" || type === "tool_use") {
		out.toolCalls.push({
			name: String(part.name ?? ""),
			arguments: part.arguments ?? part.input ?? {},
		});
	} else {
		out.unknownPartTypes.push(String(type ?? typeof part));
	}
}

export function normalizeAssistantContent(content: unknown): NormalizedMessage {
	const out: NormalizedMessage = {
		text: "",
		thinking: "",
		toolCalls: [],
		unknownPartTypes: [],
	};
	if (content === null || content === undefined) return out;
	if (typeof content === "string") {
		out.text = content;
		return out;
	}
	if (Array.isArray(content)) {
		for (const part of content) {
			if (typeof part === "string") out.text += part;
			else if (part && typeof part === "object") normalizePart(part as RawPart, out);
			else out.unknownPartTypes.push(typeof part);
		}
		return out;
	}
	if (typeof content === "object") {
		const obj = content as RawPart;
		// 嵌套 {content: [...]}（某些兼容层多包一层）。
		if (Array.isArray(obj.content) || typeof obj.content === "string") {
			return normalizeAssistantContent(obj.content);
		}
		if (obj.type) {
			normalizePart(obj, out);
			return out;
		}
		out.unknownPartTypes.push("object-without-type");
		return out;
	}
	out.unknownPartTypes.push(typeof content);
	return out;
}

/** 归一化消息数组 → 最终公开文本（最后一条 assistant）。 */
export function finalTextOfMessages(messages: Array<Record<string, unknown>>): string {
	for (let i = messages.length - 1; i >= 0; i -= 1) {
		const msg = messages[i] as { role?: string; content?: unknown };
		if (msg.role !== "assistant") continue;
		const normalized = normalizeAssistantContent(msg.content);
		if (normalized.text) return normalized.text;
	}
	return "";
}
