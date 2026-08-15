/** ACP session/update → ROSClaw TaskEvent 映射（十五审 PR-RF-3，
 *  总纲 §6.1 表）。协议事件是唯一状态源——不解析文本。 */

export interface TaskEvent {
	kind: string;
	sessionId: string;
	[key: string]: unknown;
}

export function mapAcpUpdate(
	sessionId: string,
	update: Record<string, unknown>,
): TaskEvent {
	const tag = String(update.sessionUpdate ?? "");
	switch (tag) {
		case "agent_message_chunk": {
			const content = (update.content ?? {}) as { text?: string };
			return {
				kind: "worker.message.delta",
				sessionId,
				text: String(content.text ?? ""),
			};
		}
		case "agent_thought_chunk":
			// 思维链不外显——折叠为遥测。
			return { kind: "worker.telemetry", sessionId, note: "thought" };
		case "plan":
			return {
				kind: "worker.plan.updated",
				sessionId,
				entries: update.entries ?? [],
			};
		case "tool_call":
			return {
				kind: "worker.tool.started",
				sessionId,
				tool_call_id: String(update.toolCallId ?? ""),
				title: String(update.title ?? ""),
				status: String(update.status ?? "pending"),
			};
		case "tool_call_update": {
			const status = String(update.status ?? "");
			return {
				kind:
					status === "completed" || status === "failed"
						? "worker.tool.completed"
						: "worker.tool.updated",
				sessionId,
				tool_call_id: String(update.toolCallId ?? ""),
				status,
			};
		}
		case "available_commands_update":
			return { kind: "worker.session.updated", sessionId };
		case "current_mode_update":
			return {
				kind: "worker.session.updated",
				sessionId,
				mode: String(update.currentModeId ?? ""),
			};
		default:
			return { kind: `worker.acp.${tag || "unknown"}`, sessionId };
	}
}
