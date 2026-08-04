/** AgentService control API client (批次 C).
 *
 * The TUI only ever talks to the AgentService control plane — never to MCP,
 * workers, modeld, or rosclawd.
 */

export interface CommandSpec {
	name: string;
	aliases: string[];
	description: string;
	argument_hint: string;
	category: string;
	owner: string;
	availability: string[];
	during_turn: boolean;
	mutability: string;
	confirmation: string;
	required_capabilities: string[];
	handler: string;
	disabled_reason: string;
	args_schema?: import("../commands/args-parser.js").ArgsSchema;
}

export interface CommandResult {
	request_id: string;
	command_name: string;
	ok: boolean;
	message: string;
	data: Record<string, unknown>;
	error_code: string;
}

export interface MissionSnapshot {
	mission_id: string;
	name: string;
	goal_text: string;
	state: string;
	mode: string;
	body_id: string;
	context_id: string;
	context_revision: number;
	last_event_sequence: number;
	turn_in_flight: boolean;
	pending_approvals: Array<Record<string, unknown>>;
	active_grants: Array<Record<string, unknown>>;
	open_work_orders: Array<Record<string, unknown>>;
	usage: Record<string, unknown>;
	compaction_count: number;
	tool_count: number;
	captured_at: string;
}

export interface MissionInfo {
	mission_id: string;
	goal?: { text?: string };
	state?: string;
	mode?: string;
	updated_at?: string;
	[key: string]: unknown;
}

export interface TranscriptBlock {
	block_id: string;
	kind: "user" | "assistant" | "tool_call" | "tool_result" | "card" | "decision" | "receipt" | "error";
	sequence: number;
	text?: string;
	card?: Record<string, unknown>;
	decision?: Record<string, unknown>;
	receipt?: Record<string, unknown>;
	error?: string;
}

export interface TranscriptPage {
	mission_id: string;
	blocks: TranscriptBlock[];
	latest_sequence: number;
	oldest_sequence: number;
	has_more: boolean;
}

export class AgentClient {
	constructor(
		private readonly baseUrl: string,
		private readonly controlToken = "",
	) {}

	private authHeaders(extra?: Record<string, string>): Record<string, string> | undefined {
		const headers: Record<string, string> = { ...(extra ?? {}) };
		// P1-4：ephemeral control token（0600 文件/env 获取，永不打印）。
		if (this.controlToken) headers["x-rosclaw-token"] = this.controlToken;
		return Object.keys(headers).length > 0 ? headers : undefined;
	}

	private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
		const res = await fetch(`${this.baseUrl}${path}`, {
			method,
			headers: this.authHeaders(
				body !== undefined ? { "content-type": "application/json" } : undefined,
			),
			body: body !== undefined ? JSON.stringify(body) : undefined,
		});
		if (!res.ok) {
			const text = await res.text().catch(() => "");
			throw new Error(`HTTP ${res.status} ${method} ${path}: ${text.slice(0, 200)}`);
		}
		return (await res.json()) as T;
	}

	health(): Promise<Record<string, unknown>> {
		return this.request("GET", "/health");
	}

	listMissions(): Promise<MissionInfo[]> {
		return this.request("GET", "/missions");
	}

	createMission(goal: string, mode?: string): Promise<{ mission_id: string }> {
		return this.request("POST", "/missions", { goal, mode });
	}

	submitTurn(missionId: string, text: string): Promise<{ turn_id: string }> {
		return this.request("POST", `/v2/missions/${missionId}/turns`, { text });
	}

	cancelTurn(missionId: string): Promise<unknown> {
		return this.request("POST", `/v2/missions/${missionId}/cancel`, {});
	}

	capabilities(missionId: string): Promise<{ commands: CommandSpec[] }> {
		return this.request("GET", `/v1/capabilities?mission_id=${missionId}`);
	}

	command(
		missionId: string,
		name: string,
		args: Record<string, unknown>,
		idempotencyKey: string,
	): Promise<CommandResult> {
		return this.request("POST", `/v1/missions/${missionId}/commands`, {
			request_id: `req_${idempotencyKey}`,
			idempotency_key: idempotencyKey,
			command_name: name,
			arguments: args,
		});
	}

	snapshot(missionId: string): Promise<MissionSnapshot> {
		return this.request("GET", `/v1/missions/${missionId}/snapshot`);
	}

	pendingApprovals(missionId: string): Promise<Array<Record<string, unknown>>> {
		return this.request("GET", `/approvals/pending?mission_id=${missionId}`);
	}

	/** R4/P1-1：transcript projection（服务端权威投影 + 分页）。 */
	async transcript(
		missionId: string,
		beforeSeq?: number,
		limit = 500,
	): Promise<TranscriptPage> {
		const params = new URLSearchParams({ limit: String(limit) });
		if (beforeSeq !== undefined && beforeSeq > 0) params.set("before_seq", String(beforeSeq));
		const res = await fetch(
			`${this.baseUrl}/v2/missions/${missionId}/transcript?${params}`,
			{ headers: this.authHeaders() },
		);
		if (!res.ok) {
			throw new Error(`HTTP ${res.status} transcript: ${(await res.text()).slice(0, 200)}`);
		}
		return (await res.json()) as TranscriptPage;
	}

	/** 有界事件重放（resume 恢复 transcript 用；不走 SSE 长连接）。 */
	async replayEvents(
		missionId: string,
		afterSequence = 0,
	): Promise<Array<Record<string, unknown>>> {
		const res = await fetch(
			`${this.baseUrl}/v2/missions/${missionId}/events?follow=false&after_sequence=${afterSequence}`,
			{ headers: this.authHeaders({ accept: "text/event-stream" }) },
		);
		if (!res.ok || !res.body) return [];
		const text = await res.text();
		const events: Array<Record<string, unknown>> = [];
		for (const line of text.split("\n")) {
			if (line.startsWith("data: ")) {
				try {
					events.push(JSON.parse(line.slice(6)) as Record<string, unknown>);
				} catch {
					/* skip malformed frame */
				}
			}
		}
		return events;
	}
}

let counter = 0;
export function idempotencyKey(): string {
	counter += 1;
	return `tui_${Date.now().toString(36)}_${counter}`;
}
