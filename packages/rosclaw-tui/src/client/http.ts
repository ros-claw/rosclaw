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

export class AgentClient {
	constructor(private readonly baseUrl: string) {}

	private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
		const res = await fetch(`${this.baseUrl}${path}`, {
			method,
			headers: body !== undefined ? { "content-type": "application/json" } : undefined,
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
}

let counter = 0;
export function idempotencyKey(): string {
	counter += 1;
	return `tui_${Date.now().toString(36)}_${counter}`;
}
