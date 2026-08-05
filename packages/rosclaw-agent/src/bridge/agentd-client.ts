/** agentd 状态客户端（PNA-0）：HTTP + ephemeral control token（0600 文件）。
 *
 * 只读。PNA-1 会换成专用 pi-bridge UDS；PNA-0 复用现有 HTTP 面。
 */

import { readFileSync } from "node:fs";

export interface AgentdStatus {
	reachable: boolean;
	baseUrl: string;
	status?: Record<string, unknown>;
	error?: string;
}

function controlToken(home: string): string {
	if (process.env.ROSCLAW_CONTROL_TOKEN) return process.env.ROSCLAW_CONTROL_TOKEN;
	try {
		return readFileSync(`${home}/run/agentd-control.token`, "utf-8").trim();
	} catch {
		return "";
	}
}

export async function fetchAgentdStatus(
	home: string,
	baseUrl = process.env.ROSCLAW_AGENTD_URL ?? "http://127.0.0.1:8765",
): Promise<AgentdStatus> {
	const token = controlToken(home);
	try {
		const res = await fetch(`${baseUrl}/status`, {
			headers: token ? { "x-rosclaw-token": token } : {},
			signal: AbortSignal.timeout(3000),
		});
		if (!res.ok) {
			return {
				reachable: false,
				baseUrl,
				error: `HTTP ${res.status}${res.status === 401 ? " (control token required)" : ""}`,
			};
		}
		return { reachable: true, baseUrl, status: (await res.json()) as Record<string, unknown> };
	} catch (err) {
		return { reachable: false, baseUrl, error: (err as Error).message };
	}
}
