/** pi-bridge UDS 客户端（PNA-1/PNA-2）：JSONL，token 从 0600 文件读。 */

import { readFileSync } from "node:fs";
import { connect } from "node:net";

export class BridgeError extends Error {
	constructor(
		message: string,
		readonly code: string = "",
	) {
		super(message);
	}
}

export function bridgeToken(rosclawHome: string): string {
	if (process.env.ROSCLAW_CONTROL_TOKEN) return process.env.ROSCLAW_CONTROL_TOKEN;
	try {
		return readFileSync(`${rosclawHome}/run/agentd-control.token`, "utf-8").trim();
	} catch {
		return "";
	}
}

export async function bridgeCall(
	rosclawHome: string,
	method: string,
	params: Record<string, unknown> = {},
): Promise<Record<string, unknown>> {
	const socketPath = `${rosclawHome}/run/pi-bridge.sock`;
	const token = bridgeToken(rosclawHome);
	return await new Promise((resolve, reject) => {
		const conn = connect(socketPath);
		let buffer = "";
		const timeout = setTimeout(() => {
			conn.destroy();
			reject(new BridgeError("bridge call timeout", "TIMEOUT"));
		}, 5000);
		conn.on("connect", () => {
			conn.write(JSON.stringify({ method, params: { token, ...params } }) + "\n");
		});
		conn.on("data", (chunk) => {
			buffer += chunk.toString();
			const idx = buffer.indexOf("\n");
			if (idx === -1) return;
			clearTimeout(timeout);
			conn.end();
			try {
				resolve(JSON.parse(buffer.slice(0, idx)) as Record<string, unknown>);
			} catch (err) {
				reject(err);
			}
		});
		conn.on("error", (err) => {
			clearTimeout(timeout);
			reject(new BridgeError(`bridge unavailable: ${err.message}`, "BRIDGE_UNAVAILABLE"));
		});
	});
}
