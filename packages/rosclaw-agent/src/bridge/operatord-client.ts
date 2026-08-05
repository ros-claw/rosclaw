/** operatord UDS 客户端（PNA-5）：JSONL，SO_PEERCRED 身份（无 token） */

import { connect } from "node:net";

export function defaultOperatorSocket(rosclawHome: string): string {
	return `${rosclawHome}/run/operatord.sock`;
}

export async function operatorCall(
	socketPath: string,
	method: string,
	params: Record<string, unknown> = {},
): Promise<Record<string, unknown>> {
	return await new Promise((resolve, reject) => {
		const conn = connect(socketPath);
		let buffer = "";
		const timeout = setTimeout(() => {
			conn.destroy();
			reject(new Error("operatord call timeout"));
		}, 8000);
		conn.on("connect", () => {
			conn.write(JSON.stringify({ method, params }) + "\n");
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
			reject(err);
		});
	});
}
