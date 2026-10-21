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

/** pi.tools.execute 是长任务工具路径（确定性 SIM 闭环/仿真渲染可达
 *  分钟级）——不能用状态 ping 的 5s 超时。 */
const TOOL_EXECUTE_TIMEOUT_MS = 900_000;

export async function bridgeCall(
	rosclawHome: string,
	method: string,
	params: Record<string, unknown> = {},
): Promise<Record<string, unknown>> {
	const socketPath = `${rosclawHome}/run/pi-bridge.sock`;
	const token = bridgeToken(rosclawHome);
	const timeoutMs = method === "pi.tools.execute" ? TOOL_EXECUTE_TIMEOUT_MS : 5000;
	return await new Promise((resolve, reject) => {
		const conn = connect(socketPath);
		let buffer = "";
		let settled = false;
		const fail = (error: BridgeError) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			conn.destroy();
			reject(error);
		};
		const timeout = setTimeout(() => {
			fail(new BridgeError("bridge call timeout", "TIMEOUT"));
		}, timeoutMs);
		// Decode across socket chunks, including split multibyte input echoes.
		conn.setEncoding("utf8");
		conn.on("connect", () => {
			if (settled) return;
			conn.write(JSON.stringify({ method, params: { token, ...params } }) + "\n");
		});
		conn.on("data", (chunk) => {
			if (settled) return;
			buffer += chunk;
			const idx = buffer.indexOf("\n");
			if (idx === -1) return;
			try {
				const response: unknown = JSON.parse(buffer.slice(0, idx));
				if (response === null || typeof response !== "object" || Array.isArray(response)) {
					throw new Error("invalid response shape");
				}
				settled = true;
				clearTimeout(timeout);
				conn.destroy();
				resolve(response as Record<string, unknown>);
			} catch {
				fail(new BridgeError("invalid bridge response", "INVALID_RESPONSE"));
			}
		});
		const incomplete = () => {
			fail(new BridgeError("bridge closed before complete response", "INCOMPLETE_RESPONSE"));
		};
		conn.on("end", incomplete);
		conn.on("close", incomplete);
		conn.on("error", (err) => {
			fail(new BridgeError(`bridge unavailable: ${err.message}`, "BRIDGE_UNAVAILABLE"));
		});
	});
}
