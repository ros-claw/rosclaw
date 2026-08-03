/** operator.sock JSONL client (PR-11)：/estop 与审批的专用安全通道。
 *
 * TUI 只是 operator 终端——身份由 operator.sock 的 SO_PEERCRED 派生；
 * 此客户端不携带任何 principal。
 */

import { connect } from "node:net";

export function operatorCall(
	socketPath: string,
	method: string,
	params: Record<string, unknown> = {},
	timeoutMs = 5000,
): Promise<Record<string, unknown>> {
	return new Promise((resolve, reject) => {
		const socket = connect(socketPath);
		let buffer = "";
		const timer = setTimeout(() => {
			socket.destroy();
			reject(new Error("operator socket timeout"));
		}, timeoutMs);
		socket.on("connect", () => {
			socket.write(JSON.stringify({ method, params }) + "\n");
		});
		socket.on("data", (chunk) => {
			buffer += chunk.toString();
			const idx = buffer.indexOf("\n");
			if (idx !== -1) {
				clearTimeout(timer);
				try {
					resolve(JSON.parse(buffer.slice(0, idx)) as Record<string, unknown>);
				} catch (err) {
					reject(err);
				}
				socket.end();
			}
		});
		socket.on("error", (err) => {
			clearTimeout(timer);
			reject(err);
		});
	});
}

export function defaultOperatorSocket(): string {
	const home = process.env.ROSCLAW_HOME ?? `${process.env.HOME}/.rosclaw`;
	return `${home}/run/operator.sock`;
}
