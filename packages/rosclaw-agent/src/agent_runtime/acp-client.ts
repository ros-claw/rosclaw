/** ACP Client Runtime（十五审 PR-RF-3，ADR-0011）。
 *
 * ROSClaw 作为 ACP **client** 驱动完整 Agent Harness（Claude Code/
 * Pi/Codex-via-acpx/…）：stdio ndjson JSON-RPC——initialize →
 * session/new → session/prompt（流式 session/update）→ cancel/close。
 *
 * 红线：不解析 ANSI/PTY，不从文案猜状态，不以 exit code 代替语义——
 * 一切状态来自协议消息。Harness 侧请求（permission/fs/terminal）在
 * MVP 阶段一律结构化拒绝（RF-4 sandbox 后开放白名单）。
 */

import type { ChildProcess } from "node:child_process";

interface JsonRpcRequest {
	jsonrpc: "2.0";
	id: number;
	method: string;
	params?: unknown;
}

interface PendingRequest {
	resolve: (value: Record<string, unknown>) => void;
	reject: (error: Error) => void;
	timer: NodeJS.Timeout;
}

const REQUEST_TIMEOUT_MS = 30_000;

export class AcpClient {
	private nextId = 1;
	private pending = new Map<number, PendingRequest>();
	private buffer = "";
	private updateHandlers = new Set<
		(sessionId: string, update: Record<string, unknown>) => void
	>();
	private initialized = false;

	constructor(private readonly proc: ChildProcess) {
		if (!proc.stdout || !proc.stdin) {
			throw new Error("ACP harness process requires piped stdio");
		}
		proc.stdout.setEncoding("utf-8");
		proc.stdout.on("data", (chunk: string) => this.onData(chunk));
		proc.on("exit", (code) => {
			// 进程退出：所有挂起请求诚实失败（绝不悬挂）。
			for (const [, p] of this.pending) {
				clearTimeout(p.timer);
				p.reject(new Error(`harness exited (code ${code})`));
			}
			this.pending.clear();
		});
	}

	private onData(chunk: string): void {
		this.buffer += chunk;
		const lines = this.buffer.split("\n");
		this.buffer = lines.pop() ?? "";
		for (const line of lines) {
			if (!line.trim()) continue;
			let msg: Record<string, unknown>;
			try {
				msg = JSON.parse(line);
			} catch {
				continue; // 非 JSON 行（harness 日志混入）——忽略
			}
			if (typeof msg.id === "number" && (msg.result !== undefined || msg.error !== undefined)) {
				const p = this.pending.get(msg.id);
				if (p) {
					this.pending.delete(msg.id);
					clearTimeout(p.timer);
					if (msg.error !== undefined) {
						const err = msg.error as { code?: number; message?: string };
						p.reject(new Error(`ACP error ${err.code}: ${err.message ?? ""}`));
					} else {
						p.resolve(msg.result as Record<string, unknown>);
					}
				}
			} else if (typeof msg.method === "string") {
				this.onServerMessage(msg.method, msg.params as Record<string, unknown> | undefined, msg.id);
			}
		}
	}

	private onServerMessage(
		method: string,
		params: Record<string, unknown> | undefined,
		id: unknown,
	): void {
		if (method === "session/update" && params) {
			const sessionId = String(params.sessionId ?? "");
			const update = (params.update ?? {}) as Record<string, unknown>;
			for (const handler of this.updateHandlers) handler(sessionId, update);
			return;
		}
		// Harness → client 请求（permission/fs/terminal）：MVP 结构化拒绝
		// （RF-4 sandbox 后按白名单开放；绝不默认放权）。
		if (typeof id === "number" || typeof id === "string") {
			this.sendRaw({
				jsonrpc: "2.0",
				id: id as number,
				error: { code: -32601, message: `${method} not granted by ROSClaw policy (RF-4 pending)` },
			} as never);
		}
	}

	private sendRaw(msg: JsonRpcRequest | Record<string, unknown>): void {
		this.proc.stdin!.write(`${JSON.stringify(msg)}\n`);
	}

	private request(method: string, params?: unknown): Promise<Record<string, unknown>> {
		const id = this.nextId++;
		return new Promise((resolvePromise, rejectPromise) => {
			const timer = setTimeout(() => {
				this.pending.delete(id);
				rejectPromise(new Error(`ACP ${method} timeout`));
			}, REQUEST_TIMEOUT_MS);
			this.pending.set(id, { resolve: resolvePromise, reject: rejectPromise, timer });
			this.sendRaw({ jsonrpc: "2.0", id, method, params });
		});
	}

	async initialize(): Promise<Record<string, unknown>> {
		const result = await this.request("initialize", {
			protocolVersion: 1,
			clientCapabilities: { fs: { readTextFile: false, writeTextFile: false }, terminal: false },
			clientInfo: { name: "rosclaw", version: "1.0" },
		});
		this.initialized = true;
		return result;
	}

	async newSession(params: { cwd: string; mcpServers?: unknown[] }): Promise<{ sessionId: string }> {
		if (!this.initialized) throw new Error("initialize first");
		const result = await this.request("session/new", {
			cwd: params.cwd,
			mcpServers: params.mcpServers ?? [],
		});
		return { sessionId: String(result.sessionId) };
	}

	async prompt(sessionId: string, text: string): Promise<{ stopReason: string }> {
		const result = await this.request("session/prompt", {
			sessionId,
			prompt: [{ type: "text", text }],
		});
		return { stopReason: String(result.stopReason ?? "unknown") };
	}

	async cancel(sessionId: string): Promise<void> {
		// session/cancel 是 notification（无 id）。
		this.sendRaw({ jsonrpc: "2.0", method: "session/cancel", params: { sessionId } } as never);
	}

	/** 可选：加载既有会话（crash 恢复同一 session——ADR-0011）。 */
	async loadSession(sessionId: string, cwd: string): Promise<void> {
		await this.request("session/load", { sessionId, cwd, mcpServers: [] });
	}

	onSessionUpdate(
		handler: (sessionId: string, update: Record<string, unknown>) => void,
	): () => void {
		this.updateHandlers.add(handler);
		return () => this.updateHandlers.delete(handler);
	}

	dispose(): void {
		this.proc.kill("SIGTERM");
		for (const [, p] of this.pending) {
			clearTimeout(p.timer);
			p.reject(new Error("client disposed"));
		}
		this.pending.clear();
	}
}
