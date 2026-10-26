/** Pi 会话存取 + 运行入口（PR-HP2）——main.ts 的 Pi SDK 调用集中此处。
 *
 * main.ts 不再 import '@earendil-works/pi-coding-agent'：会话打开/
 * 列出/续接与 InteractiveMode/print 模式的装配全部在本模块。
 */

import {
	InteractiveMode,
	type AgentSession,
	type AgentSessionRuntime,
	SessionManager,
	runPrintMode,
} from "@earendil-works/pi-coding-agent";

export { SessionManager };

/** 打开既有 session（精确路径由调用方经 resolveSessionQuery 解析）。 */
export function openPiSession(path: string, sessionDir: string): SessionManager {
	return SessionManager.open(path, sessionDir);
}

export function listPiSessions(
	workspaceRoot: string,
	sessionDir: string,
	onProgress?: ( scanned: number, total: number) => void,
) {
	return SessionManager.list(workspaceRoot, sessionDir, onProgress);
}

export function listAllPiSessions(
	sessionDir: string,
	onProgress?: (scanned: number, total: number) => void,
) {
	return SessionManager.listAll(sessionDir, onProgress);
}

export function continueRecentPiSession(
	workspaceRoot: string,
	sessionDir: string,
): SessionManager | undefined {
	return SessionManager.continueRecent(workspaceRoot, sessionDir);
}

/** 交互模式（TUI）。 */
export async function runPiInteractive(
	runtime: unknown,
	options: { verbose?: boolean; initialMessage?: string },
): Promise<number> {
	const mode = new InteractiveMode(runtime as never, {
		verbose: options.verbose ?? false,
		...(options.initialMessage ? { initialMessage: options.initialMessage } : {}),
	});
	await mode.run();
	return 0;
}

/** The SDK still owns prompting, rebinding, output, signals and disposal.
 * Bind methods to their real receivers (SDK objects have private fields).
 */
function settledPrintRuntime(
	runtime: AgentSessionRuntime,
	settle: () => Promise<void>,
): AgentSessionRuntime {
	const sessions = new WeakMap<AgentSession, AgentSession>();
	let settling = false;
	return new Proxy(runtime, {
		get(target, key) {
			if (key !== "session") {
				const value = Reflect.get(target, key, target);
				return typeof value === "function" ? value.bind(target) : value;
			}
			const session = target.session;
			let view = sessions.get(session);
			if (!view) {
				view = new Proxy(session, {
					get(owned, property) {
						if (property === "prompt") {
							return async (...args: Parameters<AgentSession["prompt"]>) => {
								await owned.prompt(...args);
								// SDK prompt() may resolve with an error/abort message.
								// Preserve its failure exit and disposal; waiting for an
								// unrelated operation would mask a provider timeout.
								const last = owned.state.messages.at(-1);
								if (last?.role === "assistant"
									&& (last.stopReason === "error" || last.stopReason === "aborted")) return;
								if (!settling) {
									settling = true;
									try { await settle(); }
									finally { settling = false; }
								}
							};
						}
						const value = Reflect.get(owned, property, owned);
						return typeof value === "function" ? value.bind(owned) : value;
					},
				});
				sessions.set(session, view);
			}
			return view;
		},
	});
}

/** 非 TTY 单发模式（冒烟/脚本）。 */
export async function runPiPrint(
	runtime: unknown,
	options: { initialMessage?: string; settleOperations?: () => Promise<void> },
): Promise<number> {
	const host = runtime as AgentSessionRuntime;
	return await runPrintMode(options.settleOperations
		? settledPrintRuntime(host, options.settleOperations) : host, {
		mode: "text",
		...(options.initialMessage ? { initialMessage: options.initialMessage } : {}),
	});
}
