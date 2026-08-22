/** Pi 会话存取 + 运行入口（PR-HP2）——main.ts 的 Pi SDK 调用集中此处。
 *
 * main.ts 不再 import '@earendil-works/pi-coding-agent'：会话打开/
 * 列出/续接与 InteractiveMode/print 模式的装配全部在本模块。
 */

import {
	InteractiveMode,
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

/** 非 TTY 单发模式（冒烟/脚本）。 */
export async function runPiPrint(
	runtime: unknown,
	options: { initialMessage?: string },
): Promise<number> {
	return await runPrintMode(runtime as never, {
		mode: "text",
		...(options.initialMessage ? { initialMessage: options.initialMessage } : {}),
	});
}
