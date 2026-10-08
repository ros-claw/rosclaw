/** Pi 会话存取 + 运行入口（PR-HP2）——main.ts 的 Pi SDK 调用集中此处。
 *
 * main.ts 不再 import '@earendil-works/pi-coding-agent'：会话打开/
 * 列出/续接与 InteractiveMode/print 模式的装配全部在本模块。
 */

import { statSync } from "node:fs";

import {
	InteractiveMode,
	SessionManager,
	runPrintMode,
} from "@earendil-works/pi-coding-agent";
import { SessionWriterOwnership } from "./session-writer-ownership.js";

export { SessionManager };

/** 打开既有 session（精确路径由调用方经 resolveSessionQuery 解析）。
 *  SESSION_WRITER：传入 ownership 时在任何 SessionManager.open 之前
 *  占有该文件——其他活 owner 持有 → SESSION_IN_USE，零 SDK open。 */
export function openPiSession(
	path: string,
	sessionDir: string,
	ownership?: SessionWriterOwnership,
): SessionManager {
	const stat = statSync(path);
	if (!stat.isFile() || stat.size === 0) throw new Error("NO_RECORDED_SESSION_AT_PATH");
	ownership?.acquire(path);
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

/** Select a recorded global continuation before workspace inference.
 * The public SDK list excludes invalid session files. No match never creates a
 * fresh UUID; callers can offer normal chat explicitly instead.
 */
export async function resolveContinuationTarget(sessionDir: string) {
	const sessions = await SessionManager.listAll(sessionDir);
	const latest = sessions[0];
	if (!latest) return undefined;
	if (!latest.id || !latest.cwd || !latest.path) throw new Error("INVALID_CONTINUATION_TARGET");
	return { id: latest.id, path: latest.path, cwd: latest.cwd };
}

export async function continueRecentPiSession(
	_workspaceRoot: string,
	sessionDir: string,
	ownership?: SessionWriterOwnership,
): Promise<SessionManager | undefined> {
	const target = await resolveContinuationTarget(sessionDir);
	if (!target) return undefined;
	const recorded = openPiSession(target.path, sessionDir, ownership);
	if (recorded.getSessionId() !== target.id || recorded.getCwd() !== target.cwd) {
		throw new Error("CONTINUATION_TARGET_CHANGED");
	}
	return recorded;
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
