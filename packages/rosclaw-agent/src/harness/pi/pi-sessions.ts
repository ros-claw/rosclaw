/** Pi 会话存取 + 运行入口（PR-HP2）——main.ts 的 Pi SDK 调用集中此处。
 *
 * main.ts 不再 import '@earendil-works/pi-coding-agent'：会话打开/
 * 列出/续接与 InteractiveMode/print 模式的装配全部在本模块。
 */

import { statSync, readFileSync, writeFileSync, unlinkSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import type { ExplicitModelSelection } from "./pi-runtime.js";
import { requireSupportedThinking } from "./pi-runtime.js";

import {
	InteractiveMode,
	SessionManager,
	runPrintMode,
	parseSessionEntries, migrateSessionEntries,
} from "@earendil-works/pi-coding-agent";
import { SessionWriterOwnership } from "./session-writer-ownership.js";

export { SessionManager };

/** Read/migrate in memory ONLY: legacy source is never passed to open(). */
export function inspectOverrideSource(path: string) {
	const bytes = readFileSync(path);
	const stat = statSync(path);
	if (!stat.isFile() || !bytes.length) throw new Error("INVALID_OVERRIDE_SOURCE");
	const entries = parseSessionEntries(bytes.toString("utf8"));
	migrateSessionEntries(entries);
	const header = entries.find(e => e.type === "session");
	if (!header?.id || !header.cwd) throw new Error("INVALID_OVERRIDE_SOURCE");
	const history = entries.filter(e => e.type !== "session");
	// Match the SDK's persisted leaf (last entry), then follow parentId only.
	// Its context defaults to "off" when this ancestry has no recorded effort;
	// discarded entries must not turn that implicit value into a recorded one.
	const byId = new Map(history.map(e => [e.id, e]));
	// Check every identity/reference before traversing: a Map alone silently
	// overwrites duplicates, and a missing parent otherwise looks like a root.
	if (byId.size !== history.length) throw new Error("INVALID_OVERRIDE_SOURCE");
	for (const entry of history) {
		if (typeof entry.id !== "string" || !entry.id ||
			(entry.parentId !== null && (typeof entry.parentId !== "string" || !byId.has(entry.parentId)))) {
			throw new Error("INVALID_OVERRIDE_SOURCE");
		}
	}
	const seen = new Set<string>();
	let thinking: string | undefined;
	let foundThinking = false;
	for (let entry = history.at(-1); entry; entry = entry.parentId === null ? undefined : byId.get(entry.parentId)) {
		if (seen.has(entry.id)) throw new Error("INVALID_OVERRIDE_SOURCE");
		seen.add(entry.id);
		if (!foundThinking && entry.type === "thinking_level_change") {
			thinking = entry.thinkingLevel;
			foundThinking = true; // nearest wins, but validate ALL selected ancestors
		}
	}
	return { id: header.id, cwd: header.cwd, history, thinking,
		sha256: createHash("sha256").update(bytes).digest("hex"),
		mtimeMs: stat.mtimeMs, size: stat.size };
}

/** Only an explicitly selected, separately approved source may enter this seam.
 * forkFrom reads source, migrates the NEW file; no tools/provider are executed.
 * Retain the selected branch, not discarded branches or extension authority.
 */
export function forkPiSessionForOverride(
	path: string, sessionDir: string, ownership: SessionWriterOwnership,
	selection: ExplicitModelSelection, defaultThinking: string,
) {
	const source = inspectOverrideSource(path); // reject invalid graphs before ownership/SDK side effects
	ownership.check(path); // live/unknown source: require a frozen snapshot instead
	const thinking = requireSupportedThinking(selection, source.thinking ?? defaultThinking);
	let fork = SessionManager.forkFrom(path, source.cwd, sessionDir);
	const file = fork.getSessionFile()!;
	try {
		ownership.acquire(file);
		const after = inspectOverrideSource(path);
		if (after.sha256 !== source.sha256 || after.mtimeMs !== source.mtimeMs || after.size !== source.size) {
			throw new Error("OVERRIDE_SOURCE_CHANGED");
		}
		// forkFrom stamps a current header even for an unnumbered v1 source.
		// Persist our in-memory SDK migration on the owned fork before branching.
		writeFileSync(file, [fork.getHeader(), ...source.history].map(e => JSON.stringify(e)).join("\n") + "\n", { mode: 0o600 });
		fork = SessionManager.open(file, sessionDir);
		const retained = fork.getBranch().filter(e => e.type !== "custom" && e.type !== "custom_message");
		let parentId: string | null = null;
		const safe = retained.map(e => {
			const clean = { ...e, parentId };
			parentId = e.id;
			return clean;
		});
		writeFileSync(file, [fork.getHeader(), ...safe].map(e => JSON.stringify(e)).join("\n") + "\n", { mode: 0o600 });
		// Only the NEW owned file is opened. No original legacy migration.
		fork = SessionManager.open(file, sessionDir);
		fork.appendModelChange(selection.model.provider, selection.model.id);
		fork.appendThinkingLevelChange(thinking);
		return { session: fork, sourceId: source.id, sourcePath: resolve(path), thinking };
	} catch (err) {
		ownership.release(file);
		unlinkSync(file);
		throw err;
	}
}

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
