/** Native transcript ownership. New claims use O_EXCL. Existing foreign claims
 * are NEVER moved, restored, unlinked or overwritten by acquire/check, even if
 * birth-death diagnostics say dead. This deliberately sacrifices automatic
 * crash recovery: see docs/session-writer-ownership.md for quiescent recovery.
 * No SDK patch, TTL steal, PID-only steal or external-process signalling.
 */
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, realpathSync, unlinkSync, writeSync } from "node:fs";
import { dirname, resolve, basename } from "node:path";
import { randomUUID } from "node:crypto";

export const SESSION_IN_USE = "SESSION_IN_USE";
export class SessionInUseError extends Error {
	readonly code = SESSION_IN_USE;
	readonly sessionFile: string;
	constructor(sessionFile: string, detail: string) {
		super(`SESSION_IN_USE: session transcript is already owned or recovery is required — file: ${sessionFile} (${detail}). Use a different session or close the owning session first.`);
		this.name = "SessionInUseError";
		this.sessionFile = sessionFile;
	}
}
export function isSessionInUse(err: unknown): err is SessionInUseError {
	return err instanceof SessionInUseError || (typeof err === "object" && err !== null && (err as { code?: unknown }).code === SESSION_IN_USE);
}
export function canonicalSessionPath(sessionFile: string): string {
	const absolute = resolve(sessionFile);
	if (existsSync(absolute)) return realpathSync(absolute);
	return resolve(realpathSync(dirname(absolute)), basename(absolute));
}
export function ownerLockPath(canonicalFile: string): string { return `${canonicalFile}.owner.json`; }
/** Legacy marker path retained for diagnostics and conservative refusal.
 * This version never creates a reclaim marker and never runs a reclaimer.
 */
export function ownerReclaimMarkerPath(lockPath: string): string { return `${lockPath}.reclaim`; }
interface LockRecord { token: string; pid: number; birth: string; acquiredAt: string }
type BirthProbe = { state: "alive"; birth: string } | { state: "dead" } | { state: "unknown" };
const BOOT_ID: string | undefined = (() => {
	try { return readFileSync("/proc/sys/kernel/random/boot_id", "utf-8").trim() || undefined; }
	catch { return undefined; }
})();
function probeProcessBirth(pid: number): BirthProbe {
	if (!Number.isSafeInteger(pid) || pid <= 0 || !BOOT_ID) return { state: "unknown" };
	let stat: string;
	try { stat = readFileSync(`/proc/${pid}/stat`, "utf-8"); }
	catch (err) {
		const code = (err as NodeJS.ErrnoException).code;
		return { state: code === "ENOENT" || code === "ENOTDIR" ? "dead" : "unknown" };
	}
	const close = stat.lastIndexOf(")");
	if (close < 0) return { state: "unknown" };
	const fields = stat.slice(close + 2).split(" ");
	if (!fields[19]) return { state: "unknown" };
	return { state: "alive", birth: `${BOOT_ID}:${fields[19]}` };
}
const SELF_BIRTH = (() => {
	const probe = probeProcessBirth(process.pid);
	return probe.state === "alive" ? probe.birth : "unknown";
})();
function probeLockOwner(record: Pick<LockRecord, "pid" | "birth">): "alive" | "dead" | "unknown" {
	const probe = probeProcessBirth(record.pid);
	if (probe.state !== "alive") return probe.state;
	return probe.birth === record.birth ? "alive" : "dead";
}
/** Diagnostic only: false is NOT authorization to reclaim a lock. */
export function lockOwnerAlive(record: Pick<LockRecord, "pid" | "birth">): boolean {
	return probeLockOwner(record) !== "dead";
}
type LockRead = { state: "absent" } | { state: "unknown"; detail: string } | { state: "ok"; record: LockRecord };
function readLock(lockPath: string): LockRead {
	let text: string;
	try { text = readFileSync(lockPath, "utf-8"); }
	catch (err) {
		const code = (err as NodeJS.ErrnoException).code;
		if (code === "ENOENT" || code === "ENOTDIR") return { state: "absent" };
		return { state: "unknown", detail: `unreadable lock (${code ?? "error"})` };
	}
	try {
		const raw = JSON.parse(text) as Partial<LockRecord> | null;
		if (!raw || typeof raw.token !== "string" || !raw.token || typeof raw.pid !== "number" || !Number.isSafeInteger(raw.pid) || raw.pid <= 0 || typeof raw.birth !== "string" || !raw.birth) {
			return { state: "unknown", detail: "malformed lock" };
		}
		return { state: "ok", record: { token: raw.token, pid: raw.pid, birth: raw.birth, acquiredAt: typeof raw.acquiredAt === "string" ? raw.acquiredAt : "" } };
	} catch { return { state: "unknown", detail: "malformed lock (json)" }; }
}
/** A token identifies one manager, not one PID. Cooperating product seams only. */
export class SessionWriterOwnership {
	readonly ownerToken: string = randomUUID();
	private readonly claims = new Map<string, string>();
	get claimCount(): number { return this.claims.size; }
	has(sessionFile: string): boolean { return this.claims.has(canonicalSessionPath(sessionFile)); }
	private tryCreateLock(lockPath: string): boolean {
		mkdirSync(dirname(lockPath), { recursive: true });
		let fd: number;
		try { fd = openSync(lockPath, "wx", 0o600); }
		catch (err) {
			if ((err as NodeJS.ErrnoException).code === "EEXIST") return false;
			throw err;
		}
		try {
			const record: LockRecord = { token: this.ownerToken, pid: process.pid, birth: SELF_BIRTH, acquiredAt: new Date().toISOString() };
			writeSync(fd, `${JSON.stringify(record)}\n`);
		} finally { closeSync(fd); }
		// Failed/partial writes leave a fail-closed record, never an unlink gap.
		return true;
	}
	private refuseMarker(canonical: string, lockPath: string): void {
		// Read rather than existsSync: inaccessible authority must also refuse.
		const marker = readLock(ownerReclaimMarkerPath(lockPath));
		if (marker.state !== "absent") throw new SessionInUseError(canonical, "reclaim marker persisted; fail-closed; quiescent manual recovery required");
	}
	private foreignDetail(record: LockRecord): string {
		const state = probeLockOwner(record);
		return state === "dead"
			? "stale owner; automatic reclaim disabled; fail-closed, lock left unchanged; quiescent manual recovery required"
			: `${state === "alive" ? "live owner" : "owner liveness UNKNOWN"} pid=${record.pid}; fail-closed, lock left unchanged`;
	}
	/** No existing lock movement: the sole acquire transition is O_EXCL create.
	 * Legacy marker precheck is not a mutex. Safety does not depend on it: no
	 * current participant can create a restore gap. Mixed old/new writers are
	 * unsupported; stop all writers before upgrade or manual recovery.
	 */
	acquire(sessionFile: string): string {
		const canonical = canonicalSessionPath(sessionFile);
		if (this.claims.has(canonical)) return canonical;
		const lockPath = ownerLockPath(canonical);
		for (let attempt = 0; attempt < 8; attempt += 1) {
			this.refuseMarker(canonical, lockPath);
			if (this.tryCreateLock(lockPath)) {
				this.claims.set(canonical, lockPath);
				return canonical;
			}
			const existing = readLock(lockPath);
			if (existing.state === "absent") continue; // legitimate release; retry O_EXCL
			if (existing.state === "unknown") throw new SessionInUseError(canonical, `${existing.detail}; fail-closed, lock left unchanged`);
			if (existing.record.token === this.ownerToken) {
				this.claims.set(canonical, lockPath);
				return canonical;
			}
			throw new SessionInUseError(canonical, this.foreignDetail(existing.record));
		}
		throw new SessionInUseError(canonical, "concurrent acquisition did not settle; fail-closed");
	}
	/** Advisory only, NOT a reservation. Same stale refusal policy as acquire. */
	check(sessionFile: string): void {
		const canonical = canonicalSessionPath(sessionFile);
		if (this.claims.has(canonical)) return;
		const lockPath = ownerLockPath(canonical);
		this.refuseMarker(canonical, lockPath);
		const existing = readLock(lockPath);
		if (existing.state === "unknown") throw new SessionInUseError(canonical, `${existing.detail}; fail-closed`);
		if (existing.state === "ok" && existing.record.token !== this.ownerToken) throw new SessionInUseError(canonical, this.foreignDetail(existing.record));
	}
	/** Only own token may release. External replacement/recovery must be quiescent. */
	release(sessionFile: string): void {
		const canonical = canonicalSessionPath(sessionFile);
		const lockPath = this.claims.get(canonical);
		if (!lockPath) return;
		this.claims.delete(canonical);
		const existing = readLock(lockPath);
		if (existing.state === "ok" && existing.record.token === this.ownerToken) {
			try { unlinkSync(lockPath); } catch { /* leave failed release fail-closed */ }
		}
	}
	releaseAll(): void { for (const canonical of [...this.claims.keys()]) this.release(canonical); }
}
