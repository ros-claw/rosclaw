import { readFileSync, readdirSync, statSync } from "node:fs";
import type { ChildProcess } from "node:child_process";

export interface ProcessBirth {
	pid: number;
	ppid: number;
	start: string;
	uid: number;
	boot: string;
	state: string;
}

export function sameProcessBirth(a: ProcessBirth, b: ProcessBirth): boolean {
	return a.pid === b.pid && a.start === b.start && a.uid === b.uid && a.boot === b.boot;
}

export type ProcessObservation = { kind: "observed"; birth: ProcessBirth }
	| { kind: "gone" } | { kind: "unresolved" };

export function readProcessBirth(pid: number, boot: string,
	read: (path: string) => string = (path) => readFileSync(path, "utf8"),
	uid: (path: string) => number = (path) => statSync(path).uid): ProcessObservation {
	try {
		const text = read(`/proc/${pid}/stat`);
		const fields = text.slice(text.lastIndexOf(")") + 2).split(" ");
		if (!text.startsWith(`${pid} (`) || !/^[A-Za-z]$/.test(fields[0])
			|| !/^\d+$/.test(fields[1]) || !/^\d+$/.test(fields[19])) return { kind: "unresolved" };
		return { kind: "observed", birth: { pid, ppid: Number(fields[1]), state: fields[0], start: fields[19],
			uid: uid(`/proc/${pid}`), boot } };
	} catch (err) {
		return { kind: missing(err) ? "gone" : "unresolved" };
	}
}

function missing(err: unknown): boolean {
	return ["ENOENT", "ESRCH"].includes((err as NodeJS.ErrnoException | null)?.code ?? "");
}

const alive = (p: ProcessBirth) => p.state !== "Z" && p.state !== "X";
const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/** Preserve the existing non-Linux Unix process-group fallback. Linux uses
 * captured births instead; Windows retains ChildProcess.kill(). */
export function signalBashFallback(child: Pick<ChildProcess, "pid" | "kill">, sig: NodeJS.Signals,
	platform: NodeJS.Platform = process.platform,
	send: (pid: number, sig: NodeJS.Signals) => unknown = process.kill): void {
	try {
		if (platform !== "win32" && child.pid) send(-child.pid, sig);
		else child.kill(sig);
	} catch (err) {
		if ((err as NodeJS.ErrnoException).code !== "ESRCH") child.kill(sig);
	}
}

/** Capture ancestry while the shell is alive, retaining births across setsid and
 * reparenting. Normal successful completion deliberately leaves detached work
 * alone. Cancellation signals only individually revalidated captured births;
 * neither a reused PID nor a reused process group grants signal authority. */
export class BashProcessOwnership {
	private readonly owned = new Map<number, ProcessBirth>();
	private readonly boot: string;
	private readonly rootCaptured: boolean;
	private readonly unreadable = new Set<number>();
	private readonly discoveryUnresolved = new Set<number>();
	private readonly timer: NodeJS.Timeout;
	private stopping?: Promise<{ status: "captured_descendants_stopped" | "unresolved"; unresolvedPids: number[] }>;
	constructor(pid: number, private readonly readBirth = readProcessBirth) {
		try { this.boot = readFileSync("/proc/sys/kernel/random/boot_id", "utf8").trim(); }
		catch { this.boot = ""; }
		const root = this.boot ? this.read(pid) : { kind: "unresolved" } as const;
		this.rootCaptured = root.kind === "observed";
		if (root.kind === "observed") this.owned.set(pid, root.birth);
		else if (root.kind === "unresolved") this.unreadable.add(pid);
		this.capture();
		this.timer = setInterval(() => this.capture(), 100);
		this.timer.unref();
	}

	private read(pid: number): ProcessObservation {
		try { return this.readBirth(pid, this.boot); }
		catch { return { kind: "unresolved" }; }
	}

	private observe(expected: ProcessBirth): ProcessObservation {
		const result = this.read(expected.pid);
		// A UID change on the same birth is not evidence that it stopped, and
		// grants no authority to signal the new identity.
		if (result.kind === "unresolved" || (result.kind === "observed"
			&& result.birth.start === expected.start && result.birth.boot === expected.boot
			&& result.birth.uid !== expected.uid)) {
			this.unreadable.add(expected.pid);
			return { kind: "unresolved" };
		}
		this.unreadable.delete(expected.pid);
		return result;
	}

	private capture(): void {
		// Only inspect owned processes and their kernel children lists, never
		// scan host /proc. Double-fork children that escape before capture remain
		// unobservable; the result deliberately refers to captured births only.
		const queue = [...this.owned.values()];
		const visited = new Set<number>();
		for (const parent of queue) {
			if (visited.has(parent.pid)) continue;
			visited.add(parent.pid);
			const observation = this.observe(parent);
			if (observation.kind === "unresolved") {
				this.discoveryUnresolved.add(parent.pid);
				continue;
			}
			if (observation.kind === "gone" || !sameProcessBirth(parent, observation.birth) || !alive(observation.birth)) {
				this.owned.delete(parent.pid);
				continue;
			}
			let tids: string[];
			try { tids = readdirSync(`/proc/${parent.pid}/task`); }
			catch (err) { if (!missing(err)) this.discoveryUnresolved.add(parent.pid); continue; }
			for (const tid of tids) {
				if (!/^\d+$/.test(tid)) continue;
				let children: string;
				try { children = readFileSync(`/proc/${parent.pid}/task/${tid}/children`, "utf8"); }
				catch (err) { if (!missing(err)) this.discoveryUnresolved.add(parent.pid); continue; }
				for (const id of children.trim().split(/\s+/)) {
					if (!/^\d+$/.test(id)) continue;
					const child = this.read(Number(id));
					const stillParent = this.observe(parent);
					if (child.kind === "unresolved" || stillParent.kind === "unresolved") this.discoveryUnresolved.add(parent.pid);
					if (child.kind !== "observed" || child.birth.ppid !== parent.pid || stillParent.kind !== "observed" || !sameProcessBirth(parent, stillParent.birth)) continue;
					this.owned.set(child.birth.pid, child.birth);
					queue.push(child.birth);
				}
			}
		}
	}

	private live(): ProcessBirth[] {
		return [...this.owned.values()].filter((p) => {
			const now = this.observe(p);
			return now.kind === "unresolved" || (now.kind === "observed" && sameProcessBirth(p, now.birth) && alive(now.birth));
		});
	}

	private signal(sig: NodeJS.Signals): void {
		for (const p of this.live()) {
			// Revalidate immediately before signalling, after the ancestry scan.
			const now = this.observe(p);
			if (now.kind !== "observed" || !sameProcessBirth(p, now.birth) || !alive(now.birth)) continue;
			try { process.kill(p.pid, sig); } catch { /* Drain reports survivors. */ }
		}
	}

	stop(immediate = false): Promise<{ status: "captured_descendants_stopped" | "unresolved"; unresolvedPids: number[] }> {
		if (this.stopping) return this.stopping;
		this.stopping = (async () => {
			const started = performance.now();
			this.capture(); // Must precede terminating the shell/reparenting.
			do {
				this.capture();
				this.signal(immediate || performance.now() - started >= 1000 ? "SIGKILL" : "SIGTERM");
				await delay(20);
				if (!this.live().length) break;
			} while (performance.now() - started < 2000);
			const unresolvedPids = [...new Set([...this.live().map((p) => p.pid), ...this.unreadable, ...this.discoveryUnresolved])];
			return { status: unresolvedPids.length || !this.rootCaptured ? "unresolved" as const : "captured_descendants_stopped" as const, unresolvedPids };
		})().catch(() => ({ status: "unresolved" as const,
			unresolvedPids: [...new Set([...this.owned.keys(), ...this.unreadable, ...this.discoveryUnresolved])] }))
			.finally(() => this.dispose());
		return this.stopping;
	}

	dispose(): void { clearInterval(this.timer); }
}
