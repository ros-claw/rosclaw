/** Observe public PI compaction hooks without treating absent token events as a stall. */
// HP2-COMPAT: Type-only ExtensionAPI binds public extension hooks inside PI's extension host; no session creation or private runtime access.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { randomUUID } from "node:crypto";

export interface CompactionObserverOptions {
	notice: (text: string) => void;
	log: (record: Record<string, unknown>) => void;
	owner?: () => { session_id?: string; pid?: number };
	firstNoticeMs?: number;
	repeatNoticeMs?: number;
}

export class CompactionObserver {
	private timer: ReturnType<typeof setTimeout> | undefined;
	private active: { id: string; owner: { session_id?: string; pid?: number }; reason: string; started: number; signal: AbortSignal; abort: () => void } | undefined;
	constructor(private readonly options: CompactionObserverOptions) {}

	started(reason: string, signal: AbortSignal): void {
		this.ended("superseded");
		const id = randomUUID();
		let owner: { session_id?: string; pid?: number } = {};
		try { owner = this.options.owner?.() ?? {}; } catch { /* Owner lookup is observational. */ }
		if (signal.aborted) {
			this.record({ ...owner, compaction_id: id, status: "cancelled", reason, elapsed_ms: 0 });
			return;
		}
		const abort = () => this.ended("cancelled");
		this.active = { id, owner, reason, signal, abort, started: performance.now() };
		signal.addEventListener("abort", abort, { once: true });
		this.record({ ...owner, compaction_id: id, status: "started", reason });
		this.arm(this.options.firstNoticeMs ?? 30_000);
	}

	ended(status: "completed" | "failed" | "cancelled" | "shutdown" | "superseded", error?: string): void {
		if (this.timer) clearTimeout(this.timer);
		this.timer = undefined;
		const current = this.active;
		this.active = undefined;
		if (!current) return;
		current.signal.removeEventListener("abort", current.abort);
		this.record({ ...current.owner, compaction_id: current.id, status, reason: current.reason,
			elapsed_ms: Math.round(performance.now() - current.started),
			...(error ? { error: error.slice(0, 512) } : {}) });
	}

	private arm(delay: number): void {
		this.timer = setTimeout(() => {
			const current = this.active;
			if (!current) return;
			const elapsed = Math.round((performance.now() - current.started) / 1_000);
			this.record({ ...current.owner, compaction_id: current.id, status: "waiting", reason: current.reason, elapsed_s: elapsed,
				token_progress_observable: false, automatic_abort: false });
			try {
				this.options.notice(`上下文整理已持续 ${elapsed} 秒，仍在等待模型完成。可继续等待，或手动取消；目前没有逐步进度信息，尚无法判断请求是否停止响应。`);
			} catch { /* Notification failure must not change compaction. */ }
			if (this.active === current) this.arm(this.options.repeatNoticeMs ?? 60_000);
		}, delay);
		this.timer.unref();
	}

	private record(record: Record<string, unknown>): void {
		try { this.options.log(record); } catch { /* Observability never blocks execution. */ }
	}
}

export function registerCompactionObserver(pi: ExtensionAPI, options: CompactionObserverOptions): void {
	const observer = new CompactionObserver(options);
	pi.on("session_before_compact", async event => { observer.started(event.reason, event.signal); });
	pi.on("session_compact", async () => { observer.ended("completed"); });
	pi.on("session_compact_failed", async event => {
		observer.ended(event.aborted ? "cancelled" : "failed", event.errorMessage);
	});
	pi.on("session_shutdown", async () => { observer.ended("shutdown"); });
}
