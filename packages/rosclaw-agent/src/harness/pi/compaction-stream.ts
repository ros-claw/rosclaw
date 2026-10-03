/** Observe the actual public provider stream; session JSONL silence is not progress evidence. */
import type { StreamFn } from "@earendil-works/pi-agent-core";
import { createAssistantMessageEventStream, type AssistantMessage } from "@earendil-works/pi-ai";
import { randomUUID } from "node:crypto";

export interface CompactionStreamOptions {
    enabled: () => boolean;
    record: (record: Record<string, unknown>) => void;
    waitingNoticeMs?: number;
    progressRecordMs?: number;
    owner?: () => Record<string, unknown>;
}

export function observeCompactionStream(original: StreamFn, observation: CompactionStreamOptions): StreamFn {
    return function(this: unknown, ...args: Parameters<StreamFn>) {
        let enabled = false;
        try { enabled = observation.enabled(); } catch { /* Failed observation never changes ordinary calls. */ }
        if (!enabled) return original.apply(this, args);
        let owner: Record<string, unknown> = {};
        try { owner = { ...observation.owner?.() }; } catch { /* Observational owner lookup only. */ }
        const requestId = randomUUID();
        const started = performance.now();
        let lastContent = started;
        let lastProgressRecord = started;
        let contentEvents = 0;
        let events = 0;
        let terminal = false;
        let partial: AssistantMessage | undefined;
        const output = createAssistantMessageEventStream();
        let timer: ReturnType<typeof setTimeout> | undefined;
        const record = (value: Record<string, unknown>) => {
            try { observation.record({ ...owner, request_id: requestId,
                elapsed_ms: Math.round(performance.now() - started), ...value }); } catch { /* Logging never breaks requests. */ }
        };
        const finish = (message: AssistantMessage, source: string) => {
            if (terminal) return;
            terminal = true;
            if (timer) clearTimeout(timer);
            record({ status: message.stopReason === "aborted" ? "stream_cancelled"
                : message.stopReason === "error" ? "stream_failed" : "stream_completed",
                terminal_source: source, events, content_events: contentEvents,
                ...(source === "protocol_guard" ? { usage_status: "NOT_RECORDED" }
                    : { usage_status: "PROVIDER_RESULT", usage: message.usage }),
                stop_reason: message.stopReason });
        };
        const protocolFailure = (detail: string) => {
            const model = args[0];
            const failure: AssistantMessage = { role: "assistant", api: model.api, provider: model.provider,
                model: model.id, content: partial?.content ?? [],
                usage: partial?.usage ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
                    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
                stopReason: args[2]?.signal?.aborted ? "aborted" : "error",
                errorMessage: `ROSCLAW_STREAM_PROTOCOL_ERROR: ${detail}`, timestamp: Date.now() };
            finish(failure, "protocol_guard");
            output.push({ type: "error", reason: failure.stopReason as "error" | "aborted", error: failure });
            output.end();
        };
        const armWaiting = () => {
            timer = setTimeout(() => {
                if (terminal) return;
                record({ status: "waiting", phase: contentEvents ? "between_content" : "before_first_content",
                    idle_ms: Math.round(performance.now() - lastContent), events,
                    content_events: contentEvents, automatic_abort: false });
                armWaiting();
            }, observation.waitingNoticeMs ?? 30_000);
            timer.unref();
        };
        record({ status: "request_started", automatic_abort: false });
        armWaiting();
        void (async () => {
            try {
                const source = await original.apply(this, args);
                let resultOnly: AssistantMessage | undefined;
                // A public EventStream.end(result) may legally resolve without
                // a terminal event; attach before iterating to preserve it.
                void source.result().then(result => { resultOnly = result; }, () => {});
                for await (const event of source) {
                    events++;
                    if ("partial" in event) partial = event.partial;
                    if ((event.type === "text_delta" || event.type === "thinking_delta"
                        || event.type === "toolcall_delta") && event.delta.length > 0) {
                        contentEvents++;
                        lastContent = performance.now();
                        if (contentEvents === 1 || lastContent - lastProgressRecord >= (observation.progressRecordMs ?? 15_000)) {
                            record({ status: contentEvents === 1 ? "first_content" : "content_progress",
                                event_type: event.type, events, content_events: contentEvents });
                            lastProgressRecord = lastContent;
                        }
                    }
                    if (event.type === "done" || event.type === "error") {
                        finish(event.type === "done" ? event.message : event.error, "provider_event");
                    }
                    output.push(event);
                }
                if (!terminal) {
                    if (resultOnly) {
                        finish(resultOnly, "result_only");
                        output.end(resultOnly);
                    } else {
                        protocolFailure("provider stream ended without a terminal event or result");
                    }
                } else output.end();
            } catch (error) {
                if (!terminal) protocolFailure(error instanceof Error ? error.message : "provider stream failed");
            }
        })();
        return output;
    };
}
