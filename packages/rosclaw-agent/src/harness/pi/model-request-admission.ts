import type { StreamFn } from "@earendil-works/pi-agent-core";
import type { AssistantMessage } from "@earendil-works/pi-ai";

/** Local token admission, NOT a provider billing bound.
 * v1 intentionally stops future dispatch after any UNKNOWN settlement, even
 * after an allowance increase. An aborted SDK message's zero usage fields do
 * not establish zero provider spend. No external-clone reconciliation API is
 * defined here: stream terminal identity belongs to the wrapped invocation.
 * See the cancellation feasibility appendix in PI_120_MIGRATION.md.
 */
export interface ModelRequestAdmissionPolicy {
    version: 1;
    lifetime: "runtime";
    guarantee: "local_reservation";
    inclusiveInputLimit: number;
    outputLimit: number;
    requestReserveInput: number;
    requestReserveOutput: number;
}
export class ModelRequestAdmissionError extends Error {
    readonly recoverable = true;
    constructor(readonly code: string) { super(code); this.name = "ModelRequestAdmissionError"; }
}
export function validateModelRequestAdmission(value: unknown): asserts value is ModelRequestAdmissionPolicy {
    const keys = ["version", "lifetime", "guarantee", "inclusiveInputLimit", "outputLimit", "requestReserveInput", "requestReserveOutput"];
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new ModelRequestAdmissionError("INVALID_MODEL_REQUEST_ADMISSION");
    const v = value as Record<string, unknown>;
    if (Object.keys(v).length !== keys.length || keys.some(k => !Object.hasOwn(v, k))
        || v.version !== 1 || v.lifetime !== "runtime") throw new ModelRequestAdmissionError("INVALID_MODEL_REQUEST_ADMISSION");
    if (v.guarantee === "strict_billing") throw new ModelRequestAdmissionError("MODEL_REQUEST_BILLING_GUARANTEE_UNSUPPORTED");
    if (v.guarantee !== "local_reservation" || keys.slice(3).some(k => typeof v[k] !== "number" || !Number.isSafeInteger(v[k]) || (v[k] as number) <= 0)
        || (v.requestReserveInput as number) > (v.inclusiveInputLimit as number)
        || (v.requestReserveOutput as number) > (v.outputLimit as number)) throw new ModelRequestAdmissionError("INVALID_MODEL_REQUEST_ADMISSION");
}

export class ModelRequestAdmission {
    private policy: ModelRequestAdmissionPolicy;
    private input = 0;
    private output = 0;
    private unknown = 0;
    private dispatched = 0;
    private settled = 0;
    private pending = new Set<object>();
    private lastDenial: string | undefined;
    constructor(policy: ModelRequestAdmissionPolicy) {
        validateModelRequestAdmission(policy);
        this.policy = Object.freeze({ ...policy });
    }
    snapshot() {
        return { knownInclusiveInput: this.input, knownOutput: this.output, unknownUsageRequests: this.unknown,
            dispatchedRequests: this.dispatched, settledRequests: this.settled, pendingRequests: this.pending.size,
            reservedInput: this.pending.size * this.policy.requestReserveInput,
            reservedOutput: this.pending.size * this.policy.requestReserveOutput, lastDenial: this.lastDenial,
            providerCostBoundVerified: false };
    }
    /** Explicit local recovery: raise allowance only; never erase spend/UNKNOWN. */
    reconfigure(policy: ModelRequestAdmissionPolicy): void {
        validateModelRequestAdmission(policy);
        if (this.pending.size || policy.inclusiveInputLimit < this.policy.inclusiveInputLimit || policy.outputLimit < this.policy.outputLimit
            || policy.requestReserveInput !== this.policy.requestReserveInput || policy.requestReserveOutput !== this.policy.requestReserveOutput)
            throw new ModelRequestAdmissionError("MODEL_REQUEST_RECOVERY_UNSAFE");
        this.policy = Object.freeze({ ...policy });
    }
    private deny(code: string): never { this.lastDenial = code; throw new ModelRequestAdmissionError(code); }
    /** Reject unsupported public summary entry before SDK abort/wait/auth work. */
    refuseSummary(): never { return this.deny("MODEL_REQUEST_SUMMARY_UNSUPPORTED"); }
    private reserve(summary: boolean, aborted: boolean): object {
        if (summary) this.deny("MODEL_REQUEST_SUMMARY_UNSUPPORTED");
        if (aborted) this.deny("MODEL_REQUEST_CANCELLED_BEFORE_DISPATCH");
        if (this.unknown) this.deny("MODEL_REQUEST_USAGE_UNKNOWN");
        const p = this.policy;
        // Subtraction avoids unsafe integer overflow, including concurrent reservations.
        if (this.input >= p.inclusiveInputLimit || this.output >= p.outputLimit
            || p.requestReserveInput > p.inclusiveInputLimit - this.input - this.pending.size * p.requestReserveInput
            || p.requestReserveOutput > p.outputLimit - this.output - this.pending.size * p.requestReserveOutput)
            this.deny("MODEL_REQUEST_ALLOWANCE_EXHAUSTED");
        const token = {};
        this.pending.add(token);
        return token;
    }
    private settle(token: object, message?: AssistantMessage): void {
        if (!this.pending.delete(token)) return;
        this.settled++;
        let unknown = !message || message.stopReason === "error" || message.stopReason === "aborted";
        for (const key of ["input", "cacheRead", "cacheWrite", "output"] as const) {
            const n = message?.usage?.[key];
            if (typeof n !== "number" || !Number.isSafeInteger(n) || n < 0) { unknown = true; continue; }
            const old = key === "output" ? this.output : this.input;
            if (!Number.isSafeInteger(old + n)) { unknown = true; continue; }
            if (key === "output") this.output += n; else this.input += n;
        }
        if (unknown) this.unknown++;
    }
    wrap(original: StreamFn, isSummary: () => boolean = () => false): StreamFn {
        const gate = this;
        return async function(this: unknown, ...args: Parameters<StreamFn>) {
            const token = gate.reserve(isSummary(), args[2]?.signal?.aborted === true);
            let create: typeof import("@earendil-works/pi-ai").createAssistantMessageEventStream;
            try { create = (await import("@earendil-works/pi-ai")).createAssistantMessageEventStream; }
            catch (error) { gate.pending.delete(token); throw error; }
            const output = create();
            // Reserve is synchronous before any await. Once invoked even a
            // synchronous throw has unknown cost; only a proven pre-dispatch abort releases.
            if (args[2]?.signal?.aborted) {
                gate.pending.delete(token);
                return gate.deny("MODEL_REQUEST_CANCELLED_BEFORE_DISPATCH");
            }
            gate.dispatched++;
            let source: ReturnType<StreamFn>;
            try { source = original.apply(this, args); }
            catch (error) { gate.settle(token); throw error; }
            void (async () => {
                let partial: AssistantMessage | undefined;
                let terminal = false;
                const failure = () => {
                    const model = args[0];
                    const msg: AssistantMessage = { role: "assistant", api: model.api, provider: model.provider, model: model.id,
                        content: partial?.content ?? [], usage: partial?.usage ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
                            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
                        stopReason: args[2]?.signal?.aborted ? "aborted" : "error", errorMessage: "MODEL_REQUEST_STREAM_USAGE_UNKNOWN", timestamp: Date.now() };
                    gate.settle(token, msg);
                    output.push({ type: "error", reason: msg.stopReason as "error" | "aborted", error: msg });
                    output.end();
                };
                try {
                    const stream = await source;
                    let resultOnly: AssistantMessage | undefined;
                    void stream.result().then(result => { resultOnly = result; }, () => {});
                    for await (const event of stream) {
                        if ("partial" in event) partial = event.partial;
                        if (event.type === "done" || event.type === "error") {
                            if (terminal) continue; // duplicate end, same dispatch only
                            terminal = true;
                            gate.settle(token, event.type === "done" ? event.message : event.error);
                        }
                        output.push(event);
                    }
                    if (!terminal) {
                        if (resultOnly) { gate.settle(token, resultOnly); output.end(resultOnly); }
                        else failure();
                    } else output.end();
                } catch { if (!terminal) failure(); else output.end(); }
            })();
            return output;
        };
    }
}
