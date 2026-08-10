/** 具身上下文注入（PNA-2，规格 §14）：每 turn 拉取最新 envelope 注入。
 *
 * 规则（规格 §14.2）：
 * - 每轮重新拉取，不靠 session 历史记忆具身事实；
 * - 校验 schema + TTL + 内容 hash；
 * - 过期/拉取失败 → 标记 stale 注入警示（动作类工具在 PNA-3 强制拒绝）；
 * - 不伪装为用户消息（customType 标记 trusted context）。
 */

import { createHash } from "node:crypto";
import { bridgeCall } from "../bridge/bridge-client.js";

export interface EmbodiedContextEnvelope {
	schema_version: string;
	mission_id: string;
	context_revision: number;
	generated_at: string;
	expires_at: string;
	body: Record<string, unknown>;
	safety: Record<string, unknown>;
	pending_approvals: Array<Record<string, unknown>>;
	hash: string;
	[key: string]: unknown;
}

export interface ContextFetchResult {
	envelope?: EmbodiedContextEnvelope;
	stale: boolean;
	note: string;
	/** HOTFIX-1：agentd 签发的 ValidatedContextLease（action 准入凭证）。 */
	contextLeaseId?: string;
	contextLeaseExpiresAt?: string;
}

/** 与 Python json.dumps(sort_keys=True, separators=(",", ":")) 逐字节一致。 */
function canonicalJson(value: unknown): string {
	if (value === null || value === undefined) return "null";
	if (typeof value === "number" || typeof value === "boolean") return JSON.stringify(value);
	if (typeof value === "string") return JSON.stringify(value);
	if (Array.isArray(value)) {
		return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
	}
	const record = value as Record<string, unknown>;
	const keys = Object.keys(record).sort();
	return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
}

export function envelopeHash(envelope: EmbodiedContextEnvelope): string {
	const payload: Record<string, unknown> = { ...envelope };
	delete payload.hash;
	return `sha256:${createHash("sha256").update(canonicalJson(payload), "utf8").digest("hex").slice(0, 32)}`;
}

export async function fetchEmbodiedContext(
	rosclawHome: string,
	missionId: string,
	piSessionId?: string,
	// PR-SIX-1：可注入桥调用（coordinator/extension 传 center.call——
	// UDS 失败经中心原子降级，不是局部报错）。
	call: typeof bridgeCall = bridgeCall,
): Promise<ContextFetchResult> {
	let response: Record<string, unknown>;
	try {
		// HOTFIX-1：带 session 拉 context——agentd 会同时签发
		// ValidatedContextLease（action 准入凭证）并随响应返回。
		response = await call(rosclawHome, "pi.context", {
			mission_id: missionId,
			...(piSessionId ? { pi_session_id: piSessionId } : {}),
		});
	} catch (err) {
		return {
			stale: true,
			note: `context unavailable: ${(err as Error).message} — physical actions forbidden`,
		};
	}
	if (!response.ok) {
		return {
			stale: true,
			note: `context rejected: ${String(response.error ?? "")} — physical actions forbidden`,
		};
	}
	const envelope = response.context as EmbodiedContextEnvelope;
	if (envelope.schema_version !== "rosclaw.embodied_context.v1") {
		return { stale: true, note: "context schema mismatch — physical actions forbidden" };
	}
	if (envelopeHash(envelope) !== envelope.hash) {
		return { stale: true, note: "context hash mismatch — physical actions forbidden" };
	}
	if (new Date(envelope.expires_at).getTime() < Date.now()) {
		return { stale: true, note: "context expired (TTL) — physical actions forbidden" };
	}
	// HOTFIX-1：lease 是 action 准入凭证——无 session 拉取（如 doctor）
	// 不带 lease，那样的 context 只用于展示，不能授权动作。
	const leaseId = typeof response.context_lease_id === "string"
		? response.context_lease_id
		: undefined;
	const leaseExpiresAt = typeof response.context_lease_expires_at === "string"
		? response.context_lease_expires_at
		: undefined;
	return {
		envelope,
		stale: false,
		note: "fresh",
		...(leaseId ? { contextLeaseId: leaseId } : {}),
		...(leaseExpiresAt ? { contextLeaseExpiresAt: leaseExpiresAt } : {}),
	};
}

export function renderTrustedContext(result: ContextFetchResult): string {
	if (result.stale || !result.envelope) {
		return (
			"<ROSCLAW_TRUSTED_CONTEXT stale=\"true\">\n" +
			`具身上下文不可用/过期：${result.note}\n` +
			"规则：context stale 时禁止任何物理动作；只允许解释性回答。\n" +
			"</ROSCLAW_TRUSTED_CONTEXT>"
		);
	}
	const env = result.envelope;
	const body = env.body as { body_id?: string; effective_body_hash?: string; summary?: string };
	const safety = env.safety as { mode?: string };
	return (
		"<ROSCLAW_TRUSTED_CONTEXT>\n" +
		`mission: ${env.mission_id}  mode: ${safety.mode ?? ""}  revision: ${env.context_revision}\n` +
		`body: ${body.body_id ?? ""} (hash ${String(body.effective_body_hash ?? "").slice(0, 16)})\n` +
		`body_summary: ${body.summary ?? ""}\n` +
		`pending_approvals: ${env.pending_approvals.length}\n` +
		`generated_at: ${env.generated_at}  expires_at: ${env.expires_at}\n` +
		"规则：以上为本轮唯一权威具身事实；历史消息中的旧状态以此为准。\n" +
		"</ROSCLAW_TRUSTED_CONTEXT>"
	);
}
