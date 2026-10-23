/** Refresh before each LLM call, after the complete parallel tool batch.
 * Tool-end observations can expire while another tool is still running.
 * This observes state; it never approves or replays a physical action.
 */
import { appendFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { BridgeToolContext } from "../../tools/bridge-tools.js";
import { fetchEmbodiedContext, renderTrustedContext } from "../../extension/context-injection.js";

export async function refreshModelContext(ctx: BridgeToolContext, messages: AgentMessage[]): Promise<AgentMessage[]> {
	const prior = ctx.active.current;
	if (!prior.missionId) return messages;
	const fetched = await fetchEmbodiedContext(ctx.rosclawHome, prior.missionId, prior.sessionId,
		(_home, method, parameters) => ctx.center.call(method, parameters));
	const now = ctx.active.current;
	const bindingChanged = now.sessionId !== prior.sessionId || now.missionId !== prior.missionId
		|| now.bodyHash !== prior.bodyHash || now.mode !== prior.mode || now.bindingId !== prior.bindingId;
	const wrongEnvelope = fetched.envelope && (fetched.envelope.mission_id !== prior.missionId
		|| fetched.envelope.safety.mode !== prior.mode
		|| (prior.bodyHash !== undefined && fetched.envelope.body.effective_body_hash !== prior.bodyHash)
		|| (prior.bodyId !== undefined && fetched.envelope.body.body_id !== prior.bodyId));
	const validLease = fetched.contextLeaseId && fetched.contextLeaseExpiresAt
		&& Date.parse(fetched.contextLeaseExpiresAt) > Date.now();
	if (bindingChanged || wrongEnvelope || !validLease) {
		fetched.stale = true;
		fetched.note = "Model context binding/lease changed or could not be verified; actions remain forbidden";
	}
	if (!fetched.stale && fetched.envelope) {
		ctx.active.applyEnvelope(fetched.envelope, fetched.contextLeaseId);
		if (fetched.activeRun?.run_dir) ctx.active.patch({ taskRunDir: fetched.activeRun.run_dir });
	} else ctx.active.markContextStale(fetched.note);
	// Preserve the exact public ROS diagnosis seen at each decision. Leases,
	// writer credentials and approval tokens are deliberately excluded.
	try {
		const directory = join(ctx.rosclawHome, "logs");
		mkdirSync(directory, { recursive: true });
		appendFileSync(join(directory, "model-context-refresh.jsonl"), JSON.stringify({
			timestamp: new Date().toISOString(), mission_id: prior.missionId,
			context_revision: fetched.envelope?.context_revision, stale: fetched.stale,
			note: fetched.note, ros_observations: (fetched.envelope?.self_state as Record<string, unknown> | undefined)?.ros_observations,
		}) + "\n");
	} catch { console.warn("Native context diagnostic log unavailable; admission checks remain authoritative"); }
	return [...messages.filter(m => !(m.role === "custom" && m.customType === "rosclaw.embodied_context")), {
		role: "custom", customType: "rosclaw.embodied_context", display: false,
		content: renderTrustedContext(fetched), timestamp: Date.now(),
		details: { stale: fetched.stale, note: fetched.note, refresh_scope: "each_model_request" },
	}];
}
