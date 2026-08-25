/** P1-A1（0824 总纲 §10.1）：模型探测单源——经 Pi ModelRuntime。
 *
 * setup/doctor 的 probe 与 chat 走同一引擎、同一配置（agentDir 下
 * settings.json + models.json + auth.json/env）。不再有第二条
 * Python HTTP chat probe 栈。
 *
 * 探测四步（与 K0 对齐）：auth 配置 → models listing → 短 chat →
 * 严格 tool call。输出为无 secret 的 JSON 报告（apiKey 的 $ENV
 * 引用绝不展开打印）。
 */

import type { ModelRuntime } from "@earendil-works/pi-coding-agent";

import { createSharedModelRuntime } from "./pi-model-runtime.js";

export interface PiProbeReport {
	engine: "pi";
	reachable: boolean;
	auth_configured: boolean;
	models_visible: string[];
	expected_model_present: boolean;
	chat_ok: boolean;
	tool_call_ok: boolean;
	provider?: string;
	model?: string;
	error?: string;
}

interface ProbeOptions {
	agentDir: string;
	cwd: string;
	profile: "developer" | "robot";
	/** 单步网络超时（chat/tool call）；默认 60s。 */
	timeoutMs?: number;
	/** 测试注入：替代真实 ModelRuntime。 */
	runtime?: ModelRuntime;
	/** 测试注入：替代 SettingsManager 读取。 */
	defaults?: { provider?: string; model?: string };
}

function classifyError(err: unknown): string {
	const message = err instanceof Error ? err.message : String(err);
	if (/401|403|unauthorized|forbidden/i.test(message)) return `AUTH_FAILED: ${message}`;
	if (/402|payment|quota|insufficient/i.test(message)) return `QUOTA_EXHAUSTED: ${message}`;
	if (/429|rate.?limit/i.test(message)) return `RATE_LIMITED: ${message}`;
	if (/ECONNREFUSED|ENOTFOUND|ETIMEDOUT|fetch failed|network/i.test(message)) {
		return `NETWORK_UNREACHABLE: ${message}`;
	}
	return `PROBE_FAILED: ${message}`;
}

async function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
	let timer: ReturnType<typeof setTimeout> | undefined;
	try {
		return await Promise.race([
			promise,
			new Promise<never>((_, reject) => {
				timer = setTimeout(() => reject(new Error(`${label} timeout after ${ms}ms`)), ms);
			}),
		]);
	} finally {
		if (timer) clearTimeout(timer);
	}
}

export async function probePiModel(options: ProbeOptions): Promise<PiProbeReport> {
	const report: PiProbeReport = {
		engine: "pi",
		reachable: false,
		auth_configured: false,
		models_visible: [],
		expected_model_present: false,
		chat_ok: false,
		tool_call_ok: false,
	};
	const timeoutMs = options.timeoutMs ?? 60_000;
	let defaults = options.defaults;
	if (!defaults) {
		const { SettingsManager } = await import("@earendil-works/pi-coding-agent");
		const settings = SettingsManager.create(options.cwd, options.agentDir);
		defaults = {
			provider: settings.getDefaultProvider() ?? undefined,
			model: settings.getDefaultModel() ?? undefined,
		};
	}
	if (!defaults.provider || !defaults.model) {
		report.error = "MODEL_NOT_CONFIGURED: settings.json 缺 defaultProvider/defaultModel——运行 rosclaw setup model";
		return report;
	}
	report.provider = defaults.provider;
	report.model = defaults.model;
	let runtime = options.runtime;
	if (!runtime) {
		try {
			runtime = await createSharedModelRuntime(options.agentDir, options.profile);
		} catch (err) {
			report.error = classifyError(err);
			return report;
		}
	}
	// 1) auth 配置（models.json $ENV 引用 / auth.json / env——任一命中）。
	report.auth_configured = runtime.hasConfiguredAuth(defaults.provider);
	if (!report.auth_configured) {
		report.error = `AUTH_NOT_CONFIGURED: provider ${defaults.provider} 无可用凭据（env/models.json/auth.json 均无）`;
		return report;
	}
	// 2) models listing（可用性快照=凭据+目录都通）。
	try {
		const available = await withTimeout(
			runtime.getAvailable(defaults.provider), timeoutMs, "models listing",
		);
		report.models_visible = available.map((m) => m.id);
		report.reachable = true;
	} catch (err) {
		report.error = classifyError(err);
		return report;
	}
	const model = runtime.getModel(defaults.provider, defaults.model);
	report.expected_model_present =
		report.models_visible.length === 0 ||
		report.models_visible.includes(defaults.model);
	if (!model) {
		report.error = `MODEL_UNKNOWN: ${defaults.provider}/${defaults.model} 不在 provider 目录`;
		report.expected_model_present = false;
		return report;
	}
	// 3) 短 chat。
	try {
		const reply = await withTimeout(
			runtime.completeSimple(model, {
				messages: [{
					role: "user",
					content: "Reply with exactly the two letters: OK",
					timestamp: 0,
				}],
			}),
			timeoutMs,
			"chat probe",
		);
		// stopReason=error 时 detail 在 errorMessage（如 401）——分类
		// 透传，不报空泛的 EMPTY。
		const replyError = (reply as { errorMessage?: string }).errorMessage;
		if (reply.stopReason === "error" || replyError) {
			report.error = classifyError(replyError ?? `stopReason=${reply.stopReason}`);
			return report;
		}
		const text = reply.content
			.filter((b): b is { type: "text"; text: string } => b.type === "text")
			.map((b) => b.text)
			.join("");
		report.chat_ok = text.trim().length > 0;
		if (!report.chat_ok) {
			report.error = `CHAT_PROBE_EMPTY: 模型未给出有效回复（stopReason=${reply.stopReason}）`;
			return report;
		}
	} catch (err) {
		report.error = classifyError(err);
		return report;
	}
	// 4) 严格 tool call。
	try {
		const reply = await withTimeout(
			runtime.completeSimple(model, {
				systemPrompt: "You must call the report_ok tool exactly once. Do not answer in text.",
				messages: [{
					role: "user",
					content: "Call report_ok with ok=true now.",
					timestamp: 0,
				}],
				tools: [{
					name: "report_ok",
					description: "Report probe success.",
					parameters: {
						type: "object",
						properties: { ok: { type: "boolean" } },
						required: ["ok"],
					},
				}],
			}),
			timeoutMs,
			"tool call probe",
		);
		const replyError = (reply as { errorMessage?: string }).errorMessage;
		if (reply.stopReason === "error" || replyError) {
			report.error = classifyError(replyError ?? `stopReason=${reply.stopReason}`);
			return report;
		}
		report.tool_call_ok = reply.content.some(
			(b) => b.type === "toolCall" && (b as { name?: string }).name === "report_ok",
		);
		if (!report.tool_call_ok) {
			report.error = "TOOL_CALL_PROBE_FAILED: 模型未发起 report_ok 调用";
		}
	} catch (err) {
		report.error = classifyError(err);
	}
	return report;
}
