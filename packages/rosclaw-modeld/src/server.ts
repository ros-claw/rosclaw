/** rosclaw-modeld UDS server (批次 D §7.3)。
 *
 * 安全边界：
 * - socket 目录 0700、socket 0600；
 * - 所有请求要求启动时生成的 bearer token（经进程环境传递，不落盘）；
 * - 错误统一 redact；
 * - 本进程没有任何 Mission/工具/硬件能力——只有 pi-ai provider 调用。
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { timingSafeEqual } from "node:crypto";
import { chmodSync, mkdirSync, rmSync } from "node:fs";
import { dirname } from "node:path";
import { FileCredentialStore } from "./credentials.js";
import { getProvider, providerIds, providerSpec } from "./providers.js";
import { redact } from "./redact.js";
import { streamTurn, type ModeldTurnRequest } from "./stream.js";

export interface ModeldOptions {
	socketPath: string;
	token: string;
	homeDir: string;
}

export function startModeld(options: ModeldOptions): Promise<Server> {
	const store = new FileCredentialStore(options.homeDir);
	mkdirSync(dirname(options.socketPath), { recursive: true, mode: 0o700 });
	chmodSync(dirname(options.socketPath), 0o700);
	rmSync(options.socketPath, { force: true });

	function json(res: ServerResponse, code: number, body: unknown): void {
		res.writeHead(code, { "content-type": "application/json" });
		res.end(JSON.stringify(body));
	}

	async function readBody(req: IncomingMessage): Promise<Record<string, unknown>> {
		const chunks: Buffer[] = [];
		for await (const chunk of req) chunks.push(chunk as Buffer);
		if (chunks.length === 0) return {};
		return JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>;
	}

	function envKeyFor(providerId: string): string | undefined {
		const spec = providerSpec(providerId);
		if (!spec) return undefined;
		for (const name of spec.envKeys) {
			const value = process.env[name];
			if (value) return value;
		}
		return undefined;
	}

	const server = createServer(async (req, res) => {
		try {
			const auth = req.headers.authorization ?? "";
			const expected = Buffer.from(`Bearer ${options.token}`);
			const provided = Buffer.from(auth);
			if (provided.length !== expected.length || !timingSafeEqual(provided, expected)) {
				json(res, 401, { error: "unauthorized" });
				return;
			}
			const url = new URL(req.url ?? "/", "http://localhost");
			const path = url.pathname;

			if (req.method === "GET" && path === "/v1/providers") {
				const stored = new Set(store.list().map((c) => c.provider));
				json(res, 200, {
					providers: providerIds().map((id) => {
						const spec = providerSpec(id);
						return {
							id,
							name: spec?.name ?? id,
							base_url: spec?.baseUrl,
							env_keys: spec?.envKeys ?? [],
							auth: stored.has(id)
								? "stored"
								: envKeyFor(id)
									? "env"
									: (spec?.envKeys.length ?? 0) === 0
										? "none_required"
										: "missing",
						};
					}),
				});
				return;
			}

			if (req.method === "GET" && path === "/v1/models") {
				const providerId = url.searchParams.get("provider") ?? "";
				const provider = getProvider(providerId);
				json(res, 200, {
					models: provider.getModels().map((m) => ({
						id: m.id,
						provider: providerId,
						context_window: m.contextWindow,
						max_tokens: m.maxTokens,
						reasoning: m.reasoning,
					})),
				});
				return;
			}

			if (req.method === "GET" && path === "/v1/auth") {
				json(res, 200, { credentials: store.list() });
				return;
			}

			const loginMatch = path.match(/^\/v1\/auth\/([a-z0-9-]+)\/login$/);
			if (req.method === "POST" && loginMatch) {
				const providerId = loginMatch[1];
				if (!providerSpec(providerId)) {
					json(res, 404, { error: `unknown provider ${providerId}` });
					return;
				}
				const body = await readBody(req);
				if (body.mode === "oauth") {
					json(res, 501, {
						error: "oauth_not_implemented",
						message: "OAuth 登录将在后续批次提供；请使用 API Key（mode=api_key）。",
					});
					return;
				}
				const key = String(body.api_key ?? "").trim();
				if (!key) {
					json(res, 400, { error: "missing api_key" });
					return;
				}
				store.set(providerId, key);
				json(res, 200, { ok: true, provider: providerId });
				return;
			}

			const logoutMatch = path.match(/^\/v1\/auth\/([a-z0-9-]+)\/logout$/);
			if (req.method === "POST" && logoutMatch) {
				json(res, 200, { ok: true, deleted: store.delete(logoutMatch[1]) });
				return;
			}

			if (req.method === "POST" && path === "/v1/probe") {
				const body = await readBody(req);
				const providerId = String(body.provider ?? "");
				const modelId = String(body.model ?? "");
				const apiKey = store.resolve(providerId) ?? envKeyFor(providerId);
				if (!apiKey && (providerSpec(providerId)?.envKeys.length ?? 0) > 0) {
					json(res, 200, { ok: false, error: "no_credential", message: "未配置凭据" });
					return;
				}
				const provider = getProvider(providerId);
				const model = provider.getModels().find((m) => m.id === modelId);
				if (!model) {
					json(res, 200, { ok: false, error: "unknown_model", message: modelId });
					return;
				}
				const events = [];
				for await (const event of streamTurn(
					provider,
					model,
					{
						provider: providerId,
						model: modelId,
						messages: [{ role: "user", content: "Reply with exactly: ok" }],
						max_tokens: 8,
					},
					apiKey,
					AbortSignal.timeout(30_000),
				)) {
					events.push(event);
				}
				const failed = events.find((e) => e.type === "error");
				json(res, 200, {
					ok: !failed,
					error: failed ? (failed as { kind: string }).kind : undefined,
					message: failed ? (failed as { message: string }).message : "probe ok",
				});
				return;
			}

			if (req.method === "POST" && path === "/v1/stream") {
				const body = (await readBody(req)) as unknown as ModeldTurnRequest;
				const providerId = String(body.provider ?? "");
				const apiKey = store.resolve(providerId) ?? envKeyFor(providerId);
				res.writeHead(200, {
					"content-type": "text/event-stream",
					"cache-control": "no-cache",
				});
				const send = (event: unknown) => {
					res.write(`data: ${JSON.stringify(event)}\n\n`);
				};
				if (!apiKey && (providerSpec(providerId)?.envKeys.length ?? 0) > 0) {
					send({ type: "error", kind: "no_credential", message: "未配置凭据" });
					res.end();
					return;
				}
				let provider;
				let model;
				try {
					provider = getProvider(providerId);
					model = provider.getModels().find((m) => m.id === String(body.model));
					if (!model) throw new Error(`unknown model ${String(body.model)}`);
				} catch (err) {
					send({ type: "error", kind: "unknown_model", message: redact((err as Error).message) });
					res.end();
					return;
				}
				const abort = new AbortController();
				req.on("close", () => abort.abort());
				for await (const event of streamTurn(provider, model, body, apiKey, abort.signal)) {
					send(event);
				}
				res.end();
				return;
			}

			json(res, 404, { error: "not_found" });
		} catch (err) {
			json(res, 500, { error: redact((err as Error).message) });
		}
	});

	return new Promise((resolve) => {
		server.listen(options.socketPath, () => {
			chmodSync(options.socketPath, 0o600);
			resolve(server);
		});
	});
}
