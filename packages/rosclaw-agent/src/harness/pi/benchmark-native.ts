/** Thin native PI SDK launcher for the benchmark A leg, not the PI CLI.
 * The CLI has no credential-store injection seam. Keep PI's original session,
 * coding tools, resource loader and TUI; add no ROSClaw tools or model requests.
 */
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { assertReadonlyOAuthOutsideWritableRoots, BenchmarkOAuthError, createReadonlyBenchmarkModelRuntime, OAUTH_MIN_VALIDITY_MS } from "./readonly-benchmark-oauth.js";

export async function createNativeBenchmarkRuntime(cwd: string, agentDir: string, authPath: string, modelId: string) {
	assertReadonlyOAuthOutsideWritableRoots(authPath, [cwd, agentDir]);
	const { createAgentSessionRuntime, createAgentSessionServices, createAgentSessionFromServices, SessionManager, SettingsManager } = await import("@earendil-works/pi-coding-agent");
	const modelRuntime = await createReadonlyBenchmarkModelRuntime(authPath);
	const model = modelRuntime.getModel("openai-codex", modelId);
	if (!model) throw new BenchmarkOAuthError("BENCH_BUILTIN_MODEL_UNKNOWN");
	return await createAgentSessionRuntime(async ({ cwd, agentDir, sessionManager, sessionStartEvent }) => {
		const services = await createAgentSessionServices({ cwd, agentDir, modelRuntime, settingsManager: SettingsManager.create(cwd, agentDir) });
		const created = await createAgentSessionFromServices({ services, sessionManager, sessionStartEvent, model });
		return { ...created, services, diagnostics: services.diagnostics };
	}, { cwd, agentDir, sessionManager: SessionManager.create(cwd, resolve(agentDir, "sessions")) });
}

export async function benchmarkNativeMain(args: string[]): Promise<number> {
	let started = false;
	const check = args[0] === "--check";
	const values = check ? args.slice(1) : args;
	const [authPath, modelId, validityText] = values;
	const minimumValidityMs = validityText === undefined ? OAUTH_MIN_VALIDITY_MS : Number(validityText);
	try {
		if (!authPath || !modelId || modelId.trim() !== modelId || values.length > 3) throw new BenchmarkOAuthError("BENCH_OAUTH_ARGUMENT_INVALID");
		const modelRuntime = await createReadonlyBenchmarkModelRuntime(authPath, minimumValidityMs);
		const model = modelRuntime.getModel("openai-codex", modelId);
		if (!model || model.provider !== "openai-codex" || model.id !== modelId) throw new BenchmarkOAuthError("BENCH_BUILTIN_MODEL_UNKNOWN");
		if (!await modelRuntime.getAuth(model)) throw new BenchmarkOAuthError("BENCH_OAUTH_NOT_CONFIGURED");
		if (check) {
			console.log(JSON.stringify({ status: "READY", provider: "openai-codex", model_id: modelId, credential_mode: "EXPLICIT_READONLY_NO_REFRESH", remote_auth_verified: false }));
			return 0;
		}
		const agentDir = process.env.PI_CODING_AGENT_DIR;
		if (!agentDir) throw new BenchmarkOAuthError("BENCH_ISOLATED_AGENT_DIR_REQUIRED");
		const { InteractiveMode } = await import("@earendil-works/pi-coding-agent");
		const runtime = await createNativeBenchmarkRuntime(process.cwd(), agentDir, authPath, modelId);
		const ui = new InteractiveMode(runtime);
		try {
			await ui.init();
			started = true;
			console.log("HARNESSBENCH_NATIVE_PI_READY");
			await ui.run();
		} finally {
			ui.stop();
			await runtime.dispose();
		}
		return 0;
	} catch (error) {
		// Never serialize credential/parser/provider errors, causes, or token data.
		const code = error instanceof BenchmarkOAuthError ? error.code : "BENCH_READONLY_OAUTH_OR_RUNTIME_FAILED";
		console.log(JSON.stringify({ status: started ? "ERROR" : "NOT_RUN", code }));
		return 2;
	}
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
	process.env.PI_SKIP_VERSION_CHECK = "1";
	process.exitCode = await benchmarkNativeMain(process.argv.slice(2));
}
