/** Opt-in benchmark credential reference: one operator-owned source, no copies.
 * PI parses the credential. This store never runs configured commands, writes,
 * logs tokens, or invokes the OAuth refresh callback. Renew through the existing
 * operator login workflow outside the benchmark if the source expires.
 */
import { lstatSync, realpathSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { ModelRuntime, readStoredCredential } from "@earendil-works/pi-coding-agent";
import type { AuthOperationOptions, Credential, CredentialInfo, CredentialStore } from "@earendil-works/pi-ai";

export const READONLY_OAUTH_ENV = "ROSCLAW_PI_READONLY_OAUTH_PATH";
const PROVIDER = "openai-codex";
// PI starts refreshing at five minutes. Reject before that branch can run.
export const OAUTH_MIN_VALIDITY_MS = 5 * 60 * 1000 + 1000;

export class BenchmarkOAuthError extends Error {
	constructor(readonly code: string) { super(code); }
}

function canonicalWritableRoot(path: string): string {
	let current = resolve(path);
	const missing: string[] = [];
	for (;;) {
		try {
			lstatSync(current); // broken symlinks must fail, not be treated as absent
			return resolve(realpathSync(current), ...missing);
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
			// realpath ENOENT after lstat succeeded means an existing broken link.
			try { lstatSync(current); throw new Error("broken writable-root link"); }
			catch (statError) { if ((statError as NodeJS.ErrnoException).code !== "ENOENT") throw statError; }
			const parent = dirname(current);
			if (parent === current) throw error;
			missing.unshift(relative(parent, current));
			current = parent;
		}
	}
}

export function assertReadonlyOAuthOutsideWritableRoots(authPath: string, roots: readonly string[]): void {
	const within = (source: string, root: string) => {
		const rel = relative(root, source);
		return rel === "" || (!isAbsolute(rel) && rel !== ".." && !rel.startsWith(`..${sep}`));
	};
	try {
		const source = resolve(authPath);
		const canonicalSource = realpathSync(source);
		for (const root of roots) {
			if (within(source, resolve(root)) || within(canonicalSource, canonicalWritableRoot(root))) {
				throw new BenchmarkOAuthError("BENCH_OAUTH_SOURCE_OVERLAPS_WRITABLE_SCOPE");
			}
		}
	} catch (error) {
		if (error instanceof BenchmarkOAuthError) throw error;
		throw new BenchmarkOAuthError("BENCH_OAUTH_WRITABLE_SCOPE_INVALID");
	}
}

export class ReadOnlyBenchmarkOAuthStore implements CredentialStore {
	constructor(readonly authPath: string, private readonly minimumValidityMs = OAUTH_MIN_VALIDITY_MS) {
		if (!isAbsolute(authPath) || resolve(authPath) !== authPath ||
			!Number.isFinite(minimumValidityMs) || minimumValidityMs < OAUTH_MIN_VALIDITY_MS) {
			throw new BenchmarkOAuthError("BENCH_OAUTH_REFERENCE_INVALID");
		}
		if ([join(homedir(), ".pi", "agent", "auth.json"), join(homedir(), ".rosclaw", "agent", "auth.json")].includes(authPath)) {
			throw new BenchmarkOAuthError("BENCH_OAUTH_DEDICATED_SOURCE_REQUIRED");
		}
	}

	private revision(): string {
		try {
			const st = lstatSync(this.authPath);
			if (!st.isFile() || st.isSymbolicLink() || st.nlink !== 1 || (st.mode & 0o777) !== 0o600 ||
				st.uid !== process.getuid?.() || st.size < 1 || st.size > 1048576 || realpathSync(this.authPath) !== this.authPath) {
				throw new Error();
			}
			return [st.dev, st.ino, st.size, st.mtimeMs, st.ctimeMs, st.mode].join(":");
		} catch {
			throw new BenchmarkOAuthError("BENCH_OAUTH_SOURCE_UNAVAILABLE_OR_UNSAFE");
		}
	}

	async read(providerId: string, options?: AuthOperationOptions): Promise<Credential | undefined> {
		options?.signal?.throwIfAborted();
		if (providerId !== PROVIDER) throw new BenchmarkOAuthError("BENCH_OAUTH_PROVIDER_FORBIDDEN");
		const before = this.revision();
		// Public PI entrypoint parses only the selected provider; no env/API-key
		// command resolution, login, or refresh occurs in this call.
		const credential = readStoredCredential(PROVIDER, this.authPath);
		if (this.revision() !== before) throw new BenchmarkOAuthError("BENCH_OAUTH_SOURCE_CHANGED_DURING_READ");
		options?.signal?.throwIfAborted();
		if (!credential || credential.type !== "oauth" || typeof credential.access !== "string" || !credential.access ||
			typeof credential.refresh !== "string" || !credential.refresh || !Number.isFinite(credential.expires)) {
			throw new BenchmarkOAuthError("BENCH_OAUTH_CREDENTIAL_MISSING_OR_INVALID");
		}
		if (Date.now() + this.minimumValidityMs >= credential.expires) {
			throw new BenchmarkOAuthError("BENCH_OAUTH_RENEW_EXTERNALLY_REQUIRED");
		}
		return structuredClone(credential);
	}

	async list(options?: AuthOperationOptions): Promise<readonly CredentialInfo[]> {
		await this.read(PROVIDER, options);
		return [{ providerId: PROVIDER, type: "oauth" }];
	}

	async modify(_provider: string, _fn: (current: Credential | undefined) => Promise<Credential | undefined>): Promise<Credential | undefined> {
		// Do not invoke fn: it may be a network refresh with token rotation.
		throw new BenchmarkOAuthError("BENCH_OAUTH_READONLY_REFRESH_OR_LOGIN_FORBIDDEN");
	}
	async delete(): Promise<void> {
		throw new BenchmarkOAuthError("BENCH_OAUTH_READONLY_LOGOUT_FORBIDDEN");
	}
}

export async function createReadonlyBenchmarkModelRuntime(authPath: string, minimumValidityMs = OAUTH_MIN_VALIDITY_MS): Promise<ModelRuntime> {
	const credentials = new ReadOnlyBenchmarkOAuthStore(authPath, minimumValidityMs);
	await credentials.read(PROVIDER); // fail before catalog/session initialization
	return await ModelRuntime.create({
		credentials,
		modelsPath: null, // builtin catalog; never shadow with trial/global models.json
		allowModelNetwork: false,
	});
}
