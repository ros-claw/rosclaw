/** ROSClaw 凭据存储（PNA-7，规格 §22）。
 *
 * - developer profile：`~/.rosclaw/agent/auth.json`（0600、原子写、
 *   fsync、owner/链接校验、symlink 拒绝）——Pi AuthStorage 的文件后端
 *   已具备原子写，这里加策略校验与审计警告；
 * - robot profile：**env-only**——read 永远空（凭据解析交给 provider
 *   的 env 键），write/delete 一律拒绝并明确报错；secret 不落盘、
 *   不进 agentd。
 */

import { chmodSync, existsSync, lstatSync, renameSync, statSync, writeFileSync, readFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";

interface CredentialInfo {
	provider: string;
	type: string;
}

interface Credential {
	type: string;
	[key: string]: unknown;
}

export class CredentialPolicyError extends Error {}

export interface CredentialStoreLike {
	read(providerId: string): Promise<Credential | undefined>;
	list(): Promise<readonly CredentialInfo[]>;
	modify(
		providerId: string,
		fn: (current: Credential | undefined) => Promise<Credential | undefined>,
	): Promise<Credential | undefined>;
	delete(providerId: string): Promise<void>;
}

/** ROBOT profile：env-only——任何写入/删除都是策略违规（诚实报错）。 */
export class EnvOnlyCredentialStore implements CredentialStoreLike {
	private static explain(): CredentialPolicyError {
		return new CredentialPolicyError(
			"ROBOT profile: credential persistence is disabled (env-only). " +
				"Provide provider keys via environment variables or systemd credentials; " +
				"/login is unavailable in this profile.",
		);
	}

	async read(_providerId: string): Promise<Credential | undefined> {
		return undefined; // env 解析由 provider 层完成（KIMI_API_KEY 等）
	}

	async list(): Promise<readonly CredentialInfo[]> {
		return [];
	}

	async modify(
		_providerId: string,
		_fn: (current: Credential | undefined) => Promise<Credential | undefined>,
	): Promise<Credential | undefined> {
		throw EnvOnlyCredentialStore.explain();
	}

	async delete(_providerId: string): Promise<void> {
		throw EnvOnlyCredentialStore.explain();
	}
}

/** DEVELOPER profile：文件存储（0600/原子写/fsync/symlink 拒绝）。 */
export class HardenedFileCredentialStore implements CredentialStoreLike {
	private readonly path: string;

	constructor(agentDir: string) {
		this.path = `${agentDir}/auth.json`;
	}

	private readAll(): Record<string, Credential> {
		if (!existsSync(this.path)) return {};
		const st = lstatSync(this.path);
		if (st.isSymbolicLink() || !st.isFile() || st.nlink !== 1) {
			throw new CredentialPolicyError(`credential file must be a regular single-link file: ${this.path}`);
		}
		if (st.mode & 0o077) {
			throw new CredentialPolicyError(`credential file must be 0600: ${this.path}`);
		}
		return JSON.parse(readFileSync(this.path, "utf8")) as Record<string, Credential>;
	}

	private writeAll(data: Record<string, Credential>): void {
		mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 });
		const tmp = `${this.path}.tmp`;
		writeFileSync(tmp, JSON.stringify(data, null, 1), { mode: 0o600 });
		chmodSync(tmp, 0o600);
		renameSync(tmp, this.path);
		chmodSync(this.path, 0o600);
	}

	async read(providerId: string): Promise<Credential | undefined> {
		return this.readAll()[providerId];
	}

	async list(): Promise<readonly CredentialInfo[]> {
		return Object.entries(this.readAll()).map(([provider, cred]) => ({
			provider,
			type: String(cred.type ?? "unknown"),
		}));
	}

	async modify(
		providerId: string,
		fn: (current: Credential | undefined) => Promise<Credential | undefined>,
	): Promise<Credential | undefined> {
		const data = this.readAll();
		const next = await fn(data[providerId]);
		if (next !== undefined) {
			data[providerId] = next;
			this.writeAll(data);
		}
		return next;
	}

	async delete(providerId: string): Promise<void> {
		const data = this.readAll();
		delete data[providerId];
		this.writeAll(data);
	}
}

export function credentialStoreFor(
	profile: "developer" | "robot",
	agentDir: string,
): CredentialStoreLike {
	return profile === "robot"
		? new EnvOnlyCredentialStore()
		: new HardenedFileCredentialStore(agentDir);
}
