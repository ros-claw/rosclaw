/** 凭据后端（审计 P0-04 凭据策略统一）。
 *
 * 策略（唯一来源，/doctor 可见）：
 * 1. Headless/robot 默认：env:VAR 引用，不落任何文件；
 * 2. /login 的 API key 默认只进内存（session-scoped，modeld 重启即失）；
 * 3. 0600 明文文件必须显式 opt-in（ROSCLAW_MODELD_ALLOW_FILE_CREDENTIALS=1），
 *    激活时 /v1/auth 与 doctor 显示警告；写入 tmp+fsync+atomic rename；
 *    损坏文件报错隔离——绝不静默当作空凭据。
 * modeld 是 provider secret 的唯一所有者；其他进程只见 status/fingerprint。
 */

import { createHash } from "node:crypto";
import {
	chmodSync,
	closeSync,
	constants,
	existsSync,
	mkdirSync,
	openSync,
	fsyncSync,
	readFileSync,
	renameSync,
	statSync,
	writeFileSync,
} from "node:fs";
import { dirname } from "node:path";

interface StoredCredential {
	type: "api_key";
	key: string;
	created_at: string;
}

export interface CredentialInfo {
	provider: string;
	type: string;
	fingerprint: string;
	created_at: string;
	scope: "session" | "file";
}

export class CredentialStoreError extends Error {}

export class UnifiedCredentialStore {
	private readonly path: string;
	private readonly fileEnabled: boolean;
	private readonly memory = new Map<string, StoredCredential>();

	constructor(homeDir: string, opts: { allowFileCredentials?: boolean } = {}) {
		this.path = `${homeDir}/modeld-credentials.json`;
		this.fileEnabled =
			opts.allowFileCredentials ?? process.env.ROSCLAW_MODELD_ALLOW_FILE_CREDENTIALS === "1";
	}

	get policy(): { file_credentials: boolean; warning: string | null } {
		return {
			file_credentials: this.fileEnabled,
			warning: this.fileEnabled
				? "明文 0600 凭据文件已显式启用（opt-in）——生产/headless 推荐 env:VAR 或 systemd credentials"
				: null,
		};
	}

	private readFile(): Record<string, StoredCredential> {
		if (!this.fileEnabled || !existsSync(this.path)) return {};
		// P1-6：校验 owner/mode/link count——凭据文件不得是链接或多链接。
		const st = statSync(this.path);
		if (!st.isFile() || st.nlink !== 1) {
			throw new CredentialStoreError(`credential file must be a regular single-link file: ${this.path}`);
		}
		if (st.mode & 0o077) {
			throw new CredentialStoreError(`credential file must be 0600: ${this.path}`);
		}
		try {
			return JSON.parse(readFileSync(this.path, "utf8")) as Record<string, StoredCredential>;
		} catch (err) {
			// P1-6：损坏文件真实隔离到带时间戳的 quarantine（仍 0600），
			// 不静默当作空凭据（审计 P0-04.3）。
			const quarantined = `${this.path}.corrupt-${new Date().toISOString().replace(/[:.]/g, "")}`;
			try {
				renameSync(this.path, quarantined);
				chmodSync(quarantined, 0o600);
			} catch {
				// quarantine 失败不掩盖原始损坏事实
			}
			throw new CredentialStoreError(
				`credential file corrupt: ${this.path} — ${(err as Error).message}. quarantined to ${quarantined}; fix or delete it.`,
			);
		}
	}

	private writeFile(data: Record<string, StoredCredential>): void {
		// P1-6：O_NOFOLLOW + 原子替换 + 文件/目录双 fsync（崩溃一致性）。
		mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 });
		const tmp = `${this.path}.tmp`;
		const nofollow = "O_NOFOLLOW" in constants ? constants.O_NOFOLLOW : 0;
		const fd = openSync(tmp, constants.O_WRONLY | constants.O_CREAT | constants.O_TRUNC | nofollow, 0o600);
		try {
			writeFileSync(fd, JSON.stringify(data));
			fsyncSync(fd);
		} finally {
			closeSync(fd);
		}
		chmodSync(tmp, 0o600);
		renameSync(tmp, this.path);
		chmodSync(this.path, 0o600);
		const dirFd = openSync(dirname(this.path), constants.O_RDONLY);
		try {
			fsyncSync(dirFd);
		} finally {
			closeSync(dirFd);
		}
	}

	set(provider: string, key: string): { scope: "session" | "file" } {
		const cred: StoredCredential = { type: "api_key", key, created_at: new Date().toISOString() };
		if (this.fileEnabled) {
			const data = this.readFile();
			data[provider] = cred;
			this.writeFile(data);
			return { scope: "file" };
		}
		this.memory.set(provider, cred);
		return { scope: "session" };
	}

	delete(provider: string): boolean {
		const hadMemory = this.memory.delete(provider);
		let hadFile = false;
		if (this.fileEnabled) {
			const data = this.readFile();
			if (provider in data) {
				delete data[provider];
				this.writeFile(data);
				hadFile = true;
			}
		}
		return hadMemory || hadFile;
	}

	/** 内存路径取 key —— 仅供 stream 调用，不出现在任何 API 响应。 */
	resolve(provider: string): string | undefined {
		return this.memory.get(provider)?.key ?? this.readFile()[provider]?.key;
	}

	list(): CredentialInfo[] {
		const fingerprint = (key: string) =>
			createHash("sha256").update(key).digest("hex").slice(0, 8);
		const items: CredentialInfo[] = [...this.memory.entries()].map(([provider, cred]) => ({
			provider,
			type: cred.type,
			fingerprint: fingerprint(cred.key),
			created_at: cred.created_at,
			scope: "session",
		}));
		for (const [provider, cred] of Object.entries(this.readFile())) {
			items.push({
				provider,
				type: cred.type,
				fingerprint: fingerprint(cred.key),
				created_at: cred.created_at,
				scope: "file",
			});
		}
		return items;
	}
}
