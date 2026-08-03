/** File credential store (批次 D §7.3)。
 *
 * - 文件 0600、目录 0700；
 * - list/status 只返回 metadata（provider、created_at、sha256 前 8 位
 *   指纹）——永远不返回 secret 本体；
 * - secret 只经内存路径传给 pi-ai stream 调用。
 */

import { createHash } from "node:crypto";
import { chmodSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
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
}

export class FileCredentialStore {
	private readonly path: string;

	constructor(homeDir: string) {
		this.path = `${homeDir}/modeld-credentials.json`;
	}

	private readAll(): Record<string, StoredCredential> {
		if (!existsSync(this.path)) return {};
		try {
			return JSON.parse(readFileSync(this.path, "utf8")) as Record<string, StoredCredential>;
		} catch {
			return {}; // 解析失败不伪造成功——当作空（调用方会按未配置处理）
		}
	}

	private writeAll(data: Record<string, StoredCredential>): void {
		mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 });
		// tmp + fsync + atomic rename（§8.4 写入要求）。
		const tmp = `${this.path}.tmp`;
		writeFileSync(tmp, JSON.stringify(data), { mode: 0o600 });
		chmodSync(tmp, 0o600);
		renameSync(tmp, this.path);
		chmodSync(this.path, 0o600);
	}

	set(provider: string, key: string): void {
		const data = this.readAll();
		data[provider] = { type: "api_key", key, created_at: new Date().toISOString() };
		this.writeAll(data);
	}

	delete(provider: string): boolean {
		const data = this.readAll();
		if (!(provider in data)) return false;
		delete data[provider];
		this.writeAll(data);
		return true;
	}

	/** 内存路径取 key —— 仅供 stream 调用使用，不出现在任何 API 响应。 */
	resolve(provider: string): string | undefined {
		return this.readAll()[provider]?.key;
	}

	list(): CredentialInfo[] {
		return Object.entries(this.readAll()).map(([provider, cred]) => ({
			provider,
			type: cred.type,
			fingerprint: createHash("sha256").update(cred.key).digest("hex").slice(0, 8),
			created_at: cred.created_at,
		}));
	}
}
