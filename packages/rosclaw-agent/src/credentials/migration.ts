/** Provider 迁移与映射（PNA-7，规格 §22.3）：一套配置来源。
 *
 * 从 ROSClaw config.yaml（legacy agentd 配置）推导 Pi 运行时的
 * settings（defaultProvider/defaultModel）与 env 键映射：
 * - kimi_code → kimi-coding（KIMI_API_KEY）
 * - kimi_api/moonshot → moonshotai-cn（MOONSHOT_API_KEY）
 * - openai/anthropic/openrouter/ollama 同名
 * api_key_ref: env:X 的 X 若存在而目标 env 缺失 → 进程内补设
 * （值不落地、不进 journal）。
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const PROVIDER_MAP: Record<string, { provider: string; env: string }> = {
	kimi_code: { provider: "kimi-coding", env: "KIMI_API_KEY" },
	kimi_api: { provider: "moonshotai-cn", env: "MOONSHOT_API_KEY" },
	moonshot: { provider: "moonshotai-cn", env: "MOONSHOT_API_KEY" },
	openai: { provider: "openai", env: "OPENAI_API_KEY" },
	anthropic: { provider: "anthropic", env: "ANTHROPIC_API_KEY" },
	openrouter: { provider: "openrouter", env: "OPENROUTER_API_KEY" },
	ollama: { provider: "ollama", env: "" },
};

export interface ProviderMigration {
	provider: string;
	model: string;
	envKeySet: boolean;
	source: string;
}

/** 极简 YAML 提取（仅支持 config.yaml 的 models.profiles 已知结构）。 */
export function parseRosclawConfig(text: string): {
	defaultProfile: string;
	profiles: Record<string, { provider: string; model: string; apiKeyRef: string }>;
} {
	const profiles: Record<string, { provider: string; model: string; apiKeyRef: string }> = {};
	let defaultProfile = "";
	let current: string | null = null;
	let inProfiles = false;
	for (const raw of text.split("\n")) {
		const line = raw.replace(/\s+#.*$/, "");
		if (/^\s*default_profile:\s*/.test(line)) {
			defaultProfile = line.split(":")[1].trim();
		}
		if (/^\s*profiles:\s*$/.test(line)) {
			inProfiles = true;
			continue;
		}
		if (inProfiles) {
			const profileMatch = line.match(/^\s{4}([A-Za-z0-9_-]+):\s*$/);
			if (profileMatch) {
				current = profileMatch[1];
				profiles[current] = { provider: "", model: "", apiKeyRef: "" };
				continue;
			}
			const fieldMatch = line.match(/^\s{6}(provider|model|api_key_ref):\s*(\S+)\s*$/);
			if (fieldMatch && current) {
				const [, key, value] = fieldMatch;
				if (key === "provider") profiles[current].provider = value;
				if (key === "model") profiles[current].model = value;
				if (key === "api_key_ref") profiles[current].apiKeyRef = value;
			}
		}
	}
	return { defaultProfile, profiles };
}

export function migrateProviders(
	rosclawHome: string,
	env: NodeJS.ProcessEnv = process.env,
): ProviderMigration | null {
	const configPath = join(rosclawHome, "config.yaml");
	if (!existsSync(configPath)) return null;
	const settingsPath = join(rosclawHome, "agent", "settings.json");
	let settings: Record<string, unknown> = {};
	if (existsSync(settingsPath)) {
		try {
			settings = JSON.parse(readFileSync(settingsPath, "utf8")) as Record<string, unknown>;
		} catch {
			settings = {};
		}
	}
	if (settings.defaultProvider && settings.defaultModel) return null; // 已配置
	const parsed = parseRosclawConfig(readFileSync(configPath, "utf8"));
	const profileName = parsed.defaultProfile || Object.keys(parsed.profiles)[0];
	const profile = parsed.profiles[profileName];
	if (!profile) return null;
	const mapping = PROVIDER_MAP[profile.provider];
	if (!mapping) return null;
	// env 键补设（进程内，不落地）。
	let envKeySet = false;
	const refMatch = profile.apiKeyRef.match(/^env:([A-Z0-9_]+)$/);
	if (mapping.env && refMatch) {
		const sourceValue = env[refMatch[1]];
		if (sourceValue && !env[mapping.env]) {
			env[mapping.env] = sourceValue;
			envKeySet = true;
		} else if (env[mapping.env]) {
			envKeySet = true;
		}
	}
	settings.defaultProvider = mapping.provider;
	settings.defaultModel = profile.model;
	mkdirSync(join(rosclawHome, "agent"), { recursive: true, mode: 0o700 });
	writeFileSync(settingsPath, JSON.stringify(settings, null, 1), { mode: 0o600 });
	return {
		provider: mapping.provider,
		model: profile.model,
		envKeySet,
		source: `config.yaml:${profileName}`,
	};
}
