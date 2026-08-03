/** Provider registry (批次 D §7.2/§7.4)。
 *
 * 只注册 ROSClaw 验收过的 provider；按需懒加载子路径——永不 import
 * pi-ai providers/all（§14.13）。每个 provider 声明其 env key 名（引用，
 * 不是值）与内置模型目录。
 */

import { createProvider, type Model, type Provider } from "@earendil-works/pi-ai";
import { envApiKeyAuth } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";

export interface ProviderSpec {
	id: string;
	name: string;
	/** env var 名（引用；值只从进程环境读取）。 */
	envKeys: string[];
	baseUrl: string;
	build: () => Provider;
}

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };

function openAiModel(
	provider: string,
	id: string,
	baseUrl: string,
	contextWindow: number,
	maxTokens: number,
	reasoning = true,
): Model<"openai-completions"> {
	return {
		id,
		name: id,
		api: "openai-completions",
		provider,
		baseUrl,
		reasoning,
		input: ["text"],
		cost: ZERO_COST,
		contextWindow,
		maxTokens,
		compat: {
			// Kimi/OpenAI-compat 接受 system 而非 developer 角色（实测 400）。
			supportsDeveloperRole: false,
			supportsStore: false,
		},
	};
}

function moonshotProvider(): Provider {
	return createProvider({
		id: "moonshot",
		name: "Moonshot AI (Kimi 开放平台)",
		baseUrl: "https://api.moonshot.cn/v1",
		auth: { apiKey: envApiKeyAuth("Moonshot API key", ["MOONSHOT_API_KEY"]) },
		models: [openAiModel("moonshot", "kimi-k3", "https://api.moonshot.cn/v1", 262_144, 16_384)],
		api: openAICompletionsApi(),
	});
}

function kimiCodeProvider(): Provider {
	// Kimi Coding Plan 的 OpenAI 兼容面（agentd 全栈验证过的协议面）。
	return createProvider({
		id: "kimi-code",
		name: "Kimi For Coding",
		baseUrl: "https://api.kimi.com/coding/v1",
		auth: { apiKey: envApiKeyAuth("Kimi Code API key", ["KIMI_API_KEY", "ROSCLAW_KIMI_API_KEY"]) },
		models: [
			openAiModel("kimi-code", "k3", "https://api.kimi.com/coding/v1", 1_048_576, 16_384),
			openAiModel("kimi-code", "k3-256k", "https://api.kimi.com/coding/v1", 262_144, 16_384),
		],
		api: openAICompletionsApi(),
	});
}

function ollamaProvider(): Provider {
	const baseUrl = process.env.ROSCLAW_OLLAMA_BASE_URL ?? "http://127.0.0.1:11434/v1";
	return createProvider({
		id: "ollama",
		name: "Ollama (local)",
		baseUrl,
		auth: { apiKey: envApiKeyAuth("none (local)", []) },
		models: [openAiModel("ollama", process.env.ROSCLAW_OLLAMA_MODEL ?? "qwen3:8b", baseUrl, 32_768, 8_192, false)],
		api: openAICompletionsApi(),
	});
}

const REGISTRY: Record<string, ProviderSpec> = {
	moonshot: {
		id: "moonshot",
		name: "Moonshot AI (Kimi 开放平台)",
		envKeys: ["MOONSHOT_API_KEY"],
		baseUrl: "https://api.moonshot.cn/v1",
		build: moonshotProvider,
	},
	"kimi-code": {
		id: "kimi-code",
		name: "Kimi For Coding",
		envKeys: ["KIMI_API_KEY", "ROSCLAW_KIMI_API_KEY"],
		baseUrl: "https://api.kimi.com/coding/v1",
		build: kimiCodeProvider,
	},
	ollama: {
		id: "ollama",
		name: "Ollama (local)",
		envKeys: [],
		baseUrl: "http://127.0.0.1:11434/v1",
		build: ollamaProvider,
	},
};

const providers = new Map<string, Provider>();

export function providerIds(): string[] {
	return Object.keys(REGISTRY);
}

export function providerSpec(id: string): ProviderSpec | undefined {
	return REGISTRY[id];
}

export function getProvider(id: string): Provider {
	const spec = REGISTRY[id];
	if (!spec) throw new Error(`unknown provider ${id}`);
	let provider = providers.get(id);
	if (!provider) {
		provider = spec.build();
		providers.set(id, provider);
	}
	return provider;
}
