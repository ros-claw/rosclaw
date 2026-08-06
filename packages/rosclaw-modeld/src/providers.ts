/** Provider registry（审计 P0-04）：上游 pi-ai provider 直接复用。
 *
 * - 上游能枚举的 provider/model 不手抄：moonshot/kimi-code/openai/
 *   anthropic/openrouter 全部用 pi-ai 子路径 provider（懒加载，
 *   永不 import providers/all，§14.13）。
 * - ROSClaw 只保留策略覆盖（ollama 本地 profile）与 registry/状态面。
 * - 每个 provider 声明 env key 引用与能力矩阵来源（provider.getModels()）。
 */

import type { Model, Provider } from "@earendil-works/pi-ai";
import { createProvider, envApiKeyAuth } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";

export interface ProviderSpec {
	id: string;
	name: string;
	envKeys: string[];
	/** 懒加载子路径（不用 providers/all）。 */
	load: () => Promise<Provider>;
}

async function loadMoonshot(): Promise<Provider> {
	const { moonshotaiCnProvider } = await import(
		"@earendil-works/pi-ai/providers/moonshotai-cn"
	);
	return moonshotaiCnProvider();
}

async function loadKimiCode(): Promise<Provider> {
	const { kimiCodingProvider } = await import("@earendil-works/pi-ai/providers/kimi-coding");
	return kimiCodingProvider();
}

async function loadOpenAI(): Promise<Provider> {
	const { openaiProvider } = await import("@earendil-works/pi-ai/providers/openai");
	return openaiProvider();
}

async function loadAnthropic(): Promise<Provider> {
	const { anthropicProvider } = await import("@earendil-works/pi-ai/providers/anthropic");
	return anthropicProvider();
}

async function loadOpenRouter(): Promise<Provider> {
	const { openrouterProvider } = await import("@earendil-works/pi-ai/providers/openrouter");
	return openrouterProvider();
}

/** ROSClaw 特有的本地 profile（策略覆盖，非上游 catalog）。 */
async function loadOllama(): Promise<Provider> {
	const baseUrl = process.env.ROSCLAW_OLLAMA_BASE_URL ?? "http://127.0.0.1:11434/v1";
	const model: Model<"openai-completions"> = {
		id: process.env.ROSCLAW_OLLAMA_MODEL ?? "qwen3:8b",
		name: process.env.ROSCLAW_OLLAMA_MODEL ?? "qwen3:8b",
		api: "openai-completions",
		provider: "ollama",
		baseUrl,
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 32_768,
		maxTokens: 8_192,
	};
	return createProvider({
		id: "ollama",
		name: "Ollama (local)",
		baseUrl,
		auth: { apiKey: envApiKeyAuth("none (local)", []) },
		models: [model],
		api: openAICompletionsApi(),
	});
}

const REGISTRY: Record<string, ProviderSpec> = {
	moonshot: {
		id: "moonshot",
		name: "Moonshot AI (Kimi 开放平台)",
		envKeys: ["MOONSHOT_API_KEY"],
		load: loadMoonshot,
	},
	"kimi-code": {
		id: "kimi-code",
		name: "Kimi For Coding",
		envKeys: ["KIMI_API_KEY", "ROSCLAW_KIMI_API_KEY"],
		load: loadKimiCode,
	},
	openai: {
		id: "openai",
		name: "OpenAI",
		envKeys: ["OPENAI_API_KEY"],
		load: loadOpenAI,
	},
	anthropic: {
		id: "anthropic",
		name: "Anthropic",
		envKeys: ["ANTHROPIC_API_KEY"],
		load: loadAnthropic,
	},
	openrouter: {
		id: "openrouter",
		name: "OpenRouter",
		envKeys: ["OPENROUTER_API_KEY"],
		load: loadOpenRouter,
	},
	ollama: {
		id: "ollama",
		name: "Ollama (local)",
		envKeys: [],
		load: loadOllama,
	},
};

const providers = new Map<string, Provider>();

export function providerIds(): string[] {
	return Object.keys(REGISTRY);
}

export function providerSpec(id: string): ProviderSpec | undefined {
	return REGISTRY[id];
}

export async function getProvider(id: string): Promise<Provider> {
	const spec = REGISTRY[id];
	if (!spec) throw new Error(`unknown provider ${id}`);
	let provider = providers.get(id);
	if (!provider) {
		provider = await spec.load();
		providers.set(id, provider);
	}
	return provider;
}
