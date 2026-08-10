/** UI locale 管理（六审 §8.2/PR-SIX-5）。
 *
 * - UI locale 会话内固定且持久化（auto/zh-CN/en-US）——不因用户偶尔
 *   说句英文就来回切换；
 * - Agent 回答语言策略独立（follow-user 默认；lock 可锁定）；
 * - 机器合约（JSON key/error code/enum）永不本地化。
 *
 * 持久化：agent/locale.json（0600 目录内；不是 pi settings——自定义
 * 键可能被上游剥离）。launcher 可经 ROSCLAW_UI_LOCALE 显式覆盖。
 */

import { readFileSync, writeFileSync, renameSync } from "node:fs";

export type UiLocale = "auto" | "zh-CN" | "en-US";
export type ReplyLanguage = "follow-user" | "zh-CN" | "en-US";

export interface LocaleConfig {
	ui_locale: UiLocale;
	reply_language: ReplyLanguage;
}

const DEFAULT: LocaleConfig = { ui_locale: "auto", reply_language: "follow-user" };

export class LocaleManager {
	private config: LocaleConfig;
	private readonly path: string;
	private readonly listeners = new Set<() => void>();

	constructor(agentDir: string) {
		this.path = `${agentDir}/locale.json`;
		this.config = { ...DEFAULT };
		// launcher 显式传入优先（服务器 LANG 经常不准）。
		const fromEnv = process.env.ROSCLAW_UI_LOCALE;
		try {
			const raw = JSON.parse(readFileSync(this.path, "utf-8")) as Partial<LocaleConfig>;
			if (raw.ui_locale) this.config.ui_locale = raw.ui_locale;
			if (raw.reply_language) this.config.reply_language = raw.reply_language;
		} catch {
			// 无文件/损坏 → 默认值（诚实回退，不猜）。
		}
		if (fromEnv === "zh-CN" || fromEnv === "en-US") {
			this.config.ui_locale = fromEnv;
		}
	}

	get current(): LocaleConfig {
		return { ...this.config };
	}

	/** 实际生效的 UI locale（auto 解析为具体值——LC_MESSAGES/LANG
	 *  只在 auto 时参考，且只在明确含 zh 时选中文）。 */
	get effective(): "zh-CN" | "en-US" {
		if (this.config.ui_locale !== "auto") return this.config.ui_locale;
		const lang = `${process.env.LC_MESSAGES ?? ""} ${process.env.LANG ?? ""}`;
		return lang.includes("zh") ? "zh-CN" : "en-US";
	}

	setUiLocale(value: UiLocale): void {
		this.config.ui_locale = value;
		this.persist();
		for (const listener of this.listeners) listener();
	}

	setReplyLanguage(value: ReplyLanguage): void {
		this.config.reply_language = value;
		this.persist();
	}

	subscribe(listener: () => void): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	private persist(): void {
		const tmp = `${this.path}.tmp`;
		writeFileSync(tmp, JSON.stringify(this.config, null, 1) + "\n", {
			encoding: "utf-8",
			mode: 0o600,
		});
		renameSync(tmp, this.path);
	}
}
