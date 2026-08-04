/** rosclaw-tui app (批次 C)：薄 Presenter + reducer + 命令路由。
 *
 * 禁止复制 Pi interactive-mode.ts —— 这里只做：
 * pi-tui 组件装配、SSE 消费、effect 渲染、命令路由。
 * TUI 永远不执行工具/Worker/物理动作，不与模型直接通信。
 */

import {
	Editor,
	Markdown,
	ProcessTerminal,
	SelectList,
	Text,
	TUI,
	matchesKey,
} from "@earendil-works/pi-tui";
import { parseArgs as parseSchemaArgs } from "./commands/args-parser.js";
import { MaskedInput } from "./components/masked-input.js";
import { AgentClient, idempotencyKey } from "./client/http.js";
import { defaultOperatorSocket, operatorCall } from "./client/operator.js";
import { streamEvents } from "./client/sse.js";
import { CommandRegistry } from "./commands/registry.js";
import { LiveAssistantMessage } from "./components/live-message.js";
import { HOTKEYS_TEXT, renderCard, statusLine } from "./components/render.js";
import { chalk, editorTheme, markdownTheme, modeColor } from "./components/theme.js";
import { reduce, shortId, type Effect } from "./state/reducer.js";
import { initialState, type UiState } from "./state/types.js";

export interface AppOptions {
	baseUrl: string;
	missionId: string;
}

export class RosclawTuiApp {
	private tui!: TUI;
	private editor!: Editor;
	private statusText!: Text;
	private state: UiState;
	private client: AgentClient;
	private registry = new CommandRegistry();
	private live: LiveAssistantMessage | null = null;
	private abort = new AbortController();
	private running = false;
	private lastDeltaAt = 0;
	private phaseTicker: ReturnType<typeof setInterval> | null = null;

	constructor(private readonly options: AppOptions) {
		this.client = new AgentClient(options.baseUrl);
		this.state = initialState(options.missionId);
	}

	async start(): Promise<void> {
		// 初始对齐：snapshot 是权威，SSE 是增量。
		try {
			const snap = await this.client.snapshot(this.options.missionId);
			this.state.missionName = snap.name;
			this.state.missionState = snap.state;
			this.state.mode = snap.mode;
			this.state.bodyId = snap.body_id;
			this.state.lastSeq = snap.last_event_sequence;
			this.state.turnInFlight = snap.turn_in_flight;
			this.state.compactions = snap.compaction_count;
			this.state.pendingApprovals = snap.pending_approvals.map((a) => ({
				requestId: String(a.request_id ?? ""),
				title: String(a.title ?? ""),
				riskTier: String(a.risk_tier ?? "LOW"),
				expiresAt: a.expires_at ? String(a.expires_at) : undefined,
			}));
		} catch (err) {
			throw new Error(
				`无法连接 AgentService（${this.options.baseUrl}）：${(err as Error).message}\n` +
					"请先启动 rosclaw-agentd（rosclaw chat 会自动启动），或用 --url 指定地址。",
			);
		}
		try {
			const caps = await this.client.capabilities(this.options.missionId);
			this.registry.loadRemote(caps.commands);
		} catch {
			// 老服务端没有 /v1/capabilities → 只用本地命令，诚实降级。
		}

		const terminal = new ProcessTerminal();
		this.tui = new TUI(terminal);
		this.statusText = new Text(this.statusString());
		this.editor = new Editor(this.tui, editorTheme);
		this.editor.onSubmit = (text) => {
			void this.handleInput(text);
		};

		// resume：先把 journal 中的可见历史恢复成 transcript（审计 P0-02.7），
		// 再进入 live 增量（sequence 已由 snapshot 对齐，不会重复）。
		await this.replayTranscript();

		this.tui.addChild(
			new Markdown(
				`**ROSClaw Native Agent** — ${this.state.missionName}\n` +
					`mode: ${this.state.mode} | body: ${this.state.bodyId}\n` +
					"输入消息开始；/help 查看命令。TUI 只是客户端，权威状态在 AgentService。",
				0,
				0,
				markdownTheme,
			),
		);
		this.tui.addChild(this.statusText);
		this.tui.addChild(this.editor);
		this.tui.setFocus(this.editor);

		this.tui.addInputListener((data) => {
			if (matchesKey(data, "ctrl+c")) {
				if (this.state.turnInFlight) {
					void this.client.cancelTurn(this.options.missionId);
					this.print(new Text(chalk.dim("已请求取消当前 turn（已派发动作不受影响）")));
				} else if (this.lastCtrlC) {
					// 双击 Ctrl+C 退出（turn 空闲时）。
					this.stop();
				} else {
					this.print(new Text(chalk.dim("再按一次 Ctrl+C 退出；输入为空时 Ctrl+D 也可退出")));
					this.lastCtrlC = true;
					const timer = setTimeout(() => (this.lastCtrlC = false), 2000);
					timer.unref?.();
				}
				return { consume: true };
			}
			if (matchesKey(data, "ctrl+d")) {
				// §6.2：仅输入为空时 Ctrl+D 退出（防误触）。
				if (this.editor.getText().trim() === "") {
					this.stop();
				} else {
					this.print(new Text(chalk.dim("Ctrl+D 仅在输入为空时退出（请先清空输入）")));
				}
				return { consume: true };
			}
			return undefined;
		});

		this.running = true;
		this.tui.start();
		// 500ms 内无 delta 时状态行仍持续显示阶段+计时（审计 P0-02.4：
		// 不能保持静默）。
		this.phaseTicker = setInterval(() => {
			if (this.state.turnInFlight) this.refreshStatus();
		}, 500);
		this.phaseTicker.unref?.();
		void this.consumeEvents();
	}

	private lastCtrlC = false;

	stop(): void {
		this.running = false;
		if (this.phaseTicker !== null) clearInterval(this.phaseTicker);
		this.live?.flush(true);
		this.abort.abort();
		this.tui.stop();
	}

	private statusString(): string {
		const line = statusLine(this.state);
		let phase = "";
		if (this.state.turnInFlight) {
			const label = this.state.phase || "排队";
			const idleMs = this.lastDeltaAt > 0 ? Date.now() - this.lastDeltaAt : 0;
			phase =
				idleMs > 500
					? ` [${label}… ${(idleMs / 1000).toFixed(0)}s]`
					: ` [${label}…]`;
		}
		return modeColor(this.state.mode, line) + chalk.dim(phase);
	}

	private refreshStatus(): void {
		this.statusText.setText?.(this.statusString());
	}

	private transcriptCount = 0;
	private static readonly MAX_TRANSCRIPT_BLOCKS = 400;

	/** 在 status bar 与 editor 之上插入一行（保持布局顺序）；超出窗口
	 * 移除最旧 transcript 块（渲染窗口与 journal 持久化分离——历史
	 * 永远在 journal，/resume 可完整恢复）。 */
	private print(component: Text | Markdown): void {
		this.tui.removeChild(this.statusText);
		this.tui.removeChild(this.editor);
		this.tui.addChild(component);
		this.tui.addChild(this.statusText);
		this.tui.addChild(this.editor);
		this.tui.setFocus(this.editor);
		this.transcriptCount += 1;
		if (this.transcriptCount > RosclawTuiApp.MAX_TRANSCRIPT_BLOCKS) {
			const children = this.tui.children;
			if (children.length > 3) {
				this.tui.removeChild(children[1]);
				this.transcriptCount -= 1;
			}
		}
	}

	private applyEffects(effects: Effect[]): void {
		for (const effect of effects) {
			switch (effect.kind) {
				case "append_markdown":
					this.print(new Markdown(effect.text, 0, 0, markdownTheme));
					break;
				case "append_delta": {
					if (this.live === null) {
						this.live = new LiveAssistantMessage(markdownTheme, () => this.refreshStatus());
						this.print(this.live.component);
					}
					this.lastDeltaAt = Date.now();
					this.live.append(effect.text);
					break;
				}
				case "flush_delta":
					if (this.live !== null) {
						this.live.flush(true);
						this.live = null;
					}
					break;
				case "append_card":
					this.print(new Text(renderCard(effect.card)));
					break;
				case "spinner":
				case "spinner_stop":
				case "status_refresh":
					this.refreshStatus();
					break;
			}
		}
	}

	/** resume 时把历史事件重放为 transcript（delta 按 turn 聚合，卡片直出）。 */
	private async replayTranscript(): Promise<void> {
		const events = await this.client.replayEvents(this.options.missionId);
		let live: LiveAssistantMessage | null = null;
		const flushLive = () => {
			if (live !== null) {
				live.flush(true);
				live = null;
			}
		};
		for (const raw of events) {
			const event = raw as { type?: string; visibility?: string; payload?: Record<string, unknown>; sequence?: number };
			if (event.visibility === "DEBUG") continue;
			const type = String(event.type ?? "");
			const payload = event.payload ?? {};
			if (type === "model.text.delta") {
				if (live === null) {
					live = new LiveAssistantMessage(markdownTheme, () => undefined);
					this.print(live.component);
				}
				live.append(String(payload.text ?? ""));
				continue;
			}
			flushLive();
			const effects = reduce(this.state, {
				event_id: `replay_${String(event.sequence ?? "")}`,
				sequence: Number(event.sequence ?? 0),
				mission_id: this.options.missionId,
				type,
				visibility: String(event.visibility ?? "USER"),
				payload,
				timestamp: "",
			});
			for (const effect of effects) {
				if (effect.kind === "append_card") this.print(new Text(renderCard(effect.card)));
			}
		}
		flushLive();
	}

	private async consumeEvents(): Promise<void> {
		const { baseUrl, missionId } = this.options;
		const generator = streamEvents(baseUrl, missionId, {
			signal: this.abort.signal,
			onGap: () => {
				// sequence 缺口：停止乐观渲染，拉权威快照对齐（§5.4）。
				void this.resync();
			},
			onReconnect: () => {
				this.state.reconnecting = true;
				this.refreshStatus();
			},
		});
		for await (const event of generator) {
			this.state.reconnecting = false;
			const effects = reduce(this.state, event);
			this.applyEffects(effects);
		}
	}

	private async resync(): Promise<void> {
		try {
			const snap = await this.client.snapshot(this.options.missionId);
			this.state.missionState = snap.state;
			this.state.mode = snap.mode;
			this.state.turnInFlight = snap.turn_in_flight;
			this.print(new Text(chalk.yellow("事件流出现缺口，已用权威快照重新对齐。")));
			this.refreshStatus();
		} catch {
			// 快照也拉不到 → 等 SSE 重连
		}
	}

	/** /resume：切换 mission——停旧流、对齐 snapshot、恢复 transcript、重开事件流。 */
	private async switchMission(missionId: string): Promise<void> {
		this.abort.abort();
		this.abort = new AbortController();
		this.options.missionId = missionId;
		this.state = initialState(missionId);
		try {
			const snap = await this.client.snapshot(missionId);
			this.state.missionName = snap.name;
			this.state.missionState = snap.state;
			this.state.mode = snap.mode;
			this.state.bodyId = snap.body_id;
			this.state.lastSeq = snap.last_event_sequence;
			this.state.turnInFlight = snap.turn_in_flight;
			const caps = await this.client.capabilities(missionId);
			this.registry.loadRemote(caps.commands);
		} catch (err) {
			this.print(new Text(chalk.red(`无法切换到 ${missionId}：${(err as Error).message}`)));
			return;
		}
		this.print(new Text(chalk.dim(`—— 已切换到 ${this.state.missionName} ——`)));
		await this.replayTranscript();
		this.refreshStatus();
		void this.consumeEvents();
	}

	private async handleInput(raw: string): Promise<void> {
		const text = raw.trim();
		if (!text) return;
		// //text 转义：以 / 开头的普通文本显式发送（审计 P0-03.6）。
		if (text.startsWith("//")) {
			const literal = text.slice(1);
			this.print(new Text(chalk.cyan("> " + literal)));
			await this.client.submitTurn(this.options.missionId, literal);
			return;
		}
		const route = this.registry.parse(text);
		switch (route.kind) {
			case "not_a_command":
				this.print(new Text(chalk.cyan("> " + text)));
				try {
					await this.client.submitTurn(this.options.missionId, text);
				} catch (err) {
					this.print(new Text(chalk.red(`发送失败：${(err as Error).message}`)));
				}
				return;
			case "unknown":
				this.print(
					new Text(
						chalk.yellow(
							`未知命令 /${route.name}。直接作为文本发送请去掉开头的 /，或 /help 查看命令。`,
						),
					),
				);
				return;
			case "local":
				await this.handleLocal(route.name, route.args);
				return;
			case "remote":
				await this.handleRemote(route.spec, route.args);
				return;
		}
	}

	/** 远程命令：共享 args_schema 解析 + interaction（select/secret/confirm/path）。 */
	private async handleRemote(
		spec: import("./client/http.js").CommandSpec,
		rawArgs: string,
	): Promise<void> {
		if (spec.disabled_reason) {
			this.print(new Text(chalk.yellow(`/${spec.name} 不可用：${spec.disabled_reason}`)));
			return;
		}
		const parsed = parseSchemaArgs(spec.args_schema, rawArgs);
		if (!parsed.ok) {
			this.print(
				new Text(chalk.yellow(`/${spec.name} ${parsed.error}（${spec.argument_hint || "见 /help"}）`)),
			);
			return;
		}
		const interaction = spec.args_schema?.interaction ?? "none";
		if (interaction === "secret" && parsed.args.api_key === undefined) {
			await this.runSecretInteraction(spec, parsed.args);
			return;
		}
		if (interaction === "select" && !rawArgs.trim()) {
			await this.runSelectInteraction(spec, parsed.args);
			return;
		}
		await this.executeRemote(spec.name, parsed.args);
	}

	private async executeRemote(name: string, args: Record<string, unknown>): Promise<void> {
		const result = await this.client.command(
			this.options.missionId,
			name,
			args,
			idempotencyKey(),
		);
		this.print(
			new Text(
				result.ok
					? chalk.green(result.message || "ok")
					: chalk.red(`${result.error_code}: ${result.message}`),
			),
		);
		if (name === "model" && result.ok) this.refreshStatus();
	}

	/** secret 交互：masked 输入，值只进命令参数，不进 transcript/history。 */
	private async runSecretInteraction(
		spec: import("./client/http.js").CommandSpec,
		baseArgs: Record<string, unknown>,
	): Promise<void> {
		const provider = String(baseArgs.provider ?? "");
		if (!provider) {
			this.print(new Text(chalk.yellow(`/${spec.name} 需要 provider（如 /login kimi-code）`)));
			return;
		}
		const masked = new MaskedInput(this.tui, `${provider} API key:`);
		const handle = this.tui.showOverlay(masked, { width: "60%" });
		this.tui.setFocus(masked);
		masked.onSubmit = (secret) => {
			handle.hide();
			this.tui.setFocus(this.editor);
			if (secret) {
				void this.executeRemote(spec.name, { ...baseArgs, api_key: secret });
			}
		};
		masked.onCancel = () => {
			handle.hide();
			this.tui.setFocus(this.editor);
			this.print(new Text(chalk.dim("已取消登录（未发送任何凭据）")));
		};
	}

	/** select 交互：从命令数据或 modeld 列表构建 SelectList。 */
	private async runSelectInteraction(
		spec: import("./client/http.js").CommandSpec,
		baseArgs: Record<string, unknown>,
	): Promise<void> {
		let items: Array<{ value: string; label: string; description?: string }> = [];
		if (spec.args_schema?.interaction_source === "models") {
			const listing = await this.client.command(
				this.options.missionId,
				"model",
				{},
				idempotencyKey(),
			);
			const models = (listing.data?.models ?? {}) as Record<string, string[]>;
			for (const [provider, ids] of Object.entries(models)) {
				for (const id of ids) {
					items.push({ value: `${provider}/${id}`, label: `${provider}/${id}` });
				}
			}
		} else if (spec.args_schema?.interaction_source === "workers") {
			const listing = await this.client.command(
				this.options.missionId,
				"workers",
				{},
				idempotencyKey(),
			);
			const workers = (listing.data?.workers ?? []) as Array<{ worker_id: string; status: string }>;
			items = workers.map((w) => ({
				value: w.worker_id,
				label: w.worker_id,
				description: w.status,
			}));
		}
		if (items.length === 0) {
			this.print(new Text(chalk.yellow("没有可选项（modeld/worker 列表为空）")));
			return;
		}
		const list = new SelectList(items, Math.min(items.length, 10), {
			selectedPrefix: (t) => chalk.blue(t),
			selectedText: (t) => chalk.bold(t),
			description: (t) => chalk.dim(t),
			scrollInfo: (t) => chalk.dim(t),
			noMatch: (t) => chalk.dim(t),
		});
		const handle = this.tui.showOverlay(list, { width: "70%" });
		this.tui.setFocus(list);
		list.onSelect = (item) => {
			handle.hide();
			this.tui.setFocus(this.editor);
			if (spec.args_schema?.interaction_source === "models") {
				void this.executeRemote(spec.name, { target: item.value });
			} else if (spec.args_schema?.interaction_source === "workers") {
				const sub = String((this.registry.parse(`/worker ${item.value}`).kind === "remote" && "inspect") || "inspect");
				void this.executeRemote(spec.name, { subcommand: sub, worker_id: item.value });
			}
		};
		list.onCancel = () => {
			handle.hide();
			this.tui.setFocus(this.editor);
		};
	}

	private async handleLocal(name: string, args: string): Promise<void> {
		switch (name) {
			case "help": {
				const rows = this.registry
					.all()
					.filter((c) => !args || c.name === args)
					.map((c) => {
						const disabled = c.disabled ? chalk.yellow(`（不可用：${c.disabled}）`) : "";
						return `/${c.name} ${c.hint}\n  ${c.description} [${c.owner}] ${disabled}`;
					});
				this.print(new Text(rows.join("\n") || "无匹配命令"));
				return;
			}
			case "hotkeys":
				this.print(new Text(HOTKEYS_TEXT));
				return;
			case "clear-screen":
				process.stdout.write("\x1b[2J\x1b[H");
				return;
			case "quit": {
				const active = this.state.workers.filter(
					(w) => !["accepted", "failed", "expired"].includes(w.status),
				);
				if (active.length > 0) {
					this.print(
						new Text(
							chalk.yellow(
								`注意：${active.length} 个 WorkOrder 未终态。退出 TUI 不会停止已派发工作；` +
									"权威状态保留在 AgentService。",
							),
						),
					);
				}
				this.stop();
				return;
			}
			case "approve":
			case "deny": {
				// 经 operator.sock：peer identity（SO_PEERCRED）+ display hash——
				// 不走 HTTP principal 字段（那可以被伪造/与环境 uid 不符）。
				const listed = (await operatorCall(defaultOperatorSocket(), "approvals.list")) as {
					ok: boolean;
					approvals?: Array<{ request_id: string; display_hash: string }>;
					error?: string;
				};
				const entry = (listed.approvals ?? []).find(
					(a) => a.request_id === args || a.request_id.startsWith(args),
				);
				if (!entry) {
					this.print(new Text(chalk.yellow(`没有匹配的待批准请求：${args || "(空)"}`)));
					return;
				}
				const decided = (await operatorCall(defaultOperatorSocket(), "approvals.decide", {
					request_id: entry.request_id,
					display_hash: entry.display_hash,
					approve: name === "approve",
				})) as { ok: boolean; error?: string };
				if (!decided.ok) {
					this.print(new Text(chalk.red(`决定被拒：${decided.error ?? "unknown"}`)));
				}
				return;
			}
			case "estop": {
				try {
					const result = await operatorCall(defaultOperatorSocket(), "estop", {
						reason: "operator /estop from rosclaw-tui",
					});
					this.print(
						new Text(
							result.ok
								? chalk.red.bold("E-STOP 已请求 rosclawd 执行。")
								: chalk.yellow(`E-STOP 未执行：${String(result.error ?? "unknown")}`),
						),
					);
				} catch (err) {
					this.print(
						new Text(
							chalk.yellow(
								`E-STOP 通道不可用：${(err as Error).message}（未假装已停止）`,
							),
						),
					);
				}
				return;
			}
			case "missions": {
				const missions = await this.client.listMissions();
				const lines = missions.map(
					(m) => `${m.mission_id}  ${String(m.state ?? "")}  ${String((m.goal as { text?: string })?.text ?? "").slice(0, 40)}`,
				);
				this.print(new Text(lines.join("\n") || "(无 Mission)"));
				return;
			}
			case "resume": {
				if (args) {
					await this.switchMission(args);
					return;
				}
				const missions = await this.client.listMissions();
				const items = missions.map((m) => ({
					value: m.mission_id,
					label: `${m.mission_id}  ${String((m.goal as { text?: string })?.text ?? "").slice(0, 30)}`,
					description: String(m.state ?? ""),
				}));
				if (items.length === 0) {
					this.print(new Text(chalk.yellow("没有可恢复的 Mission")));
					return;
				}
				const list = new SelectList(items, Math.min(items.length, 10), {
					selectedPrefix: (t) => chalk.blue(t),
					selectedText: (t) => chalk.bold(t),
					description: (t) => chalk.dim(t),
					scrollInfo: (t) => chalk.dim(t),
					noMatch: (t) => chalk.dim(t),
				});
				const handle = this.tui.showOverlay(list, { width: "80%" });
				this.tui.setFocus(list);
				list.onSelect = (item) => {
					handle.hide();
					this.tui.setFocus(this.editor);
					void this.switchMission(item.value);
				};
				list.onCancel = () => {
					handle.hide();
					this.tui.setFocus(this.editor);
				};
				return;
			}
		}
	}
}
