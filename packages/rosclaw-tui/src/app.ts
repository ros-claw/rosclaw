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
	Text,
	TUI,
	matchesKey,
} from "@earendil-works/pi-tui";
import { AgentClient, idempotencyKey } from "./client/http.js";
import { streamEvents } from "./client/sse.js";
import { CommandRegistry } from "./commands/registry.js";
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
	private deltaBuffer = "";
	private abort = new AbortController();
	private running = false;

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
				} else {
					this.print(new Text(chalk.dim("再按一次 Ctrl+C 退出；输入为空时 Ctrl+D 也可退出")));
					this.confirmExit = true;
					setTimeout(() => (this.confirmExit = false), 2000);
					if (this.confirmExit && this.lastCtrlC) this.stop();
					this.lastCtrlC = true;
					setTimeout(() => (this.lastCtrlC = false), 2000);
				}
				return { consume: true };
			}
			if (matchesKey(data, "ctrl+d")) {
				this.stop();
				return { consume: true };
			}
			return undefined;
		});

		this.running = true;
		this.tui.start();
		void this.consumeEvents();
	}

	private confirmExit = false;
	private lastCtrlC = false;

	stop(): void {
		this.running = false;
		this.abort.abort();
		this.tui.stop();
	}

	private statusString(): string {
		const line = statusLine(this.state);
		const phase = this.state.phase ? ` [${this.state.phase}…]` : "";
		return modeColor(this.state.mode, line) + chalk.dim(phase);
	}

	private refreshStatus(): void {
		this.statusText.setText?.(this.statusString());
	}

	/** 在 status bar 与 editor 之上插入一行（保持布局顺序）。 */
	private print(component: Text | Markdown): void {
		this.tui.removeChild(this.statusText);
		this.tui.removeChild(this.editor);
		this.tui.addChild(component);
		this.tui.addChild(this.statusText);
		this.tui.addChild(this.editor);
		this.tui.setFocus(this.editor);
	}

	private applyEffects(effects: Effect[]): void {
		for (const effect of effects) {
			switch (effect.kind) {
				case "append_markdown":
					this.print(new Markdown(effect.text, 0, 0, markdownTheme));
					break;
				case "append_delta":
					this.deltaBuffer += effect.text;
					break;
				case "flush_delta":
					if (this.deltaBuffer.trim()) {
						this.print(new Markdown(this.deltaBuffer, 0, 0, markdownTheme));
					}
					this.deltaBuffer = "";
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

	private async handleInput(raw: string): Promise<void> {
		const text = raw.trim();
		if (!text) return;
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
			case "remote": {
				if (route.spec.disabled_reason) {
					this.print(new Text(chalk.yellow(`/${route.spec.name} 不可用：${route.spec.disabled_reason}`)));
					return;
				}
				const args = this.parseRemoteArgs(route.spec.name, route.args);
				const result = await this.client.command(
					this.options.missionId,
					route.spec.name,
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
				return;
			}
		}
	}

	private parseRemoteArgs(name: string, args: string): Record<string, unknown> {
		if (name === "rename") return { name: args };
		if (name === "compact") {
			if (args === "dry-run") return { dry_run: true };
			if (args.startsWith("focus ")) return { focus: args.slice(6) };
			return {};
		}
		return {};
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
				const item = this.state.pendingApprovals.find(
					(a) => a.requestId === args || a.requestId.startsWith(args),
				);
				if (!item) {
					this.print(new Text(chalk.yellow(`没有匹配的待批准请求：${args || "(空)"}`)));
					return;
				}
				await this.client.decideApproval(item.requestId, name === "approve");
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
		}
	}
}
