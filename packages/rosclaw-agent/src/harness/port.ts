/** NativeHarnessBackend 最小 SPI（PR-H1，ADR-0012）。
 *
 * 边界规则（总纲 v2 §7.2）：
 * - 接口只包含 ROSClaw 确实使用的能力——不抽象每个 Pi 内部类型；
 * - capability 缺失必须在 session 创建时明确报错（不运行中静默切换）；
 * - Backend 实现不能访问 Task 数据库私有写接口；
 * - Backend 只能通过注册的 ToolGateway 调用 ROSClaw 能力；
 * - 这不是多引擎生态——用户永远只跑 `rosclaw chat`。
 */

/** Harness 能力声明——缺能力的 Backend 在创建期 fail-fast。 */
export interface HarnessCapabilities {
	persistentSessions: boolean;
	resume: boolean;
	steering: boolean;
	followUp: boolean;
	compaction: boolean;
	modelSwitching: boolean;
	customTools: boolean;
	toolStreaming: boolean;
	toolPolicyHook: boolean;
	interactiveUi: boolean;
}

/** 统一错误码（§8.4 模型/Provider 分类 + Harness 错误）。 */
export type HarnessErrorCode =
	| "MODEL_CREDENTIAL_MISSING"
	| "MODEL_CREDENTIAL_INVALID"
	| "PROVIDER_QUOTA_EXHAUSTED"
	| "PROVIDER_RATE_LIMITED"
	| "PROVIDER_UNAVAILABLE"
	| "MODEL_NOT_FOUND"
	| "MODEL_TOOL_CALL_UNSUPPORTED"
	| "MODEL_CONTEXT_LIMIT"
	| "MODEL_REQUEST_CANCELLED"
	| "HARNESS_CAPABILITY_MISSING"
	| "HARNESS_SESSION_LOST";

export interface HarnessError {
	code: HarnessErrorCode;
	message: string;
	retryable: boolean;
}

/** 统一事件流（§7.4）——TUI/TaskCoordinator/事件存储不得 switch
 *  Backend 私有事件类型。 */
export type HarnessEvent =
	| { type: "session.ready"; sessionRef: string }
	| { type: "turn.started"; turnId: string }
	| { type: "assistant.delta"; turnId: string; text: string }
	| { type: "assistant.completed"; turnId: string; messageId: string }
	| { type: "tool.started"; callId: string; tool: string; args: unknown }
	| { type: "tool.updated"; callId: string; update: unknown }
	| { type: "tool.completed"; callId: string; result: unknown }
	| { type: "tool.failed"; callId: string; error: HarnessError }
	| { type: "compaction.started" }
	| { type: "compaction.completed"; snapshotRef?: string }
	| { type: "turn.cancelled"; turnId: string }
	| { type: "turn.failed"; turnId: string; error: HarnessError }
	| { type: "session.idle" };

export interface HarnessSessionRef {
	backendId: string;
	/** Backend 原生 session 标识（只在 binding 里，不进产品 UI）。 */
	nativeRef: string;
}

export interface HarnessCreateOptions {
	cwd: string;
	/** 产品侧注入的 Harness 原生选项（Backend 私有，不透出）。 */
	backendOptions?: Record<string, unknown>;
}

export interface HarnessInput {
	text: string;
	/** steer/followUp 的语义角色（同一 session 内注入）。 */
	role?: "user" | "steer" | "followUp";
}

export interface HarnessSession {
	readonly sessionRef: HarnessSessionRef;
	readonly cwd: string;

	prompt(input: HarnessInput): Promise<void>;
	steer(input: HarnessInput): Promise<void>;
	followUp(input: HarnessInput): Promise<void>;

	events(): AsyncIterable<HarnessEvent>;

	compact(instruction?: string): Promise<{ ok: boolean; detail?: string }>;
	setModel(model: { provider: string; model: string }): Promise<void>;
	setThinking(level: string): Promise<void>;

	cancelTurn(): Promise<void>;
	waitUntilIdle(): Promise<void>;
	close(): Promise<void>;
}

export interface NativeHarnessBackend {
	readonly backendId: string;
	readonly capabilities: HarnessCapabilities;

	create(options: HarnessCreateOptions): Promise<HarnessSession>;
	resume(ref: HarnessSessionRef): Promise<HarnessSession>;
}

/** capability 核对——缺项 fail-fast（创建期，不是运行中）。 */
export function assertHarnessCapabilities(
	backend: NativeHarnessBackend,
	required: (keyof HarnessCapabilities)[],
): void {
	const missing = required.filter((key) => !backend.capabilities[key]);
	if (missing.length) {
		throw new Error(
			`HARNESS_CAPABILITY_MISSING: ${backend.backendId} 缺 ${missing.join(", ")}`,
		);
	}
}
