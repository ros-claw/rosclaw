/** PiHarnessBackend（PR-HP2，ADR-0012/0013）——当前唯一默认 Native Harness。
 *
 * 边界：
 * - 直接 import Pi SDK（不启动 pi CLI 子进程）；
 * - Pi 原生 session id 只在 HarnessSessionRef.nativeRef（不进产品 UI）；
 * - 事件经 adapter 统一为 HarnessEvent（产品侧不 switch Pi 私有类型）；
 * - create()/resume() 真实工作——装配本体在同目录 pi-runtime.ts
 *   （自 runtime/create-runtime.ts 迁入）；
 * - 能力声明来自运行时 probe（probePiCapabilities）——禁止硬编码
 *   全 true；probe 缺失的能力在创建期 fail-fast。
 */

import { AgentSession, InteractiveMode, SessionManager } from "@earendil-works/pi-coding-agent";

import type {
	HarnessCapabilities,
	HarnessCreateOptions,
	HarnessSession,
	HarnessSessionRef,
	NativeHarnessBackend,
} from "../port.js";
import { PiHarnessSession } from "./pi-session-adapter.js";

export const PI_BACKEND_ID = "pi";

/** 运行时 probe——从 Pi SDK 真实能力面推导，不硬编码。版本升级砍掉
 *  API 时对应能力自动变 false（创建期 fail-fast），不静默降级。 */
export function probePiCapabilities(): HarnessCapabilities {
	const proto = AgentSession.prototype as unknown as Record<string, unknown>;
	const has = (obj: Record<string, unknown>, name: string) =>
		typeof obj[name] === "function";
	const sm = SessionManager as unknown as Record<string, unknown>;
	return {
		persistentSessions: typeof sm.open === "function",
		resume: typeof sm.open === "function"
			&& typeof sm.continueRecent === "function",
		steering: has(proto, "steer"),
		followUp: has(proto, "followUp"),
		compaction: has(proto, "compact"),
		modelSwitching: has(proto, "setModel"),
		customTools: has(proto, "setActiveToolsByName"),
		toolStreaming: has(proto, "subscribe"),
		toolPolicyHook: has(proto, "subscribe"),  // 扩展钩子经事件订阅面
		interactiveUi: typeof InteractiveMode === "function",
	};
}

/** backendOptions 里产品侧传入的完整装配选项（Backend 私有，不透出）。 */
interface PiBackendOptions extends Record<string, unknown> {
	rosclawHome?: string;
	profile?: "developer" | "robot";
	version?: string;
	missionId?: string;
	resumed?: boolean;
	taskContext?: unknown;
	workspaceStore?: unknown;
	workspaceAutoBound?: boolean;
}

export function createPiBackend(): NativeHarnessBackend {
	return {
		backendId: PI_BACKEND_ID,
		capabilities: probePiCapabilities(),
		async create(options: HarnessCreateOptions): Promise<HarnessSession> {
			const extra = (options.backendOptions ?? {}) as PiBackendOptions;
			const { createRosclawRuntime } = await import("./pi-runtime.js");
			const { resolveTaskContext } = await import(
				"../../native/active-task-context.js"
			);
			const rosclawHome = extra.rosclawHome
				?? `${process.env.HOME}/.rosclaw`;
			const runtime = await createRosclawRuntime({
				cwd: options.cwd,
				rosclawHome,
				profile: extra.profile ?? "developer",
				version: extra.version ?? "0.0.0",
				taskContext: (extra.taskContext ?? resolveTaskContext({
					rosclawHome,
					cwd: options.cwd,
					mode: "SIMULATION",
				})) as never,
				...(extra.missionId ? { missionId: extra.missionId } : {}),
				...(extra.resumed ? { resumed: true } : {}),
				...(extra.workspaceStore
					? { workspaceStore: extra.workspaceStore as never }
					: {}),
				...(extra.workspaceAutoBound ? { workspaceAutoBound: true } : {}),
			});
			// RosclawRuntime.runtime = AgentSessionRuntime → .session 才是
			// AgentSession（与 main.ts 的 runtime.session 同一对象）。
			const session = (runtime as unknown as {
				runtime: { session: import("@earendil-works/pi-coding-agent").AgentSession };
			}).runtime.session;
			return new PiHarnessSession(session, options.cwd);
		},
		async resume(ref: HarnessSessionRef): Promise<HarnessSession> {
			if (ref.backendId !== PI_BACKEND_ID) {
				throw new Error(
					`HARNESS_SESSION_LOST: backend ${ref.backendId} 的 session 不能由 pi resume`,
				);
			}
			const { SessionManager } = await import("@earendil-works/pi-coding-agent");
			const rosclawHome = `${process.env.ROSCLAW_HOME ?? `${process.env.HOME}/.rosclaw`}`;
			const sessionDir = `${rosclawHome}/agent/sessions`;
			// nativeRef（session id）→ 真实路径（精确 ID/唯一前缀——
			// 不手拼路径，与 main.ts 的 --resume 解析同一实现）。
			const { resolveSessionQuery } = await import("./pi-resolve.js");
			const sessions = await SessionManager.listAll(sessionDir);
			const hit = resolveSessionQuery(ref.nativeRef, sessions);
			if (!hit.ok) {
				throw new Error(
					`HARNESS_SESSION_LOST: session ${ref.nativeRef} ${hit.error}`,
				);
			}
			const opened = SessionManager.open(hit.path, sessionDir);
			const runtime = await (await import("./pi-runtime.js")).createRosclawRuntime({
				cwd: opened.getCwd() ?? process.cwd(),
				rosclawHome,
				profile: "developer",
				version: "0.0.0",
				taskContext: (await import("../../native/active-task-context.js"))
					.resolveTaskContext({
						rosclawHome,
						cwd: opened.getCwd() ?? process.cwd(),
						mode: "SIMULATION",
						resumedWorkspace: opened.getCwd() ?? undefined,
					}) as never,
				sessionManager: opened,
				resumed: true,
			});
			const session = (runtime as unknown as {
				runtime: { session: import("@earendil-works/pi-coding-agent").AgentSession };
			}).runtime.session;
			return new PiHarnessSession(session, opened.getCwd() ?? "");
		},
	};
}
