/** PiHarnessBackend（PR-H1，ADR-0012）——当前唯一默认 Native Harness。
 *
 * 边界：
 * - 直接 import Pi SDK（不启动 pi CLI 子进程）；
 * - Pi 原生 session id 只在 HarnessSessionRef.nativeRef（不进产品 UI）；
 * - 事件经 adapter 统一为 HarnessEvent（产品侧不 switch Pi 私有类型）；
 * - 这里只做"装配边界"——create-runtime.ts 的具体装配逐步迁入，
 *   本 PR 先建立类型边界与唯一 backendId 事实源。
 */

import type {
	HarnessCapabilities,
	HarnessCreateOptions,
	HarnessSession,
	HarnessSessionRef,
	NativeHarnessBackend,
} from "../port.js";

export const PI_BACKEND_ID = "pi";

export const PI_CAPABILITIES: HarnessCapabilities = {
	persistentSessions: true,
	resume: true,
	steering: true,
	followUp: true,
	compaction: true,
	modelSwitching: true,
	customTools: true,
	toolStreaming: true,
	toolPolicyHook: true,
	interactiveUi: true,
};

/** 当前装配仍在 runtime/create-runtime.ts（Pi SDK 直装）——backend
 * 对象是该装配的边界声明；H 系列后续 PR 把装配本体迁入此处。 */
export function createPiBackend(): NativeHarnessBackend {
	return {
		backendId: PI_BACKEND_ID,
		capabilities: PI_CAPABILITIES,
		async create(_options: HarnessCreateOptions): Promise<HarnessSession> {
			// 装配本体在 runtime/create-runtime.ts（createRosclawRuntime）——
			// 该路径已直接 import Pi SDK（无 pi CLI 子进程）。本方法在
			// 装配迁入前显式报错，不静默提供第二路径。
			throw new Error(
				"HARNESS_CAPABILITY_MISSING: pi backend 装配本体在 " +
				"create-runtime.ts（迁入 Harness Backend 前不得另起路径）",
			);
		},
		async resume(_ref: HarnessSessionRef): Promise<HarnessSession> {
			throw new Error(
				"HARNESS_CAPABILITY_MISSING: resume 装配在 create-runtime.ts",
			);
		},
	};
}
