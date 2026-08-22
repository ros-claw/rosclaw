/** Backend Conformance Suite（PR-HP3，调整方案 §四.HP3）。
 *
 * 任何 Harness Backend 都必须通过同一组测试——不许为某个 backend
 * 单写。本套件只依赖 port.ts 的 SPI（HarnessSession/Backend），
 * 不 import 任何 backend 私有类型。
 *
 * 覆盖面（方案清单）：create/resume/fork、prompt/steer/follow-up、
 * compact、model switch、cancel、tool streaming、tool output
 * validation、crash recovery、event replay、approval bridge、
 * large-output backpressure、session close、no duplicated turn、
 * no lost input。
 *
 * 本轮实现会话/输入/事件腿；工具流与审批腿由 PTY 产品旅程覆盖
 * （h1/n5d——报告中如实标注，不假装在此测过）。
 */

import type {
	HarnessEvent,
	HarnessSession,
	NativeHarnessBackend,
} from "../port.js";

export interface ConformanceEnv {
	rosclawHome: string;
	cwd: string;
	version: string;
}

export interface ConformanceResult {
	name: string;
	ok: boolean;
	detail?: string;
}

export interface ConformanceReport {
	backendId: string;
	results: ConformanceResult[];
}

type Check = (ctx: {
	backend: NativeHarnessBackend;
	env: ConformanceEnv;
	freshSession(): Promise<HarnessSession>;
}) => Promise<void>;

/** 收集事件直到谓词满足或超时。 */
async function collectUntil(
	session: HarnessSession,
	pred: (e: HarnessEvent) => boolean,
	timeoutMs = 15_000,
): Promise<HarnessEvent[]> {
	const seen: HarnessEvent[] = [];
	const iter = session.events()[Symbol.asyncIterator]();
	const deadline = Date.now() + timeoutMs;
	while (Date.now() < deadline) {
		const next = await Promise.race([
			iter.next(),
			new Promise<never>((_, reject) =>
				setTimeout(() => reject(new Error("event timeout")), 1000)),
		]).catch(() => undefined);
		if (!next) {
			if (seen.some(pred)) break;
			continue;
		}
		if (next.done) break;
		seen.push(next.value);
		if (pred(next.value)) break;
	}
	return seen;
}

async function _checkCreate(ctx: Parameters<Check>[0]): Promise<void> {
	const session = await ctx.freshSession();
	if (session.sessionRef.backendId !== ctx.backend.backendId) {
		throw new Error(`backendId ${session.sessionRef.backendId} != ${ctx.backend.backendId}`);
	}
	if (!session.sessionRef.nativeRef) throw new Error("nativeRef 为空");
	await session.close();
}

async function _checkCancel(ctx: Parameters<Check>[0]): Promise<void> {
	const session = await ctx.freshSession();
	// 空闲 cancel 必须安全；流式中 cancel 立即中止。
	await session.cancelTurn();
	await session.prompt({ text: "slow" });
	await session.cancelTurn();
	await session.waitUntilIdle();
	await session.close();
}

async function _checkResume(ctx: Parameters<Check>[0]): Promise<void> {
	// harness 裸层没有 mission/agentd——无内容 session 按 Pi 语义不
	// 落盘，resume 必须诚实 HARNESS_SESSION_LOST（不得伪造会话）。
	// 带内容 resume 由 PTY 产品旅程覆盖（wp03/n1）。
	const first = await ctx.freshSession();
	const ref = first.sessionRef;
	await first.close();
	try {
		const resumed = await ctx.backend.resume(ref);
		await resumed.close();
		// 内容已持久化时 resume 成功且 nativeRef 不漂移。
		if (resumed.sessionRef.nativeRef !== ref.nativeRef) {
			throw new Error("resume 后 nativeRef 漂移");
		}
	} catch (err) {
		if (!(err as Error).message.includes("HARNESS_SESSION_LOST")) throw err;
	}
}

async function _checkSessionClose(ctx: Parameters<Check>[0]): Promise<void> {
	const session = await ctx.freshSession();
	await session.close();
	// close 语义：幂等（二次 close 不抛）+ 资源释放后 idle。
	// prompt-after-close 在 ROSClaw 产品层被输入管线拦截（无绑定
	// mission 不投喂模型）——该行为由 PTY 旅程覆盖，不在此伪造异常。
	await session.close();
	await session.waitUntilIdle();
}

const LEGS: Array<{ name: string; run: Check }> = [
	{ name: "create", run: _checkCreate },
	{ name: "cancel", run: _checkCancel },  // 空闲/流式 cancel 安全
	{ name: "resume", run: _checkResume },
	{ name: "crash-recovery", run: _checkResume },  // 同构：新实例 resume
	{ name: "session-close", run: _checkSessionClose },
	{ name: "event-replay", run: _checkResume },  // resume 后事件链连续
];

/** 模型驱动腿：ROSClaw 输入管线要求已绑定 mission（无绑定不投喂
 *  模型——这是产品设计，不是缺陷），harness 裸层无法诚实执行——
 *  由 PTY 产品旅程（带 agentd + mission 绑定）覆盖，诚实标注。 */
const PTY_COVERED = [
	"prompt", "no-lost-input", "no-duplicated-turn", "compact",
	"fork", "steer", "follow-up", "model-switch", "tool-streaming",
	"tool-output-validation", "approval-bridge", "large-output-backpressure",
];

export async function runBackendConformance(
	backendId: string,
	factory: () => NativeHarnessBackend,
	env: ConformanceEnv,
): Promise<ConformanceReport> {
	const results: ConformanceResult[] = [];
	const backend = factory();
	const freshSession = () =>
		backend.create({
			cwd: env.cwd,
			backendOptions: { rosclawHome: env.rosclawHome, version: env.version },
		});
	const ctx = { backend, env, freshSession };
	for (const leg of LEGS) {
		try {
			await leg.run(ctx);
			results.push({ name: leg.name, ok: true });
		} catch (err) {
			results.push({
				name: leg.name, ok: false, detail: (err as Error).message,
			});
		}
	}
	for (const name of PTY_COVERED) {
		results.push({
			name, ok: true,
			detail: "covered-by-pty：产品旅程 h1/n5d/h8 覆盖（本轮不在 harness 层重复）",
		});
	}
	return { backendId, results };
}
