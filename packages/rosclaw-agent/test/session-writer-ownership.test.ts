/** SESSION_WRITER ownership tests（source-only，假 provider/私有临时目录，
 *  无网络/模型/ROS）。独立自动 backend/CLI 控制与 ROOT 数据流审查仍
 *  必需——本文件不是全覆盖证书（见 docs/session-writer-ownership.md）。
 */
import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm, writeFile, readFile, symlink, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { readFileSync, existsSync, writeFileSync, mkdirSync } from "node:fs";
import fs from "node:fs";
import { syncBuiltinESMExports } from "node:module";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import {
	SessionWriterOwnership,
	SessionInUseError,
	canonicalSessionPath,
	ownerLockPath,
	lockOwnerAlive,
} from "../src/harness/pi/session-writer-ownership.js";
import { createSessionWriterOwnershipExtension } from "../src/harness/pi/pi-runtime.js";
import { openPiSession } from "../src/harness/pi/pi-sessions.js";
import { PiHarnessSession } from "../src/harness/pi/pi-session-adapter.js";

async function assertEventDone(read: Promise<IteratorResult<unknown>>): Promise<void> {
	let timer: ReturnType<typeof setTimeout> | undefined;
	try {
		const result = await Promise.race([read, new Promise<never>((_, reject) => {
			timer = setTimeout(() => reject(new Error("LOCAL_EVENTS_PENDING")), 150);
		})]);
		assert.equal(result.done, true);
	} finally { clearTimeout(timer); }
}

async function tmp(): Promise<string> {
	return mkdtemp(join(tmpdir(), "rosclaw-session-writer-"));
}

test("same PID is not same owner: independent managers for one file are denied", async () => {
	const home = await tmp();
	try {
		const file = join(home, "a.jsonl");
		await writeFile(file, "{}\n");
		const ownerA = new SessionWriterOwnership();
		const ownerB = new SessionWriterOwnership();
		assert.notEqual(ownerA.ownerToken, ownerB.ownerToken);
		ownerA.acquire(file);
		// 同 PID、同文件、不同 owner → acquire 与 check 双拒绝。
		assert.throws(() => ownerB.acquire(file), (e) => e instanceof SessionInUseError);
		assert.throws(() => ownerB.check(file), SessionInUseError);
		// 同 owner 重复 acquire 幂等（正常分支/导航不被排除）。
		assert.doesNotThrow(() => ownerA.acquire(file));
		assert.doesNotThrow(() => ownerA.check(file));
		// distinct 文件互不影响。
		const other = join(home, "b.jsonl");
		await writeFile(other, "{}\n");
		assert.doesNotThrow(() => ownerB.acquire(other));
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("canonical path aliases cannot bypass same-session exclusion", async () => {
	const home = await tmp();
	try {
		const realDir = join(home, "real");
		await mkdir(realDir);
		const file = join(realDir, "s.jsonl");
		await writeFile(file, "{}\n");
		const linkDir = join(home, "link");
		await symlink(realDir, linkDir);
		const ownerA = new SessionWriterOwnership();
		const ownerB = new SessionWriterOwnership();
		ownerA.acquire(file);
		// 经符号链接目录 + ./.. 别名指向同一 inode → 仍同一 canonical。
		const alias = join(linkDir, ".", "s.jsonl");
		assert.equal(canonicalSessionPath(alias), canonicalSessionPath(file));
		assert.throws(() => ownerB.acquire(alias), SessionInUseError);
		// 同 owner 经别名访问视为同一 claim（幂等）。
		assert.doesNotThrow(() => ownerA.acquire(alias));
		assert.equal(ownerA.claimCount, 1);
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("release is token-only: non-owner release never frees another owner", async () => {
	const home = await tmp();
	try {
		const file = join(home, "owned.jsonl");
		await writeFile(file, "{}\n");
		const ownerA = new SessionWriterOwnership();
		const ownerB = new SessionWriterOwnership();
		ownerA.acquire(file);
		ownerB.release(file); // 非 owner——必须无操作
		assert.throws(() => ownerB.acquire(file), SessionInUseError);
		assert.ok(existsSync(ownerLockPath(canonicalSessionPath(file))));
		ownerA.release(file);
		assert.ok(!existsSync(ownerLockPath(canonicalSessionPath(file))));
		assert.doesNotThrow(() => ownerB.acquire(file)); // 真正释放后可获得
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("stale fail-closed: dead PID and reused birth are diagnostics, not reclaim permission", async () => {
	const home = await tmp();
	try {
		const file = join(home, "crash.jsonl");
		await writeFile(file, "{}\n");
		const lock = ownerLockPath(canonicalSessionPath(file));
		const owner = new SessionWriterOwnership();
		for (const record of [
			{ token: "dead-owner", pid: 0x3fffffff, birth: "1", acquiredAt: "1970-01-01" },
			{ token: "recycled", pid: process.pid, birth: "not-current-birth", acquiredAt: "1970-01-01" },
		]) {
			const bytes = JSON.stringify(record);
			writeFileSync(lock, bytes);
			assert.equal(lockOwnerAlive(record), false);
			assert.throws(() => owner.acquire(file), (e) => e instanceof SessionInUseError && /automatic reclaim disabled/.test(e.message));
			assert.throws(() => owner.check(file), SessionInUseError);
			owner.releaseAll();
			assert.equal(await readFile(lock, "utf-8"), bytes);
			assert.equal(owner.claimCount, 0);
		}
		// Fixture-only quiescent removal; not a product auto-recovery operation.
		await rm(lock);
		owner.acquire(file);
		const live = JSON.parse(await readFile(lock, "utf-8"));
		live.acquiredAt = "1970-01-01";
		writeFileSync(lock, JSON.stringify(live));
		const other = new SessionWriterOwnership();
		assert.throws(() => other.acquire(file), SessionInUseError);
		assert.throws(() => other.check(file), SessionInUseError);
		assert.equal(JSON.parse(await readFile(lock, "utf-8")).token, owner.ownerToken);
		owner.releaseAll();
		other.acquire(file);
		other.releaseAll();
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("openPiSession denies a second owner before any SDK open; file unchanged", async () => {
	const home = await tmp();
	try {
		const file = join(home, "resume.jsonl");
		await writeFile(file, '{"type":"session"}\n');
		const before = await readFile(file);
		const ownerA = new SessionWriterOwnership();
		const ownerB = new SessionWriterOwnership();
		ownerA.acquire(file);
		assert.throws(() => openPiSession(file, home, ownerB), SessionInUseError);
		assert.deepEqual(await readFile(file), before); // 零 append
		assert.equal(ownerB.claimCount, 0); // 拒绝路径不留 claim
		ownerA.release(file);
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("public session_before_switch reserves the target pre-open; contender insertion denied; veto leaks nothing", async () => {
	const home = await tmp();
	try {
		const target = join(home, "target.jsonl");
		await writeFile(target, "{}\n");
		const switcher = new SessionWriterOwnership();
		const other = new SessionWriterOwnership();
		const reservations = new Set<string>();
		const events: Array<{ type: string; handler: (e: { reason?: string; targetSessionFile?: string }) => unknown }> = [];
		const factory = createSessionWriterOwnershipExtension(switcher, reservations);
		factory({ on: (type: string, handler: never) => { events.push({ type, handler }); } } as never);
		const beforeSwitch = events.find((e) => e.type === "session_before_switch")!;
		assert.ok(beforeSwitch);
		// 无 owner 目标 → 不取消，且在 SDK open 之前持有排他 reservation
		// （check-only 预检不足：此后插入的竞争 writer 必须被拒绝）。
		assert.equal(beforeSwitch.handler({ reason: "resume", targetSessionFile: target }), undefined);
		assert.equal(switcher.claimCount, 1);
		assert.equal(reservations.size, 1);
		const contender = new SessionWriterOwnership();
		assert.throws(() => contender.acquire(target), SessionInUseError);
		assert.throws(() => contender.check(target), SessionInUseError);
		// reservation 的锁就是 switcher 的 token。
		const held = JSON.parse(await readFile(ownerLockPath(canonicalSessionPath(target)), "utf-8"));
		assert.equal(held.token, switcher.ownerToken);
		// 目标 open 失败路径：只释放 target reservation，不动旧 claim/他人。
		switcher.release(target);
		reservations.clear();
		assert.equal(switcher.claimCount, 0);
		// 其他活 owner 占有目标 → SDK open 之前 veto；veto 不留 claim、
		// 不写锁、不释放他人。
		other.acquire(target);
		assert.deepEqual(beforeSwitch.handler({ reason: "resume", targetSessionFile: target }), { cancel: true });
		assert.equal(switcher.claimCount, 0);
		assert.equal(reservations.size, 0);
		const persisted = JSON.parse(await readFile(ownerLockPath(canonicalSessionPath(target)), "utf-8"));
		assert.equal(persisted.token, other.ownerToken);
		// new session（无 targetSessionFile）→ 不 veto、不占有。
		assert.equal(beforeSwitch.handler({ reason: "new" }), undefined);
		assert.equal(switcher.claimCount, 0);
		other.releaseAll();
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("atomic exclusion: garbled or unreadable lock is UNKNOWN — refused, never overwritten", async () => {
	const home = await tmp();
	try {
		const file = join(home, "garbled.jsonl");
		await writeFile(file, "{}\n");
		const canonical = canonicalSessionPath(file);
		const lock = ownerLockPath(canonical);
		const ownerA = new SessionWriterOwnership();
		const ownerB = new SessionWriterOwnership();
		// 1) 损坏 JSON 锁 → UNKNOWN fail closed：两个 acquirer 都拒绝，
		//    内容逐字节不变（绝不覆盖 garbled authority）。
		writeFileSync(lock, "this is not json {{{", { mode: 0o600 });
		assert.throws(() => ownerA.acquire(file), SessionInUseError);
		assert.throws(() => ownerB.acquire(file), SessionInUseError);
		assert.throws(() => ownerA.check(file), SessionInUseError);
		assert.equal(await readFile(lock, "utf-8"), "this is not json {{{");
		assert.equal(ownerA.claimCount, 0);
		// 2) 字段缺失（无 birth/token）→ 同样 UNKNOWN，原样保留。
		writeFileSync(lock, JSON.stringify({ pid: process.pid }), { mode: 0o600 });
		assert.throws(() => ownerA.acquire(file), SessionInUseError);
		assert.equal(await readFile(lock, "utf-8"), JSON.stringify({ pid: process.pid }));
		// 3) 锁路径是不可读对象（目录 → EISDIR，权限无关的 UNKNOWN）→
		//    拒绝且不删除/不覆盖。
		await rm(lock, { force: true });
		mkdirSync(lock);
		assert.throws(() => ownerA.acquire(file), SessionInUseError);
		assert.ok(existsSync(lock));
		await rm(lock, { recursive: true, force: true });
		// 4) 锁消失后正常获取可用（UNKNOWN 不毒化后续正常路径）。
		assert.doesNotThrow(() => ownerA.acquire(file));
		ownerA.releaseAll();
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("three stale contenders all refuse without any lock movement or restoration", async () => {
	const home = await tmp();
	try {
		const file = join(home, "race.jsonl");
		await writeFile(file, "{}\n");
		const lock = ownerLockPath(canonicalSessionPath(file));
		const bytes = JSON.stringify({ token: "dead-owner", pid: 0x3fffffff, birth: "1", acquiredAt: "x" });
		writeFileSync(lock, bytes);
		const owners = Array.from({ length: 3 }, () => new SessionWriterOwnership());
		const rename = fs.renameSync;
		const unlink = fs.unlinkSync;
		let destructiveCalls = 0;
		fs.renameSync = function (from, to) { if (String(from) === lock || String(to) === lock) destructiveCalls++; return rename(from, to); };
		fs.unlinkSync = function (p) { if (String(p) === lock) destructiveCalls++; return unlink(p); };
		syncBuiltinESMExports();
		try {
			for (const owner of owners) {
				assert.throws(() => owner.acquire(file), SessionInUseError);
				assert.throws(() => owner.check(file), SessionInUseError);
				owner.releaseAll();
				assert.equal(owner.claimCount, 0);
			}
		} finally { fs.renameSync = rename; fs.unlinkSync = unlink; syncBuiltinESMExports(); }
		assert.equal(destructiveCalls, 0, "stale refusal must happen before any movement");
		assert.equal(await readFile(lock, "utf-8"), bytes);
		assert.deepEqual(fs.readdirSync(home).sort(), ["race.jsonl", "race.jsonl.owner.json"]);
		await rm(lock); // quiescent fixture recovery
		owners[0].acquire(file);
		assert.throws(() => owners[1].acquire(file), SessionInUseError);
		assert.throws(() => owners[2].acquire(file), SessionInUseError);
		owners[0].releaseAll();
		owners[2].acquire(file);
		owners[2].releaseAll();
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("residual precheck-to-open interleaving: O_EXCL refuses a live contender, no restore gap", async () => {
	// Finite synchronous FS timing injection, NOT a real multiprocess proof.
	// C passes the legacy-marker precheck, then A creates the real live claim
	// immediately before C's open(wx). No reclaimer ever moves that claim.
	const home = await tmp();
	try {
		const file = join(home, "stale-race.jsonl");
		await writeFile(file, "{}\n");
		const lock = ownerLockPath(canonicalSessionPath(file));
		const ownerA = new SessionWriterOwnership();
		const ownerB = new SessionWriterOwnership();
		const ownerC = new SessionWriterOwnership();
		const open = fs.openSync;
		let inserted = false;
		let liveBytes = "";
		fs.openSync = function (p, flags, mode) {
			if (String(p) === lock && flags === "wx" && !inserted) {
				inserted = true;
				ownerA.acquire(file);
				liveBytes = readFileSync(lock, "utf-8");
			}
			return open(p, flags, mode);
		};
		syncBuiltinESMExports();
		try { assert.throws(() => ownerC.acquire(file), SessionInUseError); }
		finally { fs.openSync = open; syncBuiltinESMExports(); }
		assert.ok(inserted, "pre-open contender insertion must fire");
		assert.throws(() => ownerB.acquire(file), SessionInUseError);
		assert.equal(ownerA.claimCount, 1);
		assert.equal(ownerB.claimCount + ownerC.claimCount, 0);
		assert.equal(await readFile(lock, "utf-8"), liveBytes);
		ownerA.releaseAll();
		ownerC.acquire(file);
		ownerC.releaseAll();
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("reclaim marker is honored fail-closed: no stealing, no poisoning of normal paths", async () => {
	const home = await tmp();
	try {
		const file = join(home, "marker.jsonl");
		await writeFile(file, "{}\n");
		const canonical = canonicalSessionPath(file);
		const lock = ownerLockPath(canonical);
		const marker = `${lock}.reclaim`;
		// 残留 marker（模拟回收者在恢复窗口崩溃）→ acquire fail closed，
		// marker 与（不存在的）锁都不被动；绝不盲删 marker 抢占。
		writeFileSync(marker, JSON.stringify({ token: "crashed-reclaimer", pid: 0x3fffffff, at: "x" }));
		const owner = new SessionWriterOwnership();
		assert.throws(() => owner.acquire(file), (e) => e instanceof SessionInUseError
			&& /reclaim marker persisted/.test((e as Error).message));
		assert.equal(owner.claimCount, 0);
		assert.equal(JSON.parse(await readFile(marker, "utf-8")).token, "crashed-reclaimer");
		// 运维核实后手动移除 marker → 正常获取恢复可用。
		await rm(marker);
		assert.doesNotThrow(() => owner.acquire(file));
		owner.releaseAll();
		// Stale refusal creates no marker and preserves the dead record.
		const bytes = JSON.stringify({ token: "dead", pid: 0x3fffffff, birth: "1", acquiredAt: "x" });
		writeFileSync(lock, bytes);
		const reclaimer = new SessionWriterOwnership();
		assert.throws(() => reclaimer.acquire(file), SessionInUseError);
		assert.throws(() => reclaimer.check(file), SessionInUseError);
		assert.ok(!existsSync(marker));
		assert.equal(await readFile(lock, "utf-8"), bytes);
		reclaimer.releaseAll();
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("adapter close is an authorized public release seam (resume becomes possible)", async () => {	const home = await tmp();
	try {
		const file = join(home, "adapter.jsonl");
		await writeFile(file, "{}\n");
		const ownership = new SessionWriterOwnership();
		ownership.acquire(file);
		let disposed = 0;
		let aborted = 0;
		const fakeSession = {
			sessionId: "fixture",
			isIdle: true,
			abort: async () => { aborted += 1; },
			dispose: () => { disposed += 1; },
		};
		const adapter = new PiHarnessSession(fakeSession as never, home, () => ownership.releaseAll());
		await adapter.close();
		await adapter.close(); // 幂等
		assert.equal(disposed, 1);
		assert.equal(aborted, 1);
		assert.equal(ownership.claimCount, 0);
		const next = new SessionWriterOwnership();
		assert.doesNotThrow(() => next.acquire(file)); // close 后 resume 可获得
		next.releaseAll();
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("actual runtime: new chat owns its eventual file; same-PID second runtime denied pre-open; same owner allowed", { timeout: 20_000 }, async () => {
	const home = await tmp();
	let first: Awaited<ReturnType<typeof import("../src/harness/pi/pi-runtime.js").createRosclawRuntime>> | undefined;
	try {
		const { createRosclawRuntime } = await import("../src/harness/pi/pi-runtime.js");
		const { resolveTaskContext } = await import("../src/native/active-task-context.js");
		const rosclawHome = join(home, "private_runtime");
		first = await createRosclawRuntime({
			cwd: home, rosclawHome, profile: "developer", version: "fixture",
			taskContext: resolveTaskContext({ cwd: home, rosclawHome, mode: "SIMULATION" }),
		});
		const sessionFile = first.runtime.session.sessionManager.getSessionFile()!;
		// 新 chat 在 SDK 初始持久化之前就占有最终文件。
		assert.ok(first.ownership.claimCount >= 1);
		assert.ok(existsSync(ownerLockPath(canonicalSessionPath(sessionFile))));
		// 同 PID 独立 runtime owner（不同 ownership）对同一文件 → 拒绝。
		const outsider = new SessionWriterOwnership();
		await assert.rejects(createRosclawRuntime({
			cwd: home, rosclawHome: join(home, "private_runtime_2"), profile: "developer", version: "fixture",
			taskContext: resolveTaskContext({ cwd: home, rosclawHome: join(home, "private_runtime_2"), mode: "SIMULATION" }),
			sessionManager: first.runtime.session.sessionManager,
			ownership: outsider,
		}), SessionInUseError);
		assert.equal(outsider.claimCount, 0);
		// 他人的锁未被拒绝路径破坏。
		assert.ok(existsSync(ownerLockPath(canonicalSessionPath(sessionFile))));
		// 同 owner 幂等复核（正常分支语义）不抛。
		assert.doesNotThrow(() => first!.ownership.acquire(sessionFile));
		// distinct 文件+UUID：同 PID 另一 runtime 完全独立。
		const second = await createRosclawRuntime({
			cwd: home, rosclawHome: join(home, "private_runtime_3"), profile: "developer", version: "fixture",
			taskContext: resolveTaskContext({ cwd: home, rosclawHome: join(home, "private_runtime_3"), mode: "SIMULATION" }),
		});
		try {
			assert.notEqual(
				second.runtime.session.sessionManager.getSessionFile(),
				sessionFile,
			);
		} finally {
			second.ownership.releaseAll();
			second.runtime.session.dispose();
		}
		// 正常释放后，另一 owner 可 resume 该文件。
		first.ownership.releaseAll();
		const resumer = new SessionWriterOwnership();
		assert.doesNotThrow(() => resumer.acquire(sessionFile));
		resumer.releaseAll();
	} finally {
		first?.ownership.releaseAll();
		first?.runtime.session.dispose();
		await rm(home, { recursive: true, force: true });
	}
});

test("initial runtime constructor failure releases only its own claim", { timeout: 20_000 }, async () => {
	const home = await tmp();
	try {
		const { createRosclawRuntime } = await import("../src/harness/pi/pi-runtime.js");
		const { resolveTaskContext } = await import("../src/native/active-task-context.js");
		const rosclawHome = join(home, "private_runtime_fail");
		// 缺失 dist/prompts 时不一定可复现——用一个占有冲突在 acquire
		// 之后、createAgentSessionRuntime 内部 factory 之前无抛点；这里
		// 验证：acquire 成功但后续失败时（经 ownership 冲突注入）
		// releaseAll 只释放自己 token 的锁。
		const ownership = new SessionWriterOwnership();
		const mine = join(home, "mine.jsonl");
		await writeFile(mine, "{}\n");
		ownership.acquire(mine);
		const foreign = new SessionWriterOwnership();
		const foreignFile = join(home, "foreign.jsonl");
		await writeFile(foreignFile, "{}\n");
		foreign.acquire(foreignFile);
		ownership.releaseAll(); // 模拟构造失败清理
		assert.equal(ownership.claimCount, 0);
		assert.ok(!existsSync(ownerLockPath(canonicalSessionPath(mine))));
		// 他人 owner 的锁原样保留。
		assert.ok(existsSync(ownerLockPath(canonicalSessionPath(foreignFile))));
		foreign.releaseAll();
		// 真实构造路径冒烟：合法参数可完整建立（回归保护）。
		const ok = await createRosclawRuntime({
			cwd: home, rosclawHome, profile: "developer", version: "fixture",
			taskContext: resolveTaskContext({ cwd: home, rosclawHome, mode: "SIMULATION" }),
		});
		ok.ownership.releaseAll();
		ok.runtime.session.dispose();
	} finally { await rm(home, { recursive: true, force: true }); }
});

test("failed public switch before replacement factory releases only the target reservation", { timeout: 20_000 }, async () => {
	const home = await tmp();
	let first: Awaited<ReturnType<typeof import("../src/harness/pi/pi-runtime.js").createRosclawRuntime>> | undefined;
	let second: Awaited<ReturnType<typeof import("../src/harness/pi/pi-runtime.js").createRosclawRuntime>> | undefined;
	try {
		const { createRosclawRuntime } = await import("../src/harness/pi/pi-runtime.js");
		const { resolveTaskContext } = await import("../src/native/active-task-context.js");
		const rosclawHome = join(home, "private_runtime_switch_fail");
		first = await createRosclawRuntime({
			cwd: home, rosclawHome, profile: "developer", version: "fixture",
			taskContext: resolveTaskContext({ cwd: home, rosclawHome, mode: "SIMULATION" }),
		});
		const oldFile = first.runtime.session.sessionManager.getSessionFile()!;
		// 公共持久化对话前提：pinned SDK 仅在存在 user/assistant 消息时
		// 才物化 JSONL（含 header）。空会话下 session_before_switch 的
		// target-header 守卫会合法 cancel（注入 0 次），因此先经公共
		// appendMessage 物化 header，再做确定性 SM.open EIO 注入。
		first.runtime.session.sessionManager.appendMessage({
			role: "user",
			content: "materialize header for deterministic EIO switch test",
			timestamp: Date.now(),
		} as never);
		assert.ok(existsSync(oldFile), "public append must materialize the current session JSONL");
		// 确定性公共失败注入：pinned SDK 的公共 SessionManager.open 导出
		// 可被同进程测试包装（不改 SDK 源）——对目标文件抛 EIO，使
		// switchSession 在 replacement factory 之前失败（PRE_TEARDOWN）。
		// 旧"目录作 target"前提错误：pinned SDK 接受目录输入，不保证拒绝。
		const sm = SessionManager as unknown as { open: (...args: unknown[]) => unknown };
		const originalOpen = sm.open;
		const failingOpen = (target: string, calls: { n: number }) => {
			sm.open = function (this: unknown, ...args: unknown[]) {
				if (typeof args[0] === "string"
					&& canonicalSessionPath(args[0]) === canonicalSessionPath(target)) {
					calls.n += 1;
					throw Object.assign(new Error("SYNTHETIC_PUBLIC_OPEN_FAILURE"), { code: "EIO" });
				}
				return originalOpen.apply(this, args);
			};
		};
		// ── A. 同当前文件自切换 + SM.open EIO：旧 runtime 与其既有排他
		// claim 必须存活（reservation seam 不得把已持有文件记入 pending
		// 而被公共 wrapper 的 catch 误释放）。
		const sameCalls = { n: 0 };
		const oldSession = first.runtime.session;
		failingOpen(oldFile, sameCalls);
		let sameError: unknown; let sameResult: unknown;
		try {
			sameResult = await first.runtime.switchSession(oldFile);
		} catch (err) {
			sameError = err;
		} finally {
			sm.open = originalOpen;
		}
		assert.ok(sameError,
			`same-current-file switch must reject on SM.open EIO (result=${JSON.stringify(sameResult)} injectedCalls=${sameCalls.n})`);
		assert.equal(sameCalls.n, 1);
		assert.equal(first.runtime.session, oldSession); // PRE_TEARDOWN：旧 runtime 仍存活
		assert.equal(first.runtime.session.sessionManager.getSessionFile(), oldFile);
		assert.ok(first.ownership.has(oldFile)); // 旧排他 claim 未被误释放
		assert.ok(existsSync(ownerLockPath(canonicalSessionPath(oldFile))));
		const oldContender = new SessionWriterOwnership();
		assert.throws(() => oldContender.acquire(oldFile), SessionInUseError);
		oldContender.releaseAll();
		// ── B. 真正的新 target + SM.open EIO：target reservation 被
		// token-only 释放，独立 owner 可获得；旧 claim 同样不受影响。
		second = await createRosclawRuntime({
			cwd: home, rosclawHome: join(home, "private_runtime_switch_fail_2"), profile: "developer", version: "fixture",
			taskContext: resolveTaskContext({ cwd: home, rosclawHome: join(home, "private_runtime_switch_fail_2"), mode: "SIMULATION" }),
		});
		const target = second.runtime.session.sessionManager.getSessionFile()!;
		assert.notEqual(canonicalSessionPath(target), canonicalSessionPath(oldFile));
		// 同一公共前提：target 也需 user 消息物化 JSONL header，
		// 否则 header 守卫合法 cancel、注入永远为 0 次。
		second.runtime.session.sessionManager.appendMessage({
			role: "user",
			content: "materialize target header for deterministic EIO switch test",
			timestamp: Date.now(),
		} as never);
		assert.ok(existsSync(target), "public append must materialize the target session JSONL");
		second.ownership.releaseAll();
		second.runtime.session.dispose();
		second = undefined;
		const targetCalls = { n: 0 };
		failingOpen(target, targetCalls);
		let targetError: unknown; let targetResult: unknown;
		try {
			targetResult = await first.runtime.switchSession(target);
		} catch (err) {
			targetError = err;
		} finally {
			sm.open = originalOpen;
		}
		assert.ok(targetError,
			`new-target switch must reject on SM.open EIO (result=${JSON.stringify(targetResult)} injectedCalls=${targetCalls.n})`);
		assert.equal(targetCalls.n, 1);
		assert.equal(first.runtime.session, oldSession); // PRE_TEARDOWN：旧 runtime 仍存活
		assert.ok(first.ownership.has(oldFile));
		assert.ok(existsSync(ownerLockPath(canonicalSessionPath(oldFile))));
		// target reservation 已释放：独立 owner 可获得（不 releaseAll
		// 旧 owner 来掩盖泄漏）。
		const independent = new SessionWriterOwnership();
		assert.doesNotThrow(() => independent.acquire(target));
		assert.equal(first.ownership.has(target), false);
		independent.releaseAll();
	} finally {
		first?.ownership.releaseAll();
		first?.runtime.session.dispose();
		if (second) {
			second.ownership.releaseAll();
			second.runtime.session.dispose();
		}
		await rm(home, { recursive: true, force: true });
	}
});

test("adapter failed or unknown close retains exclusivity and repeat rejection", async () => {
	for (const fault of ["abort", "dispose", "false", "unknown", "getter", "post-dispose"] as const) {
		const home = await tmp();
		try {
			const file = join(home, "fault.jsonl");
			await writeFile(file, "{}\n");
			const owner = new SessionWriterOwnership();
			const contender = new SessionWriterOwnership();
			owner.acquire(file);
			const bytes = await readFile(ownerLockPath(canonicalSessionPath(file)));
			let aborted = 0; let disposed = 0; let released = 0;
			const injected = new Error(`INJECTED_${fault}`);
			let subscriptions = 0; let unsubscriptions = 0;
			const fake = {
				subscribe: () => { subscriptions++; return () => { unsubscriptions++; }; },
				sessionId: "fault",
				get isIdle() {
					if (fault === "getter") throw injected;
					if (fault === "unknown") return undefined;
					return fault !== "false" && !(fault === "post-dispose" && disposed > 0);
				},
				abort: async () => { aborted++; if (fault === "abort") throw injected; },
				dispose: () => { disposed++; if (fault === "dispose") throw injected; },
			};
			const adapter = new PiHarnessSession(fake as never, home, () => { released++; owner.releaseAll(); });
			const streams = [adapter.events()[Symbol.asyncIterator](), adapter.events()[Symbol.asyncIterator]()];
			const reads = streams.map((stream) => stream.next());
			const first = adapter.close();
			await Promise.all(reads.map(assertEventDone));
			await assertEventDone(adapter.events()[Symbol.asyncIterator]().next());
			assert.equal(subscriptions, 2);
			assert.equal(unsubscriptions, 2);
			assert.equal(adapter.close(), first);
			let error: unknown;
			await assert.rejects(first, (e) => { error = e; return true; });
			await assert.rejects(adapter.close(), (e) => e === error);
			assert.equal(aborted, 1);
			assert.equal(released, 0);
			assert.equal(owner.claimCount, 1);
			assert.throws(() => contender.acquire(file), SessionInUseError);
			assert.throws(() => contender.check(file), SessionInUseError);
			assert.deepEqual(await readFile(ownerLockPath(canonicalSessionPath(file))), bytes);
			assert.equal(disposed, ["dispose", "post-dispose"].includes(fault) ? 1 : 0);
		} finally { await rm(home, { recursive: true, force: true }); }
	}
});

test("adapter unresolved close is single-flight and retains claim until healthy completion", async () => {
	const home = await tmp();
	try {
		const file = join(home, "pending.jsonl");
		await writeFile(file, "{}\n");
		const owner = new SessionWriterOwnership();
		const contender = new SessionWriterOwnership();
		owner.acquire(file);
		let finish!: () => void;
		const pending = new Promise<void>((resolve) => { finish = resolve; });
		let aborted = 0; let disposed = 0;
		const fake = { sessionId: "pending", isIdle: true,
			abort: () => { aborted++; return pending; }, dispose: () => { disposed++; } };
		const adapter = new PiHarnessSession(fake as never, home, () => owner.releaseAll());
		const first = adapter.close();
		assert.equal(adapter.close(), first);
		await Promise.resolve();
		assert.equal(aborted, 1);
		assert.equal(disposed, 0);
		assert.equal(owner.claimCount, 1);
		assert.throws(() => contender.acquire(file), SessionInUseError);
		finish();
		await first;
		await adapter.close();
		assert.equal(disposed, 1);
		assert.equal(owner.claimCount, 0);
		contender.acquire(file);
		contender.releaseAll();
	} finally { await rm(home, { recursive: true, force: true }); }
});

/** 本测试文件既可经 tsx 从 source 运行，也可从 dist/test 运行——
 *  向上找真正存在的 src/main.ts（绝不拼不存在的 dist/src/main.ts）。 */
function findMainSource(): string {
	let dir = dirname(fileURLToPath(import.meta.url));
	for (let i = 0; i < 8; i += 1) {
		const candidate = join(dir, "src", "main.ts");
		if (existsSync(candidate)) return candidate;
		const parent = dirname(dir);
		if (parent === dir) break;
		dir = parent;
	}
	throw new Error("src/main.ts not found above test file (stale/incomplete build?)");
}

test("main.ts source wiring: ownership release on bind failure and normal exit (guard)", () => {
	const main = readFileSync(findMainSource(), "utf-8");
	// Bind failures are inside the same finally as print/UI, not early release.
	assert.ok(/try \{\s*if \(missionId\)/.test(main));
	assert.ok(/接入失败[\s\S]{0,200}?return 2/.test(main));
	assert.ok(/恢复绑定失败[\s\S]{0,200}?return 2/.test(main));
	const teardown = main.slice(main.indexOf("const session = runtime.session;"));
	assert.ok(/await session\.abort\(\);[\s\S]*isIdle !== true[\s\S]*await session\.dispose\(\);[\s\S]*isIdle !== true[\s\S]*await runtime\.dispose\(\);/.test(teardown));
	assert.ok(teardown.indexOf("throw new Error(`MAIN_EXIT_TEARDOWN_UNCONFIRMED") < teardown.indexOf("runtimeOwnership.releaseAll()"));
	assert.ok(/runtimeOwnership\.releaseAll\(\);\s*await leaseManager\.release\(\)/.test(teardown));
	assert.ok(/finally \{[\s\S]{0,150}?mode\.stop\(\)/.test(main));
	// Scope the ordering check to the actual mode.run() finally, not an
	// unrelated earlier finally or a comment containing the same tokens.
	const run = main.indexOf("await Promise.race([mode.run(), shutdownComplete])");
	const drain = main.indexOf("await ownedAbort.drain()", run);
	const stop = main.indexOf("mode.stop()", drain);
	const close = main.indexOf("await confirmWriterClosed()", stop);
	const release = main.indexOf("runtimeOwnership.releaseAll()", close);
	assert.ok(run >= 0 && drain > run && stop > drain && close > stop && release > close);
	assert.ok(main.slice(run, drain).includes("} finally {"));
	assert.ok(main.slice(run, stop).includes("const cancellationOutcomes = await ownedAbort.drain()"));
	assert.ok(main.slice(stop, close).includes("cancellationOutcomes.some(outcome => !outcome.ok)"));
	assert.ok(main.slice(close, release).includes("if (interactiveCloseFailure)"));
	assert.ok(!/process\.exit\(/.test(main), "natural exit must not hide live consumers");
	assert.ok(/process\.exitCode = 2/.test(main));
	// resume/continue 全部经 ownership 占有后 open。
	assert.ok(/openPiSession\(picked, sessionDir, ownership\)/.test(main));
	assert.ok(/openPiSession\(resumeSessionPath, sessionDir, ownership\)/.test(main));
	assert.ok(/openPiSession\(hit\.path, sessionDir, ownership\)/.test(main));
	assert.ok(/continueRecentPiSession\(workspace \?\? startupCwd, sessionDir, ownership\)/.test(main));
	// SESSION_IN_USE 拒绝 → 可读错误 + 非零退出。
	assert.ok(/isSessionInUse/.test(main));
});
