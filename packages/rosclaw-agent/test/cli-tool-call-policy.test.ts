/** Stage A：`--tool-call-policy JSON_FILE` CLI plumbing 回归测试。
 *
 * 通过真实 spawn `dist/main.js` 验证：策略文件在任何 runtime/session/
 * provider import 与 auth 读取之前被校验——非法策略/缺失参数/不可读文件
 * 一律以 INVALID_TOOL_CALL_POLICY 提前 exit 2；合法策略（含 block-all
 * 空 allowlist、空 map、保留字段名）通过校验并继续进入正常 CLI 流程
 * （此处用一个必然失败但不触碰 provider 的 --resume 解析错误作为
 * “已越过策略校验”的可观测哨兵）。默认不带 flag 的行为不变。
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const mainJs = join(dirname(fileURLToPath(import.meta.url)), "..", "src", "main.js");
const RESUME_SENTINEL = "nonexistent-stage-a-sentinel-session";

interface CliResult {
	status: number | null;
	stdout: string;
	stderr: string;
}

function runCli(args: string[], home: string): CliResult {
	const result = spawnSync(process.execPath, [mainJs, ...args], {
		encoding: "utf-8",
		timeout: 30_000,
		env: {
			...process.env,
			ROSCLAW_HOME: home,
			PI_OFFLINE: "1",
			// 拒绝路径不得读取 auth——即使存在 key 也不能被用到；
			// 这里不提供任何 key，保证早失败不依赖网络/凭证。
		},
	});
	return { status: result.status, stdout: result.stdout, stderr: result.stderr };
}

function withTempHome(fn: (home: string) => void): void {
	const home = mkdtempSync(join(tmpdir(), "cli-policy-test-"));
	try {
		fn(home);
	} finally {
		rmSync(home, { recursive: true, force: true });
	}
}

function writePolicy(home: string, content: string): string {
	const path = join(home, "policy.json");
	writeFileSync(path, content, "utf-8");
	return path;
}

function assertPolicyRejected(home: string, args: string[]): void {
	const { status, stderr } = runCli(args, home);
	assert.equal(status, 2, `expected early exit 2; stderr=${stderr}`);
	assert.match(stderr, /INVALID_TOOL_CALL_POLICY/, stderr);
}

test("missing --tool-call-policy argument rejects before runtime/auth", () => {
	withTempHome((home) => {
		assertPolicyRejected(home, ["--tool-call-policy"]);
	});
});

test("unreadable policy file rejects before runtime/auth", () => {
	withTempHome((home) => {
		assertPolicyRejected(home, ["--tool-call-policy", join(home, "does-not-exist.json")]);
	});
});

const INVALID_POLICIES: Array<[string, string]> = [
	...[null, true, 1, "", "FULL", " compact", [], {}].map((mode): [string, string] =>
		[`invalid visibleBudgetMode ${JSON.stringify(mode)}`, JSON.stringify({ allowedTools: [], visibleBudgetMode: mode })]),
	["malformed JSON", "{"],
	["root non-object (array)", "[]"],
	["root non-object (string)", "\"x\""],
	["root null", "null"],
	["missing allowedTools", "{}"],
	["unknown top-level key", JSON.stringify({ allowedTools: [], bogus: 1 })],
	["allowedTools not array", JSON.stringify({ allowedTools: "a" })],
	["duplicate names", JSON.stringify({ allowedTools: ["a", "a"] })],
	["blank name", JSON.stringify({ allowedTools: [""] })],
	["untrimmed name", JSON.stringify({ allowedTools: [" a"] })],
	["non-string name", JSON.stringify({ allowedTools: [1] })],
	["bool limit", JSON.stringify({ allowedTools: ["a"], maxCalls: { a: true } })],
	["fraction limit", JSON.stringify({ allowedTools: ["a"], maxCalls: { a: 1.5 } })],
	["negative limit", JSON.stringify({ allowedTools: ["a"], maxCalls: { a: -1 } })],
	["unsafe limit", JSON.stringify({ allowedTools: ["a"], maxCalls: { a: 9007199254740993 } })],
	["undeclared maxCalls key", JSON.stringify({ allowedTools: ["a"], maxCalls: { b: 1 } })],
	["bool maxTotalCalls", JSON.stringify({ allowedTools: ["a"], maxTotalCalls: true })],
	["fraction maxTotalCalls", JSON.stringify({ allowedTools: ["a"], maxTotalCalls: 0.5 })],
	["negative maxTotalCalls", JSON.stringify({ allowedTools: ["a"], maxTotalCalls: -1 })],
	["unsafe maxTotalCalls", JSON.stringify({ allowedTools: ["a"], maxTotalCalls: 9007199254740993 })],
	["undeclared exactCommands key", JSON.stringify({ allowedTools: ["a"], exactCommands: { b: ["x"] } })],
	["empty exactCommands list", JSON.stringify({ allowedTools: ["a"], exactCommands: { a: [] } })],
	["duplicate exactCommands", JSON.stringify({ allowedTools: ["a"], exactCommands: { a: ["x", "x"] } })],
	["blank exactCommand", JSON.stringify({ allowedTools: ["a"], exactCommands: { a: [""] } })],
	["non-string exactCommand", JSON.stringify({ allowedTools: ["a"], exactCommands: { a: [1] } })],
	["visibleBudget non-boolean", JSON.stringify({ allowedTools: ["a"], visibleBudget: "true" })],
	["non-standard Infinity token", JSON.stringify({ allowedTools: ["a"], maxTotalCalls: 1e999 })],
];

for (const [label, content] of INVALID_POLICIES) {
	test(`invalid policy rejects before runtime/auth: ${label}`, () => {
		withTempHome((home) => {
			const path = writePolicy(home, content);
			assertPolicyRejected(home, ["--tool-call-policy", path, "--resume", RESUME_SENTINEL]);
		});
	});
}

const VALID_POLICIES: Array<[string, string]> = [
	["explicit full", JSON.stringify({ allowedTools: [], visibleBudgetMode: "full" })],
	["compact hidden", JSON.stringify({ allowedTools: [], visibleBudgetMode: "compact", visibleBudget: false })],
	["block-all empty allowedTools", JSON.stringify({ allowedTools: [] })],
	["optional empty maps", JSON.stringify({ allowedTools: ["bash"], maxCalls: {}, exactCommands: {} })],
	["full valid policy", JSON.stringify({
		allowedTools: ["bash", "read"],
		maxCalls: { bash: 2 },
		maxTotalCalls: 3,
		exactCommands: { bash: ["ls"] },
		visibleBudget: true,
	})],
	["integral float 1.0 accepted like Number.isSafeInteger", JSON.stringify({ allowedTools: ["a"], maxCalls: { a: 1.0 }, maxTotalCalls: 2.0 })],
	// 保留字段名作为 own data key 保留——不得崩溃、不得原型污染。
	["reserved own names retained", JSON.stringify({ allowedTools: ["__proto__"], maxCalls: { __proto__: 1 } })],
];

for (const [label, content] of VALID_POLICIES) {
	test(`valid policy passes early validation and flow continues: ${label}`, () => {
		withTempHome((home) => {
			const path = writePolicy(home, content);
			// 哨兵：--resume 不存在的会话必然在 session 解析处 exit 2 并
			// 打印“不存在”——证明已越过策略校验（且无 INVALID_TOOL_CALL_POLICY）。
			const { status, stderr } = runCli(["--tool-call-policy", path, "--resume", RESUME_SENTINEL], home);
			assert.equal(status, 2, `stderr=${stderr}`);
			assert.doesNotMatch(stderr, /INVALID_TOOL_CALL_POLICY/, stderr);
			assert.match(stderr, /不存在/, stderr);
		});
	});
}

test("reserved names do not pollute prototypes", () => {
	withTempHome((home) => {
		const path = writePolicy(home, JSON.stringify({
			allowedTools: ["__proto__", "constructor"],
			maxCalls: { __proto__: 1, constructor: 2 },
		}));
		runCli(["--tool-call-policy", path, "--resume", RESUME_SENTINEL], home);
		assert.equal(({} as Record<string, unknown>).__proto__ !== 1, true);
		assert.equal(({} as Record<string, unknown>).polluted, undefined);
	});
});

test("exactPaths schema rejects early before model override/auth", () => {
	withTempHome(home => {
		for (const exactPaths of [[], null, { write: [] }, { write: [42] }, { write: [""] },
			{ write: ["../outside"] }, { write: ["file", "file"] }, { bash: ["file"] }, { read: ["file"] }]) {
			const path = writePolicy(home, JSON.stringify({ allowedTools: ["write"], exactPaths }));
			assertPolicyRejected(home, ["--tool-call-policy", path, "--provider", "invalid", "--model", "invalid"]);
		}
	});
});

test("exactPaths valid schema passes early parser without binding to policy parent", () => {
	withTempHome(home => {
		const path = writePolicy(home, JSON.stringify({ allowedTools: ["read", "write"],
			exactPaths: { read: ["missing.txt"], write: ["new/leaf.txt"] }, maxTotalCalls: 0 }));
		const { status, stderr } = runCli(["--tool-call-policy", path, "--resume", RESUME_SENTINEL], home);
		assert.equal(status, 2); assert.doesNotMatch(stderr, /INVALID_TOOL_CALL_POLICY/); assert.match(stderr, /不存在/);
	});
});

test("cross-language exactPaths lexical schema rejects all malformed path forms", () => {
	withTempHome(home => {
		for (const exactPaths of [false, { read: "file" }, { read: [false] },
			{ read: ["   "] }, { read: ["\ufeff"] }, { read: ["x\0"] },
			{ read: ["a/../x"] }, { read: ["a\\..\\x"] }, { read: ["x/"] },
			JSON.parse('{"__proto__":["file"]}')]) {
			const path = writePolicy(home, JSON.stringify({ allowedTools: ["read", "__proto__"], exactPaths }));
			assertPolicyRejected(home, ["--tool-call-policy", path, "--provider", "invalid", "--model", "invalid"]);
		}
	});
});

test("cross-language exactPaths accepts empty map and explicit lexical files before binding", () => {
	withTempHome(home => {
		for (const exactPaths of [{}, { read: ["./relative.txt", "/tmp/explicit.txt", "\u0085", " file "] }]) {
			const path = writePolicy(home, JSON.stringify({ allowedTools: ["read"], exactPaths }));
			const { status, stderr } = runCli(["--tool-call-policy", path, "--resume", RESUME_SENTINEL], home);
			assert.equal(status, 2);
			assert.doesNotMatch(stderr, /INVALID_TOOL_CALL_POLICY/);
			assert.match(stderr, /不存在/);
		}
	});
});

test("default without flag unchanged (no policy validation errors)", () => {
	withTempHome((home) => {
		const { status, stderr } = runCli(["--resume", RESUME_SENTINEL], home);
		assert.equal(status, 2, `stderr=${stderr}`);
		assert.doesNotMatch(stderr, /INVALID_TOOL_CALL_POLICY/, stderr);
		assert.match(stderr, /不存在/, stderr);
	});
});
