// Native coverage for the guarded bash output boundary:
// per-stream incremental UTF8 decode, raw-byte accounting against the fixed
// 65536-byte preview cap, valid codepoint-boundary prefix, explicit
// \n[TRUNCATED notice, truthful details.truncated/originalOutputBytes/
// outputBytes/outputLimitBytes, preserved exit/signal/timeout/abort/isError,
// and real PI runToolCall retention of upstream truncation flags.
import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runToolCall } from "@earendil-works/pi-agent-core";
import { buildWorkspacePackTools } from "../src/tools/workspace-pack.js";

const LIMIT = 65536;

type BashTool = ReturnType<typeof buildWorkspacePackTools>[number];
type ToolResult = Awaited<ReturnType<BashTool["execute"]>>;

function payloadOf(result: ToolResult): { payload: string; notice: string } {
	const text = result.content
		.filter(c => c.type === "text")
		.map(c => (c as { text: string }).text)
		.join("\n");
	const pos = text.indexOf("exit=");
	assert.ok(pos >= 0, "execution header missing");
	const rendered = text.slice(text.indexOf("\n", pos) + 1);
	const split = rendered.indexOf("\n[TRUNCATED");
	return split < 0
		? { payload: rendered, notice: "" }
		: { payload: rendered.slice(0, split), notice: rendered.slice(split) };
}

async function runCase(root: string, bash: BashTool, name: string, buf: Buffer, exitCode = 0) {
	await writeFile(join(root, `${name}.txt`), buf);
	const result = await bash.execute(name, { command: `cat ${name}.txt; exit ${exitCode}`, timeout_sec: 5 }, undefined, undefined, {} as never);
	const { payload, notice } = payloadOf(result);
	const details = result.details as Record<string, unknown>;
	return { result, payload, notice, details, inputBytes: buf.length };
}

function assertValidPrefix(payload: string, buf: Buffer) {
	const payloadBytes = Buffer.byteLength(payload, "utf-8");
	assert.ok(payloadBytes <= LIMIT, "payload exceeds fixed cap");
	assert.ok(Buffer.from(payload, "utf-8").equals(buf.subarray(0, payloadBytes)), "payload is not an exact raw-byte prefix");
	assert.ok(!payload.includes(String.fromCharCode(0xfffd)), "payload contains U+FFFD replacement codepoint");
	assert.ok(!/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(payload), "payload contains lone surrogates");
}

test("bash output boundary: byte-exact completeness, truncation metadata and UTF8 prefixes", async (t) => {
	const root = await mkdtemp(join(tmpdir(), "rosclaw-bash-output-"));
	try {
		const bash = buildWorkspacePackTools({ root, bwrapPath: () => null }).find(tool => tool.name === "bash")!;

		await t.test("probe", async () => {
			const buf = Buffer.from("a".repeat(65536));
			const { result, payload, notice, details } = await runCase(root, bash, "probe_ascii", buf);
			console.log("PROBE", JSON.stringify({
				pl: payload.length, pb: Buffer.byteLength(payload), nt: notice.length, nb: Buffer.byteLength(notice),
				tr: details.truncated, ob: details.outputBytes, oob: details.originalOutputBytes,
				eq: payload === buf.toString("utf-8"), isErr: result.isError, ec: details.exitCode,
				ns: notice.slice(0, 40),
			}));
			const buf2 = Buffer.from("a".repeat(65537));
			const r2 = await runCase(root, bash, "probe_ascii2", buf2);
			console.log("PROBE2", JSON.stringify({
				pl: r2.payload.length, pb: Buffer.byteLength(r2.payload), nb: Buffer.byteLength(r2.notice),
				tr: r2.details.truncated, ob: r2.details.outputBytes, oob: r2.details.originalOutputBytes,
				pref: Buffer.from(r2.payload).equals(buf2.subarray(0, Buffer.byteLength(r2.payload))),
				ns: r2.notice.slice(0, 40),
			}));
		});

		await t.test("ASCII 65535/65536 complete, 65537 truncated with truthful details", async () => {
			for (const [name, size, truncated] of [["ascii_minus", 65535, false], ["ascii_equal", 65536, false], ["ascii_plus", 65537, true]] as const) {
				const buf = Buffer.from("a".repeat(size));
				const { result, payload, notice, details } = await runCase(root, bash, name, buf);
				assert.equal(details.exitCode, 0);
				assert.equal(result.isError, false);
				assert.equal(details.truncated, truncated, `${name}: truncated flag`);
				assert.equal(details.originalOutputBytes, size, `${name}: originalOutputBytes`);
				assert.equal(details.outputBytes, Buffer.byteLength(payload, "utf-8"), `${name}: outputBytes`);
				assert.equal(details.outputLimitBytes, LIMIT, `${name}: outputLimitBytes`);
				assertValidPrefix(payload, buf);
				if (truncated) {
					assert.ok(notice.startsWith("\n[TRUNCATED"), "truncation notice must start with \\n[TRUNCATED");
					assert.ok(Buffer.byteLength(notice, "utf-8") <= 512, "truncation notice exceeds 512 bytes");
					assert.notEqual(payload, buf.toString("utf-8"));
				} else {
					assert.equal(notice, "", "complete output must have no notice");
					assert.equal(payload, buf.toString("utf-8"), "complete output must be byte-exact");
				}
			}
		});

		await t.test("truncation preserves exit code 7 and does not hide the failure", async () => {
			const buf = Buffer.from("a".repeat(65537));
			const { result, details } = await runCase(root, bash, "exit7", buf, 7);
			assert.equal(details.exitCode, 7);
			assert.equal(result.isError, true, "actual nonzero exit must stay a failure");
			assert.equal(details.truncated, true);
			assert.equal(details.originalOutputBytes, 65537);
		});

		await t.test("multibyte boundaries: retained prefix never cuts a codepoint", async () => {
			for (const [name, buf, truncated] of [
				["utf8_equal_bytes", Buffer.from("a".repeat(65532) + "😀"), false],
				["utf8_cross_bytes", Buffer.from("a".repeat(65534) + "😀"), true],
				["utf8_surrogate_cross", Buffer.from("a".repeat(65535) + "😀"), true],
				["chinese_equal_bytes", Buffer.from("a" + "中".repeat(21845)), false],
				["chinese_plus_bytes", Buffer.from("中".repeat(21846)), true],
			] as const) {
				const { payload, notice, details } = await runCase(root, bash, name, buf);
				assert.equal(details.truncated, truncated, `${name}: truncated flag`);
				assert.equal(details.originalOutputBytes, buf.length, `${name}: originalOutputBytes`);
				assert.equal(details.outputBytes, Buffer.byteLength(payload, "utf-8"), `${name}: outputBytes`);
				assert.equal(details.outputLimitBytes, LIMIT);
				assertValidPrefix(payload, buf);
				if (truncated) assert.ok(notice.startsWith("\n[TRUNCATED"));
				else assert.equal(payload, buf.toString("utf-8"));
			}
		});

		await t.test("timed split multibyte writes decode as one character per stream", async () => {
			for (const stream of ["stdout", "stderr"] as const) {
				const emitter = `process.${stream}.write(Buffer.from([0xe4,0xb8])); setTimeout(()=>process.${stream}.write(Buffer.from([0xad])),40);`;
				await writeFile(join(root, `split_${stream}.mjs`), emitter);
				const result = await bash.execute(`split_${stream}`, { command: `node split_${stream}.mjs`, timeout_sec: 5 }, undefined, undefined, {} as never);
				const { payload, notice } = payloadOf(result);
				const details = result.details as Record<string, unknown>;
				assert.equal(payload, "中", `${stream}: split UTF8 character must survive chunking`);
				assert.equal(notice, "");
				assert.equal(details.truncated, false);
				assert.equal(details.originalOutputBytes, 3);
				assert.equal(details.outputBytes, 3);
				assert.equal(details.outputLimitBytes, LIMIT);
				assert.equal(details.exitCode, 0);
			}
		});

		await t.test("dropped codepoint closes retention: later bytes never skip past it", async () => {
			for (const stream of ["stdout", "stderr"] as const) {
				const emitter =
					`const w=(b)=>new Promise(r=>process.${stream}.write(b,r));` +
					`(async()=>{ await w("a".repeat(65535)); await w(Buffer.from([0xf0,0x9f,0x98,0x80])); ` +
					`await new Promise(r=>setTimeout(r,60)); await w("z"); })();`;
				await writeFile(join(root, `gap_${stream}.mjs`), emitter);
				const result = await bash.execute(`gap_${stream}`, { command: `node gap_${stream}.mjs`, timeout_sec: 5 }, undefined, undefined, {} as never);
				const { payload, notice } = payloadOf(result);
				const details = result.details as Record<string, unknown>;
				assert.equal(details.truncated, true, `${stream}: truncated flag`);
				assert.equal(details.originalOutputBytes, 65535 + 4 + 1, `${stream}: all raw bytes counted`);
				assert.equal(details.outputBytes, 65535, `${stream}: retained payload bytes`);
				assert.equal(details.outputLimitBytes, LIMIT);
				assert.equal(payload, "a".repeat(65535), `${stream}: preview is the exact raw prefix ending before the dropped codepoint`);
				assert.ok(!payload.includes("z"), `${stream}: later byte must not skip past the dropped codepoint`);
				assert.ok(!payload.includes("�"), `${stream}: no U+FFFD`);
				assert.ok(notice.startsWith("\n[TRUNCATED"));
				assert.ok(Buffer.byteLength(notice, "utf-8") <= 512);
			}
		});

		await t.test("cross-stream split Chinese codepoints survive via independent decoders", async () => {
			const emitter =
				`const so=(b)=>new Promise(r=>process.stdout.write(b,r));` +
				`const se=(b)=>new Promise(r=>process.stderr.write(b,r));` +
				`(async()=>{` +
				`await so(Buffer.from([0xe4,0xb8])); await se(Buffer.from([0xe4]));` +
				`await new Promise(r=>setTimeout(r,50));` +
				`await so(Buffer.from([0xad])); await se(Buffer.from([0xb8,0xad]));` +
				`})();`;
			await writeFile(join(root, "cross_split.mjs"), emitter);
			const result = await bash.execute("cross_split", { command: "node cross_split.mjs", timeout_sec: 5 }, undefined, undefined, {} as never);
			const { payload, notice } = payloadOf(result);
			const details = result.details as Record<string, unknown>;
			assert.equal(details.truncated, false);
			assert.equal(details.originalOutputBytes, 6, "six raw bytes across both streams");
			assert.equal(details.outputBytes, 6);
			assert.equal(details.outputLimitBytes, LIMIT);
			assert.equal(notice, "");
			assert.ok(!payload.includes("�"), "no U+FFFD from split codepoints");
			// No intrinsic global chronology claim: compare the codepoint multiset.
			const codepoints = [...payload].sort();
			assert.deepEqual(codepoints, ["中", "中"], "each stream's split codepoint decoded intact");
		});

		await t.test("oversized 70KB JSON is partial and never presented as complete", async () => {
			const buf = Buffer.from(JSON.stringify({ schema: "NATIVE_FIXTURE", items: "a".repeat(70000) }));
			const { payload, notice, details } = await runCase(root, bash, "json70k", buf);
			assert.equal(details.truncated, true);
			assert.equal(details.originalOutputBytes, buf.length);
			assert.ok(Buffer.byteLength(payload, "utf-8") <= LIMIT);
			assert.ok(notice.startsWith("\n[TRUNCATED"));
			assert.throws(() => JSON.parse(payload), "partial JSON preview must not parse as complete JSON");
			assertValidPrefix(payload, buf);
		});

		await t.test("timeout and abort terminal state preserved with output metadata", async () => {
			const timeoutResult = await bash.execute("timeout", { command: "printf early; sleep 60", timeout_sec: 0.05 }, undefined, undefined, {} as never);
			const td = timeoutResult.details as Record<string, unknown>;
			assert.equal(td.timedOut, true);
			assert.equal(td.aborted, false);
			assert.equal(timeoutResult.isError, true);
			assert.equal(td.truncated, false);
			assert.equal(td.outputLimitBytes, LIMIT);

			const control = new AbortController();
			const pending = bash.execute("abort", { command: "sleep 60" }, control.signal, undefined, {} as never);
			setTimeout(() => control.abort(), 40);
			const abortResult = await pending;
			const ad = abortResult.details as Record<string, unknown>;
			assert.equal(ad.aborted, true);
			assert.equal(ad.timedOut, false);
			assert.equal(abortResult.isError, true);
			assert.equal(ad.truncated, false);
		});

		await t.test("real PI runToolCall retains upstream truncation flags and native metadata", async () => {
			const wrapped = {
				...bash,
				execute: async (id: string, args: unknown, signal?: AbortSignal, onUpdate?: Parameters<BashTool["execute"]>[3]) => {
					const r = await bash.execute(id, args, signal, onUpdate, {} as never);
					return { ...r, details: { ...(r.details as Record<string, unknown>), truncated: true, truncation: { truncated: true, source: "SYNTHETIC_UPSTREAM_NATIVE_FIXTURE" } } };
				},
			};
			const sdk = await runToolCall(
				{ type: "toolCall", id: "sdk_trunc", name: "bash", arguments: { command: "printf ok", timeout_sec: 5 } },
				{
					context: { messages: [] },
					tools: [wrapped],
					assistantMessage: { role: "assistant", content: [], api: "openai-responses", provider: "fixture", model: "fixture", usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: "toolUse", timestamp: 0 },
				},
			);
			const details = sdk.result.details as Record<string, unknown>;
			assert.equal(details.truncated, true, "upstream true truncation flag must survive the PI dispatcher");
			assert.deepEqual(details.truncation, { truncated: true, source: "SYNTHETIC_UPSTREAM_NATIVE_FIXTURE" });

			const plain = await runToolCall(
				{ type: "toolCall", id: "sdk_native", name: "bash", arguments: { command: `cat ascii_plus.txt`, timeout_sec: 5 } },
				{
					context: { messages: [] },
					tools: [{ ...bash, execute: (id: string, args: unknown, signal?: AbortSignal, onUpdate?: Parameters<BashTool["execute"]>[3]) => bash.execute(id, args, signal, onUpdate, {} as never) }],
					assistantMessage: { role: "assistant", content: [], api: "openai-responses", provider: "fixture", model: "fixture", usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: "toolUse", timestamp: 0 },
				},
			);
			const nd = plain.result.details as Record<string, unknown>;
			assert.equal(nd.truncated, true, "native truncation metadata must reach the PI ToolResult");
			assert.equal(nd.originalOutputBytes, 65537);
			assert.equal(nd.outputLimitBytes, LIMIT);
			assert.equal(plain.result.isError, false, "truncation alone must not fail a successful exit");
		});
		await t.test("initial UTF8 BOM is ordinary output data and is retained per stream", async () => {
			const BOM = Buffer.from([0xef, 0xbb, 0xbf]);
			// small: BOM + 1 raw byte on each stream
			for (const stream of ["stdout", "stderr"] as const) {
				const buf = Buffer.concat([BOM, Buffer.from("a")]);
				await writeFile(join(root, `bom_small_${stream}.mjs`), `process.${stream}.write(Buffer.from(${JSON.stringify([...buf])}));`);
				const result = await bash.execute(`bom_small_${stream}`, { command: `node bom_small_${stream}.mjs`, timeout_sec: 5 }, undefined, undefined, {} as never);
				const { payload, notice } = payloadOf(result);
				const details = result.details as Record<string, unknown>;
				assert.equal(details.truncated, false, `${stream}: 4 raw bytes is complete`);
				assert.equal(details.originalOutputBytes, 4);
				assert.equal(details.outputBytes, 4);
				assert.equal(notice, "");
				assert.equal(payload, buf.toString("utf-8"), `${stream}: BOM + payload byte-exact`);
				assert.ok(payload.startsWith("﻿"), `${stream}: leading BOM retained`);
			}
			// split: BOM bytes delivered across timed chunks must survive streaming decode
			for (const stream of ["stdout", "stderr"] as const) {
				const emitter = `process.${stream}.write(Buffer.from([0xef])); setTimeout(()=>process.${stream}.write(Buffer.from([0xbb,0xbf,0x61])),40);`;
				await writeFile(join(root, `bom_split_${stream}.mjs`), emitter);
				const result = await bash.execute(`bom_split_${stream}`, { command: `node bom_split_${stream}.mjs`, timeout_sec: 5 }, undefined, undefined, {} as never);
				const { payload, notice } = payloadOf(result);
				const details = result.details as Record<string, unknown>;
				assert.equal(details.truncated, false);
				assert.equal(details.originalOutputBytes, 4);
				assert.equal(details.outputBytes, 4);
				assert.equal(notice, "");
				assert.equal(payload, "﻿a", `${stream}: split initial BOM decodes as one character`);
				assert.ok(!payload.includes("�"), `${stream}: no U+FFFD from split BOM`);
			}
			// equal/above cap: BOM counts toward raw bytes; 65536 exact is complete, 65537 is a byte-prefix
			for (const stream of ["stdout", "stderr"] as const) {
				const bufEqual = Buffer.concat([BOM, Buffer.from("a".repeat(65536 - 3))]);
				const eq = await runCase(root, bash, `bom_equal_${stream}`, bufEqual);
				assert.equal(eq.details.truncated, false, `${stream}: exactly 65536 raw bytes is not truncation`);
				assert.equal(eq.details.originalOutputBytes, 65536);
				assert.equal(eq.details.outputBytes, 65536);
				assert.equal(eq.notice, "");
				assert.equal(eq.payload, bufEqual.toString("utf-8"), `${stream}: whole payload including BOM`);
				const bufAbove = Buffer.concat([BOM, Buffer.from("a".repeat(65537 - 3))]);
				const ab = await runCase(root, bash, `bom_above_${stream}`, bufAbove);
				assert.equal(ab.details.truncated, true);
				assert.equal(ab.details.originalOutputBytes, 65537);
				assert.equal(ab.details.outputBytes, 65536);
				assert.ok(ab.notice.startsWith("\n[TRUNCATED"));
				assertValidPrefix(ab.payload, bufAbove);
				assert.ok(ab.payload.startsWith("﻿"), `${stream}: truncated prefix retains the leading BOM`);
			}
		});
	} finally {
		await rm(root, { recursive: true, force: true });
	}
});
