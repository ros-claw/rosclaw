import assert from "node:assert/strict";
import test from "node:test";
import { redact } from "../src/redact.js";

test("redacts known secrets verbatim", () => {
	const key = "sk-kimi-abcdef1234567890";
	assert.equal(redact(`Authorization failed for ${key}`, [key]), "Authorization failed for <redacted>");
});

test("redacts sk- patterns even when unknown", () => {
	assert.equal(redact("token sk-abcdefghij leaked"), "token <redacted> leaked");
});

test("redacts bearer tokens and key=value shapes", () => {
	assert.equal(redact("header Bearer abcdefgh12345"), "header <redacted>");
	const out = redact('api_key="supersecretvalue"');
	assert.ok(!out.includes("supersecretvalue"));
});

test("short strings are untouched", () => {
	assert.equal(redact("plain error message"), "plain error message");
});
