import assert from "node:assert/strict";
import test from "node:test";
import { ProviderStallWatchdog, providerWatchdogTimingFromEnv } from "../src/native/provider-watchdog.js";

test("longer provider waits keep early notices and bounded cancellation", () => {
 const timing = providerWatchdogTimingFromEnv({
  ROSCLAW_PROVIDER_FIRST_TOKEN_TIMEOUT_MS: "180000",
  ROSCLAW_PROVIDER_STREAM_IDLE_TIMEOUT_MS: "120000",
 });
 assert.deepEqual(timing, {firstTokenNoticeMs: 10000, firstTokenAbortMs: 180000,
  streamIdleStatusMs: 15000, streamIdleAbortMs: 120000});
});

test("invalid durations cannot accidentally disable or shorten the watchdog", () => {
 for (const value of ["0", "-1", "Infinity", "NaN", "60s", "999", "3600001", "1e6"]) {
  const timing = providerWatchdogTimingFromEnv({ROSCLAW_PROVIDER_FIRST_TOKEN_TIMEOUT_MS: value});
  assert.equal(timing.firstTokenAbortMs, 30000, value);
 }
});

test("notice and abort text reflect the actual configured timers", async () => {
 const notices: string[] = [];
 let canceled = 0;
 const watchdog = new ProviderStallWatchdog({notice: text => notices.push(text),
  stallAbort: () => { canceled++; }, firstTokenNoticeMs: 20, firstTokenAbortMs: 70});
 watchdog.turnStarted();
 try {
  await new Promise(resolve => setTimeout(resolve, 100));
  assert.equal(canceled, 1);
  assert.ok(notices.some(text => text.includes("0.02s") && text.includes("0.07s")));
  assert.ok(notices.some(text => text.includes("首 token 0.07s")));
 } finally { watchdog.turnEnded(); }
});

test("intermittent live content does not flood idle notices or disable abort", async () => {
 const notices: string[] = [];
 let canceled = 0;
 const watchdog = new ProviderStallWatchdog({notice: text => notices.push(text),
  stallAbort: () => { canceled++; }, firstTokenNoticeMs: 1000, firstTokenAbortMs: 2000,
  streamIdleStatusMs: 20, streamIdleAbortMs: 120});
 watchdog.turnStarted();
 try {
  for (let i = 0; i < 3; i++) {
   watchdog.contentProgress();
   await new Promise(resolve => setTimeout(resolve, 35));
  }
  assert.equal(notices.length, 1, "one slow live turn must not repeat the same warning each chunk");
  assert.equal(canceled, 0);
  await new Promise(resolve => setTimeout(resolve, 140));
  assert.equal(canceled, 1, "rate-limiting notices must not delay stall cancellation");
 } finally { watchdog.turnEnded(); }
});
