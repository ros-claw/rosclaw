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
