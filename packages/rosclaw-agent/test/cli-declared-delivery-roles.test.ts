/** Source-only authored regressions. Root must build before running; no SDK/model execution. */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import test from "node:test";
import { fileURLToPath, pathToFileURL } from "node:url";

const mainJs = join(dirname(fileURLToPath(import.meta.url)), "..", "src", "main.js");
const BOUNDARY = "OFFLINE_SESSION_IMPORT_BOUNDARY";
const CAPTURE = "OFFLINE_RUNTIME_ARGUMENT:";

// A strict import firewall, not a replacement CLI or extracted loader function.
// main.js and its policy validators are loaded unchanged from the real build.
// Every post-policy dependency is inert; unexpected imports fail closed.
const loaderSource = String.raw`
import { pathToFileURL } from 'node:url';
let main;
export function initialize(data) { main = data.main; }
const stubs = {
 './ui-owned-cancellation.js': 'export function attachOwnedUIAbort() { throw Error("UNEXPECTED_UI"); }',
 './harness/pi/pi-sessions.js': 'console.log("OFFLINE_SESSION_IMPORT_BOUNDARY"); export const continueRecentPiSession=undefined, resolveContinuationTarget=undefined, listAllPiSessions=undefined, listPiSessions=undefined, openPiSession=undefined, runPiPrint=undefined, forkPiSessionForOverride=undefined, inspectOverrideSource=undefined;',
 './harness/pi/session-writer-ownership.js': 'export class SessionWriterOwnership {} export function isSessionInUse() { return false; }',
 './harness/pi/pi-runtime.js': 'export function createRosclawRuntime(options) { console.log("OFFLINE_RUNTIME_ARGUMENT:" + JSON.stringify(options.toolCallBudget ?? null)); throw Error("OFFLINE_CONTROLLED_STOP"); } export const resolveExplicitModel=undefined, requireSupportedThinking=undefined;',
 './session/workspace.js': 'export class WorkspaceStore { current=undefined; bind() { throw Error("UNEXPECTED_BIND"); } }',
 '@earendil-works/pi-coding-agent': 'export const SettingsManager=undefined;',
 './native/active-task-context.js': 'export function resolveTaskContext(o) { return { workspaceRoot:o.cwd, workspaceSource:"offline-stub" }; }',
};
export async function resolve(specifier, context, next) {
 if (context.parentURL === main && Object.hasOwn(stubs, specifier)) {
  return { url:'data:text/javascript,' + encodeURIComponent(stubs[specifier]), shortCircuit:true };
 }
 if (specifier.startsWith('node:')) {
  if (!['node:fs','node:path','node:crypto'].includes(specifier)) throw Error('OFFLINE_IMPORT_DENIED:' + specifier);
  return next(specifier, context);
 }
 const url = new URL(specifier, context.parentURL || main).href;
 const allowed = [main, new URL('./version.js',main).href,
  new URL('./harness/pi/tool-call-budget.js',main).href,
  new URL('./harness/pi/model-request-admission.js',main).href];
 if (!allowed.includes(url)) throw Error('OFFLINE_IMPORT_DENIED:' + specifier);
 return next(specifier, context);
}
`;

// Independent of PI_OFFLINE and of the module firewall: deny network/process APIs.
const guardSource = String.raw`
import { register, syncBuiltinESMExports } from 'node:module';
import net from 'node:net';
import tls from 'node:tls';
import http from 'node:http';
import https from 'node:https';
import dgram from 'node:dgram';
import dns from 'node:dns';
import cp from 'node:child_process';
const deny = () => { throw Error('OFFLINE_EFFECT_DENIED'); };
net.connect=net.createConnection=net.Socket.prototype.connect=net.Server.prototype.listen=deny;
tls.connect=http.request=http.get=https.request=https.get=dgram.createSocket=deny;
dns.lookup=dns.resolve=deny;
for (const key of ['spawn','spawnSync','exec','execSync','execFile','execFileSync','fork']) cp[key]=deny;
globalThis.fetch=deny;
syncBuiltinESMExports();
register(new URL('./loader.mjs',import.meta.url), { data:{ main:process.env.ROLE_TEST_MAIN_URL } });
`;

function runPolicy(policy: unknown, omitFlag = false) {
 assert.ok(existsSync(mainJs), `Missing compiled CLI ${mainJs}; build package first (no skip).`);
 const home = mkdtempSync(join(tmpdir(), "cli-delivery-roles-"));
 try {
  const path = join(home, "policy.json");
  writeFileSync(path, JSON.stringify(policy));
  writeFileSync(join(home, "loader.mjs"), loaderSource);
  writeFileSync(join(home, "guard.mjs"), guardSource);
  const result = spawnSync(process.execPath, ["--import", pathToFileURL(join(home, "guard.mjs")).href,
   mainJs, ...(omitFlag ? [] : ["--tool-call-policy", path])], {
   cwd: home, encoding: "utf8", timeout: 30_000,
   // No inherited credentials, NODE_OPTIONS, proxy or provider settings.
   env: { HOME: home, ROSCLAW_HOME: home, PI_CODING_AGENT_DIR: join(home, "agent"),
    PI_OFFLINE: "1", PI_SKIP_VERSION_CHECK: "1", ROLE_TEST_MAIN_URL: pathToFileURL(mainJs).href },
  });
  assert.ifError(result.error);
  assert.equal(result.signal, null, `unexpected signal: ${result.stderr}`);
  return result;
 } finally { rmSync(home, { recursive: true, force: true }); }
}

function policyFor(roles: string[]) {
 const paths = roles.map((_, i) => `diagnostics/report-${i}.json`);
 return { allowedTools: ["rosclaw_deliver"], exactPaths: { rosclaw_deliver: paths },
  declaredDeliveryRoles: paths.map((path, i) => ({ path, role: roles[i] })),
  maxCalls: { rosclaw_deliver: 5 }, maxTotalCalls: 5, visibleBudget: false };
}

function assertAccepted(policy: unknown, omitFlag = false): void {
 const result = runPolicy(policy, omitFlag);
 assert.equal(result.status, 2, result.stderr);
 assert.doesNotMatch(result.stderr, /INVALID_TOOL_CALL_POLICY|OFFLINE_IMPORT_DENIED|OFFLINE_EFFECT_DENIED/);
 assert.match(result.stderr, /OFFLINE_CONTROLLED_STOP/);
 assert.ok(result.stdout.includes(BOUNDARY), result.stdout);
 const captures = result.stdout.split(/\r?\n/).filter(line => line.startsWith(CAPTURE));
 assert.equal(captures.length, 1, result.stdout);
 // This is the actual createRosclawRuntime argument, not a reread of policy.json.
 // Full equality catches dropped/rewritten roles and changes to other policy fields.
 assert.deepEqual(JSON.parse(captures[0].slice(CAPTURE.length)), omitFlag ? null : policy);
}

test("real CLI preserves five diagnostic_report declarations through loader to offline runtime boundary", () => {
 assertAccepted(policyFor(Array(5).fill("diagnostic_report")));
});

test("real CLI preserves all role forms and 128-entry upper boundary", () => {
 assertAccepted(policyFor(["report", "plot", "image", "video", "data", "progress", "diagnostic",
  "progress_step_1", "diagnostic_A_9", "diagnostic_" + "a".repeat(117)]));
 assertAccepted(policyFor(Array(128).fill("diagnostic_report")));
});

test("omitted roles and omitted policy retain legacy loader behavior", () => {
 assertAccepted({ allowedTools: [] });
 const { declaredDeliveryRoles: _roles, ...legacy } = policyFor(["report"]);
 assertAccepted(legacy);
 assertAccepted(null, true);
});

const invalidRoles: Array<[string, unknown]> = [
 ["null", null], ["object", {}], ["string", "report"], ["number", 1], ["boolean", false],
 ["empty", []], ["129 entries", Array.from({ length: 129 }, (_, i) => ({ path: `p${i}`, role: "report" }))],
 ...[null, false, 1, "entry", [], {}, { path: "diagnostics/report-0.json" }, { role: "report" },
  { path: "diagnostics/report-0.json", role: "report", extra: 1 }].map((entry, i): [string, unknown] => [`bad entry ${i}`, [entry]]),
 ...[null, false, 1, "", " REPORT", "Report", "general", "diagnostic-foo", "progress_", "report_extra",
  "diagnostic_é", "report\n", "diagnostic_" + "a".repeat(118)].map((role, i): [string, unknown] =>
  [`bad role ${i}`, [{ path: "diagnostics/report-0.json", role }]]),
 ...[null, false, 1, "", "   ", "\ufeff", "x\0", "../x", "a/../x", "a\\..\\x", "x/"].map((path, i): [string, unknown] =>
  [`bad path ${i}`, [{ path, role: "report" }]]),
 ["duplicate", [{ path: "diagnostics/report-0.json", role: "report" }, { path: "diagnostics/report-0.json", role: "data" }]],
];

function assertRejected(policy: unknown): void {
 const result = runPolicy(policy);
 assert.equal(result.status, 2, result.stderr);
 assert.match(result.stderr, /INVALID_TOOL_CALL_POLICY/);
 assert.doesNotMatch(result.stdout, /OFFLINE_SESSION_IMPORT_BOUNDARY|OFFLINE_RUNTIME_ARGUMENT:/);
 assert.doesNotMatch(result.stderr, /OFFLINE_CONTROLLED_STOP|OFFLINE_IMPORT_DENIED|OFFLINE_EFFECT_DENIED/);
}

for (const [label, declaredDeliveryRoles] of invalidRoles) {
 test(`real CLI rejects before first post-loader import: ${label}`, () => {
  const policy = policyFor(["report"]);
  // Give the over-limit case otherwise-valid lexical membership.
  if (label === "129 entries") policy.exactPaths.rosclaw_deliver = Array.from({ length: 129 }, (_, i) => `p${i}`);
  assertRejected({ ...policy, declaredDeliveryRoles });
 });
}

test("real CLI rejects declarations without allowed delivery and exact delivery paths", () => {
 const policy = policyFor(["report"]);
 for (const candidate of [
  { ...policy, allowedTools: [] },
  { ...policy, exactPaths: undefined },
  { ...policy, exactPaths: {} },
  { ...policy, allowedTools: ["read"], exactPaths: { read: ["diagnostics/report-0.json"] } },
 ]) assertRejected(candidate);
});

test("real CLI defers dot-relative membership without rewriting declarations", () => {
 assertAccepted({ allowedTools: ["rosclaw_deliver"],
  exactPaths: { rosclaw_deliver: ["./x.json"] },
  declaredDeliveryRoles: [{ path: "x.json", role: "diagnostic_report" }] });
});

test("real CLI passes foreign lexical path unchanged to inert runtime, not constructor authorization", () => {
 const policy = policyFor(["report"]);
 assertAccepted({ ...policy, declaredDeliveryRoles: [{ path: "foreign.json", role: "report" }] });
});

test("real CLI defers relative/absolute canonical membership in both directions", () => {
 assert.ok(existsSync(mainJs), `Missing compiled CLI ${mainJs}; build package first (no skip).`);
 const home = mkdtempSync(join(tmpdir(), "cli-role-canonical-deferral-"));
 try {
  writeFileSync(join(home, "loader.mjs"), loaderSource);
  writeFileSync(join(home, "guard.mjs"), guardSource);
  const absolute = join(home, "x.json");
  for (const [exact, declared] of [[absolute, "x.json"], ["x.json", absolute]]) {
   const policy = { allowedTools: ["rosclaw_deliver"],
    exactPaths: { rosclaw_deliver: [exact] },
    declaredDeliveryRoles: [{ path: declared, role: "diagnostic_report" }] };
   const path = join(home, "policy.json");
   writeFileSync(path, JSON.stringify(policy));
   const result = spawnSync(process.execPath, ["--import", pathToFileURL(join(home, "guard.mjs")).href,
    mainJs, "--tool-call-policy", path], {
    cwd: home, encoding: "utf8", timeout: 30_000,
    env: { HOME: home, ROSCLAW_HOME: home, PI_CODING_AGENT_DIR: join(home, "agent"),
     PI_OFFLINE: "1", PI_SKIP_VERSION_CHECK: "1", ROLE_TEST_MAIN_URL: pathToFileURL(mainJs).href },
   });
   assert.ifError(result.error);
   assert.equal(result.signal, null, `unexpected signal: ${result.stderr}`);
   assert.equal(result.status, 2, result.stderr);
   assert.doesNotMatch(result.stderr, /INVALID_TOOL_CALL_POLICY|OFFLINE_IMPORT_DENIED|OFFLINE_EFFECT_DENIED/);
   assert.match(result.stderr, /OFFLINE_CONTROLLED_STOP/);
   assert.ok(result.stdout.includes(BOUNDARY), result.stdout);
   const captures = result.stdout.split(/\r?\n/).filter(line => line.startsWith(CAPTURE));
   assert.equal(captures.length, 1, result.stdout);
   assert.deepEqual(JSON.parse(captures[0].slice(CAPTURE.length)), policy);
  }
 } finally { rmSync(home, { recursive: true, force: true }); }
});

// These are Stage A pass-through assertions only, not canonical authorization:
// runtime is deliberately inert. Existing real budget-constructor tests own
// canonical membership, alias duplicates and symlink enforcement.

