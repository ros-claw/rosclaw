/** Real compiled main + SDK editor dispatcher on a PTY. No exit interception,
 * no replacement UI.run/shutdown, no provider or robot connection. */
import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { SessionManager } from "@earendil-works/pi-coding-agent";

const main = fileURLToPath(new URL("../src/main.js", import.meta.url));
const sdk = import.meta.resolve("@earendil-works/pi-coding-agent");
const ownerModule = new URL("../src/harness/pi/session-writer-ownership.js", import.meta.url).href;

// Observations wrap real methods, with explicit fault seams only on close.
// In particular SDK run(), shutdown dispatch, drainInput and stop stay real.
const observer = `
import { appendFileSync, writeFileSync } from 'node:fs';
import { InteractiveMode, AgentSession } from ${JSON.stringify(sdk)};
import { SessionWriterOwnership } from ${JSON.stringify(ownerModule)};
const root = process.env.QUIT_TEST_ROOT;
const fault = process.env.QUIT_TEST_FAULT;
const record = (kind, extra = {}) => appendFileSync(root + '/events.jsonl', JSON.stringify({kind, ...extra}) + '\\n');
globalThis.fetch = async () => { record('NETWORK_REFUSED'); throw new Error('NO_PROVIDER_IN_QUIT_TEST'); };
let current;
let disposed = false;
const idle = Object.getOwnPropertyDescriptor(AgentSession.prototype, 'isIdle');
Object.defineProperty(AgentSession.prototype, 'isIdle', { ...idle, get() {
  return fault === 'idle' && disposed ? false : idle.get.call(this);
}});
const wait = AgentSession.prototype.waitForIdle;
AgentSession.prototype.waitForIdle = function(...args) {
  // This fixture injects an UNKNOWN idle observation, not a live agent. Keep
  // local cancellation measurable without manufacturing idle confirmation.
  if (fault === 'idle' && disposed) return this.agent.waitForIdle();
  return wait.apply(this, args);
};
const release = SessionWriterOwnership.prototype.releaseAll;
SessionWriterOwnership.prototype.releaseAll = function() {
  record('release', { idle: current?.session.isIdle, claims: this.claimCount });
  return release.call(this);
};
const stop = InteractiveMode.prototype.stop;
InteractiveMode.prototype.stop = function(...args) {
  const result = stop.apply(this, args);
  record('stop');
  return result;
};
const init = InteractiveMode.prototype.init;
InteractiveMode.prototype.init = async function(...args) {
  await init.apply(this, args);
  current = this;
  const session = this.session;
  for (const name of ['abort', 'dispose']) {
    const original = AgentSession.prototype[name];
    AgentSession.prototype[name] = function(...args) {
      record(name);
      if (fault === name) {
        if (name === 'abort') return Promise.reject(new Error('SYNTHETIC_ABORT_REJECT'));
        throw new Error('SYNTHETIC_DISPOSE_REJECT');
      }
      const result = original.apply(this, args);
      const confirmed = () => {
        record(name + '_confirmed', { idle: this.isIdle });
        if (name === 'dispose') disposed = true;
      };
      // Preserve dispose's synchronous SDK contract, including in host cleanup.
      if (result?.then) return result.then(value => { confirmed(); return value; });
      confirmed();
      return result;
    };
  }
  const host = this.runtimeHost;
  const dispose = host.dispose.bind(host);
  host.dispose = async (...args) => {
    record('runtimeDispose');
    const result = await dispose(...args);
    if (fault === 'runtime') throw new Error('SYNTHETIC_RUNTIME_REJECT');
    record('runtime_confirmed', { idle: session.isIdle });
    return result;
  };
  const drain = this.ui.terminal.drainInput.bind(this.ui.terminal);
  this.ui.terminal.drainInput = async (...args) => {
    record('drain');
    const result = await drain(...args);
    if (fault === 'terminal') throw new Error('SYNTHETIC_TERMINAL_REJECT');
    record('drain_confirmed');
    return result;
  };
  writeFileSync(root + '/ready', this.sessionManager.getSessionFile());
};
`;

// Finite supervisor: readiness is an actual SDK init observation, not a sleep.
// Every timeout is a failure; always reap the sole child and close the PTY.
const ptySupervisor = `
import os, sys, pty, subprocess, time, select, pathlib, json, signal, socket, threading
root = pathlib.Path(sys.argv[1]); main = sys.argv[2]; source = sys.argv[3]
# Private offline bridge fixture, not an agentd or any robot transport.
# Unknown RPCs fail closed; only synthetic session bookkeeping is accepted.
run = root/'home/run'; run.mkdir()
server = socket.socket(socket.AF_UNIX); server.bind(str(run/'pi-bridge.sock')); server.listen(); server.settimeout(.1)
stopping = threading.Event()
def serve():
 while not stopping.is_set():
  try: conn, _ = server.accept()
  except socket.timeout: continue
  with conn:
   conn.settimeout(1); buf = b''
   try:
    while b'\\n' not in buf: buf += conn.recv(65536)
    method = json.loads(buf.split(b'\\n')[0])['method']
    response = {'ok': False, 'code': 'OFFLINE_FIXTURE_UNAVAILABLE'}
    if method == 'pi.session.binding.get': response = {'ok': True, 'binding': None}
    elif method == 'pi.mission.create': response = {'ok': True, 'mission_id': 'synthetic_quit_mission'}
    elif method == 'pi.session.bind': response = {'ok': True, 'binding': {'binding_id': 'synthetic_quit_binding'}, 'lease_token': 'synthetic_fixture_only'}
    elif method in ['pi.session.release', 'pi.session.heartbeat']: response = {'ok': True}
    conn.sendall((json.dumps(response) + '\\n').encode())
   except (OSError, ValueError): pass
thread = threading.Thread(target=serve); thread.start()
master, slave = pty.openpty()
p = subprocess.Popen([os.environ['QUIT_TEST_NODE'], '--import', str(root/'observer.mjs'), main, '--workspace', str(root/'workspace'), '--resume-path', source], stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
os.close(slave)
output = bytearray(); sent = False; before = None; claim = None; deadline = time.monotonic() + 9
try:
 while p.poll() is None and time.monotonic() < deadline:
  if not sent and (root/'ready').exists():
   before = pathlib.Path(source).read_bytes(); claim = pathlib.Path(source + '.owner.json').read_bytes()
   os.write(master, b'/quit\\r'); sent = True
  if select.select([master], [], [], .02)[0]:
   try: output.extend(os.read(master, 65536))
   except OSError: pass
 timedout = p.poll() is None
 if timedout: os.killpg(p.pid, signal.SIGKILL)
 code = p.wait(timeout=2)
 while select.select([master], [], [], 0)[0]:
  try:
   data = os.read(master, 65536)
   if not data: break
   output.extend(data)
  except OSError: break
 lock = pathlib.Path(source + '.owner.json')
 print(json.dumps({'code': code, 'timeout': timedout, 'sent': sent, 'unchanged': before is not None and pathlib.Path(source).read_bytes() == before, 'claimBefore': claim is not None, 'claimAfter': lock.exists(), 'claimUnchanged': claim is not None and lock.exists() and lock.read_bytes() == claim, 'failureTyped': b'MAIN_EXIT_TEARDOWN_UNCONFIRMED' in output, 'tail': output[-3000:].decode(errors='replace')}))
finally:
 if p.poll() is None:
  os.killpg(p.pid, signal.SIGKILL); p.wait(timeout=2)
 os.close(master)
 stopping.set(); thread.join(timeout=2); server.close()
`;

for (const fault of ["none", "abort", "dispose", "idle", "runtime", "terminal"]) {
	test(`actual InteractiveMode /quit: ${fault}`, { timeout: 15000 }, () => {
		const root = mkdtempSync(join(tmpdir(), "rosclaw-interactive-quit-"));
		try {
			const agent = join(root, "home", "agent");
			const workspace = join(root, "workspace");
			mkdirSync(agent, { recursive: true });
			mkdirSync(workspace);
			writeFileSync(join(agent, "auth.json"), JSON.stringify({ "kimi-coding": { type: "api_key", key: "SYNTHETIC_NO_REQUEST" } }), { mode: 0o600 });
			writeFileSync(join(agent, "settings.json"), JSON.stringify({ defaultProvider: "kimi-coding", defaultModel: "kimi-for-coding", defaultThinkingLevel: "low", quietStartup: true, retry: { enabled: false }, compaction: { enabled: false } }));
			const source = SessionManager.create(workspace, join(agent, "sessions"));
			source.appendModelChange("kimi-coding", "kimi-for-coding");
			source.appendThinkingLevelChange("low");
			source.appendMessage({ role: "user", content: "SYNTHETIC_QUIT_HISTORY_NOT_TO_REPLAY", timestamp: 1 });
			source.appendMessage({ role: "assistant", content: [{ type: "text", text: "STOP" }], api: "anthropic-messages", provider: "kimi-coding", model: "kimi-for-coding", usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } }, stopReason: "stop", timestamp: 2 });
			const path = source.getSessionFile()!;
			const foreign = join(agent, "sessions", "foreign.jsonl.owner.json");
			const foreignBytes = '{"token":"UNKNOWN_KEEP","pid":1,"birth":"unknown"}\n';
			writeFileSync(foreign, foreignBytes, { mode: 0o600 });
			writeFileSync(join(root, "observer.mjs"), observer);
			const env = Object.fromEntries(Object.entries(process.env).filter(([key]) => !/KEY|TOKEN|PASSWORD|SECRET/.test(key) && !["NODE_OPTIONS", "ROSCLAW_HOME", "PI_CODING_AGENT_DIR", "SSH_AUTH_SOCK"].includes(key)));
			const result = spawnSync("python3", ["-B", "-c", ptySupervisor, root, main, path], {
				encoding: "utf8", timeout: 12000, maxBuffer: 1024 * 1024,
				env: { ...env, HOME: join(root, "home"), ROSCLAW_HOME: join(root, "home"), PI_CODING_AGENT_DIR: agent, PI_OFFLINE: "1", TERM: "xterm-256color", QUIT_TEST_NODE: process.execPath, QUIT_TEST_ROOT: root, QUIT_TEST_FAULT: fault },
			});
			assert.equal(result.error, undefined);
			assert.equal(result.status, 0, result.stderr);
			const receipt = JSON.parse(result.stdout.trim());
			assert.equal(receipt.timeout, false, receipt.tail);
			assert.equal(receipt.sent, true, receipt.tail);
			assert.equal(receipt.claimBefore, true);
			assert.equal(receipt.unchanged, true);
			assert.equal(receipt.code, fault === "none" ? 0 : 2, receipt.tail);
			assert.equal(readFileSync(foreign, "utf8"), foreignBytes);
			assert.equal(existsSync(join(root, "events.jsonl")), true);
			const events = readFileSync(join(root, "events.jsonl"), "utf8").trim().split("\n").map(line => JSON.parse(line));
			const kinds = events.map(event => event.kind);
			assert.equal(kinds.includes("NETWORK_REFUSED"), false);
			assert.ok(kinds.includes("stop"), "actual SDK terminal consumer must stop even on failure");
			if (fault === "none") {
				assert.equal(receipt.claimAfter, false);
				for (const [earlier, later] of [["drain_confirmed", "stop"], ["stop", "abort"], ["abort_confirmed", "dispose"], ["dispose_confirmed", "runtimeDispose"], ["runtime_confirmed", "release"]]) {
					assert.ok(kinds.indexOf(earlier) >= 0 && kinds.indexOf(earlier) < kinds.indexOf(later), `${earlier} must precede ${later}: ${kinds}`);
				}
				const releases = events.filter(event => event.kind === "release");
				assert.equal(releases.length, 1);
				assert.equal(releases[0].idle, true);
			} else {
				assert.equal(receipt.failureTyped, true, receipt.tail);
				assert.equal(receipt.claimUnchanged, true, "failed teardown must retain exact own claim");
				assert.equal(kinds.includes("release"), false);
			}
		} finally { rmSync(root, { recursive: true, force: true }); }
	});
}
