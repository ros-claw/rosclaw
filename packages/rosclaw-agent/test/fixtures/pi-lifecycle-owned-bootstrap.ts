// Deliberately no SDK/product imports: tools-manager caches getBinDir at evaluation.
import { createHash } from "node:crypto";
import { accessSync, chmodSync, constants, copyFileSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { isAbsolute, join } from "node:path";

const pins = {
 rg: { path: "/home/nvidia/.codex/packages/standalone/releases/0.162.0-aarch64-unknown-linux-musl/codex-path/rg", size: 4506992, sha256: "e36d0eb52e70696bdf1781392722e05a21bb91d3b7b762ef5ec20e5df2ec687b" },
 fd: { path: "/home/nvidia/.local/bin/fd", size: 3544856, sha256: "b96a9c5eb1f619efe836e7d5176a9aa15d6643f11c46cda6709ff8abb43eeed4" },
} as const;
const keys = ["PI_LIFECYCLE_OWNED_HOME", "PI_CODING_AGENT_DIR", "PI_OFFLINE", "PI_TELEMETRY", "ROSCLAW_HOME", "ROSCLAW_CONTROL_TOKEN"] as const;
type Env = Record<string, string | undefined>;
function directory(path: string, mode: number) {
 if (!isAbsolute(path) || realpathSync(path) !== path) throw new Error("PI_BOOTSTRAP_CANONICAL_DIRECTORY");
 const st = lstatSync(path);
 if (!st.isDirectory() || st.isSymbolicLink() || (st.mode & 0o777) !== mode || st.uid !== process.getuid?.()) throw new Error("PI_BOOTSTRAP_OWNED_DIRECTORY");
}
function binary(path: string, pin: { size: number; sha256: string }) {
 const st = lstatSync(path);
 if (!st.isFile() || st.isSymbolicLink() || st.size !== pin.size || createHash("sha256").update(readFileSync(path)).digest("hex") !== pin.sha256) throw new Error("PI_BOOTSTRAP_REAL_TOOL_PIN");
 accessSync(path, constants.X_OK);
}

// Only explicit system directories or an explicit caller source; never PATH or downloads.
function inspectLocalTool(source: string, supplied: boolean) {
 if (process.platform !== "linux" || !isAbsolute(source)) throw new Error("PI_BOOTSTRAP_LOCAL_PLATFORM_OR_PATH");
 const canonical = realpathSync(source);
 if (supplied && canonical !== source) throw new Error("PI_BOOTSTRAP_LOCAL_CANONICAL_TOOL");
 const st = lstatSync(canonical);
 if (!st.isFile() || st.isSymbolicLink() || (st.mode & 0o022) !== 0 ||
     (st.uid !== 0 && st.uid !== process.getuid?.())) throw new Error("PI_BOOTSTRAP_LOCAL_UNSAFE_TOOL");
 accessSync(canonical, constants.X_OK);
 const bytes = readFileSync(canonical);
 const machine = process.arch === "x64" ? 62 : process.arch === "arm64" ? 183 : 0;
 if (!machine || bytes.length < 64 || !bytes.subarray(0, 4).equals(Buffer.from([0x7f, 0x45, 0x4c, 0x46])) ||
     bytes[4] !== 2 || bytes[5] !== 1 || bytes[6] !== 1 || bytes.readUInt32LE(20) !== 1 ||
     ![2, 3].includes(bytes.readUInt16LE(16)) || bytes.readUInt16LE(18) !== machine || bytes.length !== st.size)
  throw new Error("PI_BOOTSTRAP_LOCAL_ELF_REQUIRED");
 return { source, canonical, size: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") };
}
function discoverLocalTool(name: "rg" | "fd", supplied: string | undefined) {
 if (supplied !== undefined) return inspectLocalTool(supplied, true);
 const names = name === "fd" ? ["fd", "fdfind"] : ["rg"];
 for (const directory of ["/usr/bin", "/usr/local/bin", "/bin"]) {
  for (const executable of names) {
   const source = join(directory, executable);
   try { return inspectLocalTool(source, false); }
   catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code !== "ENOENT" && code !== "ENOTDIR") throw error;
   }
  }
 }
 throw new Error(`PI_BOOTSTRAP_LOCAL_TOOL_MISSING:${name}`);
}

// A separate env map lets focused tests exercise cleanup without changing a live SDK.
export function establishOwnedFixture(env: Env, sources?: Record<keyof typeof pins, string>) {
 const previous = Object.fromEntries(keys.map(key => [key, env[key]]));
 const suppliedHome = env.PI_LIFECYCLE_OWNED_HOME;
 const suppliedAgent = env.PI_CODING_AGENT_DIR;
 if (!!suppliedHome !== !!suppliedAgent) throw new Error("PI_BOOTSTRAP_PARTIAL_WORKER_ENV");
 let home: string;
 let agentDir: string;
 const workerOwned = !!suppliedHome;
 if (suppliedHome && suppliedAgent) {
  home = suppliedHome; agentDir = suppliedAgent;
  directory(home, 0o700);
  if (agentDir !== join(home, "agent")) throw new Error("PI_BOOTSTRAP_EXACT_WORKER_AGENT");
  directory(agentDir, 0o700); directory(join(home, "run"), 0o700); directory(join(agentDir, "bin"), 0o700);
  for (const name of ["rg", "fd"] as const) binary(join(agentDir, "bin", name), pins[name]);
 } else {
  // Validate platform tools BEFORE allocating; explicit caller sources are also validated.
  const discovered = { rg: discoverLocalTool("rg", sources?.rg), fd: discoverLocalTool("fd", sources?.fd) };
  // Never use worker TMPDIR, PATH, a downloader or a fake/version-only executable.
  home = mkdtempSync(join(realpathSync("/tmp"), "pi-life-")); chmodSync(home, 0o700);
  agentDir = join(home, "agent");
  try {
   for (const path of [join(home, "run"), agentDir, join(agentDir, "bin")]) mkdirSync(path, { mode: 0o700 });
   for (const name of ["rg", "fd"] as const) {
    const tool = discovered[name], target = join(agentDir, "bin", name);
    copyFileSync(tool.canonical, target, constants.COPYFILE_EXCL);
    chmodSync(target, 0o555); binary(target, tool);
   }
   writeFileSync(join(agentDir, "tool-provenance.json"), JSON.stringify(discovered) + "\n", { mode: 0o600, flag: "wx" });
  } catch (error) { rmSync(home, { recursive: true, force: true }); throw error; }
 }
 env.PI_LIFECYCLE_OWNED_HOME = home; env.PI_CODING_AGENT_DIR = agentDir;
 env.PI_OFFLINE = "1"; env.PI_TELEMETRY = "0";
 if (!workerOwned) { env.ROSCLAW_HOME = home; env.ROSCLAW_CONTROL_TOKEN = "SYNTH_PI_LIFECYCLE_local_source_fixture"; }
 let disposed = false;
 return { home, agentDir, workerOwned, dispose() {
  if (disposed) return;
  // Worker owns its files/archival; only locally-created roots may be removed here.
  if (!workerOwned) rmSync(home, { recursive: true, force: true });
  for (const key of keys) { if (previous[key] === undefined) delete env[key]; else env[key] = previous[key]; }
  disposed = true;
 } };
}

// First side-effect import in BOTH modules; synchronous setup finishes before any SDK import.
export const lifecycleFixture = establishOwnedFixture(process.env);
// Isolated default-npm execution may load only the original module. Local roots
// are cleaned on exit too; worker-owned roots are never removed by this helper.
process.once("exit", () => lifecycleFixture.dispose());
let users = 0;
export function retainLifecycleFixture() { users++; }
export function releaseLifecycleFixture() {
 if (users <= 0) throw new Error("PI_BOOTSTRAP_UNBALANCED_RELEASE");
 if (--users === 0) lifecycleFixture.dispose();
}
