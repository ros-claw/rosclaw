/** Real subprocess fixtures: explicit timeout/cancellation must not hang on pipes. */
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import test from "node:test";
import { buildWorkspacePackTools } from "../src/tools/workspace-pack.js";

const pause = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
type Result = { content: Array<{ text: string }>; isError?: boolean };
function bash(root: string, extra: Record<string, unknown> = {}) {
 const tool = buildWorkspacePackTools({ root, rosclawHome: root,
  mode: () => "SIMULATION", bwrapPath: () => null, ...extra } as never)
  .find((tool) => tool.name === "bash")!;
 return (command: string, timeout_sec?: number, signal = new AbortController().signal) =>
  tool.execute("fixture", { command, timeout_sec }, signal, undefined, {} as never) as Promise<Result>;
}
function alive(pid: number) {
 try {
  const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
  return stat.slice(stat.lastIndexOf(")") + 1).trim().split(/\s+/)[0] !== "Z";
 } catch { return false; }
}
async function pids(root: string): Promise<number[]> {
 const path = join(root, "pids.json");
 for (let i = 0; i < 100; i++) {
  if (existsSync(path)) return JSON.parse(readFileSync(path, "utf8"));
  await pause(10);
 }
 throw new Error("Owned fixture did not start");
}
function tree(root: string, escape = false, parentExit = false) {
 const marker = basename(root);
 const script = join(root, "tree.mjs");
 writeFileSync(script, `import {spawn} from 'node:child_process';
import {writeFileSync} from 'node:fs';
const child=spawn(process.execPath,['-e','setInterval(()=>{},1000)',${JSON.stringify(marker)}],{stdio:'inherit',detached:${escape}});
writeFileSync(${JSON.stringify(join(root, "pids.json"))},JSON.stringify([process.pid,child.pid]));
${parentExit ? "process.exit(0);" : "setInterval(()=>{},1000);"}`);
 // Keep sh as a separate ancestor, reproducing the ROS CLI compound command.
 return `"${process.execPath}" "${script}"; :`;
}
function cleanup(root: string) {
 const path = join(root, "pids.json");
 if (existsSync(path)) {
  for (const pid of JSON.parse(readFileSync(path, "utf8")) as number[]) {
   if (!alive(pid)) continue;
   try {
    const cmd = readFileSync(`/proc/${pid}/cmdline`, "utf8");
    // Only our own unique temporary fixture; never signal an unrelated PID.
    if (cmd.includes(basename(root))) process.kill(pid, "SIGKILL");
   } catch { /* already exited */ }
  }
 }
 rmSync(root, { recursive: true, force: true });
}
async function bounded<T>(promise: Promise<T>, ms = 2000): Promise<T> {
 let timer: ReturnType<typeof setTimeout> | undefined;
 try { return await Promise.race([promise, new Promise<never>((_, reject) => {
  timer=setTimeout(() => reject(new Error("tool did not return within fixture bound")), ms);
 })]); } finally { if (timer) clearTimeout(timer); }
}

test("explicit timeout kills the owned shell descendants and returns promptly", {skip:process.platform!=="linux"}, async () => {
 const root=mkdtempSync(join(tmpdir(),"workspace-timeout-"));
 try {
  const result=bash(root)(tree(root),.5);const children=await pids(root);
  const out=await bounded(result);assert.match(out.content[0].text,/TIMEOUT\(explicit\)/);
  for(let i=0;i<50&&children.some(alive);i++)await pause(10);
  assert.ok(children.every(pid=>!alive(pid)),"live command descendant survived its timeout");
 } finally {cleanup(root);}
});

test("cancellation kills the owned descendants without an explicit timeout", {skip:process.platform!=="linux"}, async () => {
 const root=mkdtempSync(join(tmpdir(),"workspace-cancel-"));
 try {
  const signal=new AbortController();const result=bash(root)(tree(root),undefined,signal.signal);
  const children=await pids(root);signal.abort();const out=await bounded(result);
  assert.match(out.content[0].text,/CANCELLED/);
  for(let i=0;i<50&&children.some(alive);i++)await pause(10);
  assert.ok(children.every(pid=>!alive(pid)));
 } finally {cleanup(root);}
});

test("a deliberately escaped pipe holder returns bounded UNKNOWN, never success", {skip:process.platform!=="linux"}, async () => {
 const root=mkdtempSync(join(tmpdir(),"workspace-escaped-"));
 try {
  const result=bash(root)(tree(root,true),.5);const children=await pids(root);
  const out=await bounded(result);assert.match(out.content[0].text,/TIMEOUT\(explicit\)/);
  assert.match(out.content[0].text,/descendant_settlement=UNKNOWN/);
  assert.ok(alive(children[1]),"fixture must actually escape the original process group");
 } finally {cleanup(root);}
});

test("an exited leader is never signalled again using a potentially reused PID", {skip:process.platform!=="linux"}, async () => {
 const root=mkdtempSync(join(tmpdir(),"workspace-exited-"));
 try {
  const signal=new AbortController();const result=bash(root)(tree(root,true,true),undefined,signal.signal);
  await pids(root);await pause(100);signal.abort();const out=await bounded(result);
  assert.match(out.content[0].text,/leader_exited_UNKNOWN/);
  assert.match(out.content[0].text,/descendant_settlement=UNKNOWN/);
 } finally {cleanup(root);}
});

test("a timed out command cannot kill a concurrent independent command", {skip:process.platform!=="linux"}, async () => {
 const root=mkdtempSync(join(tmpdir(),"workspace-parallel-"));
 try {
  const run=bash(root);const failed=run(tree(root),.5);await pids(root);
  const independent=run(`"${process.execPath}" -e 'setTimeout(()=>console.log("independent-ok"),700)'`);
  assert.match((await bounded(failed)).content[0].text,/TIMEOUT/);
  const out=await bounded(independent);assert.match(out.content[0].text,/exit=0/);
  assert.match(out.content[0].text,/independent-ok/);assert.doesNotMatch(out.content[0].text,/TIMEOUT/);
 } finally {cleanup(root);}
});

test("already cancelled work creates no process or admission side effect",async()=>{
 const root=mkdtempSync(join(tmpdir(),"workspace-before-"));
 try {
  let effects=0;const signal=new AbortController();signal.abort();
  const out=await bash(root,{beforeEffect:async()=>{effects++;}})("touch forbidden",1,signal.signal);
  assert.ok(out.isError);assert.equal(effects,0);assert.ok(!existsSync(join(root,"forbidden")));
 } finally {cleanup(root);}
});

test("ordinary commands and the no-default-timeout policy remain",async()=>{
 const root=mkdtempSync(join(tmpdir(),"workspace-normal-"));
 try {
  const out=await bounded(bash(root)(`"${process.execPath}" -e 'setTimeout(()=>console.log("normal-ok"),250)'`));
  assert.match(out.content[0].text,/exit=0/);assert.match(out.content[0].text,/normal-ok/);
  assert.doesNotMatch(out.content[0].text,/TIMEOUT|CANCELLED/);
 } finally {cleanup(root);}
});

test("spawn failure returns promptly and clears the explicit timer",async()=>{
 const root=mkdtempSync(join(tmpdir(),"workspace-spawn-"));
 try {
  const out=await bounded(bash(root,{bwrapPath:()=>join(root,"nonexistent")})("echo never",10));
  assert.match(out.content[0].text,/spawn error:.*ENOENT/);
 } finally {cleanup(root);}
});
