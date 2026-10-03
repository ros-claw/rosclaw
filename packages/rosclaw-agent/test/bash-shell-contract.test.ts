import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { buildWorkspacePackTools } from "../src/tools/workspace-pack.js";

test("bash tool supports brace expansion and reports pipeline failures", async () => {
 const root = await mkdtemp(join(tmpdir(), "rosclaw-bash-contract-"));
 try {
  const bash = buildWorkspacePackTools({root, bwrapPath: () => null}).find(tool => tool.name === "bash")!;
  const expanded = await bash.execute("brace", {command: "printf '%s\\n' {left,right}"}, undefined, undefined, {} as never);
  const text = expanded.content.filter(item => item.type === "text").map(item => (item as {text:string}).text).join("\n");
  assert.match(text, /left\nright/);
  const failed = await bash.execute("pipe", {command: "exit_code=7; (exit $exit_code) | cat"}, undefined, undefined, {} as never);
  const failure = failed.content.filter(item => item.type === "text").map(item => (item as {text:string}).text).join("\n");
  assert.match(failure, /exit=7/);
 } finally { await rm(root, {recursive: true, force: true}); }
});
