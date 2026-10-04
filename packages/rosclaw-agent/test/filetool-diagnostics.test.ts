import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createReadTool } from "@earendil-works/pi-coding-agent";
import { buildWorkspacePackTools } from "../src/tools/workspace-pack.js";

test("absolute PI read then relative edit diagnoses workspace root; shell cd does not rebind", async () => {
 const base = mkdtempSync(join(tmpdir(), "filetool-diagnostic-"));
 try {
  const root = join(base, "default"), stage = join(base, "stage"); mkdirSync(root); mkdirSync(stage);
  writeFileSync(join(root, "driver.py"), "default body"); writeFileSync(join(stage, "driver.py"), "stage body");
  assert.match(JSON.stringify((await createReadTool(root).execute("read", {path: join(stage, "driver.py")})).content), /stage body/);
  const tools = buildWorkspacePackTools({root, mode: () => "SIMULATION", bwrapPath: () => null});
  const call = (name: string, params: Record<string, unknown>) => tools.find(t => t.name === name)!.execute("fixture", params as never, undefined, undefined, {} as never);
  await call("bash", {command: `cd '${stage}' && pwd`});
  const relative = await call("edit", {path: "driver.py", oldText: "stage body", newText: "replacement"});
  assert.equal(relative.isError, true); assert.equal((relative.details as Record<string, unknown>).code, "EDIT_OLD_TEXT_NOT_FOUND");
  assert.equal((relative.details as Record<string, unknown>).workspace_root, root); assert.equal((relative.details as Record<string, unknown>).resolved_target, join(root, "driver.py"));
  assert.match(JSON.stringify(relative.content), /shell.*cd.*does not change/i);
  const outside = await call("edit", {path: join(stage, "driver.py"), oldText: "stage body", newText: "replacement"});
  assert.equal((outside.details as Record<string, unknown>).code, "PATH_SCOPE_DENIED");
  assert.equal(readFileSync(join(stage, "driver.py"), "utf8"), "stage body"); assert.equal(readFileSync(join(root, "driver.py"), "utf8"), "default body");
 } finally {rmSync(base, {recursive: true, force: true});}
});

test("extra-root edits retain semantics and symlink escape remains denied", async () => {
 const root = mkdtempSync(join(tmpdir(), "filetool-root-")), scratch = mkdtempSync(join(tmpdir(), "filetool-scratch-"));
 try {
  writeFileSync(join(scratch, "driver.py"), "duplicate duplicate"); symlinkSync(scratch, join(root, "link"));
  const tools = buildWorkspacePackTools({root, extraRoots: () => [scratch]});
  const call = (name: string, params: Record<string, unknown>) => tools.find(t => t.name === name)!.execute("fixture", params as never, undefined, undefined, {} as never);
  const multiple = await call("edit", {path: join(scratch, "driver.py"), oldText: "duplicate", newText: "new"});
  assert.equal((multiple.details as Record<string, unknown>).code, "EDIT_OLD_TEXT_NOT_UNIQUE"); assert.equal((multiple.details as Record<string, unknown>).resolved_target, join(scratch, "driver.py"));
  const denied = await call("write", {path: join(root, "link/driver.py"), content: "overwrite"});
  assert.equal((denied.details as Record<string, unknown>).code, "PATH_SCOPE_DENIED"); assert.match(JSON.stringify(denied.content), /escapes workspace/);
  assert.equal(readFileSync(join(scratch, "driver.py"), "utf8"), "duplicate duplicate");
  const good = await call("edit", {path: join(scratch, "driver.py"), oldText: "duplicate duplicate", newText: "new"});
  assert.equal(good.isError, undefined); assert.equal(readFileSync(join(scratch, "driver.py"), "utf8"), "new");
 } finally {rmSync(root, {recursive: true, force: true}); rmSync(scratch, {recursive: true, force: true});}
});
