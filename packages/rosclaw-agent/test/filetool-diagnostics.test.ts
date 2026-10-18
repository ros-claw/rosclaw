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

test("empty oldText is rejected before any mutation; file unchanged", async () => {
 const base = mkdtempSync(join(tmpdir(), "filetool-empty-"));
 try {
  const root = join(base, "root"); mkdirSync(root);
  writeFileSync(join(root, "a.txt"), "ab");
  const tools = buildWorkspacePackTools({root, mode: () => "SIMULATION", bwrapPath: () => null});
  const call = (name: string, params: Record<string, unknown>) => tools.find(t => t.name === name)!.execute("fixture", params as never, undefined, undefined, {} as never);
  const res = await call("edit", {path: "a.txt", oldText: "", newText: "X"});
  assert.equal(res.isError, true);
  assert.equal((res.details as Record<string, unknown>).code, "EDIT_OLD_TEXT_EMPTY");
  assert.equal(readFileSync(join(root, "a.txt"), "utf8"), "ab");
 } finally {rmSync(base, {recursive: true, force: true});}
});

test("write reports UTF-8 byte length, not UTF-16 code units", async () => {
 const base = mkdtempSync(join(tmpdir(), "filetool-utf8-"));
 try {
  const root = join(base, "root"); mkdirSync(root);
  const tools = buildWorkspacePackTools({root, mode: () => "SIMULATION", bwrapPath: () => null});
  const call = (name: string, params: Record<string, unknown>) => tools.find(t => t.name === name)!.execute("fixture", params as never, undefined, undefined, {} as never);
  const res = await call("write", {path: "u.txt", content: "汉字"});
  assert.equal(res.isError, undefined);
  assert.match(JSON.stringify(res.content), /\(6 bytes\)/);
  assert.equal(readFileSync(join(root, "u.txt"), "utf8"), "汉字");
 } finally {rmSync(base, {recursive: true, force: true});}
});

test("edit reports complete UTF-8 bytes across literal and sequential CRLF edits", async () => {
 const root = mkdtempSync(join(tmpdir(), "filetool-edit-bytes-"));
 try {
  const path = join(root, "u.txt");
  writeFileSync(path, "汉字😀\r\nTARGET\r\n尾", "utf8");
  const edit = buildWorkspacePackTools({root}).find(t => t.name === "edit")!;
  const literal = "$&|$$|$`|$'|中文🚀";
  const steps = [
   {oldText: "TARGET", newText: literal, expected: `汉字😀\r\n${literal}\r\n尾`},
   {oldText: literal, newText: "短", expected: "汉字😀\r\n短\r\n尾"},
   {oldText: "汉字😀\r\n短\r\n尾", newText: "", expected: ""},
  ];
  const counts: number[] = [];
  for (const step of steps) {
   const result = await edit.execute("bytes", {path: "u.txt", oldText: step.oldText, newText: step.newText} as never, undefined, undefined, {} as never);
   assert.equal(result.isError, undefined);
   const bytes = readFileSync(path);
   assert.deepEqual(bytes, Buffer.from(step.expected, "utf8"));
   const expectedBytes = Buffer.byteLength(step.expected, "utf8");
   assert.equal(bytes.length, expectedBytes);
   assert.deepEqual(result.details, {path, writtenBytes: expectedBytes});
   assert.deepEqual(result.content, [{type: "text", text: `edited ${path} (${expectedBytes} UTF-8 bytes written)`}]);
   counts.push(expectedBytes);
  }
  assert.ok(counts[0] > counts[1]);
  assert.ok(counts[1] > counts[2]);
  assert.equal(counts[2], 0);
 } finally {rmSync(root, {recursive: true, force: true});}
});

test("rejected edits keep bytes unchanged and never report writtenBytes", async () => {
 const base = mkdtempSync(join(tmpdir(), "filetool-edit-reject-"));
 try {
  const root = join(base, "root"), outside = join(base, "outside");
  mkdirSync(root); mkdirSync(outside);
  const original = Buffer.from("汉😀\r\ndup dup\r\n", "utf8");
  const path = join(root, "u.txt"), external = join(outside, "u.txt");
  writeFileSync(path, original); writeFileSync(external, original);
  symlinkSync(outside, join(root, "escape"));
  const edit = buildWorkspacePackTools({root}).find(t => t.name === "edit")!;
  for (const fixture of [
   {path: "u.txt", oldText: "", code: "EDIT_OLD_TEXT_EMPTY", occurrences: 0},
   {path: "u.txt", oldText: "absent", code: "EDIT_OLD_TEXT_NOT_FOUND", occurrences: 0},
   {path: "u.txt", oldText: "dup", code: "EDIT_OLD_TEXT_NOT_UNIQUE", occurrences: 2},
   {path: external, oldText: "dup dup", code: "PATH_SCOPE_DENIED"},
   {path: "escape/u.txt", oldText: "dup dup", code: "PATH_SCOPE_DENIED"},
   {path: "missing.txt", oldText: "dup dup", code: "FILE_IO_ERROR"},
  ]) {
   const result = await edit.execute("reject", {path: fixture.path, oldText: fixture.oldText, newText: "changed"} as never, undefined, undefined, {} as never);
   assert.equal(result.isError, true);
   const details = result.details as Record<string, unknown>;
   assert.equal(details.code, fixture.code);
   if (fixture.occurrences !== undefined) assert.equal(details.occurrences, fixture.occurrences);
   assert.equal(Object.hasOwn(details, "writtenBytes"), false);
   assert.doesNotMatch(JSON.stringify(result.content), /UTF-8 bytes written/);
   assert.deepEqual(readFileSync(path), original);
   assert.deepEqual(readFileSync(external), original);
  }
  assert.throws(() => readFileSync(join(root, "missing.txt")), {code: "ENOENT"});
 } finally {rmSync(base, {recursive: true, force: true});}
});

test("beforeEffect refusal propagates without editing or returning byte success", async () => {
 const root = mkdtempSync(join(tmpdir(), "filetool-edit-admission-"));
 try {
  const path = join(root, "u.txt"), original = Buffer.from("汉😀\r\nunique", "utf8");
  writeFileSync(path, original);
  const refusal = new Error("fixture admission refused");
  let admissions = 0;
  const edit = buildWorkspacePackTools({root, beforeEffect: async () => { admissions++; throw refusal; }}).find(t => t.name === "edit")!;
  await assert.rejects(
   edit.execute("refused", {path: "u.txt", oldText: "unique", newText: "replacement"} as never, undefined, undefined, {} as never),
   error => error === refusal,
  );
  assert.equal(admissions, 1);
  assert.deepEqual(readFileSync(path), original);
 } finally {rmSync(root, {recursive: true, force: true});}
});
