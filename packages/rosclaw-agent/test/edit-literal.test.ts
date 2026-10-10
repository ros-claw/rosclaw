import assert from "node:assert/strict";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { buildWorkspacePackTools } from "../src/tools/workspace-pack.js";
import { buildWorkbenchTools } from "../src/workers/workbench.js";

type Result = {
	isError?: boolean;
	content: Array<{ type: string; text?: string }>;
	details: Record<string, unknown>;
};
type Edit = { execute: (...args: any[]) => Promise<Result> };
type Factory = "workspace_pack" | "workbench";

function fixture(factory: Factory) {
	const parent = mkdtempSync(join(tmpdir(), "literal-edit-"));
	const root = join(parent, "workspace");
	const outside = join(parent, "outside");
	mkdirSync(root);
	mkdirSync(outside);
	const log = join(root, "artifacts", "bash-log.txt");
	const records: Array<{ kind: string; payload: Record<string, unknown> }> = [];
	let admissions = 0;
	const tools = factory === "workspace_pack"
		? buildWorkspacePackTools({ root, beforeEffect: async () => { admissions++; } })
		: buildWorkbenchTools({
			root, bashLogPath: log,
			emitRecord: (kind, payload) => { records.push({ kind, payload }); },
		});
	const edit = tools.find(tool => tool.name === "edit") as Edit | undefined;
	assert.ok(edit, "use the genuine production edit tool");
	const execute = (path: string, oldText: string, newText: string) => edit.execute(
		"literal-edit-test",
		factory === "workspace_pack"
			? { path, oldText, newText }
			: { path, old_text: oldText, new_text: newText },
		undefined, undefined, {} as never,
	);
	return {
		parent, root, outside, log, records, execute,
		admissions: () => admissions,
		cleanup: () => {
			rmSync(parent, { recursive: true, force: true });
			assert.equal(existsSync(parent), false, "private test tree removed");
		},
	};
}

// Expected bytes are concatenated independently; no substitute replacement
// implementation, patched factory, SDK mock, shell, or provider is used.
const literals: Array<[string, string]> = [
	["dollar ampersand", "$&"],
	["dollar prefix", "$`"],
	["dollar suffix", "$'"],
	["double dollar", "$$"],
	["capture-like dollars", "$1 $2 $99 $<name>"],
	["regex end quote", "^case_[0-9]+$'"],
	["combined tokens", "$$|$&|$`|$'|$1|$<name>|$$$&"],
	["empty deletion", ""],
	["Unicode", "回球 🎾 café e\u0301 \uFEFF"],
	["multiline", "first\r\nsecond\n第三行\n$& $$ $` $'\n"],
];

for (const factory of ["workspace_pack", "workbench"] as const) {
	for (const [name, newText] of literals) {
		test(`${factory}: literal bytes for ${name}`, async () => {
			const f = fixture(factory);
			try {
				const path = join(f.root, "owned.txt");
				const prefix = "PREFIX 🎾\n";
				const suffix = "\nSUFFIX café\n";
				const oldText = "OLD_SENTINEL";
				writeFileSync(path, prefix + oldText + suffix, "utf-8");
				const result = await f.execute("owned.txt", oldText, newText);
				assert.notEqual(result.isError, true);
				assert.equal(result.details.path, path);
				assert.deepEqual(readFileSync(path), Buffer.from(prefix + newText + suffix, "utf-8"));
				if (factory === "workspace_pack") {
					assert.equal(f.admissions(), 1, "admission still runs before edit");
				} else {
					assert.equal(result.details.error, null);
					assert.equal(readFileSync(f.log, "utf-8"), `edit ${path}\n`);
					assert.deepEqual(f.records, [{ kind: "file_change", payload: {
						op: "edit", path: "owned.txt", old_bytes: oldText.length, new_bytes: newText.length,
					} }]);
				}
			} finally {
				f.cleanup();
			}
		});
	}

	for (const [name, text, oldText, workspaceCode, workbenchError] of [
		["missing", "PREFIX OLD SUFFIX", "absent", "EDIT_OLD_TEXT_NOT_FOUND", "not_found"],
		["multiple", "OLD and OLD", "OLD", "EDIT_OLD_TEXT_NOT_UNIQUE", "ambiguous"],
		["empty oldText", "nonempty original", "", "EDIT_OLD_TEXT_EMPTY", "ambiguous"],
	] as const) {
		test(`${factory}: ${name} denied without changing bytes or success audit`, async () => {
			const f = fixture(factory);
			try {
				const path = join(f.root, "owned.txt");
				writeFileSync(path, text, "utf-8");
				const before = readFileSync(path);
				const result = await f.execute("owned.txt", oldText, "$& $$ $` $'");
				assert.equal(result.isError, true);
				assert.deepEqual(readFileSync(path), before);
				assert.equal(result.details[factory === "workspace_pack" ? "code" : "error"],
					factory === "workspace_pack" ? workspaceCode : workbenchError);
				assert.deepEqual(f.records, []);
				assert.equal(existsSync(f.log), false);
				if (factory === "workspace_pack") assert.equal(f.admissions(), 1);
			} finally {
				f.cleanup();
			}
		});
	}

	for (const escape of ["relative root", "absolute root", "symlink"] as const) {
		test(`${factory}: ${escape} escape denied and owned outside bytes preserved`, async () => {
			const f = fixture(factory);
			try {
				const outsidePath = join(f.outside, "owned.txt");
				writeFileSync(outsidePath, "PREFIX OLD SUFFIX", "utf-8");
				const before = readFileSync(outsidePath);
				symlinkSync(f.outside, join(f.root, "link-out"), "dir");
				const target = escape === "relative root" ? "../outside/owned.txt"
					: escape === "absolute root" ? outsidePath : "link-out/owned.txt";
				const result = await f.execute(target, "OLD", "$& $$");
				assert.equal(result.isError, true);
				assert.equal(result.details.error, "denied");
				assert.match(result.content[0].text ?? "", /escapes workspace/);
				if (factory === "workspace_pack") {
					assert.equal(result.details.code, "PATH_SCOPE_DENIED");
					assert.equal(result.details.workspace_root, f.root);
					assert.equal(result.details.requested_path, target);
					assert.equal(f.admissions(), 1);
				}
				assert.deepEqual(readFileSync(outsidePath), before);
				assert.deepEqual(f.records, []);
				assert.equal(existsSync(f.log), false);
			} finally {
				f.cleanup();
			}
		});
	}
}
