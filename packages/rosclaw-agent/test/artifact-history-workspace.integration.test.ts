import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { buildWorkspacePackTools } from "../src/tools/workspace-pack.js";

// Only a subprocess transport: real migrations, TaskKernel registration and CLI
// export. No replacement kernel, writer, exporter, daemon or provider.
const registerScript = `
import json, sqlite3, sys
from pathlib import Path
from rosclaw.storage.migrations import MigrationRunner
from rosclaw.task_kernel.service import TaskKernel
home, workspace, source = map(Path, sys.argv[1:4])
(home / 'agentd').mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(home / 'agentd' / 'missions.db')
conn.row_factory = sqlite3.Row
MigrationRunner().apply(conn, 'sqlite')
kernel = TaskKernel(conn, home)
kernel.persist_input(mission_id='history', session_ref='s', message_id='first', text='report')
bound = kernel.ensure_task_for_effect(mission_id='history', session_ref='s', backend_native_id='s', cwd=str(workspace))
record = kernel.register_artifact(task_id=str(bound['task_id']), path=str(source), media_type='text/plain', metadata={'role':'diagnostic_source'})
row = dict(conn.execute('SELECT * FROM artifacts WHERE artifact_id = ?', (record['artifact_id'],)).fetchone())
conn.commit()
conn.close()
print(json.dumps(row))
`;
const exportScript = `
import sys
from rosclaw.agentd.cli import main
raise SystemExit(main(['--home',sys.argv[1], 'artifact','export',sys.argv[2],sys.argv[3]]))
`;
type RecordRow = { artifact_id: string; task_id: string; path: string; sha256: string; size_bytes: number; metadata_json: string };

function repository(): string {
	let root = dirname(fileURLToPath(import.meta.url));
	while (!existsSync(join(root, "src/rosclaw/task_kernel/service.py"))) {
		const parent = dirname(root);
		assert.notEqual(parent, root, "source repository must be available");
		root = parent;
	}
	return root;
}

for (const method of ["edit", "write"] as const) {
	for (const detached of [false, true]) {
		test(`ordinary history: real workspace ${method}, detached=${detached}`, async () => {
			const parent = mkdtempSync(join(tmpdir(), "artifact-history-"));
			try {
				const root = join(parent, "workspace");
				const home = join(parent, "home");
				mkdirSync(root);
				const source = join(root, "report.txt");
				const a = "回球 A café\n";
				const b = "回球 B café\n";
				assert.equal(Buffer.byteLength(a), Buffer.byteLength(b));
				const python = process.env.ROSCLAW_HISTORY_PYTHON ?? process.env.ROSCLAW_TEST_PYTHON ?? process.env.PYTHON ?? "python3";
				const run = (script: string, args: string[]) => {
					const result = spawnSync(python, ["-B", "-c", script, ...args], {
						cwd: root, encoding: "utf8", timeout: 60000, maxBuffer: 1024 * 1024,
						env: { PATH: process.env.PATH, HOME: home, PYTHONPATH: join(repository(), "src"), PYTHONDONTWRITEBYTECODE: "1" },
					});
					assert.equal(result.error, undefined);
					assert.equal(result.status, 0, result.stderr + result.stdout);
					return result.stdout;
				};
				let admissions = 0;
				const tools = buildWorkspacePackTools({ root, beforeEffect: async () => { admissions++; } });
				const execute = async (name: string, params: Record<string, string>) => {
					const tool = tools.find(t => t.name === name);
					assert.ok(tool);
					const result = await tool.execute("history", params as never, undefined, undefined, {} as never);
					assert.notEqual(result.isError, true, JSON.stringify(result));
				};
				await execute("write", { path: source, content: a });
				const register = (): RecordRow => JSON.parse(run(registerScript, [home, root, source]));
				const old = register();
				assert.notEqual(resolve(old.path), resolve(source));
				assert.equal(JSON.parse(old.metadata_json).source_path, source);
				const projected = join(home, "runs", old.task_id, "r1", "outputs", "report.txt");
				assert.deepEqual(readFileSync(projected), Buffer.from(a));
				assert.notEqual(statSync(old.path).ino, statSync(projected).ino);
				if (detached) {
					const replacement = join(root, "replacement");
					writeFileSync(replacement, readFileSync(source));
					renameSync(replacement, source);
				}
				await execute(method, method === "edit"
					? { path: source, oldText: " A ", newText: " B " }
					: { path: source, content: b });
				assert.equal(admissions, 2);
				assert.deepEqual(readFileSync(source), Buffer.from(b));
				assert.deepEqual(readFileSync(old.path), Buffer.from(a));
				assert.deepEqual(readFileSync(projected), Buffer.from(a));
				const current = register();
				assert.notEqual(old.artifact_id, current.artifact_id);
				for (const [record, expected, suffix] of [[old, a, "old"], [current, b, "new"]] as const) {
					assert.equal(record.size_bytes, Buffer.byteLength(expected));
					assert.equal(record.sha256, createHash("sha256").update(expected).digest("hex"));
					const dest = join(root, `${suffix}-export.txt`);
					run(exportScript, [home, record.artifact_id, dest]);
					assert.deepEqual(readFileSync(dest), Buffer.from(expected));
				}
				assert.equal(register().artifact_id, current.artifact_id);
				writeFileSync(projected, b);
				assert.deepEqual(readFileSync(old.path), Buffer.from(a));
				unlinkSync(projected);
				unlinkSync(source);
				const dest = join(root, "old-after-delete.txt");
				run(exportScript, [home, old.artifact_id, dest]);
				assert.deepEqual(readFileSync(dest), Buffer.from(a));
			} finally {
				rmSync(parent, { recursive: true, force: true });
				assert.equal(existsSync(parent), false);
			}
		});
	}
}
