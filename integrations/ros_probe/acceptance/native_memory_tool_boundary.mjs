/** Actual Native read/workspace tools inside an independently confined container.
 * No model, TaskKernel admission, MCP route or causal benefit is asserted.
 * Run only via the operator's frozen, per-group Docker mount manifest.
 */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { createReadTool } from '/native/node_modules/@earendil-works/pi-coding-agent/dist/index.js';
import { buildWorkspacePackTools } from '/native/dist/src/tools/workspace-pack.js';

const mode = process.argv[2];
assert(['M0', 'M1', 'M2'].includes(mode));
const output = '/workspace';
const source = '/projected/data/memory/projection.json';
const text = result => result.content.filter(x => x.type === 'text').map(x => x.text).join('\n');
const original = {};
const status = readFileSync('/proc/self/status', 'utf8');
const fields = Object.fromEntries(status.split('\n').filter(x => x.includes(':')).map(x => x.split(/:\s*/, 2)));
assert.equal(process.getuid(), 1000);
assert.equal(BigInt('0x' + fields.CapEff), 0n);
assert.equal(fields.NoNewPrivs, '1');
const net = readFileSync('/proc/net/dev', 'utf8');
assert.deepEqual(net.split('\n').filter(x => x.includes(':')).map(x => x.split(':')[0].trim()), ['lo']);
writeFileSync(output + '/original-proc-status.txt', status);
writeFileSync(output + '/original-proc-mountinfo.txt', readFileSync('/proc/self/mountinfo'));
writeFileSync(output + '/original-proc-net-dev.txt', net);

const read = createReadTool(output);
original.allowedRead = await read.execute('allowed-projection', { path: source });
const projection = JSON.parse(text(original.allowedRead));
assert.equal(projection.mode, mode);
assert.equal(projection.authorization, false);
assert.equal(projection.physical_acceptance, 'NOT_VERIFIED');
assert(!text(original.allowedRead).includes('historical source advice'));
if (mode === 'M0') assert.deepEqual(projection.items, []);
else {
  assert.equal(projection.items.length, 1);
  assert.equal(projection.items[0].facts.failure_code, 'INACTIVE');
  assert.equal(Object.hasOwn(projection.items[0], 'repair_pattern'), mode === 'M2');
}

const spec = JSON.parse(readFileSync('/probe-spec.json', 'utf8'));
assert.equal(spec.mode, mode);
assert(spec.forbidden_paths.length >= 6);
original.forbiddenReads = [];
for (const path of spec.forbidden_paths) {
  let failure;
  try { await read.execute('forbidden-read', { path }); }
  catch (error) { failure = String(error); }
  assert(failure && /ENOENT|EACCES/.test(failure), 'OS denial required: ' + path);
  original.forbiddenReads.push({ path, error: failure });
}

let beforeEffectCalls = 0;
const tools = buildWorkspacePackTools({
  root: output, mode: () => 'SIMULATION', rosclawHome: '/empty-home',
  defaultBashTimeoutMs: 5000,
  beforeEffect: async () => { beforeEffectCalls++; },
});
const tool = name => tools.find(x => x.name === name);
original.write = await tool('write').execute('allowed-write', { path: 'owned.txt', content: 'before' });
assert.equal(readFileSync(output + '/owned.txt', 'utf8'), 'before');
original.edit = await tool('edit').execute('allowed-edit', { path: 'owned.txt', oldText: 'before', newText: 'after' });
assert.equal(readFileSync(output + '/owned.txt', 'utf8'), 'after');
original.outsideWrite = await tool('write').execute('outside-write', { path: source, content: 'invalid' });
original.outsideEdit = await tool('edit').execute('outside-edit', { path: source, oldText: mode, newText: 'invalid' });
assert.equal(original.outsideWrite.isError, true);
assert.equal(original.outsideEdit.isError, true);

// This Python child is spawned by the actual Native main-session bash tool.
// No raw shell substitute is used to claim Native tool isolation.
const python = `import json,os,socket,errno
s=json.load(open('/probe-spec.json'))
results=[]
for p in s['forbidden_paths']:
 try: open(p,'rb').read(); raise AssertionError('forbidden readable: '+p)
 except OSError as e:
  assert e.errno in (errno.ENOENT,errno.EACCES)
  results.append({'path':p,'errno':e.errno})
try:
 open('/projected/data/memory/projection.json','ab').write(b'bad')
 raise AssertionError('readonly projection writable')
except OSError as e:
 assert e.errno in (errno.EROFS,errno.EACCES)
 denied_write=e.errno
c=socket.socket(); c.settimeout(1)
try:
 c.connect(('198.51.100.1',443)); raise AssertionError('external network available')
except OSError as e:
 assert e.errno in (errno.ENETUNREACH,errno.EHOSTUNREACH,errno.EACCES,errno.EPERM)
 network_errno=e.errno
finally: c.close()
print('NATIVE_CHILD_BOUNDARY='+json.dumps({'uid':os.getuid(),'denied':results,'readonly_errno':denied_write,'network_errno':network_errno}))
`;
const command = "python3 - <<'NATIVE_BOUNDARY_PY'\n" + python + '\nNATIVE_BOUNDARY_PY';
original.bash = await tool('bash').execute('native-bash-boundary', { command, timeout_sec: 5 });
const bashText = text(original.bash);
const marker = bashText.split('\n').find(x => x.startsWith('NATIVE_CHILD_BOUNDARY='));
assert(marker, 'actual Native bash child did not complete boundary checks: ' + bashText);
const child = JSON.parse(marker.slice('NATIVE_CHILD_BOUNDARY='.length));
assert.equal(child.uid, 1000);
assert.equal(child.denied.length, spec.forbidden_paths.length);
assert.deepEqual(JSON.parse(readFileSync(source, 'utf8')), projection);
assert.equal(beforeEffectCalls, 5);
writeFileSync(output + '/original-native-tool-responses.json', JSON.stringify(original, null, 2) + '\n');
const review = {
  status: 'PASS_ACTUAL_NATIVE_FILE_AND_SHELL_CONTAINER_BOUNDARY_SMOKE', mode,
  Node_version: process.version, actual_read_tool: 'Pi SDK createReadTool',
  actual_effect_tools: 'Native buildWorkspacePackTools bash/write/edit',
  shell_reports_tool_layer_only: bashText.includes('TOOL_LAYER_ONLY'),
  outer_container_boundary_checked: true, forbidden_read_paths: spec.forbidden_paths.length,
  own_projection_readonly: true, own_workspace_write_edit: true,
  separate_group_mounts: 'REQUIRES_OPERATOR_DOCKER_INSPECT_REVIEW',
  full_Native_worker_isolation: 'NOT_VERIFIED', MCP_route: 'NOT_TESTED_HERE',
  model_called: false, TaskKernel_admission: 'NOT_TESTED_HERE',
  World_started: false, robot_action_started: false, causal_benefit: 'NOT_MEASURED',
};
writeFileSync(output + '/review.json', JSON.stringify(review, null, 2) + '\n');
console.log(JSON.stringify(review));
