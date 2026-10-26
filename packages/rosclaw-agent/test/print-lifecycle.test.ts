/** Exercise the real pinned SDK print runner, without a model or robot. */
import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import {once} from "node:events";
import test from "node:test";

function fixture(body: string) {
	const result = spawnSync(process.execPath, ["--input-type=module", "-e", `
		import {runPiPrint} from './dist/src/harness/pi/pi-sessions.js';
		const phases = [];
		function session(name) { return {
			name, state: {messages: []},
			async bindExtensions() { phases.push('bind:' + this.name); },
			subscribe() { return () => phases.push('unsubscribe:' + this.name); },
			async prompt(text) { phases.push('prompt:' + this.name + ':' + text);
				this.state.messages.push({role:'assistant',stopReason:'stop',content:[{type:'text',text:'EARLY'}]}); },
		}; }
		const original = session('first');
		let current = original;
		const host = {
			get session() { return current; },
			setRebindSession(callback) { this.rebind = callback; },
			async dispose() { phases.push('dispose:' + this.session.name); },
		};
		${body}
		console.error(JSON.stringify(phases));
	`], {cwd: process.cwd(), encoding: "utf8", timeout: 5000});
	assert.equal(result.status, 0, result.stderr);
	return {stdout: result.stdout, phases: JSON.parse(result.stderr.trim().split("\n").at(-1)!) as string[]};
}

test("print settles owned work before final text and SDK disposal", () => {
	const result = fixture(`
		await runPiPrint(host, {initialMessage:'work', settleOperations:async () => {
			phases.push('settle:' + current.name);
			current.state.messages.push({role:'assistant',stopReason:'stop',content:[{type:'text',text:'FINAL'}]});
		}});
	`);
	assert.equal(result.stdout, "FINAL\n");
	assert.deepEqual(result.phases, ['bind:first','prompt:first:work','settle:first','unsubscribe:first','dispose:first']);
});

test("print binds private methods to their original receiver", () => {
	const result = fixture(`
		class Host {
			#session = original;
			get session() { return this.#session; }
			setRebindSession(callback) { this.callback = callback; }
			async dispose() { phases.push('private-dispose:' + this.#session.name); }
		}
		await runPiPrint(new Host(), {initialMessage:'work', settleOperations:async () => phases.push('settle')});
	`);
	assert.equal(result.stdout, "EARLY\n");
	assert.ok(result.phases.indexOf('settle') >= 0);
	assert.ok(result.phases.indexOf('settle') < result.phases.indexOf('private-dispose:first'));
});

test("print does not settle or replay an unsuccessful prompt", () => {
	const result = fixture(`
		original.prompt = async () => { phases.push('prompt-failed'); throw Error('TRANSPORT_UNKNOWN'); };
		const code = await runPiPrint(host, {initialMessage:'work', settleOperations:async () => phases.push('settle')});
		if (code !== 1) throw Error('failed prompt accepted');
	`);
	assert.equal(result.stdout, "");
	assert.deepEqual(result.phases, ['bind:first','prompt-failed','unsubscribe:first','dispose:first']);
});

test("print settles a rebound session without another runtime", () => {
	const result = fixture(`
		original.prompt = async () => { phases.push('switch'); current = session('second');
			await host.rebind(); await current.prompt('continuation'); };
		await runPiPrint(host, {initialMessage:'work', settleOperations:async () => {
			phases.push('settle:' + current.name);
			current.state.messages.push({role:'assistant',stopReason:'stop',content:[{type:'text',text:'REBOUND_FINAL'}]});
		}});
	`);
	assert.equal(result.stdout, "REBOUND_FINAL\n");
	assert.equal(result.phases.filter(p => p === 'settle:second').length, 1);
	assert.equal(result.phases.at(-1), 'dispose:second');
});


test("a settlement failure suppresses premature success text", () => {
    const result = fixture(`
        const code = await runPiPrint(host, {initialMessage:'work', settleOperations:async () => {
            phases.push('settle-unknown'); throw Error('OPERATION_UNKNOWN');
        }});
        if (code !== 1) throw Error('unknown outcome accepted');
    `);
    assert.equal(result.stdout, "");
    assert.deepEqual(result.phases, ['bind:first','prompt:first:work','settle-unknown','unsubscribe:first','dispose:first']);
});

test("SDK SIGTERM disposes during settlement without waiting for completion", {timeout:5000}, async () => {
    const child = spawn(process.execPath, ['--input-type=module','-e', `
        import {runPiPrint} from './dist/src/harness/pi/pi-sessions.js';
        const keepAlive = setInterval(() => {}, 1000);
        const session = {async bindExtensions(){}, subscribe(){return () => {};},
            async prompt(){}, state:{messages:[]}};
        const host = {session,setRebindSession(){},async dispose(){
            clearInterval(keepAlive); console.error('DISPOSED');
        }};
        await runPiPrint(host, {initialMessage:'work',settleOperations:async () => {
            console.error('SETTLING'); await new Promise(() => {});
        }});
    `], {cwd:process.cwd(),stdio:['ignore','pipe','pipe']});
    let output = '', diagnostics = '';
    child.stdout.on('data', data => {output += data;});
    const ready = new Promise<void>(resolve => child.stderr.on('data', data => {
        diagnostics += data;
        if (diagnostics.includes('SETTLING')) resolve();
    }));
    try {
        const ended = once(child, 'close');
        await ready;
        child.kill('SIGTERM');
        const [code, signal] = await ended;
        assert.equal(code, 143);
        assert.equal(signal, null);
        assert.match(diagnostics, /DISPOSED/);
        assert.equal(output, '');
    } finally {
        if (child.exitCode === null) child.kill('SIGKILL');
    }
});


for (const reason of ["error", "aborted"]) {
    test(`SDK-resolved ${reason} exits without settling background work`, () => {
        const result = fixture(`
            original.prompt = async () => {
                phases.push('resolved-model-failure');
                original.state.messages.push({role:'assistant',stopReason:${JSON.stringify(reason)},errorMessage:'MODEL_FAILED',content:[]});
            };
            const code = await runPiPrint(host, {initialMessage:'work',settleOperations:async () => {
                phases.push('incorrect-wait-for-background');
            }});
            if (code !== 1) throw Error('resolved model failure accepted');
        `);
        assert.equal(result.stdout, '');
        assert.deepEqual(result.phases, ['bind:first','resolved-model-failure','unsubscribe:first','dispose:first']);
    });
}

test("an SDK retry recovered before prompt return still settles its actual work", () => {
    const result = fixture(`
        original.prompt = async () => {
            original.state.messages.push({role:'assistant',stopReason:'error',errorMessage:'FIRST_ATTEMPT',content:[]});
            original.state.messages.push({role:'assistant',stopReason:'stop',content:[{type:'text',text:'RECOVERED'}]});
        };
        const code = await runPiPrint(host, {initialMessage:'work',settleOperations:async () => phases.push('settle-recovered')});
        if (code !== 0) throw Error('SDK recovery rejected');
    `);
    assert.equal(result.stdout, 'RECOVERED\n');
    assert.ok(result.phases.includes('settle-recovered'));
});
