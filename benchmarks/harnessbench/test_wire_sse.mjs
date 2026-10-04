import assert from 'node:assert/strict';
import test from 'node:test';
import { SseMetadataParser, observeSseMetadata } from './wire_sse.mjs';

for (const newline of ['\n', '\r\n', '\r']) {
  test(`metadata across every byte with ${JSON.stringify(newline)} framing`, async () => {
    const events = [];
    const s = '\uFEFF: keepalive' + newline +
      'event: message_start' + newline + 'data:{"type":"message_start",' + newline +
      'data: "message":{"model":"k3","content":"SECRET_🎾"}}' + newline + newline +
      'data: {"type":"content_block_delta","delta":{"text":"SECRET_PROMPT"}}' + newline + newline +
      'data: {"type":"message_stop"}' + newline + newline + 'data: [DONE]' + newline + newline;
    const bytes = new TextEncoder().encode(s);
    const body = new ReadableStream({start(controller) {
      for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
      controller.close();
    }});
    const summary = await observeSseMetadata(body, event => events.push(event));
    assert.deepEqual(events, [
      {type:'message_start',backend_model:'k3',response_id:null,status:null},
      {type:'message_stop',backend_model:null,response_id:null,status:null},
    ]);
    assert.deepEqual(summary, {metadata_events:2,dispatched_events:4,incomplete_event:false});
    assert.ok(!JSON.stringify(events).includes('SECRET'));
  });
}

test('OpenAI metadata with mixed line endings and UTF8 stream chunks', async () => {
  const events = [];
  const response = new Response('data:{"type":"response.created","response":{"id":"r1","model":"gpt-6.1-sol","status":"in_progress"}}\r\n\r\n' +
    'data: {"type":"response.completed","response":{"id":"r1","model":"gpt-6.1-sol","status":"completed"}}\n\n');
  const result = await observeSseMetadata(response.body, x => events.push(x));
  assert.equal(result.metadata_events, 2);
  assert.equal(events[1].status, 'completed');
  assert.equal(events[1].backend_model, 'gpt-6.1-sol');
});

test('EOF never certifies a truncated event', () => {
  const events = [];
  const p = new SseMetadataParser(x => events.push(x));
  p.push('data: {"type":"message_stop"}\n');
  assert.deepEqual(p.finish(), {metadata_events:0,dispatched_events:0,incomplete_event:true});
  assert.deepEqual(events, []);
});

test('malformed JSON produces typed error without exposing data', () => {
  const p = new SseMetadataParser(() => assert.fail());
  assert.throws(() => p.push('data: SECRET_PROMPT\n\n'),
    e => e.code === 'SSE_JSON_INVALID' && !String(e).includes('SECRET'));
});

test('long unterminated line and repeated data lines remain bounded', () => {
  for (const data of ['data:' + 'x'.repeat(64), ('data: x\n').repeat(16)]) {
    const p = new SseMetadataParser(() => {}, {maxEventChars:32});
    assert.throws(() => p.push(data), e => e.code === 'SSE_EVENT_LIMIT');
  }
});

test('invalid UTF8 does not silently corrupt metadata', async () => {
  const body = new ReadableStream({start(c) {c.enqueue(Uint8Array.of(0xff));c.close();}});
  await assert.rejects(observeSseMetadata(body, () => {}), e => e.code === 'SSE_UTF8_OR_READ_ERROR');
});

test('clone observation does not consume the provider response', async () => {
  const text = 'data: {"type":"message_stop"}\r\n\r\n';
  const response = new Response(text);
  const summary = await observeSseMetadata(response.clone().body, () => {});
  assert.equal(summary.metadata_events, 1);
  assert.equal(await response.text(), text);
});
