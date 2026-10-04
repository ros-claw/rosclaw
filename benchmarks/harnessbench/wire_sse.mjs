/** Bounded, metadata-only SSE observation. Never records prompts or deltas. */
const SELECTED = new Set([
  'message_start', 'message_stop', 'response.created', 'response.completed',
  'response.failed', 'error',
]);

export class SseMetadataError extends Error {
  constructor(code) { super(code); this.name = 'SseMetadataError'; this.code = code; }
}

export class SseMetadataParser {
  constructor(emit, { maxEventChars = 1048576 } = {}) {
    if (typeof emit !== 'function' || !Number.isSafeInteger(maxEventChars) || maxEventChars < 1) {
      throw new SseMetadataError('INVALID_CONFIGURATION');
    }
    this.emit = emit;
    this.limit = maxEventChars;
    this.line = '';
    this.data = [];
    this.eventChars = 0;
    this.afterCR = false;
    this.firstLine = true;
    this.finished = false;
    this.metadataEvents = 0;
    this.dispatchedEvents = 0;
  }

  push(text) {
    if (this.finished || typeof text !== 'string') throw new SseMetadataError('INVALID_INPUT');
    for (const ch of text) {
      if (this.afterCR) {
        this.afterCR = false;
        if (ch === '\n') continue;
      }
      if (ch === '\r' || ch === '\n') {
        this.consumeLine();
        this.afterCR = ch === '\r';
      } else {
        this.line += ch;
        if (this.eventChars + this.line.length > this.limit) throw new SseMetadataError('SSE_EVENT_LIMIT');
      }
    }
  }

  consumeLine() {
    let line = this.line;
    this.line = '';
    if (this.firstLine) { line = line.replace(/^\uFEFF/, ''); this.firstLine = false; }
    if (line === '') {
      if (this.data.length) this.dispatch();
      this.data = [];
      this.eventChars = 0;
      return;
    }
    this.eventChars += line.length + 1;
    if (this.eventChars > this.limit) throw new SseMetadataError('SSE_EVENT_LIMIT');
    const colon = line.indexOf(':');
    if (colon === 0) return;
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'data') this.data.push(value);
  }

  dispatch() {
    const raw = this.data.join('\n');
    this.dispatchedEvents++;
    if (raw === '[DONE]') return;
    let event;
    try { event = JSON.parse(raw); } catch { throw new SseMetadataError('SSE_JSON_INVALID'); }
    if (!event || typeof event !== 'object' || !SELECTED.has(event.type)) return;
    // Only these fields may leave the parser. Discard all text/reasoning/tool deltas.
    const scalar = value => typeof value === 'string' ? value : null;
    this.emit({
      type: event.type,
      backend_model: scalar(event.message?.model ?? event.response?.model),
      response_id: scalar(event.response?.id),
      status: scalar(event.response?.status),
    });
    this.metadataEvents++;
  }

  finish() {
    if (this.finished) throw new SseMetadataError('ALREADY_FINISHED');
    this.finished = true;
    // EOF alone does not dispatch an unterminated event. Make truncation explicit.
    return {
      metadata_events: this.metadataEvents,
      dispatched_events: this.dispatchedEvents,
      incomplete_event: this.line.length > 0 || this.data.length > 0,
    };
  }
}

export async function observeSseMetadata(body, emit, options) {
  if (!body) throw new SseMetadataError('MISSING_RESPONSE_BODY');
  const reader = body.getReader();
  const decoder = new TextDecoder('utf-8', { fatal: true });
  const parser = new SseMetadataParser(emit, options);
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      parser.push(decoder.decode(value, { stream: true }));
    }
    parser.push(decoder.decode());
    return parser.finish();
  } catch (error) {
    // A clone consumer must release/cancel its branch on errors, without waiting
    // for cancellation of the independent provider branch of a tee'd response.
    try { void reader.cancel().catch(() => {}); } catch {}
    if (error instanceof SseMetadataError) throw error;
    throw new SseMetadataError(error instanceof TypeError ? 'SSE_UTF8_OR_READ_ERROR' : 'SSE_READ_ERROR');
  } finally { reader.releaseLock(); }
}
