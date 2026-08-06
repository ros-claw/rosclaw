import assert from "node:assert/strict";
import test from "node:test";
import { createServer, type Server } from "node:http";
import { streamEvents, type AgentEvent } from "../src/client/sse.js";

function sseFrame(seq: number, id: string, type: string): string {
	return `id: ${seq}\ndata: ${JSON.stringify({
		event_id: id,
		sequence: seq,
		mission_id: "mis_x",
		type,
		visibility: "USER",
		payload: {},
		timestamp: "t",
	})}\n\n`;
}

async function withServer(
	handler: (req: { headers: Record<string, unknown> }, res: {
		writeHead: (code: number, headers?: Record<string, string>) => void;
		write: (chunk: string) => void;
		end: () => void;
	}) => void,
	fn: (baseUrl: string) => Promise<void>,
): Promise<void> {
	const server: Server = createServer((req, res) => {
		handler({ headers: req.headers }, res as never);
	});
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	const address = server.address();
	const port = typeof address === "object" && address ? address.port : 0;
	try {
		await fn(`http://127.0.0.1:${port}`);
	} finally {
		server.closeAllConnections?.();
		await new Promise((resolve) => server.close(resolve));
	}
}

test("replays events and dedups by event_id", async () => {
	await withServer(
		(_req, res) => {
			res.writeHead(200, { "content-type": "text/event-stream" });
			res.write(sseFrame(1, "evt_a", "agent.started"));
			res.write(sseFrame(1, "evt_a", "agent.started")); // duplicate
			res.write(sseFrame(2, "evt_b", "agent.settled"));
			res.end();
		},
		async (baseUrl) => {
			const events: AgentEvent[] = [];
			for await (const e of streamEvents(baseUrl, "mis_x", { maxAttempts: 1 })) {
				events.push(e);
			}
			assert.equal(events.length, 2);
			assert.deepEqual(events.map((e) => e.event_id), ["evt_a", "evt_b"]);
		},
	);
});

test("reconnects with Last-Event-ID after a dropped connection", async () => {
	let requestCount = 0;
	const seenLastEventIds: string[] = [];
	await withServer(
		(req, res) => {
			requestCount += 1;
			seenLastEventIds.push(String(req.headers["last-event-id"] ?? ""));
			res.writeHead(200, { "content-type": "text/event-stream" });
			if (requestCount === 1) {
				res.write(sseFrame(1, "evt_a", "agent.started"));
				res.end(); // drop
			} else {
				res.write(sseFrame(2, "evt_b", "agent.settled"));
				res.end();
			}
		},
		async (baseUrl) => {
			const events: AgentEvent[] = [];
			for await (const e of streamEvents(baseUrl, "mis_x", { maxAttempts: 3 })) {
				events.push(e);
			}
			assert.deepEqual(events.map((e) => e.event_id), ["evt_a", "evt_b"]);
			assert.equal(seenLastEventIds[1], "1", "second request must carry Last-Event-ID");
		},
	);
});

test("sequence gap triggers onGap", async () => {
	await withServer(
		(_req, res) => {
			res.writeHead(200, { "content-type": "text/event-stream" });
			res.write(sseFrame(1, "evt_a", "agent.started"));
			res.write(sseFrame(5, "evt_e", "agent.settled"));
			res.end();
		},
		async (baseUrl) => {
			const gaps: Array<[number, number]> = [];
			for await (const _e of streamEvents(baseUrl, "mis_x", {
				maxAttempts: 1,
				onGap: (expected, got) => gaps.push([expected, got]),
			})) {
				// consume
			}
			assert.deepEqual(gaps, [[2, 5]]);
		},
	);
});

test("afterSequence seeds last-event-id on the first connect (R4 exactly-once)", async () => {
	await withServer(
		(req, res) => {
			const lastEventId = String(req.headers["last-event-id"] ?? "");
			res.writeHead(200, { "content-type": "text/event-stream" });
			if (lastEventId !== "42") {
				res.write(sseFrame(1, "evt_wrong", "agent.started"));
				res.end();
				return;
			}
			res.write(sseFrame(43, "evt_after", "agent.settled"));
			res.end();
		},
		async (baseUrl) => {
			const events: AgentEvent[] = [];
			for await (const e of streamEvents(baseUrl, "mis_x", {
				maxAttempts: 1,
				afterSequence: 42,
			})) {
				events.push(e);
			}
			assert.equal(events.length, 1);
			assert.equal(events[0].event_id, "evt_after");
		},
	);
});

test("control token header is sent when provided (P1-4)", async () => {
	await withServer(
		(req, res) => {
			const token = String(req.headers["x-rosclaw-token"] ?? "");
			res.writeHead(200, { "content-type": "text/event-stream" });
			if (token !== "tok_secret") {
				res.write(sseFrame(1, "evt_noauth", "agent.started"));
				res.end();
				return;
			}
			res.write(sseFrame(1, "evt_auth", "agent.settled"));
			res.end();
		},
		async (baseUrl) => {
			const events: AgentEvent[] = [];
			for await (const e of streamEvents(baseUrl, "mis_x", {
				maxAttempts: 1,
				controlToken: "tok_secret",
			})) {
				events.push(e);
			}
			assert.equal(events.length, 1);
			assert.equal(events[0].event_id, "evt_auth");
		},
	);
});
