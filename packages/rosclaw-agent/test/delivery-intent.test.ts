import assert from "node:assert/strict";
import { test } from "node:test";
import { validateToolArguments } from "@earendil-works/pi-ai";
import { buildProductPackTools } from "../src/tools/product-pack.js";
import type { BridgeToolContext } from "../src/tools/bridge-tools.js";

test("public PI tool validator requires explicit delivery intent", () => {
	const tool = buildProductPackTools({} as BridgeToolContext)[0];
	for (const args of [{ path: "source.json" }, { path: "source.json", role: "" }]) {
		assert.throws(() => validateToolArguments(tool, {
			type: "toolCall", id: "delivery", name: tool.name, arguments: args,
		}));
	}
	for (const role of ["progress", "diagnostic", "progress_report", "diagnostic_failed_attempt", "diagnostic_progress_report_NOT_DONE", "Progress_阶段证据", " report ", "report", "plot", "image", "video", "data"]) {
		const result = validateToolArguments(tool, {
			type: "toolCall", id: "delivery", name: tool.name,
			arguments: { path: "source.json", role },
		});
		assert.equal(result.role, role);
	}
});

test("public PI coercion cannot turn ambiguous values into final intent", () => {
	const tool = buildProductPackTools({} as BridgeToolContext)[0];
	for (const role of [null, 1, true, false, [], ["report"], {}, "null", "1", "true", "false", "unknown"]) {
		assert.throws(() => validateToolArguments(tool, {
			type: "toolCall", id: "delivery", name: tool.name,
			arguments: { path: "source.json", role },
		}), `ambiguous role accepted: ${JSON.stringify(role)}`);
	}
});
