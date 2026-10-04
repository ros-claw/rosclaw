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
	for (const role of ["progress_report", "diagnostic_failed_attempt", "report", "plot"]) {
		const result = validateToolArguments(tool, {
			type: "toolCall", id: "delivery", name: tool.name,
			arguments: { path: "source.json", role },
		});
		assert.equal(result.role, role);
	}
});
