import assert from "node:assert/strict";
import test from "node:test";
import { AgentSession, ExtensionRunner } from "@earendil-works/pi-coding-agent";
import { probePiCapabilities } from "../src/harness/pi/pi-backend.js";

test("observational subscribe cannot stand in for an executable tool policy hook", () => {
	assert.equal(probePiCapabilities().toolPolicyHook, true);
	assert.equal(typeof AgentSession.prototype.subscribe, "function");
	const descriptor = Object.getOwnPropertyDescriptor(ExtensionRunner.prototype, "emitToolCall")!;
	try {
		Object.defineProperty(ExtensionRunner.prototype, "emitToolCall", { value: undefined, configurable: true });
		assert.equal(probePiCapabilities().toolPolicyHook, false);
		assert.equal(probePiCapabilities().toolStreaming, true);
	} finally {
		Object.defineProperty(ExtensionRunner.prototype, "emitToolCall", descriptor);
	}
	assert.equal(probePiCapabilities().toolPolicyHook, true);
});

test("a missing public session extension runner disables toolPolicyHook", () => {
	const descriptor = Object.getOwnPropertyDescriptor(AgentSession.prototype, "extensionRunner")!;
	try {
		Object.defineProperty(AgentSession.prototype, "extensionRunner", { value: undefined, configurable: true });
		assert.equal(probePiCapabilities().toolPolicyHook, false);
	} finally {
		Object.defineProperty(AgentSession.prototype, "extensionRunner", descriptor);
	}
});
