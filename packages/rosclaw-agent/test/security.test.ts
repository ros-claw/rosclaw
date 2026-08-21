import assert from "node:assert/strict";
import test from "node:test";

import { guardInput } from "../src/extension/input-guard.js";
import { resourcePolicy } from "../src/extension/resource-policy.js";

test("unknown slash command is handled, never sent to the model", () => {
	const verdict = guardInput("/evilcommand do something", "developer");
	assert.equal(verdict.action, "handled");
	assert.match(verdict.notice ?? "", /未知命令/);
});

test("//text escape transforms to plain text", () => {
	const verdict = guardInput("//model 其实是文本", "developer");
	assert.equal(verdict.action, "transform");
	assert.equal(verdict.text, "/model 其实是文本");
});

test("robot profile blocks /trust /share /import /reload", () => {
	for (const cmd of ["/trust", "/share", "/import", "/reload"]) {
		const verdict = guardInput(`${cmd} something`, "robot");
		assert.equal(verdict.action, "handled", cmd);
		assert.match(verdict.notice ?? "", /禁用/);
	}
	// developer profile 放行给内建语义。
	assert.equal(guardInput("/trust", "developer").action, "continue");
});

test("known commands pass through", () => {
	for (const cmd of ["/model", "/resume", "/workers", "/delegate", "/estop"]) {
		assert.equal(guardInput(cmd, "robot").action, "continue", cmd);
	}
});

test("resource policy per profile", () => {
	const robot = resourcePolicy("robot");
	assert.ok(
		robot.extensions === "off" && robot.skills === "off"
		&& robot.contextFiles === "off" && !robot.themes,
	);
	assert.equal(robot.credentialPolicy, "env-only");
	assert.equal(robot.allowBash, false);
	const developer = resourcePolicy("developer");
	assert.equal(developer.themes, true, "developer 允许用户主题");
	// PR-N2：developer 的可信上下文/内置签名 Skill 恢复（拆分面）。
	assert.equal(developer.contextFiles, "trusted-readonly");
	assert.equal(developer.skills, "bundled-signed");
	assert.equal(developer.credentialPolicy, "file-0600");
	const worker = resourcePolicy("worker");
	assert.equal(worker.allowFileTools, false);
	assert.equal(worker.credentialPolicy, "env-only");
});
