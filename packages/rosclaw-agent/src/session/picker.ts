/** ROSClaw 会话选择器（总纲 WP-P0-1 §5.1/§13）。
 *
 * 组合 Pi 0.83.0 公开 API（SessionSelectorComponent + pi-tui），不复制
 * picker/search/tree 内核。交互结构参考上游
 * packages/coding-agent/src/cli/session-picker.ts（MIT，固定
 * v0.83.0；上游 selectSession 不是公开 export，本文件是总纲允许的
 * 极薄组装层——差异测试：test/session-discovery.test.ts）。
 */

import { ProcessTerminal, TUI } from "@earendil-works/pi-tui";
import { SessionSelectorComponent } from "@earendil-works/pi-coding-agent";
import type { SessionInfo } from "@earendil-works/pi-coding-agent";

// SessionListProgress 未从包根导出——结构类型即可（loader 签名兼容）。
type SessionsLoader = (
	onProgress?: (loaded: number, total: number) => void,
) => Promise<SessionInfo[]>;

/** 打开会话选择器；返回选中的 session 文件路径，取消返回 null。 */
export async function browseSessions(
	currentSessionsLoader: SessionsLoader,
	allSessionsLoader: SessionsLoader,
): Promise<string | null> {
	const terminal = new ProcessTerminal();
	const ui = new TUI(terminal);
	return new Promise((resolve) => {
		let resolved = false;
		const finish = (path: string | null) => {
			if (resolved) return;
			resolved = true;
			ui.stop();
			resolve(path);
		};
		const selector = new SessionSelectorComponent(
			currentSessionsLoader,
			allSessionsLoader,
			(path) => finish(path),
			() => finish(null),
			() => {
				ui.stop();
				process.exit(0);
			},
			() => ui.requestRender(),
			// keybindings 为可选——缺省时组件用内建默认键位。
			{ showRenameHint: false },
		);
		ui.addChild(selector);
		ui.setFocus(selector.getSessionList());
		ui.start();
	});
}
