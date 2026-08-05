/** ROSClaw 内联扩展（PNA-0）：品牌 + 安全基线。
 *
 * - header/footer/title/working 动画替换为 ROSClaw；
 * - `!` bash 功能级关闭（user_bash full replacement）；
 * - 会话生命周期观察埋点（PNA-1 挂 SessionBinding）。
 */

import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";

export interface RosclawExtensionOptions {
	profile: "developer" | "robot";
	version: string;
}

const WORKING_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

export function createRosclawExtension(options: RosclawExtensionOptions): ExtensionFactory {
	return (pi) => {
		// -- 品牌 ----------------------------------------------------------------
		pi.on("session_start", async (_event, ctx) => {
			if (!ctx.hasUI) return;
			ctx.ui.setTitle(`ROSClaw Native Agent`);
			ctx.ui.setHeader((_tui, _theme) => {
				return new Text(
					`ROSClaw Native Agent v${options.version} ` +
						`[engine=pi profile=${options.profile}]  ·  /help 查看命令`,
				);
			});
			ctx.ui.setWorkingIndicator({ frames: WORKING_FRAMES, intervalMs: 80 });
		});

		// -- `!` bash 功能级关闭（PNA-0 即生效；PNA-9 再做 profile 化 UI 拦截） -----
		pi.on("user_bash", async () => {
			return {
				result: {
					output: "bash execution is disabled by ROSClaw policy (engine=pi)",
					exitCode: 1,
					cancelled: false,
					truncated: false,
				},
			};
		});

		// -- 生命周期观察（PNA-1 接 SessionBinding/lease） ---------------------------
		pi.on("session_before_fork", async () => {
			// PNA-6 在此强制"新 SIM Mission + 不复制 authority"；当前仅观察。
			return undefined;
		});
	};
}
