/** `rosclaw_status` 自定义工具（PNA-0）：只读状态，永不伪造可达性。 */

import { Type } from "@earendil-works/pi-ai";
import { defineTool } from "@earendil-works/pi-coding-agent";
import { fetchAgentdStatus } from "../bridge/agentd-client.js";

export function buildStatusTool(rosclawHome: string) {
	return defineTool({
		name: "rosclaw_status",
		label: "ROSClaw Status",
		description:
			"Read ROSClaw embodied-kernel status (agentd/missions/body/mode). " +
			"Read-only. Returns honest unreachable errors when agentd is down — " +
			"never invent robot state.",
		parameters: Type.Object({}),
		async execute(_toolCallId, _params, _signal, _onUpdate, _ctx) {
			const result = await fetchAgentdStatus(rosclawHome);
			const text = result.reachable
				? JSON.stringify({ agentd: "READY", ...result.status }, null, 1)
				: JSON.stringify(
						{
							agentd: "UNREACHABLE",
							base_url: result.baseUrl,
							error: result.error,
							note: "agentd 未运行或鉴权失败——不要编造机器人状态。",
						},
						null,
						1,
					);
			return {
				content: [{ type: "text", text }],
				details: { reachable: result.reachable },
			};
		},
	});
}
