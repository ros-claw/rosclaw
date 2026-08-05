/** SessionBinding 客户端（PNA-1，规格 §13.1）：启动绑定 + lease 心跳 + 退出释放。 */

import { bridgeCall } from "../bridge/bridge-client.js";

export interface SessionBinding {
	bindingId: string;
	piSessionId: string;
	missionId: string;
	leaseToken: string;
	heartbeat: ReturnType<typeof setInterval>;
}

export async function bindSession(
	rosclawHome: string,
	piSessionId: string,
	missionId: string,
): Promise<SessionBinding> {
	const response = await bridgeCall(rosclawHome, "pi.session.bind", {
		pi_session_id: piSessionId,
		mission_id: missionId,
	});
	if (!response.ok) {
		throw new Error(
			`session bind failed [${String(response.code ?? "")}]: ${String(response.error ?? "")}`,
		);
	}
	const binding = response.binding as { binding_id: string };
	const leaseToken = String(response.lease_token ?? "");
	const heartbeat = setInterval(() => {
		void bridgeCall(rosclawHome, "pi.session.heartbeat", {
			mission_id: missionId,
			pi_session_id: piSessionId,
			lease_token: leaseToken,
		}).catch(() => undefined);
	}, 30_000);
	heartbeat.unref();
	return {
		bindingId: binding.binding_id,
		piSessionId,
		missionId,
		leaseToken,
		heartbeat,
	};
}

export async function releaseSession(rosclawHome: string, binding: SessionBinding): Promise<void> {
	clearInterval(binding.heartbeat);
	await bridgeCall(rosclawHome, "pi.session.release", {
		mission_id: binding.missionId,
		pi_session_id: binding.piSessionId,
		lease_token: binding.leaseToken,
	}).catch(() => undefined);
}
