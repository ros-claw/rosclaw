/** SessionLeaseManager（NA-FIX-2）：bind + heartbeat 的唯一管理点。
 *
 * 切换事务时：旧 heartbeat 停 → 新 bind → 新 heartbeat 起。
 * lease_token 绝不丢弃（P0-2 修复）。
 */

import { bridgeCall } from "../bridge/bridge-client.js";

export interface ActiveBinding {
	bindingId: string;
	piSessionId: string;
	missionId: string;
	leaseToken: string;
}

export class SessionLeaseManager {
	private current: ActiveBinding | null = null;
	private heartbeat: ReturnType<typeof setInterval> | null = null;
	private readonly call: typeof bridgeCall;

	constructor(
		private readonly rosclawHome: string,
		call?: typeof bridgeCall,
	) {
		this.call = call ?? bridgeCall;
	}

	get active(): ActiveBinding | null {
		return this.current;
	}

	async bind(piSessionId: string, missionId: string): Promise<ActiveBinding> {
		const response = await this.call(this.rosclawHome, "pi.session.bind", {
			pi_session_id: piSessionId,
			mission_id: missionId,
		});
		if (!response.ok) {
			throw new Error(
				`session bind failed [${String(response.code ?? "")}]: ${String(response.error ?? "")}`,
			);
		}
		// 原子替换：停旧 heartbeat → 记新 binding → 起新 heartbeat。
		this.stopHeartbeat();
		const binding: ActiveBinding = {
			bindingId: (response.binding as { binding_id: string }).binding_id,
			piSessionId,
			missionId,
			leaseToken: String(response.lease_token ?? ""),
		};
		this.current = binding;
		this.startHeartbeat();
		return binding;
	}

	/** 切换前显式释放旧 lease（NA-FIX-2 事务的一步）。 */
	async release(): Promise<void> {
		this.stopHeartbeat();
		if (this.current) {
			await this.call(this.rosclawHome, "pi.session.release", {
				mission_id: this.current.missionId,
				pi_session_id: this.current.piSessionId,
				lease_token: this.current.leaseToken,
			}).catch(() => undefined);
			this.current = null;
		}
	}

	private startHeartbeat(): void {
		this.heartbeat = setInterval(() => {
			if (!this.current) return;
			void bridgeCall(this.rosclawHome, "pi.session.heartbeat", {
				mission_id: this.current.missionId,
				pi_session_id: this.current.piSessionId,
				lease_token: this.current.leaseToken,
			}).catch(() => undefined);
		}, 30_000);
		this.heartbeat.unref();
	}

	private stopHeartbeat(): void {
		if (this.heartbeat !== null) {
			clearInterval(this.heartbeat);
			this.heartbeat = null;
		}
	}
}
