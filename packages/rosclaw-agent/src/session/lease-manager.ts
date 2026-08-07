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

	private heartbeatFailures = 0;
	/** HOTFIX-3（P0-4E）：heartbeat 连续失败 → LEASE_LOST 回调。 */
	onLeaseLost: (() => void) | null = null;

	private startHeartbeat(): void {
		this.heartbeatFailures = 0;
		this.heartbeat = setInterval(() => {
			if (!this.current) return;
			void this.call(this.rosclawHome, "pi.session.heartbeat", {
				mission_id: this.current.missionId,
				pi_session_id: this.current.piSessionId,
				lease_token: this.current.leaseToken,
			})
				.then((response) => {
					// 心跳被拒（lease 过期/被抢/token 错）即失败，不等超时。
					if (response.ok === false) {
						this.noteHeartbeatFailure();
					} else {
						this.heartbeatFailures = 0;
					}
				})
				.catch(() => {
					this.noteHeartbeatFailure();
				});
		}, 30_000);
		this.heartbeat.unref();
	}

	private noteHeartbeatFailure(): void {
		this.heartbeatFailures += 1;
		// 连续 2 次失败即判 LEASE_LOST（一次可能是网络抖动；
		// 两次 = lease 真的没了——动作必须立即禁行）。
		if (this.heartbeatFailures >= 2) {
			this.stopHeartbeat();
			this.current = null;
			this.onLeaseLost?.();
		}
	}

	private stopHeartbeat(): void {
		if (this.heartbeat !== null) {
			clearInterval(this.heartbeat);
			this.heartbeat = null;
		}
	}
}
