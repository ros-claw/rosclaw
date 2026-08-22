/** 共享 ModelRuntime 构造（十审 W1，审计 §8.3.1）。
 *
 * 主 Agent（create-runtime）与内置 Worker（pi-worker-main）共用同一
 * agentDir/auth.json/models.json 配置——Worker 不需要第二份 API key。
 * WorkOrder 只携带无 secret 的 ModelExecutionSnapshot（provider/model/
 * thinking），凭据永远由子进程从同一加固凭据存储读取。
 */

import { ModelRuntime } from "@earendil-works/pi-coding-agent";

import { credentialStoreFor } from "../../credentials/store.js";

export async function createSharedModelRuntime(
	agentDir: string,
	profile: "developer" | "robot",
): Promise<ModelRuntime> {
	// 凭据后端按 profile：developer=加固文件（0600/原子写/fsync），
	// robot=env-only（写即拒）。Worker 以 developer 语义读同一份
	// auth.json/models.json（headless 不写凭据）。
	return await ModelRuntime.create({
		credentials: credentialStoreFor(profile, agentDir) as never,
		authPath: `${agentDir}/auth.json`,
		// robot=env-only 时禁 models.json；developer 允许用户自定义 provider。
		modelsPath: profile === "robot" ? null : `${agentDir}/models.json`,
	});
}

/** 无 secret 模型执行快照（十审 W1 ModelExecutionSnapshotV1）。
 *  只允许 provider/model/thinking——任何 credential 字段都不得出现。 */
export interface ModelExecutionSnapshotV1 {
	provider: string;
	model: string;
	thinking?: string;
}

export function snapshotOfModel(
	model: { provider: string; id: string } | undefined,
	thinking?: string,
): ModelExecutionSnapshotV1 | undefined {
	if (!model) return undefined;
	return { provider: model.provider, model: model.id, ...(thinking ? { thinking } : {}) };
}
