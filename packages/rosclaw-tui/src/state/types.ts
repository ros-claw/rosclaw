/** UI state (批次 C)：事件流的纯投影，不持有任何权威。 */

export interface ApprovalItem {
	requestId: string;
	title: string;
	riskTier: string;
	expiresAt?: string;
}

export interface ToolRun {
	name: string;
	source?: string;
	status: "proposed" | "running" | "completed" | "failed";
	startedAt?: string;
	durationMs?: number;
	summary?: string;
	artifactRef?: string;
}

export interface WorkerRun {
	workOrderId: string;
	workerId: string;
	status: string;
}

export interface UiState {
	missionId: string;
	missionName: string;
	missionState: string;
	mode: string;
	bodyId: string;
	model: string;
	profile: string;
	turnInFlight: boolean;
	/** 当前阶段（思考状态文案，§6.5 — 永不显示 chain-of-thought）。 */
	phase: string;
	lastSeq: number;
	pendingApprovals: ApprovalItem[];
	tools: ToolRun[];
	workers: WorkerRun[];
	compactions: number;
	degraded: string;
	reconnecting: boolean;
}

export function initialState(missionId: string): UiState {
	return {
		missionId,
		missionName: "",
		missionState: "IDLE",
		mode: "SIMULATION",
		bodyId: "",
		model: "",
		profile: "",
		turnInFlight: false,
		phase: "",
		lastSeq: 0,
		pendingApprovals: [],
		tools: [],
		workers: [],
		compactions: 0,
		degraded: "",
		reconnecting: false,
	};
}
