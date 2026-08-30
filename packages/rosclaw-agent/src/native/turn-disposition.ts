/** TurnDisposition（0827 体验审计 P0-1/P0-2）——Input Arbiter 的
 *  Harness 侧判定。
 *
 * 一条输入只能有一个 Owner：内核 pi.input.persist 返回权威
 * turn_disposition；owner=TASK_ROUTER 时 input hook 必须返回
 * handled（suppress 模型回合）——否则确定性链与 Native Agent
 * 双控制者竞争，终态互相矛盾（0827 实证事故）。
 */

export interface TurnDisposition {
	input_id: string;
	owner: "TASK_ROUTER" | "PI_CONVERSATION";
	task_id: string;
	suppress_model_turn: boolean;
}

export interface PersistedInput {
	input_id?: string;
	auto_task?: { task_id?: string; replayed?: boolean };
	/** 服务端权威判定（wire 形状——owner 未收窄为联合类型，旧/新
	 *  daemon 字段倾斜时以 suppress_model_turn/auto_task 为准）。 */
	turn_disposition?: {
		input_id?: string;
		owner?: string;
		task_id?: string;
		suppress_model_turn?: boolean;
	};
}

/** 模型回合是否被 suppress。disposition 是权威；旧 daemon 无该字段
 *  时回落 auto_task（版本倾斜期不漏 suppress——双控制者防线不能
 *  依赖单点字段）。 */
export function suppressModelTurn(persisted: PersistedInput): boolean {
	const disposition = persisted.turn_disposition;
	if (disposition && typeof disposition.suppress_model_turn === "boolean") {
		return disposition.suppress_model_turn;
	}
	return Boolean(persisted.auto_task?.task_id);
}
