/** Provider 错误分类（PR-H7，总纲 v2 §8.4）。
 *
 * 403 配额耗尽不能显示成 auth_error。分类决定错误卡的稳定码 +
 * 用户可理解说明 + 可执行恢复动作 + task 是否可继续。
 */

export type ModelErrorCode =
	| "MODEL_CREDENTIAL_MISSING"
	| "MODEL_CREDENTIAL_INVALID"
	| "PROVIDER_QUOTA_EXHAUSTED"
	| "PROVIDER_RATE_LIMITED"
	| "PROVIDER_UNAVAILABLE"
	| "MODEL_NOT_FOUND"
	| "MODEL_TOOL_CALL_UNSUPPORTED"
	| "MODEL_CONTEXT_LIMIT"
	| "MODEL_REQUEST_CANCELLED"
	| "MODEL_UNKNOWN";

export interface ClassifiedModelError {
	code: ModelErrorCode;
	/** 用户可理解说明（中文产品语言）。 */
	explanation: string;
	/** 可执行恢复动作。 */
	recovery: string;
	/** task 是否可继续（配额类 = 可等/可换模型继续，不是 FAILED）。 */
	taskRecoverable: boolean;
}

/** 分类 provider 错误文本（HTTP 状态 + 消息关键词）——确定性规则，
 *  不靠模型猜。 */
export function classifyModelError(raw: string): ClassifiedModelError {
	const text = raw.toLowerCase();
	// 顺序敏感：quota 在 generic 403/auth 之前。
	if (
		/usage limit|quota|billing|access_terminated/.test(text)
	) {
		return {
			code: "PROVIDER_QUOTA_EXHAUSTED",
			explanation: "模型配额已用完（本计费周期）",
			recovery: "配额下周期刷新；/model 换可用模型立即继续；任务保持可恢复",
			taskRecoverable: true,
		};
	}
	if (/429|rate.?limit|too many requests/.test(text)) {
		return {
			code: "PROVIDER_RATE_LIMITED",
			explanation: "请求频率超限",
			recovery: "稍等自动重试；持续超限可 /model 换模型",
			taskRecoverable: true,
		};
	}
	if (/401|unauthorized|invalid (api.?)?key|invalid authentication/.test(text)) {
		return {
			code: "MODEL_CREDENTIAL_INVALID",
			explanation: "凭据无效（401）",
			recovery: "rosclaw setup model 重新配置 key",
			taskRecoverable: true,
		};
	}
	if (/403|forbidden/.test(text)) {
		return {
			code: "MODEL_CREDENTIAL_INVALID",
			explanation: "访问被拒（403）",
			recovery: "检查 key 权限/套餐；rosclaw setup model 重配",
			taskRecoverable: true,
		};
	}
	if (/model.*not found|no such model|does not exist/.test(text)) {
		return {
			code: "MODEL_NOT_FOUND",
			explanation: "模型不存在或已下线",
			recovery: "/model 换可用模型",
			taskRecoverable: true,
		};
	}
	if (/context.*(length|limit)|too many tokens|max_tokens/.test(text)) {
		return {
			code: "MODEL_CONTEXT_LIMIT",
			explanation: "上下文超限",
			recovery: "/compact 压缩后继续",
			taskRecoverable: true,
		};
	}
	if (/econnreset|etimedout|enotfound|network|unavailable|502|503|504/.test(text)) {
		return {
			code: "PROVIDER_UNAVAILABLE",
			explanation: "provider 暂不可达",
			recovery: "自动重试中；持续失败检查网络",
			taskRecoverable: true,
		};
	}
	return {
		code: "MODEL_UNKNOWN",
		explanation: "模型调用失败（未分类）",
		recovery: "查看 /doctor 诊断",
		taskRecoverable: true,
	};
}
