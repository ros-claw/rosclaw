/** 命令注册表（P0-H，0824 总纲 §16.4/§19.P0-H）。
 *
 * /resume 与 Pi 内置冲突的实证教训（启动 [Extension issues]）——
 * 命令名与快捷键一样必须注册表化：冲突在测试期爆炸，不在运行时
 * 由 Pi loader 报 issue。
 */

/** ROSClaw 扩展命令名（唯一事实源——commands.ts/index.ts 注册的
 *  命令必须与之一致）。 */
export const ROSCLAW_COMMAND_NAMES: readonly string[] = [
	"workspace", "newtask", "done", "tasks", "taskinfo",
	"activity", "logs", "artifacts",
	"status", "mission", "body", "tools", "approvals", "revoke",
	"task", "trace", "context", "why", "sessions", "switch",
	"language", "operator-init", "safety", "robot", "robots",
	"capabilities", "tokens", "workers", "delegate", "worker-jobs",
	"memory", "doctor", "estop", "evidence",
];

/** Pi 内置命令（审计 §4 全表——与 input-guard.ts 的 PI_BUILTINS
 *  同源；Pi 升级新增时本表须同步——冲突测试会红）。 */
export const PI_BUILTIN_COMMANDS: ReadonlySet<string> = new Set([
	"settings", "model", "scoped-models", "export", "import", "share",
	"copy", "name", "session", "hotkeys", "fork", "clone", "tree",
	"trust", "login", "logout", "new", "compact", "resume", "reload",
	"quit",
]);

/** 冲突检查：返回冲突命令名列表（空 = 健康）。 */
export function commandConflicts(): string[] {
	return ROSCLAW_COMMAND_NAMES.filter((name) =>
		PI_BUILTIN_COMMANDS.has(name)
	);
}
