/** 中央快捷键注册表（WP-7，0823 审计 §四.WP-7）。
 *
 * 0823 实测：扩展 ctrl+t 与 Pi 内置冲突被静默跳过——任务活动视图
 * 根本打不开，用户只看到 [Extension issues]。所有 ROSClaw 快捷键
 * 在此一处登记；与 Pi 内置保留键的冲突在测试期爆炸，不再等到
 * 运行时由 Pi loader 报 issue。
 */

/** ROSClaw 快捷键（唯一事实源——index.ts 只能从这里取键）。 */
export const ROSCLAW_SHORTCUTS = {
	/** Task Activity 常驻 widget 开关（原 ctrl+t——Pi 内置，冲突被跳过）。 */
	taskActivity: "shift+ctrl+t",
	/** F2 Task Panel。 */
	tasksCenter: "f2",
	/** Task Panel 第二绑定。 */
	tasksCenterAlt: "alt+j",
	/** Operator 一键初始化。 */
	operatorBootstrap: "shift+ctrl+b",
} as const;

/** Pi 内置保留键（从 pi-coding-agent/pi-tui dist 枚举，0823 实证
 * ctrl+t 在列；Pi 升级新增保留键时本表须同步——冲突测试会红）。 */
export const PI_RESERVED_SHORTCUTS: ReadonlySet<string> = new Set([
	"alt+b", "alt+backspace", "alt+d", "alt+delete", "alt+down", "alt+enter",
	"alt+f", "alt+left", "alt+right", "alt+space", "alt+up", "alt+v", "alt+y",
	"ctrl+a", "ctrl+b", "ctrl+backspace", "ctrl+c", "ctrl+clear", "ctrl+d",
	"ctrl+delete", "ctrl+down", "ctrl+e", "ctrl+end", "ctrl+f", "ctrl+g",
	"ctrl+home", "ctrl+insert", "ctrl+j", "ctrl+k", "ctrl+l", "ctrl+left",
	"ctrl+n", "ctrl+o", "ctrl+p", "ctrl+r", "ctrl+right", "ctrl+s",
	"ctrl+space", "ctrl+t", "ctrl+u", "ctrl+up", "ctrl+v", "ctrl+w",
	"ctrl+x", "ctrl+y", "ctrl+z",
	"shift+ctrl+d", "shift+ctrl+o", "shift+ctrl+p",
]);

/** 注册表健康检查：返回冲突列表（空 = 健康）。 */
export function shortcutConflicts(): string[] {
	const conflicts: string[] = [];
	const seen = new Map<string, string>();
	for (const [name, key] of Object.entries(ROSCLAW_SHORTCUTS)) {
		if (PI_RESERVED_SHORTCUTS.has(key)) {
			conflicts.push(`${name}=${key} 与 Pi 内置保留键冲突`);
		}
		const prev = seen.get(key);
		if (prev) conflicts.push(`${name}=${key} 与 ${prev} 重复`);
		seen.set(key, name);
	}
	return conflicts;
}
