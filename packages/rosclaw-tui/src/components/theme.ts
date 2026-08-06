/** pi-tui themes (chalk-based, mirrors pi-tui test themes). */

import { Chalk } from "chalk";
import type { EditorTheme, MarkdownTheme, SelectListTheme } from "@earendil-works/pi-tui";

const chalk = new Chalk({ level: 3 });

export { chalk };

export const selectListTheme: SelectListTheme = {
	selectedPrefix: (text: string) => chalk.blue(text),
	selectedText: (text: string) => chalk.bold(text),
	description: (text: string) => chalk.dim(text),
	scrollInfo: (text: string) => chalk.dim(text),
	noMatch: (text: string) => chalk.dim(text),
};

export const markdownTheme: MarkdownTheme = {
	heading: (text: string) => chalk.bold.cyan(text),
	link: (text: string) => chalk.blue(text),
	linkUrl: (text: string) => chalk.dim(text),
	code: (text: string) => chalk.yellow(text),
	codeBlock: (text: string) => chalk.green(text),
	codeBlockBorder: (text: string) => chalk.dim(text),
	quote: (text: string) => chalk.italic(text),
	quoteBorder: (text: string) => chalk.dim(text),
	hr: (text: string) => chalk.dim(text),
	listBullet: (text: string) => chalk.cyan(text),
	bold: (text: string) => chalk.bold(text),
	italic: (text: string) => chalk.italic(text),
	strikethrough: (text: string) => chalk.strikethrough(text),
	underline: (text: string) => chalk.underline(text),
};

export const editorTheme: EditorTheme = {
	borderColor: (text: string) => chalk.dim(text),
	selectList: selectListTheme,
};

export function toneColor(tone: string, text: string): string {
	switch (tone) {
		case "ok":
			return chalk.green(text);
		case "warn":
			return chalk.yellow(text);
		case "error":
			return chalk.red(text);
		default:
			return chalk.dim(text);
	}
}

/** REAL 模式状态行：显眼但不刺眼的固定色（§6.3）。 */
export function modeColor(mode: string, text: string): string {
	if (mode === "REAL") return chalk.bgRed.white.bold(text);
	if (mode === "SHADOW") return chalk.yellow(text);
	return chalk.green(text);
}
