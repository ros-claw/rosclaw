/** Secret redaction (批次 D §7.3)：错误/日志统一脱敏。 */

const PATTERNS = [
	/sk-[A-Za-z0-9_-]{8,}/g,
	/Bearer\s+[A-Za-z0-9._-]{8,}/gi,
	/(api[_-]?key|token|secret)("?\s*[:=]\s*"?)[^\s",}]+/gi,
];

export function redact(text: string, knownSecrets: string[] = []): string {
	let out = text;
	for (const secret of knownSecrets) {
		if (secret && secret.length >= 4) {
			out = out.split(secret).join("<redacted>");
		}
	}
	for (const pattern of PATTERNS) {
		out = out.replace(pattern, (match, p1) =>
			p1 && match.includes(":") ? `${p1}<redacted>` : "<redacted>",
		);
	}
	return out;
}
