/** catalog 访问（六审 §8）：按生效 locale 取串，缺键 fail 到英文
 * 并可见（不出现 undefined）。 */

import { CATALOG_EN } from "./catalog.en-US.js";
import { CATALOG_ZH, type CatalogKey } from "./catalog.zh-CN.js";

export type EffectiveLocale = "zh-CN" | "en-US";

export function t(key: CatalogKey, locale: EffectiveLocale): string {
	const table = locale === "zh-CN" ? CATALOG_ZH : CATALOG_EN;
	return String(table[key] ?? CATALOG_EN[key] ?? key);
}
