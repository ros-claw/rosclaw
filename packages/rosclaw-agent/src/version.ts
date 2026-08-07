/** ROSClaw 产品版本（P0-NA-16）：由 Python launcher 经
 * ROSCLAW_PRODUCT_VERSION 显式传入（发布包构建时同源）。
 * 内部 npm 子包版本（0.1.0）只是实现细节，永不出现在产品 UI。
 * 缺失时 fail-closed 显示 unknown——不静默回退到子包版本。 */
export const VERSION = process.env.ROSCLAW_PRODUCT_VERSION || "unknown";
