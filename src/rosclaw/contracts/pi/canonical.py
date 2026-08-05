"""RFC 8785 JSON Canonicalization Scheme（二次审计 NA-FIX-1/P0-1）。

Python 的 ``json.dumps`` 与 JS ``JSON.stringify`` 的数字序列化不同
（30.0 → "30.0" vs "30"；-0.0 → "-0.0" vs "0"；1e-7 → "1e-07" vs
"1e-7"）。本模块实现与 JS Number#toString 对齐的 canonical JSON，
保证 Python 与 TypeScript 对同一 payload 得到逐字节一致的字节串。

规则（与 JS 对齐）：
- float 且整数值且 |x| < 1e21 → 按整数输出（30.0 → "30"）；
- -0.0 → "0"；
- 非整数 float → 最短 round-trip repr，指数去前导零（1e-07 → "1e-7"）；
- key 排序、紧凑分隔符、UTF-8 原样（ensure_ascii=False）。
"""

from __future__ import annotations

import json
import math
from typing import Any


def _js_number(value: int | float) -> str:
    if isinstance(value, bool):  # bool 是 int 子类——先拦
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite number cannot be canonicalized: {value!r}")
    if value == 0:
        return "0"  # JS: String(-0) === "0"
    if value.is_integer() and abs(value) < 1e21:
        return str(int(value))
    text = repr(value)
    # Python 指数带前导零（1e-07 / 1e+21）；JS 不带（1e-7 / 1e+21）。
    if "e" in text or "E" in text:
        mantissa, _, exponent = text.lower().partition("e")
        sign = ""
        if exponent.startswith(("+", "-")):
            sign, exponent = exponent[0], exponent[1:]
        exponent = exponent.lstrip("0") or "0"
        text = f"{mantissa}e{sign}{exponent}"
    return text


def canonical_dumps(value: Any) -> str:
    """JCS 风格 canonical JSON（Python 侧；与 TS canonicalJson 对齐）。"""
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return _js_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_dumps(item) for item in value) + "]"
    if isinstance(value, dict):
        parts = []
        for key in sorted(value.keys(), key=str):
            parts.append(json.dumps(str(key), ensure_ascii=False) + ":" + canonical_dumps(value[key]))
        return "{" + ",".join(parts) + "}"
    raise TypeError(f"cannot canonicalize {type(value).__name__}")
