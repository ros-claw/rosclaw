"""能力合约参数校验与规范化（五审 P0-5C）。

此前 admission 只手工检查顶层 required/additionalProperties——本模块
换用完整 JSON Schema validator，并在建卡前展开 contract/MCP 默认值：

- 人批准的是展开后的真实参数（660Hz/0.25s/18%），不是 `{}`；
- normalized arguments 同一份对象用于 hash/审批展示/grant/executor/
  receipt——canonical bytes 逐字节一致；
- 完整校验 type/enum/minimum/maximum/nested/non-finite number。
"""

from __future__ import annotations

import math
from typing import Any

from rosclaw.contracts.common import ValidationError


def expand_defaults(schema: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    """按 input_schema 展开默认值（浅层 properties 默认 + 嵌套 object）。
    缺省值来自 schema 的 default；无默认且非 required 的字段不补。"""
    if not isinstance(arguments, dict):
        raise ValidationError("arguments must be an object")
    normalized = dict(arguments)
    properties = schema.get("properties") or {}
    for key, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        if key not in normalized and "default" in prop:
            normalized[key] = prop["default"]
        elif key in normalized and prop.get("type") == "object":
            nested = normalized[key]
            nested_schema = prop.get("properties") and prop
            if nested_schema and isinstance(nested, dict):
                normalized[key] = expand_defaults(prop, nested)
    return normalized


def _check_non_finite(value: Any, path: str = "") -> None:
    """NaN/Inf 不是合法物理参数——递归拒绝。"""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValidationError(f"argument {path or '<root>'} is not a finite number")
    if isinstance(value, dict):
        for k, v in value.items():
            _check_non_finite(v, f"{path}.{k}" if path else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _check_non_finite(v, f"{path}[{i}]")


def validate_action_arguments(
    schema: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    """完整 JSON Schema 校验 + 默认值展开，返回 normalized arguments。

    fail closed：任何违规（type/enum/range/nested/non-finite/未知字段）
    都抛 ValidationError，绝不部分通过。
    """
    if not isinstance(arguments, dict):
        raise ValidationError("arguments must be an object")
    _check_non_finite(arguments)
    normalized = expand_defaults(schema, arguments)
    _check_non_finite(normalized)
    if not schema:
        return normalized
    import jsonschema

    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema)
    errors = sorted(validator.iter_errors(normalized), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        path = ".".join(str(p) for p in first.path) or "<root>"
        raise ValidationError(
            f"action arguments violate capability contract at {path}: {first.message}"
        )
    return normalized
