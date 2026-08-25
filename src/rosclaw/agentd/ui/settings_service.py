"""Settings service (批次 E §8.4)：分层、原子写、审计。

- 只允许非安全键（白名单）：agent.language、agent.context.*、
  models.backend、agent.default_mode(SIMULATION 除外——mode 升级永不
  经 settings)。body/权限/预算/安全策略一律拒绝。
- 写入：tmp + fsync + atomic rename + 文件锁；parse 失败保留旧配置。
- 每次持久改变写审计事件（不含 secret 值——settings 本来就不收 secret）。
"""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Any

import yaml

from rosclaw.contracts.common import ValidationError

#: 允许经 /settings 修改的键（点分路径 → 类型校验）。
_ALLOWED_KEYS: dict[str, type] = {
    "agent.language": str,
    "agent.context.max_input_tokens": int,
    "agent.context.dynamic_tool_limit": int,
    "models.backend": str,
}

_FORBIDDEN_PREFIXES = ("agent.body", "agent.budgets", "agent.permissions", "agent.safety")


class SettingsService:
    def __init__(self, config_path: Path) -> None:
        self._path = config_path

    def get(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        return yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}

    def get_key(self, dotted: str) -> Any:
        node: Any = self.get()
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def set_key(self, dotted: str, value: Any) -> dict:
        if dotted not in _ALLOWED_KEYS:
            if dotted.startswith(_FORBIDDEN_PREFIXES):
                raise ValidationError(f"{dotted} 属安全域，/settings 永不修改（走专用管理面）")
            raise ValidationError(f"未知的 settings 键 {dotted!r}（白名单外）")
        expected = _ALLOWED_KEYS[dotted]
        if expected is int and isinstance(value, str) and value.isdigit():
            value = int(value)
        if not isinstance(value, expected):
            raise ValidationError(f"{dotted} 需要 {expected.__name__}，得到 {type(value).__name__}")
        if dotted == "models.backend":
            raise ValidationError("models.backend 已废除（P1-A5）——模型运行时唯一=Pi")

        data = self.get()
        node = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        old = node.get(parts[-1])
        node[parts[-1]] = value
        self._atomic_write(data)
        return {"key": dotted, "old": old, "new": value}

    def _atomic_write(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".yaml.tmp")
        rendered = yaml.safe_dump(data, allow_unicode=True)
        # parse-back 校验：渲染结果必须可解析，否则不动旧配置。
        yaml.safe_load(rendered)
        with open(tmp, "w", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.write(rendered)
            fh.flush()
            os.fsync(fh.fileno())
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self._path)
