"""Local JSONL store for rosclaw-auto."""

import json
from collections.abc import Iterator
from pathlib import Path


class LocalStore:
    def __init__(self, base_path: str = "./.rosclaw.auto"):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)
        for sub in [
            "tasks",
            "failures",
            "diagnoses",
            "hypotheses",
            "proposals",
            "patches",
            "experiments",
            "evaluations",
            "champions",
            "deadends",
            "reports",
            "lineage",
        ]:
            (self.base / sub).mkdir(exist_ok=True)

    def _path(self, namespace: str, key: str) -> Path:
        ns_dir = self.base / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir / f"{key}.json"

    def save(self, namespace: str, key: str, data: dict) -> None:
        path = self._path(namespace, key)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, namespace: str, key: str) -> dict | None:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def list_keys(self, namespace: str) -> list[str]:
        d = self.base / namespace
        if not d.exists():
            return []
        return [p.stem for p in d.glob("*.json")]

    def iterate(self, namespace: str) -> Iterator[dict]:
        for key in self.list_keys(namespace):
            data = self.load(namespace, key)
            if data is not None:
                yield data

    def delete(self, namespace: str, key: str) -> bool:
        path = self._path(namespace, key)
        if path.exists():
            path.unlink()
            return True
        return False
