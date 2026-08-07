"""Construct optional Know/How transports without owning their algorithms."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .health import combine_health
from .how_client import DisabledHowClient, HttpHowClient, InProcessHowClient
from .know_client import DisabledKnowClient, HttpKnowClient, InProcessKnowClient

KnowledgeMode = Literal["disabled", "service", "inprocess"]


@dataclass(frozen=True)
class KnowledgeServiceConfig:
    mode: KnowledgeMode = "disabled"
    know_url: str | None = None
    how_url: str | None = None
    know_api_key: str = ""
    how_api_key: str = ""
    timeout: float = 15.0
    know_store_mode: Literal["embedded", "server", "memory"] = "embedded"
    know_store_path: str | None = None
    allow_test_memory: bool = False
    seekdb_host: str | None = field(default_factory=lambda: os.environ.get("SEEKDB_HOST"))
    seekdb_port: int = field(default_factory=lambda: int(os.environ.get("SEEKDB_PORT", "2881")))
    seekdb_tenant: str = field(default_factory=lambda: os.environ.get("SEEKDB_TENANT", "sys"))
    seekdb_user: str = field(default_factory=lambda: os.environ.get("SEEKDB_USER", "root"))
    seekdb_password: str = field(default_factory=lambda: os.environ.get("SEEKDB_PASSWORD", ""))
    know_database: str = field(
        default_factory=lambda: os.environ.get("ROSCLAW_KNOW_DATABASE", "rosclaw_know")
    )
    memory_database: str | None = field(
        default_factory=lambda: os.environ.get("ROSCLAW_MEMORY_DATABASE")
    )
    practice_database: str | None = field(
        default_factory=lambda: os.environ.get("ROSCLAW_PRACTICE_DATABASE")
    )
    memory_path: str | None = field(
        default_factory=lambda: os.environ.get("ROSCLAW_MEMORY_SEEKDB_PATH")
    )
    practice_path: str | None = field(
        default_factory=lambda: os.environ.get("ROSCLAW_PRACTICE_SEEKDB_PATH")
    )

    @classmethod
    def from_env(cls, *, workspace_home: str | Path | None = None) -> KnowledgeServiceConfig:
        mode = os.environ.get("ROSCLAW_KNOWLEDGE_MODE", "disabled").casefold()
        if mode not in {"disabled", "service", "inprocess"}:
            raise ValueError(f"unsupported ROSCLAW_KNOWLEDGE_MODE: {mode!r}")
        root = Path(workspace_home).expanduser() if workspace_home else Path.cwd()
        return cls(
            mode=mode,  # type: ignore[arg-type]
            know_url=os.environ.get("ROSCLAW_KNOW_URL"),
            how_url=os.environ.get("ROSCLAW_HOW_V2_URL") or os.environ.get("ROSCLAW_HOW_URL"),
            know_api_key=os.environ.get("ROSCLAW_KNOW_API_KEY", ""),
            how_api_key=os.environ.get("ROSCLAW_HOW_API_KEY", ""),
            timeout=float(os.environ.get("ROSCLAW_KNOWLEDGE_TIMEOUT", "15")),
            know_store_mode=os.environ.get(  # type: ignore[arg-type]
                "ROSCLAW_KNOW_STORE_MODE", "embedded"
            ).casefold(),
            know_store_path=os.environ.get(
                "ROSCLAW_KNOW_SEEKDB_PATH", str(root / "data" / "know" / "seekdb")
            ),
            allow_test_memory=os.environ.get("ROSCLAW_KNOW_ALLOW_TEST_MEMORY") == "1",
            seekdb_host=os.environ.get("SEEKDB_HOST"),
            seekdb_port=int(os.environ.get("SEEKDB_PORT", "2881")),
            seekdb_tenant=os.environ.get("SEEKDB_TENANT", "sys"),
            seekdb_user=os.environ.get("SEEKDB_USER", "root"),
            seekdb_password=os.environ.get("SEEKDB_PASSWORD", ""),
            know_database=os.environ.get("ROSCLAW_KNOW_DATABASE", "rosclaw_know"),
            memory_database=os.environ.get("ROSCLAW_MEMORY_DATABASE"),
            practice_database=os.environ.get("ROSCLAW_PRACTICE_DATABASE"),
            memory_path=os.environ.get("ROSCLAW_MEMORY_SEEKDB_PATH"),
            practice_path=os.environ.get("ROSCLAW_PRACTICE_SEEKDB_PATH"),
        )


class KnowledgeServiceManager:
    """Lifecycle owner for adapters only; never starts services implicitly."""

    def __init__(
        self,
        config: KnowledgeServiceConfig,
        *,
        know_client: Any | None = None,
        how_client: Any | None = None,
        inprocess_store: Any | None = None,
    ) -> None:
        self.config = config
        self._store = inprocess_store
        self._owns_store = False
        self.startup_error: str | None = None
        if know_client is not None:
            self.know = know_client
        else:
            self.know = self._build_know()
        if how_client is not None:
            self.how = how_client
        else:
            self.how = self._build_how()

    def _build_know(self) -> Any:
        if self.config.mode == "disabled":
            return DisabledKnowClient()
        if self.config.mode == "service":
            if not self.config.know_url:
                self.startup_error = "service mode requires ROSCLAW_KNOW_URL"
                return DisabledKnowClient()
            return HttpKnowClient(
                self.config.know_url,
                api_key=self.config.know_api_key,
                timeout=self.config.timeout,
            )
        try:
            if self._store is None:
                from rosclaw_know.store import create_know_store

                store_kwargs: dict[str, Any] = {
                    "mode": self.config.know_store_mode,
                    "allow_test_memory": self.config.allow_test_memory,
                }
                if self.config.know_store_mode == "embedded":
                    store_kwargs.update(
                        path=self.config.know_store_path,
                        database=self.config.know_database,
                        memory_database=self.config.memory_database,
                        practice_database=self.config.practice_database,
                        memory_path=self.config.memory_path,
                        practice_path=self.config.practice_path,
                    )
                elif self.config.know_store_mode == "server":
                    store_kwargs.update(
                        host=self.config.seekdb_host,
                        port=self.config.seekdb_port,
                        tenant=self.config.seekdb_tenant,
                        user=self.config.seekdb_user,
                        password=self.config.seekdb_password,
                        database=self.config.know_database,
                        memory_database=self.config.memory_database,
                        practice_database=self.config.practice_database,
                    )
                self._store = create_know_store(**store_kwargs)
                self._owns_store = True
            return InProcessKnowClient(self._store)
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            self.startup_error = f"Know in-process unavailable: {type(exc).__name__}: {exc}"
            return DisabledKnowClient()

    def _build_how(self) -> Any:
        if self.config.mode == "disabled":
            return DisabledHowClient()
        if self.config.mode == "service":
            if not self.config.how_url:
                self.startup_error = self.startup_error or (
                    "service mode requires ROSCLAW_HOW_V2_URL or ROSCLAW_HOW_URL"
                )
                return DisabledHowClient()
            return HttpHowClient(
                self.config.how_url,
                api_key=self.config.how_api_key,
                timeout=self.config.timeout,
            )
        try:
            if isinstance(self.know, DisabledKnowClient):
                return DisabledHowClient()
            return InProcessHowClient(self.know)
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            self.startup_error = f"How in-process unavailable: {type(exc).__name__}: {exc}"
            return DisabledHowClient()

    def health(self) -> dict[str, Any]:
        result = combine_health(self.config.mode, self.know, self.how).as_dict()
        if self.startup_error:
            result["startup_error"] = self.startup_error
            result["status"] = "degraded"
        return result

    def close(self) -> None:
        if self._owns_store and self._store is not None and hasattr(self._store, "close"):
            self._store.close()
        self._store = None
        self._owns_store = False
