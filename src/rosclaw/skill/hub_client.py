"""Thin HTTP client for the ROSClaw Skill Hub API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class SkillHubClient:
    """Client for ``/api/skills`` metadata registration."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; rosclaw-cli/1.0)",
        }

    def _request(
        self,
        method: str,
        path: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        body = json.dumps(data, ensure_ascii=False).encode("utf-8") if data is not None else None
        request = urllib.request.Request(
            url,
            data=body,
            headers=self._headers(),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                text = response.read().decode("utf-8")
                if not text:
                    return {"status_code": response.getcode()}
                return dict(json.loads(text), status_code=response.getcode())
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            return {
                "status_code": exc.code,
                "error": exc.reason,
                "body": body_text,
            }
        except urllib.error.URLError as exc:
            return {"status_code": 0, "error": str(exc.reason)}

    def create_skill(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/skills", payload)

    def update_skill(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("PUT", f"/api/skills/{name}", payload)

    def get_skill(self, name: str) -> dict[str, Any] | None:
        res = self._request("GET", f"/api/skills/{name}")
        if res.get("status_code") == 200:
            return res
        return None
