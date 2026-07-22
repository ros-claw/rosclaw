"""HTTP helpers for endpoints hosted on the local machine."""

from __future__ import annotations

import urllib.request
from typing import Any
from urllib.parse import urlparse

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def urlopen_with_loopback_bypass(
    request: str | urllib.request.Request,
    *,
    timeout: float,
) -> Any:
    """Open a URL, bypassing environment proxies only for loopback hosts."""
    url = request.full_url if isinstance(request, urllib.request.Request) else request
    if urlparse(url).hostname in _LOOPBACK_HOSTS:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)
