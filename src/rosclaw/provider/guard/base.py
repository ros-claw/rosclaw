"""Guard ABC.

Guards sit between Provider output and Runtime execution.
They validate structured outputs against safety constraints.
"""

from abc import ABC, abstractmethod
from typing import Any

from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


class Guard(ABC):
    """Abstract base for output guards."""

    name: str = ""

    @abstractmethod
    def check(
        self,
        request: ProviderRequest,
        response: ProviderResponse,
    ) -> dict[str, Any]:
        """Check a provider response.

        Returns:
            {
                "pass": bool,
                "checks": [{"name": str, "status": "pass"|"fail", "detail": str}],
                "reason": str,
                "recommended_action": str,
            }
        """
        ...
