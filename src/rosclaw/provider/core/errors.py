"""ROSClaw Provider - Custom exceptions."""


class ProviderError(Exception):
    """Base exception for all provider-related errors."""

    def __init__(self, message: str, provider: str = "", request_id: str = ""):
        super().__init__(message)
        self.provider = provider
        self.request_id = request_id


class ProviderNotFoundError(ProviderError):
    """Raised when no provider matches the requested capability."""


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is registered but unhealthy/offline."""


class CapabilityNotSupportedError(ProviderError):
    """Raised when a provider does not support the requested capability."""


class GuardBlockedError(ProviderError):
    """Raised when guard blocks a provider output from reaching runtime."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        request_id: str = "",
        checks: list[dict] | None = None,
        recommended_action: str = "",
    ):
        super().__init__(message, provider, request_id)
        self.checks = checks or []
        self.recommended_action = recommended_action


class ManifestValidationError(ProviderError):
    """Raised when a provider.yaml manifest is invalid."""


class RuntimeAdapterError(ProviderError):
    """Raised when a runtime adapter fails to invoke a model/skill."""
