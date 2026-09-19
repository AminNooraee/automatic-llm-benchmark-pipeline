"""Normalized errors emitted by the model communication layer."""

from __future__ import annotations

from llm_benchmark.exceptions import BenchmarkError


class ModelClientError(BenchmarkError):
    """Base class for model request and response failures."""


class ModelRequestValidationError(ModelClientError):
    """Raised before network access when a request is invalid."""


class ModelConnectionError(ModelClientError):
    def __init__(self, model_name: str, attempts: int) -> None:
        self.model_name = model_name
        self.attempts = attempts
        super().__init__(
            f"Unable to reach the configured endpoint for model '{model_name}' "
            f"after {attempts} attempt(s)"
        )


class ModelTimeoutError(ModelClientError):
    def __init__(self, model_name: str, timeout_seconds: float, attempts: int) -> None:
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.attempts = attempts
        super().__init__(
            f"Model '{model_name}' timed out after {timeout_seconds:g} seconds "
            f"and {attempts} attempt(s)"
        )


class ModelAPIError(ModelClientError):
    def __init__(
        self,
        *,
        model_name: str,
        status_code: int,
        message: str,
        attempts: int,
        error_type: str | None = None,
        error_code: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.status_code = status_code
        self.attempts = attempts
        self.error_type = error_type
        self.error_code = error_code
        super().__init__(
            f"Model API request for '{model_name}' failed with HTTP "
            f"{status_code}: {message}"
        )


class ModelAuthenticationError(ModelAPIError):
    """Raised for HTTP 401 and 403 responses without retrying."""


class ModelResponseError(ModelClientError):
    def __init__(self, model_name: str, reason: str, attempts: int) -> None:
        self.model_name = model_name
        self.attempts = attempts
        super().__init__(
            f"Malformed Chat Completions response for model '{model_name}': {reason}"
        )

