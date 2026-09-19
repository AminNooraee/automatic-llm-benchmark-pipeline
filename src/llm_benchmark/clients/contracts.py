"""Request, response, and retry contracts for model communication."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ClientContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatMessage(ClientContract):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    @field_validator("content")
    @classmethod
    def reject_empty_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be empty")
        return value

    @model_validator(mode="after")
    def validate_tool_message(self) -> "ChatMessage":
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "tool" and self.tool_call_id is not None:
            raise ValueError("tool_call_id is only valid for tool messages")
        return self


class GenerationUsage(ClientContract):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class GenerationResult(ClientContract):
    content: str = Field(min_length=1)
    model_name: str
    finish_reason: str | None = None
    usage: GenerationUsage | None = None
    completion_id: str | None = None
    request_id: str | None = None
    latency_ms: float = Field(ge=0)
    attempts: int = Field(ge=1)


class RetryPolicy(ClientContract):
    max_retries: int = Field(default=2, ge=0)
    initial_backoff_seconds: float = Field(default=0.5, ge=0)
    maximum_backoff_seconds: float = Field(default=8.0, ge=0)
    jitter_ratio: float = Field(default=0.1, ge=0, le=1)

    @model_validator(mode="after")
    def validate_backoff_range(self) -> "RetryPolicy":
        if self.maximum_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "maximum_backoff_seconds must be at least initial_backoff_seconds"
            )
        return self
