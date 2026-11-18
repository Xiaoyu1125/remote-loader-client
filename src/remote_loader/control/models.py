from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SessionStatus(str, Enum):
    CREATED = "CREATED"
    WARMING = "WARMING"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    CLOSED = "CLOSED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


class SessionInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str
    status: SessionStatus
    schema_hash: str
    protocol_version: int = Field(default=1, ge=1)
    data_endpoint: str | None = None
    stream_token: str | None = None
    expires_at: str | None = None
    tls: bool = True
    dataset_metadata: dict[str, Any] = Field(default_factory=dict)
    sampler_version: str | None = None
    # Validate limits after the runtime owns the session and can close it.
    frame_limits: Any = Field(default_factory=dict)
    error: str | None = None
