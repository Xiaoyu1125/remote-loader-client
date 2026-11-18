from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LoaderCheckpoint(BaseModel):
    """Position of the last fully consumed remote frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str | None = None
    last_consumed_sequence: int = Field(default=-1, ge=-1)
    last_batch_id: str | None = None
    schema_hash: str
    batch_spec_hash: str
    rank: int = Field(default=0, ge=0)
    world_size: int = Field(default=1, ge=1)

    def to_state_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> LoaderCheckpoint:
        return cls.model_validate(state)
