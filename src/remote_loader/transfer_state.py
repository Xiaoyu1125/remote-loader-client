from __future__ import annotations

import json
import os
import tempfile
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from remote_loader.batch_spec import BatchSpec, CanonicalSampleSpec, VQABatchSpec
from remote_loader.exceptions import ConfigurationError

TransferStatus = Literal[
    "STARTING",
    "RUNNING",
    "PAUSED",
    "COMPLETED",
    "STOPPED",
    "FAILED",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TransferSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    transfer_id: str = Field(min_length=1)
    status: TransferStatus
    batch_spec: dict[str, Any]
    session_id: str | None = None
    started_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None
    last_batch_at: datetime | None = None
    batches_received: int = Field(default=0, ge=0)
    batches_consumed: int = Field(default=0, ge=0)
    bytes_received: int = Field(default=0, ge=0)

    @field_validator("batch_spec")
    @classmethod
    def validate_batch_spec(cls, value: dict[str, Any]) -> dict[str, Any]:
        return BatchSpec.from_dict(value).to_request()

    @property
    def spec(self) -> BatchSpec | CanonicalSampleSpec | VQABatchSpec:
        return BatchSpec.from_dict(self.batch_spec)


class TransferState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[4] = 4
    current_transfer: TransferSnapshot | None = None
    last_transfer: TransferSnapshot | None = None


class TransferStateStore:
    """Persists the current and most recent local transfer snapshots."""

    USER_PATH: ClassVar[Path] = Path("~/.local/state/remote-loader/state.json").expanduser()

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).expanduser() if path is not None else self.USER_PATH

    def load(self) -> TransferState:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return TransferState()
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError("remote loader transfer state is unreadable") from exc
        if (
            isinstance(payload, dict)
            and type(payload.get("schema_version")) is int
            and payload["schema_version"] in {1, 2, 3}
            and payload.keys() <= {"schema_version", "current_transfer", "last_transfer"}
        ):
            # Retired history embeds dataset selectors or batched VLA requests.
            # Reading does not rewrite it; begin() persists the current schema.
            return TransferState()
        try:
            return TransferState.model_validate(payload)
        except ValidationError as exc:
            raise ConfigurationError(f"remote loader transfer state is invalid: {exc}") from exc

    def save(self, state: TransferState) -> None:
        if not isinstance(state, TransferState):
            raise ConfigurationError("TransferStateStore.save requires a TransferState")
        parent = self.path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with suppress(OSError):
                os.chmod(parent, 0o700)
            if self.path.is_symlink():
                raise ConfigurationError("refusing to replace a symlinked transfer state")
            if self.path.exists() and not self.path.is_file():
                raise ConfigurationError("transfer state path is not a regular file")
        except OSError as exc:
            raise ConfigurationError(f"could not prepare transfer state: {exc}") from exc

        temporary_path: Path | None = None
        try:
            descriptor, raw_path = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=parent)
            temporary_path = Path(raw_path)
            os.chmod(temporary_path, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(state.model_dump(mode="json"), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
            temporary_path = None
            os.chmod(self.path, 0o600)
        except OSError as exc:
            raise ConfigurationError(f"could not save transfer state: {exc}") from exc
        finally:
            if temporary_path is not None:
                with suppress(FileNotFoundError):
                    temporary_path.unlink()

    def begin(self, spec: BatchSpec | CanonicalSampleSpec | VQABatchSpec) -> TransferSnapshot:
        now = utc_now()
        snapshot = TransferSnapshot(
            transfer_id=str(uuid.uuid4()),
            status="STARTING",
            batch_spec=spec.to_request(),
            started_at=now,
            updated_at=now,
        )
        state = self.load()
        self.save(state.model_copy(update={"current_transfer": snapshot}))
        return snapshot

    def publish(self, snapshot: TransferSnapshot) -> bool:
        state = self.load()
        current = state.current_transfer
        if current is None or current.transfer_id != snapshot.transfer_id:
            return False
        self.save(state.model_copy(update={"current_transfer": snapshot}))
        return True

    def finish(self, snapshot: TransferSnapshot) -> bool:
        state = self.load()
        current = state.current_transfer
        if current is None or current.transfer_id != snapshot.transfer_id:
            return False
        self.save(
            state.model_copy(
                update={
                    "current_transfer": None,
                    "last_transfer": snapshot,
                }
            )
        )
        return True
