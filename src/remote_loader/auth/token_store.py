from __future__ import annotations

import json
import os
from contextlib import suppress
from pathlib import Path

from remote_loader.exceptions import AuthenticationError


class TokenStore:
    """Stores a pre-issued token in a mode-0600 credential file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or "~/.config/remote-loader/credentials.json").expanduser()

    def save(self, token: str) -> None:
        if not isinstance(token, str) or not token:
            raise AuthenticationError("refusing to store a token that is not a non-empty string")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with suppress(OSError):
            os.chmod(self.path.parent, 0o700)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(self.path, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"token": token}, stream)
        finally:
            with suppress(OSError):
                os.chmod(self.path, 0o600)

    def load(self) -> str:
        try:
            mode = self.path.stat().st_mode & 0o777
            if mode & 0o077:
                raise AuthenticationError(
                    f"credential file must not be accessible by group or others, got mode {mode:04o}"
                )
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            token = payload.get("token")
        except FileNotFoundError as exc:
            raise AuthenticationError("token is not configured; run 'remote-loader token save'") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise AuthenticationError(f"could not read credential store: {exc}") from exc
        if not isinstance(token, str) or not token:
            raise AuthenticationError("credential store does not contain a valid token")
        return token

    def delete(self) -> None:
        with suppress(FileNotFoundError):
            self.path.unlink()
