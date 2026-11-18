from __future__ import annotations

import logging
import re
from typing import Any

TOKEN_PATTERN = re.compile(r"(?i)(bearer\s+|stream_token[=:]\s*|access_token[=:]\s*)([^\s,;}]+)")


def redact_token(value: str) -> str:
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


class SensitiveDataFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = TOKEN_PATTERN.sub(lambda match: match.group(1) + "***", record.msg)
        if record.args:
            record.args = tuple(_redact_arg(value) for value in record.args)
        return True


def _redact_arg(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("***" if any(word in str(key).lower() for word in ("token", "secret", "key")) else item)
            for key, item in value.items()
        }
    return value


def configure_logging(level: int = logging.INFO) -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    handler = logging.StreamHandler()
    handler.addFilter(SensitiveDataFilter())
    logging.basicConfig(level=level, handlers=[handler], format="%(levelname)s %(name)s: %(message)s")
