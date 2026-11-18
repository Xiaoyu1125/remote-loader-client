"""Explicit, one-way migration of persisted schema 6 client settings."""

from __future__ import annotations

import json
import os
import ssl
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from remote_loader.client_config import (
    ClientConfig,
    ConfigStore,
    PlacementConfig,
    RuntimeConfig,
    SchedulerConfig,
    TrainingConfig,
    _reject_unknown_fields,
    normalize_beanfleet_endpoint,
    normalize_control_endpoint,
)
from remote_loader.exceptions import ConfigurationError

_LEGACY_FIELDS = {
    "schema_version",
    "control_endpoint",
    "verify_control_tls",
    "allow_insecure_data",
    "beanfleet_endpoint",
    "verify_beanfleet_tls",
    "placement",
    "runtime",
    "scheduler",
    "training",
}


def is_schema6_config(store: ConfigStore) -> bool:
    """Recognize a legacy file for actionable CLI/TUI guidance only."""
    path = store.existing_path()
    if path is None:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and type(payload.get("schema_version")) is int
        and payload["schema_version"] == 6
    )


def _legacy_values(payload: Any) -> dict[str, Any]:
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 6
    ):
        raise ConfigurationError("config migrate requires a schema 6 client configuration")
    _reject_unknown_fields(payload, _LEGACY_FIELDS, "schema 6 client configuration")
    if "control_endpoint" not in payload:
        raise ConfigurationError("schema 6 client configuration requires control_endpoint")
    normalize_control_endpoint(payload["control_endpoint"])
    for name, default in (
        ("verify_control_tls", True),
        ("allow_insecure_data", False),
        ("verify_beanfleet_tls", True),
    ):
        if type(payload.get(name, default)) is not bool:
            raise ConfigurationError(f"schema 6 {name} must be a boolean")
    beanfleet = payload.get("beanfleet_endpoint")
    if beanfleet is not None:
        beanfleet = normalize_beanfleet_endpoint(beanfleet)
        if urlparse(beanfleet).scheme == "http" and payload.get("verify_beanfleet_tls", True):
            raise ConfigurationError("plain HTTP Beanfleet requires verify_beanfleet_tls=false")
    placement_payload = payload.get("placement")
    placement = None if placement_payload is None else PlacementConfig.from_dict(placement_payload)
    runtime = RuntimeConfig.from_dict(payload.get("runtime", {}))
    training = TrainingConfig.from_dict(payload.get("training", {}))
    old_scheduler = payload.get("scheduler")
    scheduler = None
    if old_scheduler is not None:
        if not isinstance(old_scheduler, dict):
            raise ConfigurationError("schema 6 scheduler must be an object")
        _reject_unknown_fields(
            old_scheduler,
            {"endpoint", "control_scheme", "control_port", "verify_tls"},
            "schema 6 scheduler",
        )
        scheme = old_scheduler.get("control_scheme", "http")
        if not isinstance(scheme, str) or scheme not in {"http", "https"}:
            raise ConfigurationError("schema 6 scheduler control_scheme must be http or https")
        if type(old_scheduler.get("verify_tls", True)) is not bool:
            raise ConfigurationError("schema 6 scheduler verify_tls must be a boolean")
        scheduler = SchedulerConfig.from_dict(
            {**old_scheduler, "control_scheme": "https", "verify_tls": True}
        )
    return {
        "beanfleet_endpoint": beanfleet if beanfleet and urlparse(beanfleet).scheme == "https" else None,
        "placement": placement,
        "runtime": runtime,
        "training": training,
        "scheduler": scheduler,
    }


def _save_backup(path: Path, original: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ConfigurationError(f"schema 6 backup already exists: {path}") from exc
    except OSError as exc:
        raise ConfigurationError(f"could not create schema 6 backup: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(original)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise ConfigurationError(f"could not write schema 6 backup: {exc}") from exc


def migrate_schema6(
    store: ConfigStore,
    *,
    control_endpoint: str,
    tls_ca_file: str | None = None,
    tls_client_cert_file: str | None = None,
    tls_client_key_file: str | None = None,
) -> tuple[ClientConfig, Path]:
    """Validate first, save the exact old bytes, then atomically write schema 7."""
    source = store.existing_path()
    if source is None or not source.exists():
        raise ConfigurationError("config migrate requires an existing schema 6 client configuration")
    if source.is_symlink() or not source.is_file():
        raise ConfigurationError("refusing to migrate a non-regular or symlinked client configuration")
    try:
        original = source.read_bytes()
        payload = json.loads(original)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigurationError("schema 6 client configuration is unreadable") from exc
    values = _legacy_values(payload)
    config = ClientConfig(
        control_endpoint=control_endpoint,
        tls_ca_file=tls_ca_file,
        tls_client_cert_file=tls_client_cert_file,
        tls_client_key_file=tls_client_key_file,
        **values,
    )
    try:
        config.create_tls_context()
    except (OSError, ValueError, ssl.SSLError) as exc:
        raise ConfigurationError(f"invalid mTLS certificate material: {exc}") from exc
    target = store.path
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ConfigurationError("refusing to replace a non-regular or symlinked client configuration")
    backup = target.with_name(f"{target.name}.schema6.bak")
    _save_backup(backup, original)
    store.save(config)
    return config, backup
