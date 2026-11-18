from __future__ import annotations

import json
import math
import os
import ssl
import tempfile
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

from remote_loader import cloud_detection
from remote_loader.exceptions import ConfigurationError
from remote_loader.resource_catalog import find_provider, find_region

DEFAULT_SCHEDULER_ENDPOINT = cloud_detection.environment_for_provider("tencent").scheduler_endpoint
_CERTS_DIR = Path(__file__).with_name("certs")


def _normalize_endpoint(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{label} must be a string")
    normalized = value.strip().rstrip("/")
    try:
        parsed = urlparse(normalized)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ConfigurationError(f"{label} must be a valid HTTP(S) URL") from exc
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ConfigurationError(f"{label} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.params or parsed.query or parsed.fragment:
        raise ConfigurationError(f"{label} must not contain credentials, parameters, query, or fragment")
    return parsed.geturl()


def normalize_control_endpoint(value: str) -> str:
    return _normalize_endpoint(value, "control_endpoint")


def normalize_beanfleet_endpoint(value: str) -> str:
    return _normalize_endpoint(value, "beanfleet_endpoint")


def normalize_scheduler_endpoint(value: str) -> str:
    return _normalize_endpoint(value, "scheduler endpoint")


def validate_control_security(control_endpoint: str) -> None:
    if urlparse(control_endpoint).scheme.lower() != "https":
        raise ConfigurationError("control_endpoint must use HTTPS")


def validate_beanfleet_security(beanfleet_endpoint: str, verify_beanfleet_tls: bool) -> None:
    if urlparse(beanfleet_endpoint).scheme.lower() == "http" and verify_beanfleet_tls:
        raise ConfigurationError("plain HTTP requires verify_beanfleet_tls=false in the client configuration")


def validate_training_id_for_path(training_id: str) -> None:
    """Check resource-path constraints when submitting or ending a run, not loading a profile."""
    if (
        training_id in {".", ".."}
        or any(char in training_id for char in "/\\")
        or any(ord(char) < 32 for char in training_id)
    ):
        raise ConfigurationError(
            "training_id must not be '.' or '..' or contain path separators or control characters"
        )


def _reject_unknown_fields(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(payload.keys() - allowed)
    if unknown:
        raise ConfigurationError(f"{label} contains unknown fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class PlacementConfig:
    provider: str
    region: str
    zone: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (self.provider, self.region, self.zone)):
            raise ConfigurationError("placement provider, region and zone must be non-empty strings")
        provider = find_provider(self.provider)
        if provider is None:
            raise ConfigurationError(f"unknown service provider {self.provider!r}")
        region = find_region(provider, self.region)
        if region is None:
            raise ConfigurationError(f"region {self.region!r} is not available for {self.provider!r}")
        if self.zone not in {zone.id for zone in region.zones}:
            raise ConfigurationError(f"zone {self.zone!r} is not available in region {self.region!r}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlacementConfig:
        if not isinstance(payload, dict):
            raise ConfigurationError("placement must be an object")
        _reject_unknown_fields(payload, {"provider", "region", "zone"}, "placement")
        return cls(
            provider=payload.get("provider"),
            region=payload.get("region"),
            zone=payload.get("zone"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "region": self.region,
            "zone": self.zone,
        }


@dataclass(frozen=True)
class RuntimeConfig:
    prefetch_batches: int = 8

    def __post_init__(self) -> None:
        if type(self.prefetch_batches) is not int or not 1 <= self.prefetch_batches <= 1024:
            raise ConfigurationError("prefetch_batches must be an integer between 1 and 1024")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RuntimeConfig:
        if not isinstance(payload, dict):
            raise ConfigurationError("runtime must be an object")
        _reject_unknown_fields(payload, {"prefetch_batches"}, "runtime")
        return cls(prefetch_batches=payload.get("prefetch_batches", 8))

    def to_dict(self) -> dict[str, int]:
        return {"prefetch_batches": self.prefetch_batches}


@dataclass(frozen=True)
class SchedulerConfig:
    endpoint: str | None = None
    control_scheme: str = "https"
    control_port: int = 8080
    verify_tls: bool = True
    _automatic_endpoint: bool = field(default=False, repr=False, compare=False, kw_only=True)

    def __post_init__(self) -> None:
        if self.endpoint is None:
            endpoint = DEFAULT_SCHEDULER_ENDPOINT
            object.__setattr__(self, "_automatic_endpoint", True)
        else:
            endpoint = normalize_scheduler_endpoint(self.endpoint)
        if self.control_scheme != "https":
            raise ConfigurationError("loader control_scheme must be https")
        if type(self.control_port) is not int or not 1 <= self.control_port <= 65535:
            raise ConfigurationError("scheduler control_port must be an integer between 1 and 65535")
        if type(self.verify_tls) is not bool:
            raise ConfigurationError("scheduler verify_tls must be a boolean")
        object.__setattr__(self, "endpoint", endpoint)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SchedulerConfig:
        if not isinstance(payload, dict):
            raise ConfigurationError("scheduler must be an object")
        _reject_unknown_fields(
            payload,
            {"endpoint", "control_scheme", "control_port", "verify_tls"},
            "scheduler",
        )
        if "endpoint" in payload and payload["endpoint"] is None:
            raise ConfigurationError("scheduler endpoint must be a string")
        return cls(
            endpoint=payload.get("endpoint"),
            control_scheme=payload.get("control_scheme", "https"),
            control_port=payload.get("control_port", 8080),
            verify_tls=payload.get("verify_tls", True),
        )

    def to_dict(self) -> dict[str, Any]:
        settings = {
            "control_scheme": self.control_scheme,
            "control_port": self.control_port,
            "verify_tls": self.verify_tls,
        }
        if not self._automatic_endpoint:
            settings["endpoint"] = self.endpoint
        return settings


@dataclass(frozen=True)
class TrainingDataset:
    """A weighted dataset; migrated profiles retain an unknown version until edited."""

    id: str
    weight: float | None = None
    version: int | None = 2

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ConfigurationError("dataset id must be a non-empty string")
        if self.weight is not None and (
            isinstance(self.weight, bool) or not isinstance(self.weight, (int, float))
        ):
            raise ConfigurationError("dataset weight must be a finite non-negative number")
        try:
            weight = None if self.weight is None else float(self.weight)
        except OverflowError as exc:
            raise ConfigurationError("dataset weight must be a finite non-negative number") from exc
        if weight is not None and (not math.isfinite(weight) or weight < 0):
            raise ConfigurationError("dataset weight must be a finite non-negative number")
        if self.version is not None and (type(self.version) is not int or self.version <= 0):
            raise ConfigurationError("dataset version must be a positive integer when provided")
        object.__setattr__(self, "id", self.id.strip())
        object.__setattr__(self, "weight", weight)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingDataset:
        if not isinstance(payload, dict):
            raise ConfigurationError("dataset must be an object")
        _reject_unknown_fields(payload, {"id", "weight", "version"}, "dataset")
        return cls(
            id=payload.get("id"),
            weight=payload.get("weight"),
            version=payload.get("version", 2),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {"id": self.id, "version": self.version}
        if self.weight is not None:
            payload["weight"] = self.weight
        return payload


@dataclass(frozen=True)
class TrainingConfig:
    """One persisted training request; incomplete profiles may be edited before registration."""

    training_id: str | None = None
    num_nodes: int | None = None
    gpus_per_node: int | None = None
    datasets: tuple[TrainingDataset, ...] = ()
    cpu_per_gpu: int | None = None
    vqa: bool = False
    vqa_datasets: tuple[TrainingDataset, ...] = ()

    def __post_init__(self) -> None:
        if self.training_id is not None:
            if not isinstance(self.training_id, str) or not self.training_id.strip():
                raise ConfigurationError("training_id must be a non-empty string when provided")
            object.__setattr__(self, "training_id", self.training_id.strip())
        for name in ("num_nodes", "gpus_per_node", "cpu_per_gpu"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ConfigurationError(f"{name} must be a positive integer when provided")
        if type(self.vqa) is not bool:
            raise ConfigurationError("vqa must be a boolean")
        for name in ("datasets", "vqa_datasets"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or any(
                not isinstance(item, TrainingDataset) for item in values
            ):
                raise ConfigurationError(f"{name} must be a list of TrainingDataset entries")
            object.__setattr__(self, name, tuple(values))
        if not self.vqa:
            object.__setattr__(self, "vqa_datasets", ())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingConfig:
        if not isinstance(payload, dict):
            raise ConfigurationError("training must be an object")
        _reject_unknown_fields(
            payload,
            {"training_id", "num_nodes", "gpus_per_node", "datasets", "cpu_per_gpu", "vqa", "vqa_datasets"},
            "training",
        )
        values = dict(payload)
        for name in ("datasets", "vqa_datasets"):
            entries = values.get(name, ())
            if not isinstance(entries, (list, tuple)):
                raise ConfigurationError(f"{name} must be a list of dataset objects")
            values[name] = tuple(TrainingDataset.from_dict(entry) for entry in entries)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_id": self.training_id,
            "num_nodes": self.num_nodes,
            "gpus_per_node": self.gpus_per_node,
            "datasets": [entry.to_dict() for entry in self.datasets],
            "cpu_per_gpu": self.cpu_per_gpu,
            "vqa": self.vqa,
            "vqa_datasets": [entry.to_dict() for entry in self.vqa_datasets],
        }

    def to_request(self) -> dict[str, Any]:
        if (
            self.training_id is None
            or self.num_nodes is None
            or self.gpus_per_node is None
            or not self.datasets
        ):
            raise ConfigurationError("请在启动训练中填写完整的训练 ID、节点总数、节点GPU数和数据集。")
        validate_training_id_for_path(self.training_id)
        for name in ("datasets", "vqa_datasets") if self.vqa else ("datasets",):
            entries = getattr(self, name)
            if not entries:
                raise ConfigurationError(f"{name} 至少需要一个数据集。")
            if any(entry.version is None for entry in entries):
                raise ConfigurationError(f"{name} 中每个数据集都必须填写 version。")
            weights = [entry.weight for entry in entries]
            if name == "vqa_datasets" and any(weight is None for weight in weights):
                if not all(weight is None for weight in weights):
                    raise ConfigurationError("VQA weight 必须全部填写或全部留空。")
            elif not any((1.0 if weight is None else weight) > 0 for weight in weights):
                raise ConfigurationError(f"{name} 中至少一个数据集的 weight 必须大于 0。")
        request = self.to_dict()
        for dataset in request["datasets"]:
            dataset.setdefault("weight", 1.0)
        if self.cpu_per_gpu is None:
            request.pop("cpu_per_gpu")
        request["sub_epoch_size"] = 128
        return request


@dataclass(frozen=True)
class ClientConfig:
    """Validated client connection and runtime defaults."""

    control_endpoint: str = field(repr=False)
    tls_ca_file: str | None = field(default=None, repr=False)
    tls_client_cert_file: str | None = field(default=None, repr=False)
    tls_client_key_file: str | None = field(default=None, repr=False)
    placement: PlacementConfig | None = None
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    schema_version: int = 7
    beanfleet_endpoint: str | None = field(default=None, repr=False)
    verify_beanfleet_tls: bool = True
    scheduler: SchedulerConfig | None = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    # A live detection wins over the last cloud saved by the training environment.
    _saved_cloud: cloud_detection.CloudEnvironment | None = field(
        default=None, repr=False, compare=False, kw_only=True
    )
    _detected_cloud: cloud_detection.CloudEnvironment | None = field(
        default=None, repr=False, compare=False, kw_only=True
    )

    SYSTEM_PATH: ClassVar[Path] = Path("/etc/remote-loader/client.json")
    USER_PATH: ClassVar[Path] = Path("~/.config/remote-loader/client.json").expanduser()
    ENV_PATH: ClassVar[str] = "REMOTE_LOADER_CONFIG"
    DEFAULT_CONTROL_ENDPOINT: ClassVar[str] = "https://172.28.133.143:8080"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 7:
            raise ConfigurationError("unsupported client configuration schema_version")
        endpoint = normalize_control_endpoint(self.control_endpoint)
        validate_control_security(endpoint)
        tls_paths = (self.tls_ca_file, self.tls_client_cert_file, self.tls_client_key_file)
        if any(
            value is not None and (not isinstance(value, str) or not value.strip()) for value in tls_paths
        ):
            raise ConfigurationError("TLS certificate paths must be non-empty strings when provided")
        if (self.tls_client_cert_file is None) != (self.tls_client_key_file is None):
            raise ConfigurationError(
                "tls_client_cert_file and tls_client_key_file must be configured together"
            )
        beanfleet_endpoint = (
            None if self.beanfleet_endpoint is None else normalize_beanfleet_endpoint(self.beanfleet_endpoint)
        )
        if type(self.verify_beanfleet_tls) is not bool:
            raise ConfigurationError("Beanfleet TLS settings must be booleans")
        if self.scheduler is not None and not isinstance(self.scheduler, SchedulerConfig):
            raise ConfigurationError("scheduler must be a SchedulerConfig or null")
        if not isinstance(self.training, TrainingConfig):
            raise ConfigurationError("training must be a TrainingConfig")
        if beanfleet_endpoint is not None:
            validate_beanfleet_security(beanfleet_endpoint, self.verify_beanfleet_tls)
        if self.placement is not None and not isinstance(self.placement, PlacementConfig):
            raise ConfigurationError("placement must be a PlacementConfig or null")
        if not isinstance(self.runtime, RuntimeConfig):
            raise ConfigurationError("runtime must be a RuntimeConfig")
        object.__setattr__(self, "control_endpoint", endpoint)
        object.__setattr__(self, "beanfleet_endpoint", beanfleet_endpoint)
        if self.scheduler is not None and self.scheduler._automatic_endpoint:
            object.__setattr__(self, "scheduler", self.resolve_scheduler())

    @classmethod
    def load(cls, path: str | Path | None = None) -> ClientConfig:
        return ConfigStore(path).load_or_detect()

    @classmethod
    def default(cls) -> ClientConfig:
        return cls(control_endpoint=cls.DEFAULT_CONTROL_ENDPOINT)

    @property
    def _cloud(self) -> cloud_detection.CloudEnvironment:
        if self._detected_cloud is not None:
            return self._detected_cloud
        try:
            cloud = cloud_detection.detect_cloud_environment()
        except cloud_detection.CloudNotDetectedError:
            if self._saved_cloud is None:
                raise
            cloud = self._saved_cloud
        object.__setattr__(self, "_detected_cloud", cloud)
        return cloud

    def resolve_scheduler(self, endpoint: str | None = None) -> SchedulerConfig:
        settings = self.scheduler or SchedulerConfig()
        if endpoint is not None:
            return replace(settings, endpoint=endpoint, _automatic_endpoint=False)
        if not settings._automatic_endpoint:
            return settings
        try:
            endpoint = self._cloud.scheduler_endpoint
        except cloud_detection.CloudNotDetectedError:
            # Keep the legacy offline scheduler default without granting TLS trust.
            endpoint = DEFAULT_SCHEDULER_ENDPOINT
        return replace(settings, endpoint=endpoint)

    def _custom_tls_files(self) -> tuple[str, str, str] | None:
        if self.tls_ca_file is None and self.tls_client_cert_file is None:
            return None
        if self.tls_ca_file is None or self.tls_client_cert_file is None or self.tls_client_key_file is None:
            raise ConfigurationError(
                "mTLS requires tls_ca_file, tls_client_cert_file, and tls_client_key_file together; "
                "configure your credentials with remote-loader config edit --tls-ca-file, "
                "--tls-client-cert-file, and --tls-client-key-file. "
                "Cloud autodetection does not provision credentials"
            )
        return self.tls_ca_file, self.tls_client_cert_file, self.tls_client_key_file

    def _tls_files(self) -> tuple[Path, Path, Path]:
        custom = self._custom_tls_files()
        if custom is not None:
            ca, cert, key = custom
            return Path(ca).expanduser(), Path(cert).expanduser(), Path(key).expanduser()
        return (
            _CERTS_DIR / f"{self._cloud.provider}-server-ca.crt",
            _CERTS_DIR / "gpu-client.crt",
            _CERTS_DIR / "gpu-client.key",
        )

    def create_tls_context(self) -> ssl.SSLContext:
        ca, cert, key = self._tls_files()
        context = ssl.create_default_context(cafile=ca)
        context.load_cert_chain(certfile=cert, keyfile=key)
        return context

    @classmethod
    def from_path(cls, path: str | Path) -> ClientConfig:
        config_path = Path(path).expanduser()
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigurationError("remote loader client configuration was not found") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError("remote loader client configuration is unreadable") from exc
        if not isinstance(payload, dict):
            raise ConfigurationError("remote loader client configuration must be an object")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ClientConfig:
        if not isinstance(payload, dict):
            raise ConfigurationError("remote loader client configuration must be an object")
        schema_version = payload.get("schema_version", 1)
        if type(schema_version) is not int or schema_version != 7:
            raise ConfigurationError("unsupported client configuration schema_version")
        allowed = {
            "schema_version",
            "control_endpoint",
            "last_detected_cloud",
            "tls_ca_file",
            "tls_client_cert_file",
            "tls_client_key_file",
            "placement",
            "runtime",
        }
        allowed.update({"beanfleet_endpoint", "verify_beanfleet_tls", "scheduler", "training"})
        _reject_unknown_fields(payload, allowed, "client configuration")
        if "control_endpoint" not in payload:
            raise ConfigurationError("client configuration requires control_endpoint")
        placement_payload = payload.get("placement")
        scheduler_payload = payload.get("scheduler")
        saved_provider = payload.get("last_detected_cloud")
        training_payload = payload.get("training", {})
        runtime_payload = payload.get("runtime", {})
        return cls(
            control_endpoint=payload["control_endpoint"],
            _saved_cloud=(
                None if saved_provider is None else cloud_detection.environment_for_provider(saved_provider)
            ),
            beanfleet_endpoint=payload.get("beanfleet_endpoint"),
            tls_ca_file=payload.get("tls_ca_file"),
            tls_client_cert_file=payload.get("tls_client_cert_file"),
            tls_client_key_file=payload.get("tls_client_key_file"),
            verify_beanfleet_tls=payload.get("verify_beanfleet_tls", True),
            placement=None if placement_payload is None else PlacementConfig.from_dict(placement_payload),
            runtime=RuntimeConfig.from_dict(runtime_payload),
            schema_version=7,
            training=TrainingConfig.from_dict(training_payload),
            scheduler=(None if scheduler_payload is None else SchedulerConfig.from_dict(scheduler_payload)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "control_endpoint": self.control_endpoint,
            "beanfleet_endpoint": self.beanfleet_endpoint,
            "tls_ca_file": self.tls_ca_file,
            "tls_client_cert_file": self.tls_client_cert_file,
            "tls_client_key_file": self.tls_client_key_file,
            "verify_beanfleet_tls": self.verify_beanfleet_tls,
            "placement": None if self.placement is None else self.placement.to_dict(),
            "runtime": self.runtime.to_dict(),
            "training": self.training.to_dict(),
        }
        cloud = self._detected_cloud or self._saved_cloud
        if cloud is not None:
            payload["last_detected_cloud"] = cloud.provider
        if self.scheduler is not None:
            payload["scheduler"] = self.scheduler.to_dict()
        return payload


class ConfigStore:
    """Internal persistence for non-secret client configuration."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._explicit_path = None if path is None else Path(path).expanduser()

    @property
    def path(self) -> Path:
        if self._explicit_path is not None:
            return self._explicit_path
        configured_path = os.environ.get(ClientConfig.ENV_PATH)
        if configured_path is not None:
            return Path(configured_path).expanduser()
        return ClientConfig.USER_PATH

    def load(self) -> ClientConfig:
        path = self._load_path()
        if path is None:
            return ClientConfig.default()
        return ClientConfig.from_path(path)

    def load_or_detect(
        self,
        *,
        detector: Callable[[], cloud_detection.CloudEnvironment] | None = None,
        persist: bool = False,
    ) -> ClientConfig:
        path = self._load_path()
        if path is not None and path.exists():
            return ClientConfig.from_path(path)
        detected = (detector or cloud_detection.detect_cloud_environment)()
        config = ClientConfig(
            control_endpoint=ClientConfig.DEFAULT_CONTROL_ENDPOINT,
            runtime=RuntimeConfig(),
            scheduler=SchedulerConfig(),
            _detected_cloud=detected,
        )
        if persist:
            self.save(config)
        return config

    def load_optional(self) -> ClientConfig | None:
        path = self._load_path()
        if path is None or not path.exists():
            return None
        return ClientConfig.from_path(path)

    def existing_path(self) -> Path | None:
        """Return the config selected by the normal environment/user/system precedence."""
        return self._load_path()

    def save(self, config: ClientConfig) -> None:
        if not isinstance(config, ClientConfig):
            raise ConfigurationError("ConfigStore.save requires a ClientConfig")
        config._custom_tls_files()
        path = self.path
        parent = path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with suppress(OSError):
                os.chmod(parent, 0o700)
            if path.is_symlink():
                raise ConfigurationError("refusing to replace a symlinked client configuration")
            if path.exists() and not path.is_file():
                raise ConfigurationError("client configuration path is not a regular file")
        except OSError as exc:
            raise ConfigurationError(f"could not prepare client configuration: {exc}") from exc

        temporary_path: Path | None = None
        try:
            descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
            temporary_path = Path(raw_path)
            os.chmod(temporary_path, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(config.to_dict(), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        except OSError as exc:
            raise ConfigurationError(f"could not save client configuration: {exc}") from exc
        finally:
            if temporary_path is not None:
                with suppress(FileNotFoundError):
                    temporary_path.unlink()

    def _load_path(self) -> Path | None:
        if self._explicit_path is not None:
            return self._explicit_path
        configured_path = os.environ.get(ClientConfig.ENV_PATH)
        if configured_path is not None:
            return Path(configured_path).expanduser()
        for candidate in (ClientConfig.USER_PATH, ClientConfig.SYSTEM_PATH):
            if candidate.is_file():
                return candidate
        return None
