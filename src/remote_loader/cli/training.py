from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from remote_loader.auth import TokenStore
from remote_loader.client_config import ClientConfig, ConfigStore, SchedulerConfig, TrainingConfig
from remote_loader.exceptions import ConfigurationError, ServiceUnavailableError
from remote_loader.scheduler import SchedulerClient


def load_training_config(path: str | Path) -> TrainingConfig:
    try:
        with Path(path).expanduser().open("rb") as stream:
            return TrainingConfig.from_dict(yaml.safe_load(stream))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"invalid training configuration {path}: {exc}") from exc


class TrainingCommands:
    """Resource actions using the same persisted training settings as the TUI."""

    def __init__(
        self,
        *,
        config_store: ConfigStore | None = None,
        token_store: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config_store = config_store or ConfigStore()
        self.token_store = token_store or TokenStore()
        self.client_factory = client_factory or SchedulerClient

    @contextmanager
    def _client(
        self,
        *,
        training_id: str | None = None,
        scheduler_endpoint: str | None = None,
        verify_scheduler_tls: bool | None = None,
    ) -> Iterator[tuple[ClientConfig, SchedulerConfig, str, Any]]:
        config = self.config_store.load_or_detect()
        identity = config.training.training_id if training_id is None else training_id
        if not isinstance(identity, str) or not identity.strip():
            raise ConfigurationError("specify a training_id or save it using TUI 启动训练")
        previous = config.resolve_scheduler(scheduler_endpoint)
        settings = replace(
            previous,
            verify_tls=previous.verify_tls if verify_scheduler_tls is None else verify_scheduler_tls,
        )
        with closing(
            self.client_factory(
                settings.endpoint,
                token=self.token_store.load(),
                verify=settings.verify_tls,
                connect_timeout=3.0,
                read_timeout=10.0,
                max_retries=0,
            )
        ) as client:
            yield config, settings, identity.strip(), client

    @staticmethod
    def _target(settings: SchedulerConfig, identity: str) -> dict[str, Any]:
        return {"training_id": identity, "scheduler_endpoint": settings.endpoint}

    def register(self, training: TrainingConfig | None = None, **options: Any) -> dict[str, Any]:
        training = training or self.config_store.load_or_detect().training
        request = training.to_request()
        with self._client(training_id=training.training_id, **options) as (
            config,
            settings,
            identity,
            client,
        ):
            try:
                client.create_training_run(request)
            except ServiceUnavailableError as exc:
                raise ServiceUnavailableError(
                    f"registration of {identity!r} at {settings.endpoint} was not confirmed; "
                    "check training status before retrying with the same ID and parameters"
                ) from exc
        try:
            self.config_store.save(replace(config, scheduler=settings, training=training))
        except Exception as exc:
            raise ConfigurationError(
                f"training {identity!r} was registered, but saving its settings failed; "
                "use the submitted training_id and scheduler endpoint for subsequent commands"
            ) from exc
        return {**self._target(settings, identity), "status": "registered"}

    def status(self, **options: Any) -> dict[str, Any]:
        with self._client(**options) as (_config, settings, identity, client):
            resources = client.training_status(identity)
        return {**self._target(settings, identity), "resources": resources}

    def release_rank(self, local_rank: int, **options: Any) -> dict[str, Any]:
        with self._client(**options) as (_config, settings, identity, client):
            result = client.release_rank(identity, local_rank=local_rank)
        return {**self._target(settings, identity), "local_rank": local_rank, "result": result}

    def end(self, **options: Any) -> dict[str, Any]:
        with self._client(**options) as (_config, settings, identity, client):
            result = client.end_training(identity)
        return {**self._target(settings, identity), "result": result}
