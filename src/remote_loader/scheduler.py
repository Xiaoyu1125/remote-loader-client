from __future__ import annotations

import math
import os
import time
from ipaddress import ip_address
from typing import Any, ClassVar
from urllib.parse import quote

import httpx

from remote_loader.client_config import (
    DEFAULT_SCHEDULER_ENDPOINT,
    ClientConfig,
    normalize_scheduler_endpoint,
    validate_training_id_for_path,
)
from remote_loader.control.api_client import raise_for_api_error
from remote_loader.exceptions import ConfigurationError, ServiceUnavailableError


class SchedulerClient:
    RETRYABLE: ClassVar[set[int]] = {408, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        endpoint: str = DEFAULT_SCHEDULER_ENDPOINT,
        *,
        token: str | None = None,
        verify: bool = True,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        max_retries: int = 3,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if type(verify) is not bool:
            raise ConfigurationError("scheduler verify must be a boolean")
        if type(max_retries) is not int or max_retries < 0:
            raise ConfigurationError("scheduler max_retries must be a non-negative integer")
        for timeout in (connect_timeout, read_timeout):
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout)
                or timeout <= 0
            ):
                raise ConfigurationError("scheduler timeout must be a finite positive number")
        if token is not None and (not isinstance(token, str) or not token.strip()):
            raise ConfigurationError("scheduler token must be a non-empty string")
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=normalize_scheduler_endpoint(endpoint),
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
            verify=verify,
            transport=transport,
            headers={} if token is None else {"Authorization": f"Bearer {token}"},
        )

    def create_training_run(self, request: dict[str, Any]) -> str:
        """Submit one explicit registration; never retry this mutation automatically."""
        identity = self._identity(request.get("training_id"))
        validate_training_id_for_path(identity)
        request = {**request, "training_id": identity}
        try:
            response = self._client.post("/training_runs", json=request)
        except httpx.RequestError as exc:
            raise ServiceUnavailableError(
                "scheduler registration request failed; check the training ID before retrying"
            ) from exc
        raise_for_api_error(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ConfigurationError(
                "scheduler registration response is not JSON; check the training ID before retrying"
            ) from exc
        if not isinstance(payload, dict) or payload.get("training_id") != request["training_id"]:
            raise ConfigurationError(
                "scheduler registration response has a different or missing training_id; "
                "check the training state before retrying"
            )
        return payload["training_id"]

    def get_nodes(self, training_id: str | None, local_rank: int, *, vqa: bool = False) -> tuple[str, ...]:
        if not isinstance(training_id, str) or not training_id.strip():
            raise ConfigurationError(
                "scheduler requires a preassigned training_id in the client configuration"
            )
        if type(local_rank) is not int or local_rank < 0:
            raise ConfigurationError("scheduler local_rank must be a non-negative integer")
        params: dict[str, str | int] = {"training_id": training_id.strip(), "local_rank": local_rank}
        response = self._request("GET", "/dataloader_server_nodes", params=params)
        try:
            payload = response.json()
        except ValueError as exc:
            raise ConfigurationError("scheduler response must be a JSON object") from exc
        if not isinstance(payload, dict):
            raise ConfigurationError("scheduler response must be a JSON object")
        group = "vqa_nodes" if vqa else "nodes"
        nodes = payload.get(group)
        if not isinstance(nodes, list) or not nodes:
            raise ConfigurationError(f"scheduler response {group} must be a non-empty list")

        result: list[str] = []
        seen: set[str] = set()
        for node in nodes:
            if not isinstance(node, str) or "%" in node:
                raise ConfigurationError(f"scheduler response {group} must contain IP literals")
            try:
                normalized = str(ip_address(node))
            except ValueError as exc:
                raise ConfigurationError(f"scheduler response {group} must contain IP literals") from exc
            if normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return tuple(result)

    @staticmethod
    def _identity(training_id: str) -> str:
        if not isinstance(training_id, str) or not training_id.strip():
            raise ConfigurationError("training_id must be a non-empty string")
        return training_id.strip()

    @staticmethod
    def _object(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ConfigurationError("scheduler response must be a JSON object") from exc
        if not isinstance(payload, dict):
            raise ConfigurationError("scheduler response must be a JSON object")
        return payload

    def training_status(self, training_id: str) -> dict[str, Any]:
        return self._object(
            self._request("GET", "/status", params={"training_id": self._identity(training_id)})
        )

    def release_rank(self, training_id: str, local_rank: int) -> dict[str, Any]:
        if type(local_rank) is not int or local_rank < 0:
            raise ConfigurationError("scheduler local_rank must be a non-negative integer")
        return self._object(
            self._request(
                "DELETE",
                "/dataloader_server_nodes",
                params={"training_id": self._identity(training_id), "local_rank": local_rank},
            )
        )

    def end_training(self, training_id: str) -> dict[str, Any]:
        identity = self._identity(training_id)
        try:
            validate_training_id_for_path(identity)
        except ConfigurationError as exc:
            raise ConfigurationError(
                f"{exc}; for an existing run, use "
                "'remote-loader training release-rank --local-rank <local_rank>' "
                "for each local rank on each training host with the same "
                "--training-id and --scheduler-endpoint"
            ) from exc
        response = self._request("DELETE", f"/training_runs/{quote(identity, safe='')}", allow_missing=True)
        if response.status_code == 404:
            return {"released": False, "freed_cpu_nodes": []}
        return self._object(response)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str | int] | None = None,
        allow_missing: bool = False,
    ) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(method, path, params=params)
                if allow_missing and response.status_code == 404:
                    return response
                if response.status_code not in self.RETRYABLE or attempt == self.max_retries:
                    raise_for_api_error(response)
                    return response
            except httpx.RequestError:
                if attempt == self.max_retries:
                    break
            time.sleep(min(0.25 * (2**attempt), 2.0))
        raise ServiceUnavailableError("scheduler unavailable after retries") from None

    def close(self) -> None:
        self._client.close()


def resolve_control_endpoints(
    config: ClientConfig,
    *,
    local_rank: int | None = None,
    token: str | None = None,
    vqa: bool = False,
) -> tuple[str, ...]:
    if config.scheduler is None:
        return (config.control_endpoint,)

    scheduler = SchedulerClient(
        config.scheduler.endpoint,
        token=token,
        verify=config.scheduler.verify_tls,
    )
    try:
        nodes = scheduler.get_nodes(
            config.training.training_id,
            int(os.environ.get("LOCAL_RANK", "0")) if local_rank is None else local_rank,
            **({"vqa": True} if vqa else {}),
        )
    finally:
        scheduler.close()

    endpoints: list[str] = []
    for node in nodes:
        address = ip_address(node)
        host = f"[{address}]" if address.version == 6 else str(address)
        endpoint = f"{config.scheduler.control_scheme}://{host}:{config.scheduler.control_port}"
        endpoints.append(endpoint)
    return tuple(endpoints)
