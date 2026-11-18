from __future__ import annotations

import ssl
import time
import uuid
from collections.abc import Mapping
from typing import Any, ClassVar

import httpx

from remote_loader.batch_spec import BatchSpec, CanonicalSampleSpec, VQABatchSpec
from remote_loader.checkpoint import LoaderCheckpoint
from remote_loader.client_config import normalize_control_endpoint, validate_control_security
from remote_loader.exceptions import (
    AuthenticationError,
    ConfigurationError,
    PermissionDeniedError,
    RateLimitError,
    RemoteLoaderError,
    ServiceUnavailableError,
    SessionExpiredError,
    SessionNotFoundError,
    StatisticsTimeoutError,
)

from .models import SessionInfo


def raise_for_api_error(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        payload = response.json()
        detail = payload.get("detail") or payload.get("message") or payload.get("error")
    except (ValueError, AttributeError):
        detail = None
    message = str(detail or f"remote service returned HTTP {response.status_code}")
    status = response.status_code
    if status == 401:
        raise AuthenticationError(message)
    if status == 403:
        raise PermissionDeniedError(message)
    if status == 404:
        raise SessionNotFoundError(message)
    if status in {408, 410}:
        raise SessionExpiredError(message)
    if status == 422:
        raise ConfigurationError(message)
    if status == 429:
        raise RateLimitError(message)
    if status >= 500:
        raise ServiceUnavailableError(message)
    raise RemoteLoaderError(message)


class ControlClient:
    """HTTP control-plane client. Large batch payloads never pass through it."""

    RETRYABLE: ClassVar[set[int]] = {408, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        endpoint: str,
        token: str,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        max_retries: int = 3,
        ssl_context: ssl.SSLContext | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        endpoint = normalize_control_endpoint(endpoint)
        validate_control_security(endpoint)
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=endpoint.rstrip("/"),
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
            verify=ssl_context or True,
            transport=transport,
            trust_env=False,
            headers={"Authorization": f"Bearer {token}"},
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        deadline: float | None = None,
    ) -> httpx.Response:
        headers = {"X-Request-ID": str(uuid.uuid4())}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        last_error: httpx.RequestError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                options: dict[str, Any] = {}
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise StatisticsTimeoutError(
                            "statistics wait timed out; remote job was not cancelled"
                        )
                    # Bound all four HTTP timeout phases by the caller's remaining budget.
                    options["timeout"] = httpx.Timeout(remaining / 4)
                response = self._client.request(method, path, json=json, headers=headers, **options)
                if response.status_code not in self.RETRYABLE or attempt == self.max_retries:
                    raise_for_api_error(response)
                    return response
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
            delay = min(0.25 * (2**attempt), 2.0)
            if deadline is not None:
                delay = min(delay, max(0.0, deadline - time.monotonic()))
            time.sleep(delay)
        if deadline is not None and time.monotonic() >= deadline:
            raise StatisticsTimeoutError(
                "statistics wait request timed out; remote job was not cancelled"
            ) from last_error
        raise ServiceUnavailableError("control service unavailable after retries") from last_error

    def create_session(
        self,
        spec: BatchSpec | CanonicalSampleSpec | VQABatchSpec,
        *,
        prefetch_batches: int,
        rank: int,
        world_size: int,
        checkpoint: LoaderCheckpoint | None = None,
    ) -> SessionInfo:
        payload: dict[str, Any] = {
            "batch_spec": spec.to_request(),
            "runtime": {
                "prefetch_batches": prefetch_batches,
                "world_size": world_size,
                "rank": rank,
            },
        }
        if checkpoint is not None:
            payload["resume"] = checkpoint.to_state_dict()
        response = self._request("POST", "/v1/sessions", json=payload, idempotency_key=str(uuid.uuid4()))
        return SessionInfo.model_validate(response.json())

    def get_metadata(self) -> dict[str, Any]:
        """Read this server's ordered scheduler catalog without creating a session."""
        response = self._request("GET", "/v1/datasets/metadata")
        value = response.json()
        if not isinstance(value, dict):
            raise ConfigurationError("dataset metadata response must be an object")
        return value

    def submit_statistics(
        self, catalog_id: str, spec: Mapping[str, Any], *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        return self._statistics_response(
            self._request(
                "POST",
                "/v1/statistics",
                json={"catalog_id": catalog_id, "spec": spec},
                idempotency_key=idempotency_key or str(uuid.uuid4()),
            )
        )

    def get_statistics(self, job_id: str, *, deadline: float | None = None) -> dict[str, Any]:
        return self._statistics_response(
            self._request(
                "GET",
                f"/v1/statistics/{job_id}",
                deadline=deadline,
            )
        )

    def get_statistics_result(self, job_id: str, *, deadline: float | None = None) -> dict[str, Any]:
        return self._statistics_response(
            self._request(
                "GET",
                f"/v1/statistics/{job_id}/result",
                deadline=deadline,
            )
        )

    def cancel_statistics(self, job_id: str) -> dict[str, Any]:
        return self._statistics_response(
            self._request(
                "POST",
                f"/v1/statistics/{job_id}/cancel",
                idempotency_key=str(uuid.uuid4()),
            )
        )

    @staticmethod
    def _statistics_response(response: httpx.Response) -> dict[str, Any]:
        from remote_loader.exceptions import ProtocolError

        try:
            value = response.json()
        except ValueError as exc:
            raise ProtocolError("statistics response must be JSON") from exc
        if not isinstance(value, dict):
            raise ProtocolError("statistics response must be an object")
        return value

    def get_session(self, session_id: str) -> SessionInfo:
        response = self._request("GET", f"/v1/sessions/{session_id}")
        return SessionInfo.model_validate(response.json())

    def pause_session(self, session_id: str, last_consumed_sequence: int) -> SessionInfo:
        response = self._request(
            "POST",
            f"/v1/sessions/{session_id}/pause",
            json={"last_consumed_sequence": last_consumed_sequence},
            idempotency_key=str(uuid.uuid4()),
        )
        return SessionInfo.model_validate(response.json())

    def resume_session(
        self, session_id: str, *, last_consumed_sequence: int, schema_hash: str
    ) -> SessionInfo:
        response = self._request(
            "POST",
            f"/v1/sessions/{session_id}/resume",
            json={
                "last_consumed_sequence": last_consumed_sequence,
                "schema_hash": schema_hash,
            },
            idempotency_key=str(uuid.uuid4()),
        )
        return SessionInfo.model_validate(response.json())

    def close_session(self, session_id: str) -> None:
        self._request("POST", f"/v1/sessions/{session_id}/close", idempotency_key=str(uuid.uuid4()))

    def get_metrics(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/sessions/{session_id}/metrics").json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/v1/health").json()

    def close(self) -> None:
        self._client.close()
