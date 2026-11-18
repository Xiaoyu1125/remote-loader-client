from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import httpx

from remote_loader.cloud_network import CloudNetworkError


class BeanfleetPeeringStatus(str, Enum):
    CREATING = "Creating"
    ACCEPTING = "Accepting"
    UPDATING = "Updating"
    REJECTED = "Rejected"
    EXPIRED = "Expired"
    ACTIVATED = "Activated"
    READY = "Ready"
    DELETING = "Deleting"
    DELETED = "Deleted"

    def __str__(self) -> str:
        return self.value


TERMINAL_PEERING_STATUSES = frozenset(
    {
        BeanfleetPeeringStatus.REJECTED,
        BeanfleetPeeringStatus.EXPIRED,
        BeanfleetPeeringStatus.DELETING,
        BeanfleetPeeringStatus.DELETED,
    }
)


@dataclass(frozen=True)
class BeanfleetPeering:
    peering_id: str
    status: BeanfleetPeeringStatus

    def to_dict(self) -> dict[str, str]:
        return {"peering_id": self.peering_id, "status": self.status.value}


class BeanfleetApiError(CloudNetworkError):
    def __init__(self, method: str, path: str, status_code: int, detail: str) -> None:
        self.method = method
        self.path = path
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"beanfleet {method} {path} returned HTTP {status_code}: {detail}")


class BeanfleetTransportError(CloudNetworkError):
    def __init__(self, method: str, path: str, *, outcome_unknown: bool) -> None:
        self.method = method
        self.path = path
        self.outcome_unknown = outcome_unknown
        suffix = (
            "; the request outcome may be unknown, so do not retry it blindly"
            if outcome_unknown
            else ""
        )
        super().__init__(f"beanfleet {method} {path} failed at the transport layer{suffix}")


class BeanfleetClient:
    RETRYABLE_GET_STATUS_CODES: ClassVar[frozenset[int]] = frozenset(
        {408, 425, 429, 500, 502, 503, 504}
    )

    def __init__(
        self,
        endpoint: str,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        verify: bool = True,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self._client = httpx.Client(
            base_url=self.endpoint,
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
            verify=verify,
            transport=transport,
        )

    def create_vpc_peering_connection(
        self,
        *,
        region_id: str,
        vpc_id: str,
        account_id: str,
        cidr_blocks: Sequence[str],
    ) -> BeanfleetPeering:
        response = self._request(
            "POST",
            "/aliyun/vpc-peering-connections",
            json={
                "region_id": region_id,
                "vpc_id": vpc_id,
                "account_id": account_id,
                "cidr_blocks": list(cidr_blocks),
            },
        )
        peering = self._parse_peering(response)
        if peering.status != BeanfleetPeeringStatus.CREATING:
            raise CloudNetworkError(
                "beanfleet create response must have status Creating; "
                f"got {peering.status.value}"
            )
        return peering

    def get_vpc_peering_connection(self, peering_id: str, region_id: str) -> BeanfleetPeering:
        path = f"/aliyun/vpc-peering-connections/{peering_id}"
        response = self._request("GET", path, params={"region_id": region_id})
        return self._parse_peering(response, expected_id=peering_id)

    def notify_vpc_peering_connection_accepted(
        self, peering_id: str, region_id: str
    ) -> BeanfleetPeering:
        path = f"/aliyun/vpc-peering-connections/{peering_id}/accepted"
        response = self._request("POST", path, params={"region_id": region_id})
        peering = self._parse_peering(response, expected_id=peering_id)
        if peering.status not in {
            BeanfleetPeeringStatus.ACTIVATED,
            BeanfleetPeeringStatus.READY,
        }:
            raise CloudNetworkError(
                "beanfleet accepted response must have status Activated or Ready; "
                f"got {peering.status.value}"
            )
        return peering

    def close(self) -> None:
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Mapping[str, Any] | None = None,
        params: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        try:
            response = self._client.request(method, path, json=json, params=params)
        except httpx.TransportError as exc:
            raise BeanfleetTransportError(
                method,
                path,
                outcome_unknown=method != "GET",
            ) from exc
        if response.is_success:
            return response
        raise BeanfleetApiError(
            method,
            path,
            response.status_code,
            self._error_detail(response),
        )

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            for key in ("detail", "message", "error"):
                value = payload.get(key)
                if value:
                    return str(value)
        return response.text.strip() or "no error details"

    @staticmethod
    def _parse_peering(
        response: httpx.Response, *, expected_id: str | None = None
    ) -> BeanfleetPeering:
        try:
            payload = response.json()
        except ValueError as exc:
            raise CloudNetworkError("beanfleet returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise CloudNetworkError("beanfleet peering response must be an object")
        peering_id = payload.get("peering_id")
        raw_status = payload.get("status")
        if not isinstance(peering_id, str) or not peering_id.strip():
            raise CloudNetworkError("beanfleet peering response requires a non-empty peering_id")
        if expected_id is not None and peering_id != expected_id:
            raise CloudNetworkError(
                f"beanfleet returned peering_id {peering_id}, expected {expected_id}"
            )
        if not isinstance(raw_status, str):
            raise CloudNetworkError("beanfleet peering response requires a string status")
        try:
            status = BeanfleetPeeringStatus(raw_status)
        except ValueError as exc:
            raise CloudNetworkError(f"beanfleet returned unknown peering status {raw_status!r}") from exc
        return BeanfleetPeering(peering_id=peering_id, status=status)
