from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from remote_loader.beanfleet import (
    TERMINAL_PEERING_STATUSES,
    BeanfleetApiError,
    BeanfleetClient,
    BeanfleetPeering,
    BeanfleetPeeringStatus,
    BeanfleetTransportError,
)
from remote_loader.cloud_network import (
    AliyunNetworkClient,
    CloudNetworkError,
    ConnectionInspection,
    LocalNetwork,
    PeerConnection,
    RoutePlan,
    normalize_ipv4,
)


@dataclass(frozen=True)
class PeeringCreationPlan:
    beanfleet_endpoint: str
    region_id: str
    vpc_id: str
    account_id: str
    cidr_blocks: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "beanfleet_endpoint": self.beanfleet_endpoint,
            "request": {
                "region_id": self.region_id,
                "vpc_id": self.vpc_id,
                "account_id": self.account_id,
                "cidr_blocks": list(self.cidr_blocks),
            },
        }


@dataclass(frozen=True)
class PeeringCompletionPlan:
    local_network: LocalNetwork
    peering: BeanfleetPeering
    inspection: ConnectionInspection
    routes: tuple[RoutePlan, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "peering": self.peering.to_dict(),
            "connection": self.inspection.connection.to_dict(),
            "checks": [check.to_dict() for check in self.inspection.checks],
            "routes": [route.to_dict() for route in self.routes],
        }


@dataclass(frozen=True)
class PeeringWorkflowResult:
    peering: BeanfleetPeering
    connection: PeerConnection
    routes: tuple[RoutePlan, ...]

    def to_dict(self) -> dict[str, object]:
        requester_destinations = {
            normalize_ipv4(cidr) for cidr in self.connection.requester.ipv4_cidrs
        }
        route_destinations = {normalize_ipv4(route.destination) for route in self.routes}
        return {
            "ready": (
                self.peering.status == BeanfleetPeeringStatus.READY
                and self.connection.status == "Activated"
                and bool(self.routes)
                and route_destinations == requester_destinations
                and all(route.existing_status == "Available" for route in self.routes)
            ),
            "peering": {
                "peering_id": self.peering.peering_id,
                "beanfleet_status": self.peering.status.value,
                "aliyun_status": self.connection.status,
            },
            "connection": self.connection.to_dict(),
            "routes": [route.to_dict() for route in self.routes],
        }


ProgressCallback = Callable[[str, BeanfleetPeering], None]


class PeeringWorkflow:
    def __init__(
        self,
        *,
        network_client: AliyunNetworkClient,
        beanfleet_client: BeanfleetClient,
        region_id: str,
        server_poll_interval: float = 5.0,
        server_timeout: float = 120.0,
        route_poll_interval: float = 3.0,
        route_timeout: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.network_client = network_client
        self.beanfleet_client = beanfleet_client
        self.region_id = region_id
        self.server_poll_interval = server_poll_interval
        self.server_timeout = server_timeout
        self.route_poll_interval = route_poll_interval
        self.route_timeout = route_timeout
        self._clock = clock
        self._sleep = sleeper
        self._progress = progress

    def creation_plan(self, local_network: LocalNetwork) -> PeeringCreationPlan:
        if local_network.vpc.region != self.region_id:
            raise CloudNetworkError(
                f"local VPC is in {local_network.vpc.region}, not selected region {self.region_id}"
            )
        return PeeringCreationPlan(
            beanfleet_endpoint=self.beanfleet_client.endpoint,
            region_id=self.region_id,
            vpc_id=local_network.vpc.vpc_id,
            account_id=local_network.identity.account_id,
            cidr_blocks=local_network.vpc.ipv4_cidrs,
        )

    def create(self, plan: PeeringCreationPlan) -> BeanfleetPeering:
        peering = self.beanfleet_client.create_vpc_peering_connection(
            region_id=plan.region_id,
            vpc_id=plan.vpc_id,
            account_id=plan.account_id,
            cidr_blocks=plan.cidr_blocks,
        )
        self._emit("created", peering)
        return peering

    def prepare_completion(
        self, local_network: LocalNetwork, peering_id: str
    ) -> PeeringCompletionPlan:
        peering = self._wait_for_status(
            peering_id,
            {
                BeanfleetPeeringStatus.ACCEPTING,
                BeanfleetPeeringStatus.UPDATING,
                BeanfleetPeeringStatus.ACTIVATED,
                BeanfleetPeeringStatus.READY,
            },
            phase="prepare completion review",
        )
        inspection = self.network_client.inspect_connection(
            peering_id,
            expected_vpc_id=local_network.vpc.vpc_id,
            local_ip=local_network.local_address.address,
        )
        inspection.require_actionable()
        routes = self.network_client.plan_connection_routes(
            inspection,
            inspection.connection.requester.ipv4_cidrs,
            local_ip=local_network.local_address.address,
            route_table_id=local_network.vswitch.route_table_id,
        )
        return PeeringCompletionPlan(
            local_network=local_network,
            peering=peering,
            inspection=inspection,
            routes=routes,
        )

    def complete(self, plan: PeeringCompletionPlan) -> PeeringWorkflowResult:
        peering = self._wait_for_status(
            plan.peering.peering_id,
            {
                BeanfleetPeeringStatus.ACCEPTING,
                BeanfleetPeeringStatus.UPDATING,
                BeanfleetPeeringStatus.ACTIVATED,
                BeanfleetPeeringStatus.READY,
            },
            phase="resume confirmed workflow",
        )
        if peering.status == BeanfleetPeeringStatus.ACCEPTING:
            self._refresh_confirmed_routes(plan)
            self.network_client.accept_connection(
                peering.peering_id,
                expected_vpc_id=plan.local_network.vpc.vpc_id,
                local_ip=plan.local_network.local_address.address,
                expected_requester_cidrs=(
                    plan.inspection.connection.requester.ipv4_cidrs
                ),
            )
            peering = self._wait_for_status(
                peering.peering_id,
                {BeanfleetPeeringStatus.ACTIVATED, BeanfleetPeeringStatus.READY},
                phase="wait for Aliyun activation",
            )
        elif peering.status == BeanfleetPeeringStatus.UPDATING:
            peering = self._wait_for_status(
                peering.peering_id,
                {BeanfleetPeeringStatus.ACTIVATED, BeanfleetPeeringStatus.READY},
                phase="wait for Aliyun activation",
            )

        if peering.status == BeanfleetPeeringStatus.ACTIVATED:
            self._refresh_confirmed_routes(plan)
            peering = self._notify_accepted(peering.peering_id)
        if peering.status != BeanfleetPeeringStatus.READY:
            peering = self._wait_for_status(
                peering.peering_id,
                {BeanfleetPeeringStatus.READY},
                phase="wait for beanfleet platform setup",
            )

        inspection, confirmed_routes = self._refresh_confirmed_routes(plan)
        current_routes = confirmed_routes
        for route in confirmed_routes:
            submitted_route = self.network_client.apply_route(route)
            inspection, current_routes = self._wait_for_routes_available(
                plan.local_network, inspection, current_routes, target=submitted_route
            )
        inspection, available = self._wait_for_routes_available(
            plan.local_network, inspection, current_routes
        )
        peering = self._wait_for_status(
            peering.peering_id,
            {BeanfleetPeeringStatus.READY},
            phase="confirm final beanfleet readiness",
        )
        inspection, available = self._refresh_confirmed_routes(plan)
        if (
            inspection.connection.status != "Activated"
            or not available
            or any(
                route.create_required or route.existing_status != "Available"
                for route in available
            )
        ):
            states = ", ".join(
                f"{route.destination}={route.existing_status or 'missing'}"
                for route in available
            )
            raise CloudNetworkError(
                f"client routes for peering {peering.peering_id} were no longer all "
                f"Available after final beanfleet readiness ({states or 'no routes'})"
            )
        return PeeringWorkflowResult(
            peering=peering,
            connection=inspection.connection,
            routes=available,
        )

    def _notify_accepted(self, peering_id: str) -> BeanfleetPeering:
        for attempt in range(2):
            try:
                return self.beanfleet_client.notify_vpc_peering_connection_accepted(
                    peering_id, self.region_id
                )
            except BeanfleetApiError as exc:
                if exc.status_code != 409 or attempt == 1:
                    raise CloudNetworkError(
                        f"peering {peering_id} failed during accepted notification; "
                        f"last status Activated: {exc}"
                    ) from exc
                peering = self._wait_for_status(
                    peering_id,
                    {BeanfleetPeeringStatus.ACTIVATED, BeanfleetPeeringStatus.READY},
                    phase="recover from accepted notification conflict",
                )
                if peering.status == BeanfleetPeeringStatus.READY:
                    return peering
            except BeanfleetTransportError as exc:
                try:
                    return self._wait_for_status(
                        peering_id,
                        {BeanfleetPeeringStatus.READY},
                        phase="resolve accepted notification outcome",
                    )
                except CloudNetworkError as poll_error:
                    raise CloudNetworkError(
                        f"accepted notification outcome for peering {peering_id} is unknown: "
                        f"{poll_error}"
                    ) from exc
            except CloudNetworkError as exc:
                raise CloudNetworkError(
                    f"peering {peering_id} failed during accepted notification; "
                    f"last status Activated: {exc}"
                ) from exc
        raise AssertionError("accepted notification retry loop exhausted")

    def _refresh_confirmed_routes(
        self, plan: PeeringCompletionPlan
    ) -> tuple[ConnectionInspection, tuple[RoutePlan, ...]]:
        inspection = self.network_client.inspect_connection(
            plan.peering.peering_id,
            expected_vpc_id=plan.local_network.vpc.vpc_id,
            local_ip=plan.local_network.local_address.address,
        )
        refreshed = self.network_client.plan_connection_routes(
            inspection,
            inspection.connection.requester.ipv4_cidrs,
            local_ip=plan.local_network.local_address.address,
            route_table_id=plan.local_network.vswitch.route_table_id,
        )
        self._require_same_route_targets(plan.routes, refreshed)
        return inspection, refreshed

    def _wait_for_routes_available(
        self,
        local_network: LocalNetwork,
        inspection: ConnectionInspection,
        submitted: tuple[RoutePlan, ...],
        *,
        target: RoutePlan | None = None,
    ) -> tuple[ConnectionInspection, tuple[RoutePlan, ...]]:
        deadline = self._clock() + self.route_timeout
        current_inspection = inspection
        current = submitted
        while True:
            current_inspection = self.network_client.inspect_connection(
                inspection.connection.connection_id,
                expected_vpc_id=local_network.vpc.vpc_id,
                local_ip=local_network.local_address.address,
            )
            current = self.network_client.plan_connection_routes(
                current_inspection,
                current_inspection.connection.requester.ipv4_cidrs,
                local_ip=local_network.local_address.address,
                route_table_id=local_network.vswitch.route_table_id,
            )
            self._require_same_route_targets(submitted, current)
            required = current
            if target is not None:
                target_identity = self._route_target(target)
                required = tuple(
                    route for route in current if self._route_target(route) == target_identity
                )
            if (
                required
                and current_inspection.connection.status == "Activated"
                and all(
                    not route.create_required and route.existing_status == "Available"
                    for route in required
                )
            ):
                return current_inspection, current
            now = self._clock()
            if now >= deadline:
                states = ", ".join(
                    f"{route.destination}={route.existing_status or 'missing'}"
                    for route in required
                )
                raise CloudNetworkError(
                    f"client routes for peering {inspection.connection.connection_id} "
                    "did not become Available while the connection was Activated in time "
                    f"(connection={current_inspection.connection.status}; {states})"
                )
            self._sleep(min(self.route_poll_interval, deadline - now))

    def _wait_for_status(
        self,
        peering_id: str,
        targets: set[BeanfleetPeeringStatus],
        *,
        phase: str,
    ) -> BeanfleetPeering:
        deadline = self._clock() + self.server_timeout
        last_status: BeanfleetPeeringStatus | None = None
        last_error: Exception | None = None
        while True:
            try:
                peering = self.beanfleet_client.get_vpc_peering_connection(
                    peering_id, self.region_id
                )
            except BeanfleetTransportError as exc:
                last_error = exc
            except BeanfleetApiError as exc:
                if exc.status_code not in BeanfleetClient.RETRYABLE_GET_STATUS_CODES:
                    detail = (
                        f"last status {last_status.value}"
                        if last_status is not None
                        else "no valid status received"
                    )
                    raise CloudNetworkError(
                        f"peering {peering_id} failed during {phase}; {detail}: {exc}"
                    ) from exc
                last_error = exc
            except CloudNetworkError as exc:
                detail = (
                    f"last status {last_status.value}"
                    if last_status is not None
                    else "no valid status received"
                )
                raise CloudNetworkError(
                    f"peering {peering_id} failed during {phase}; {detail}: {exc}"
                ) from exc
            else:
                last_error = None
                if peering.status != last_status:
                    self._emit(phase, peering)
                    last_status = peering.status
                if peering.status in TERMINAL_PEERING_STATUSES:
                    raise CloudNetworkError(
                        f"peering {peering_id} entered terminal status "
                        f"{peering.status.value} during {phase}"
                    )
                if peering.status in targets:
                    return peering

            now = self._clock()
            if now >= deadline:
                target_names = "/".join(sorted(status.value for status in targets))
                detail = (
                    f"last status {last_status.value}"
                    if last_status is not None
                    else f"last error {last_error}"
                )
                raise CloudNetworkError(
                    f"peering {peering_id} did not reach {target_names} during {phase}; "
                    f"{detail}"
                )
            self._sleep(min(self.server_poll_interval, deadline - now))

    def _emit(self, phase: str, peering: BeanfleetPeering) -> None:
        if self._progress is not None:
            self._progress(phase, peering)

    @staticmethod
    def _route_target(route: RoutePlan) -> tuple[object, ...]:
        # Confirmation authorizes ensuring this fixed route target exists. Its
        # observed route ID/status and create-vs-reuse action may safely change.
        return (
            route.connection_id,
            route.destination,
            route.route_table_id,
            route.local_ip,
            route.vswitch_id,
            route.local_vpc_id,
        )

    @classmethod
    def _require_same_route_targets(
        cls, confirmed: tuple[RoutePlan, ...], current: tuple[RoutePlan, ...]
    ) -> None:
        if {cls._route_target(route) for route in confirmed} != {
            cls._route_target(route) for route in current
        }:
            raise CloudNetworkError(
                "the route targets changed after confirmation; review a new completion plan"
            )
