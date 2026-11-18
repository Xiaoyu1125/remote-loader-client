from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

ALIYUN_CLI_INSTALL_URL = "https://help.aliyun.com/zh/cli/install-update-alibaba-cloud-cli"
ALIYUN_PLUGIN_INSTALL = "aliyun plugin install --names sts vpc vpcpeer"
MINIMUM_ALIYUN_CLI_VERSION = (3, 3, 0)
_ALIYUN_IDENTITY_ENVIRONMENT_VARIABLES = (
    "ALIBABA_CLOUD_ACCESS_KEY_ID",
    "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
    "ALIBABA_CLOUD_SECURITY_TOKEN",
    "ALIBABA_CLOUD_CREDENTIALS_URI",
    "ALIBABA_CLOUD_OIDC_PROVIDER_ARN",
    "ALIBABA_CLOUD_OIDC_TOKEN_FILE",
    "ALIBABA_CLOUD_ROLE_ARN",
    "ALIBABA_CLOUD_EXTERNAL_ID",
    "ALIBABA_CLOUD_SSO_CLIENT_ID",
    "ALIBABA_CLOUD_PROFILE",
    "ALIBABA_CLOUD_IGNORE_PROFILE",
)


class CloudNetworkError(RuntimeError):
    """A cloud-network operation cannot continue safely."""


@dataclass(frozen=True)
class CloudIdentity:
    account_id: str
    arn: str
    identity_type: str
    principal_id: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class VpcPeerSide:
    account_id: str
    region: str
    vpc_id: str
    ipv4_cidrs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "account_id": self.account_id,
            "region": self.region,
            "vpc_id": self.vpc_id,
            "ipv4_cidrs": list(self.ipv4_cidrs),
        }


@dataclass(frozen=True)
class PeerConnection:
    connection_id: str
    name: str
    status: str
    business_status: str
    expires_at: str | None
    requester: VpcPeerSide
    accepter: VpcPeerSide

    def to_dict(self) -> dict[str, object]:
        return {
            "connection_id": self.connection_id,
            "name": self.name,
            "status": self.status,
            "business_status": self.business_status,
            "expires_at": self.expires_at,
            "requester": self.requester.to_dict(),
            "accepter": self.accepter.to_dict(),
        }


@dataclass(frozen=True)
class NetworkCheck:
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ConnectionInspection:
    identity: CloudIdentity
    connection: PeerConnection
    checks: tuple[NetworkCheck, ...]
    local_networks: tuple[LocalNetwork, ...] = ()

    @property
    def has_errors(self) -> bool:
        return any(check.status == "error" for check in self.checks)

    def require_actionable(self) -> None:
        errors = [check.detail for check in self.checks if check.status == "error"]
        if errors:
            raise CloudNetworkError("connection checks failed: " + "; ".join(errors))

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "connection": self.connection.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
            "local_networks": [network.to_dict() for network in self.local_networks],
        }


@dataclass(frozen=True)
class LocalIpv4Address:
    interface: str
    address: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class VSwitch:
    vswitch_id: str
    vpc_id: str
    name: str
    cidr: str
    status: str
    route_table_id: str
    route_table_type: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Vpc:
    vpc_id: str
    account_id: str
    name: str
    region: str
    status: str
    ipv4_cidrs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "vpc_id": self.vpc_id,
            "account_id": self.account_id,
            "name": self.name,
            "region": self.region,
            "status": self.status,
            "ipv4_cidrs": list(self.ipv4_cidrs),
        }


@dataclass(frozen=True)
class LocalNetwork:
    identity: CloudIdentity
    local_address: LocalIpv4Address
    vswitch: VSwitch
    vpc: Vpc

    def to_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_dict(),
            "local_address": self.local_address.to_dict(),
            "vswitch": self.vswitch.to_dict(),
            "vpc": self.vpc.to_dict(),
        }


@dataclass(frozen=True)
class RouteMatch:
    local_address: LocalIpv4Address
    vswitch: VSwitch

    def to_dict(self) -> dict[str, object]:
        return {
            "local_address": self.local_address.to_dict(),
            "vswitch": self.vswitch.to_dict(),
        }


@dataclass(frozen=True)
class RoutePlan:
    connection_id: str
    destination: str
    route_table_id: str
    local_ip: str | None
    vswitch_id: str | None
    existing_route_id: str | None
    existing_status: str | None
    create_required: bool
    local_vpc_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_ipv4(value: str) -> str:
    try:
        network = ipaddress.ip_network(value.strip(), strict=False)
    except ValueError as exc:
        raise CloudNetworkError(f"invalid IPv4 destination or address: {value!r}") from exc
    if not isinstance(network, ipaddress.IPv4Network):
        raise CloudNetworkError("IPv6 VPC peering routes are not supported yet")
    return str(network)


def _local_ipv4_address(value: str) -> LocalIpv4Address:
    try:
        parsed = ipaddress.ip_address(value.strip())
    except ValueError as exc:
        raise CloudNetworkError(f"invalid local IP address: {value!r}") from exc
    if not isinstance(parsed, ipaddress.IPv4Address):
        raise CloudNetworkError("--local-ip must be an IPv4 address")
    return LocalIpv4Address(interface="selected", address=str(parsed))


def discover_local_ipv4_addresses(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[LocalIpv4Address, ...]:
    command = ["ip", "-j", "-4", "address", "show", "scope", "global"]
    try:
        result = runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            stdin=subprocess.DEVNULL,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise CloudNetworkError("could not inspect local IPv4 addresses; install iproute2") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no error details"
        raise CloudNetworkError(f"could not inspect local IPv4 addresses: {detail}")
    try:
        interfaces = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CloudNetworkError("the ip command did not return valid JSON") from exc

    addresses: list[LocalIpv4Address] = []
    if not isinstance(interfaces, list):
        raise CloudNetworkError("the ip command returned an unexpected JSON value")
    for interface in interfaces:
        if not isinstance(interface, dict):
            continue
        name = str(interface.get("ifname", ""))
        if not name or name == "lo":
            continue
        for info in interface.get("addr_info", []):
            if not isinstance(info, dict) or info.get("family") != "inet":
                continue
            address = str(info.get("local", ""))
            try:
                parsed = ipaddress.ip_address(address)
            except ValueError:
                continue
            if isinstance(parsed, ipaddress.IPv4Address) and not parsed.is_loopback:
                addresses.append(LocalIpv4Address(interface=name, address=str(parsed)))
    return tuple(sorted(set(addresses), key=lambda value: (value.interface, value.address)))


class AliyunCli:
    """Small wrapper for the plugin-based Alibaba Cloud CLI 3.3+."""

    def __init__(
        self,
        *,
        profile: str,
        region: str,
        executable: str = "aliyun",
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        timeout_seconds: float = 30,
    ) -> None:
        if not profile.strip():
            raise CloudNetworkError("Alibaba Cloud CLI profile must not be empty")
        if not region.strip():
            raise CloudNetworkError("Alibaba Cloud region must not be empty")
        self.profile = profile.strip()
        self.region = region.strip()
        self.executable = executable
        self._runner = runner
        self.timeout_seconds = timeout_seconds

    def check_environment(self) -> str:
        result = self._run([self.executable, "version"])
        if result.returncode != 0:
            detail = _strip_ansi(result.stderr.strip() or result.stdout.strip() or "no error details")
            raise CloudNetworkError(f"could not query Alibaba Cloud CLI version: {detail}")
        match = re.search(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)", result.stdout)
        if match is None:
            raise CloudNetworkError(
                f"could not parse Alibaba Cloud CLI version from {result.stdout.strip()!r}"
            )
        version = tuple(int(value) for value in match.groups())
        if version < MINIMUM_ALIYUN_CLI_VERSION:
            raise CloudNetworkError(
                "Alibaba Cloud CLI 3.3.0 or newer is required; "
                f"found {'.'.join(str(value) for value in version)}"
            )

        commands = (
            ("sts", "get-caller-identity"),
            ("vpcpeer", "get-vpc-peer-connection-attribute"),
            ("vpc", "describe-vswitches"),
            ("vpcpeer", "accept-vpc-peer-connection"),
            ("vpc", "describe-vpc-attribute"),
            ("vpc", "describe-route-entry-list"),
            ("vpc", "create-route-entry"),
        )
        for product, operation in commands:
            help_result = self._run([self.executable, product, operation, "--help"])
            if help_result.returncode != 0:
                raise self._command_error(help_result, product, operation)
        return ".".join(str(value) for value in version)

    def call(self, product: str, operation: str, *parameters: str) -> dict[str, Any]:
        command = [
            self.executable,
            "--profile",
            self.profile,
            "--region",
            self.region,
            product,
            operation,
            *parameters,
        ]
        result = self._run(command)
        if result.returncode != 0:
            raise self._command_error(result, product, operation)
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise CloudNetworkError(
                f"aliyun {product} {operation} did not return JSON: {result.stdout.strip()!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise CloudNetworkError(f"aliyun {product} {operation} returned an unexpected JSON value")
        return payload

    def identity(self) -> CloudIdentity:
        payload = self.call("sts", "get-caller-identity")
        account_id = str(payload.get("AccountId", ""))
        if not account_id:
            raise CloudNetworkError("GetCallerIdentity did not return AccountId")
        return CloudIdentity(
            account_id=account_id,
            arn=str(payload.get("Arn", "")),
            identity_type=str(payload.get("IdentityType", "")),
            principal_id=str(payload.get("PrincipalId", "")),
        )

    def get_connection(self, connection_id: str) -> PeerConnection:
        payload = self.call(
            "vpcpeer",
            "get-vpc-peer-connection-attribute",
            "--instance-id",
            connection_id,
        )
        return _parse_peer_connection(payload)

    def accept_connection(self, connection_id: str) -> None:
        self.call(
            "vpcpeer",
            "accept-vpc-peer-connection",
            "--instance-id",
            connection_id,
            "--client-token",
            uuid.uuid4().hex,
        )

    def list_vswitches(self, vpc_id: str | None = None) -> tuple[VSwitch, ...]:
        page = 1
        result: list[VSwitch] = []
        while True:
            parameters = [
                "--biz-region-id",
                self.region,
                "--page-number",
                str(page),
                "--page-size",
                "50",
            ]
            if vpc_id is not None:
                parameters.extend(("--vpc-id", _required_id(vpc_id, "VPC ID")))
            payload = self.call(
                "vpc",
                "describe-vswitches",
                *parameters,
            )
            values = _extract_list(payload.get("VSwitches"), "VSwitch")
            result.extend(_parse_vswitch(value) for value in values)
            total = _integer(payload.get("TotalCount"), default=len(result))
            if len(result) >= total or len(values) < 50:
                return tuple(result)
            page += 1

    def get_vpc(self, vpc_id: str) -> Vpc:
        payload = self.call(
            "vpc",
            "describe-vpc-attribute",
            "--biz-region-id",
            self.region,
            "--vpc-id",
            _required_id(vpc_id, "VPC ID"),
        )
        return _parse_vpc(payload)

    def list_route_entries(
        self,
        route_table_id: str,
        destination: str,
    ) -> tuple[dict[str, Any], ...]:
        next_token: str | None = None
        result: list[dict[str, Any]] = []
        while True:
            parameters = [
                "--biz-region-id",
                self.region,
                "--route-table-id",
                route_table_id,
                "--destination-cidr-block",
                destination,
                "--max-result",
                "100",
            ]
            if next_token:
                parameters.extend(("--next-token", next_token))
            payload = self.call("vpc", "describe-route-entry-list", *parameters)
            container = payload.get("RouteEntrys") or payload.get("RouteEntries")
            values = _extract_list(container, "RouteEntry")
            result.extend(values)
            next_value = payload.get("NextToken")
            next_token = next_value if isinstance(next_value, str) and next_value else None
            if next_token is None:
                return tuple(result)

    def create_route(
        self,
        *,
        route_table_id: str,
        destination: str,
        connection_id: str,
    ) -> str | None:
        payload = self.call(
            "vpc",
            "create-route-entry",
            "--biz-region-id",
            self.region,
            "--route-table-id",
            route_table_id,
            "--destination-cidr-block",
            destination,
            "--next-hop-type",
            "VpcPeer",
            "--next-hop-id",
            connection_id,
            "--route-entry-name",
            "modalink-vpc-peer",
            "--description",
            "Modalink VPC peering route",
            "--client-token",
            uuid.uuid4().hex,
        )
        value = payload.get("RouteEntryId")
        return value if isinstance(value, str) and value else None

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        for variable in _ALIYUN_IDENTITY_ENVIRONMENT_VARIABLES:
            environment.pop(variable, None)
        environment["ALIBABA_CLOUD_CLI_PLUGIN_AUTO_INSTALL"] = "false"
        try:
            return self._runner(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                stdin=subprocess.DEVNULL,
                timeout=self.timeout_seconds,
                env=environment,
            )
        except FileNotFoundError as exc:
            raise CloudNetworkError(
                f"Alibaba Cloud CLI executable {self.executable!r} was not found; "
                f"install version 3.3.0 or newer from {ALIYUN_CLI_INSTALL_URL}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise CloudNetworkError(f"Alibaba Cloud CLI timed out: {' '.join(command[:4])}") from exc

    @staticmethod
    def _command_error(
        result: subprocess.CompletedProcess[str],
        product: str,
        operation: str,
    ) -> CloudNetworkError:
        detail = _strip_ansi(result.stderr.strip() or result.stdout.strip() or "no error details")
        if "plugin" in detail.lower() or "unknown command" in detail.lower():
            detail = f"{detail}; install the required plugins with: {ALIYUN_PLUGIN_INSTALL}"
        return CloudNetworkError(
            f"aliyun {product} {operation} failed with exit code {result.returncode}: {detail}"
        )


class AliyunNetworkClient:
    def __init__(
        self,
        *,
        profile: str,
        region: str,
        cli: AliyunCli | None = None,
        local_ip_provider: Callable[[], Sequence[LocalIpv4Address]] = discover_local_ipv4_addresses,
    ) -> None:
        normalized_profile = profile.strip()
        normalized_region = region.strip()
        if not normalized_profile:
            raise CloudNetworkError("Alibaba Cloud CLI profile must not be empty")
        if not normalized_region:
            raise CloudNetworkError("Alibaba Cloud region must not be empty")
        self.profile = normalized_profile
        self.region = normalized_region
        self.cli = cli or AliyunCli(
            profile=normalized_profile,
            region=normalized_region,
        )
        self._local_ip_provider = local_ip_provider
        self._environment_checked = False

    def check_environment(self) -> str:
        version = self.cli.check_environment()
        self._environment_checked = True
        return version

    def identity(self) -> CloudIdentity:
        self._ensure_environment()
        return self.cli.identity()

    def local_network_candidates(
        self,
        *,
        local_ip: str | None = None,
        vpc_id: str | None = None,
    ) -> tuple[LocalNetwork, ...]:
        self._ensure_environment()
        identity = self.cli.identity()
        return self._local_network_candidates(
            identity,
            local_ip=local_ip,
            vpc_id=vpc_id,
        )

    def discover_local_network(
        self,
        *,
        local_ip: str | None = None,
        vpc_id: str | None = None,
    ) -> LocalNetwork:
        candidates = self.local_network_candidates(local_ip=local_ip, vpc_id=vpc_id)
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            filters = []
            if local_ip is not None:
                filters.append(f"local IP {local_ip}")
            if vpc_id is not None:
                filters.append(f"VPC {vpc_id}")
            suffix = f" matching {' and '.join(filters)}" if filters else ""
            raise CloudNetworkError(
                "could not match a local IPv4 address to an available vSwitch"
                f"{suffix} in region {self.region}"
            )
        choices = ", ".join(
            f"{candidate.local_address.address}->{candidate.vpc.vpc_id}/"
            f"{candidate.vswitch.vswitch_id}/{candidate.vswitch.route_table_id}"
            for candidate in candidates
        )
        raise CloudNetworkError(
            f"multiple local network candidates were found ({choices}); "
            "re-run with --vpc-id and/or --local-ip"
        )

    def inspect_connection(
        self,
        connection_id: str,
        *,
        expected_vpc_id: str | None = None,
        local_ip: str | None = None,
        verify_local_network: bool = True,
    ) -> ConnectionInspection:
        self._ensure_environment()
        identity = self.cli.identity()
        connection = self.cli.get_connection(_required_id(connection_id, "connection ID"))
        inspection = inspect_connection(
            identity,
            connection,
            selected_region=self.region,
            expected_vpc_id=expected_vpc_id,
        )
        if not verify_local_network:
            return inspection

        try:
            local_networks = self._local_network_candidates(
                identity,
                local_ip=local_ip,
                vpc_id=connection.accepter.vpc_id,
            )
        except CloudNetworkError as exc:
            return ConnectionInspection(
                identity=identity,
                connection=connection,
                checks=(
                    *inspection.checks,
                    NetworkCheck("local_network", "error", str(exc)),
                ),
            )

        checks = list(inspection.checks)
        if not local_networks:
            checks.append(
                NetworkCheck(
                    "local_network",
                    "error",
                    "no local IPv4 address belongs to an available vSwitch in accepter VPC "
                    f"{connection.accepter.vpc_id}",
                )
            )
        else:
            addresses = ", ".join(sorted({network.local_address.address for network in local_networks}))
            checks.append(
                NetworkCheck(
                    "local_network",
                    "ok",
                    f"local address(es) {addresses} match accepter VPC {connection.accepter.vpc_id}",
                )
            )
            discovered_cidrs = _canonical_cidrs(local_networks[0].vpc.ipv4_cidrs)
            peer_cidrs = _canonical_cidrs(connection.accepter.ipv4_cidrs)
            checks.append(
                _check(
                    "accepter_vpc_cidrs",
                    discovered_cidrs == peer_cidrs,
                    (
                        "discovered VPC CIDRs match the peering connection"
                        if discovered_cidrs == peer_cidrs
                        else "discovered VPC CIDRs "
                        f"{', '.join(discovered_cidrs) or '<none>'} do not match connection CIDRs "
                        f"{', '.join(peer_cidrs) or '<none>'}"
                    ),
                )
            )
        return ConnectionInspection(
            identity=identity,
            connection=connection,
            checks=tuple(checks),
            local_networks=local_networks,
        )

    def accept_connection(
        self,
        connection_id: str,
        *,
        expected_vpc_id: str | None = None,
        local_ip: str | None = None,
        expected_requester_cidrs: Sequence[str] | None = None,
    ) -> PeerConnection:
        inspection = self.inspect_connection(
            connection_id,
            expected_vpc_id=expected_vpc_id,
            local_ip=local_ip,
        )
        inspection.require_actionable()
        if expected_requester_cidrs is not None:
            confirmed_cidrs = _canonical_cidrs(expected_requester_cidrs)
            current_cidrs = _canonical_cidrs(
                inspection.connection.requester.ipv4_cidrs
            )
            if current_cidrs != confirmed_cidrs:
                raise CloudNetworkError(
                    "requester CIDRs changed after confirmation; review a new completion plan"
                )
        status = inspection.connection.status
        if status == "Activated":
            return inspection.connection
        if status == "Accepting":
            self.cli.accept_connection(inspection.connection.connection_id)
        elif status != "Updating":
            raise CloudNetworkError(
                f"connection {inspection.connection.connection_id} cannot be accepted in state {status}"
            )
        return self.cli.get_connection(inspection.connection.connection_id)

    def find_route_matches(
        self,
        connection: PeerConnection,
        *,
        local_addresses: Sequence[LocalIpv4Address] | None = None,
    ) -> tuple[RouteMatch, ...]:
        self._ensure_environment()
        addresses = tuple(self._local_ip_provider() if local_addresses is None else local_addresses)
        switches = self.cli.list_vswitches(connection.accepter.vpc_id)
        return _match_addresses_to_vswitches(addresses, switches)

    def _local_network_candidates(
        self,
        identity: CloudIdentity,
        *,
        local_ip: str | None,
        vpc_id: str | None,
    ) -> tuple[LocalNetwork, ...]:
        addresses = tuple(self._local_ip_provider())
        if local_ip is not None:
            selected = _local_ipv4_address(local_ip)
            if not any(address.address == selected.address for address in addresses):
                raise CloudNetworkError(
                    f"selected local IP {selected.address} is not assigned to this machine"
                )
            addresses = tuple(address for address in addresses if address.address == selected.address)

        switches = self.cli.list_vswitches(vpc_id)
        matches = _match_addresses_to_vswitches(addresses, switches)
        vpcs: dict[str, Vpc] = {}
        candidates: list[LocalNetwork] = []
        for match in matches:
            matched_vpc_id = match.vswitch.vpc_id
            vpc = vpcs.get(matched_vpc_id)
            if vpc is None:
                vpc = self.cli.get_vpc(matched_vpc_id)
                if vpc.account_id != identity.account_id:
                    raise CloudNetworkError(
                        f"VPC {vpc.vpc_id} belongs to account {vpc.account_id}, "
                        f"not authenticated account {identity.account_id}"
                    )
                if vpc.region != self.region:
                    raise CloudNetworkError(f"VPC {vpc.vpc_id} is in region {vpc.region}, not {self.region}")
                if vpc.status != "Available":
                    raise CloudNetworkError(
                        f"VPC {vpc.vpc_id} is not available; current status is {vpc.status}"
                    )
                vpcs[matched_vpc_id] = vpc
            candidates.append(
                LocalNetwork(
                    identity=identity,
                    local_address=match.local_address,
                    vswitch=match.vswitch,
                    vpc=vpc,
                )
            )
        return tuple(
            sorted(
                set(candidates),
                key=lambda candidate: (
                    candidate.local_address.address,
                    candidate.vpc.vpc_id,
                    candidate.vswitch.vswitch_id,
                ),
            )
        )

    def plan_connection_routes(
        self,
        inspection: ConnectionInspection,
        destinations: Iterable[str],
        *,
        local_ip: str | None = None,
        route_table_id: str | None = None,
    ) -> tuple[RoutePlan, ...]:
        inspection.require_actionable()
        connection = inspection.connection
        if connection.status not in {"Accepting", "Updating", "Activated"}:
            raise CloudNetworkError(
                f"connection {connection.connection_id} cannot be used to plan routes "
                f"in state {connection.status}"
            )
        canonical_destinations = tuple(
            dict.fromkeys(validate_destination(connection, value) for value in destinations)
        )
        if not canonical_destinations:
            raise CloudNetworkError("at least one route destination is required")
        selected_table, selected_match = self._select_route_table(
            connection,
            local_ip=local_ip,
            route_table_id=route_table_id,
        )
        plans: list[RoutePlan] = []
        for canonical in canonical_destinations:
            entries = self.cli.list_route_entries(selected_table, canonical)
            existing = _matching_peer_route(entries, connection.connection_id)
            if entries and existing is None:
                raise CloudNetworkError(
                    f"route table {selected_table} already contains {canonical} "
                    "with a different next hop"
                )
            plans.append(
                RoutePlan(
                    connection_id=connection.connection_id,
                    destination=canonical,
                    route_table_id=selected_table,
                    local_ip=(
                        selected_match.local_address.address if selected_match else local_ip
                    ),
                    vswitch_id=(
                        selected_match.vswitch.vswitch_id if selected_match else None
                    ),
                    existing_route_id=(
                        _string(existing, "RouteEntryId") if existing else None
                    ),
                    existing_status=_string(existing, "Status") if existing else None,
                    create_required=existing is None,
                    local_vpc_id=connection.accepter.vpc_id,
                )
            )
        return tuple(plans)

    def plan_route(
        self,
        connection_id: str,
        destination: str,
        *,
        local_ip: str | None = None,
        route_table_id: str | None = None,
        expected_vpc_id: str | None = None,
    ) -> RoutePlan:
        verify_local_network = route_table_id is None or local_ip is not None
        inspection = self.inspect_connection(
            connection_id,
            expected_vpc_id=expected_vpc_id,
            local_ip=local_ip,
            verify_local_network=verify_local_network,
        )
        if inspection.connection.status != "Activated":
            raise CloudNetworkError(
                f"connection {inspection.connection.connection_id} must be Activated "
                "before configuring routes"
            )
        return self.plan_connection_routes(
            inspection,
            (destination,),
            local_ip=local_ip,
            route_table_id=route_table_id,
        )[0]

    def apply_route(self, plan: RoutePlan) -> RoutePlan:
        inspection = self.inspect_connection(
            plan.connection_id,
            expected_vpc_id=plan.local_vpc_id,
            local_ip=plan.local_ip,
            verify_local_network=plan.local_ip is not None,
        )
        inspection.require_actionable()
        connection = inspection.connection
        if connection.status != "Activated":
            raise CloudNetworkError(
                f"connection {connection.connection_id} must be Activated before configuring routes"
            )
        canonical = validate_destination(connection, plan.destination)
        switches = self.cli.list_vswitches(connection.accepter.vpc_id)
        associated = [switch for switch in switches if switch.route_table_id == plan.route_table_id]
        if not associated:
            raise CloudNetworkError(
                f"confirmed route table {plan.route_table_id} is no longer associated with "
                f"accepter VPC {connection.accepter.vpc_id}; generate a new preview"
            )
        if plan.local_ip is not None:
            matches = _match_addresses_to_vswitches(
                (_local_ipv4_address(plan.local_ip),),
                associated,
            )
            exact = next(
                (
                    match
                    for match in matches
                    if plan.vswitch_id is None or match.vswitch.vswitch_id == plan.vswitch_id
                ),
                None,
            )
            if exact is None:
                raise CloudNetworkError(
                    "the confirmed local IP, vSwitch, and route table no longer match; generate a new preview"
                )

        entries = self.cli.list_route_entries(plan.route_table_id, canonical)
        existing = _matching_peer_route(entries, connection.connection_id)
        if entries and existing is None:
            raise CloudNetworkError(
                f"route table {plan.route_table_id} now contains {canonical} with a different next hop"
            )
        if existing is not None:
            return RoutePlan(
                connection_id=plan.connection_id,
                destination=canonical,
                route_table_id=plan.route_table_id,
                local_ip=plan.local_ip,
                vswitch_id=plan.vswitch_id,
                existing_route_id=_string(existing, "RouteEntryId"),
                existing_status=_string(existing, "Status"),
                create_required=False,
                local_vpc_id=connection.accepter.vpc_id,
            )

        route_entry_id = self.cli.create_route(
            route_table_id=plan.route_table_id,
            destination=canonical,
            connection_id=plan.connection_id,
        )
        return RoutePlan(
            connection_id=plan.connection_id,
            destination=canonical,
            route_table_id=plan.route_table_id,
            local_ip=plan.local_ip,
            vswitch_id=plan.vswitch_id,
            existing_route_id=route_entry_id,
            existing_status="Submitted",
            create_required=False,
            local_vpc_id=connection.accepter.vpc_id,
        )

    def ensure_route(
        self,
        connection_id: str,
        destination: str,
        *,
        local_ip: str | None = None,
        route_table_id: str | None = None,
        expected_vpc_id: str | None = None,
    ) -> RoutePlan:
        plan = self.plan_route(
            connection_id,
            destination,
            local_ip=local_ip,
            route_table_id=route_table_id,
            expected_vpc_id=expected_vpc_id,
        )
        return self.apply_route(plan)

    def _select_route_table(
        self,
        connection: PeerConnection,
        *,
        local_ip: str | None,
        route_table_id: str | None,
    ) -> tuple[str, RouteMatch | None]:
        switches = self.cli.list_vswitches(connection.accepter.vpc_id)
        if route_table_id is not None:
            table_id = _required_id(route_table_id, "route table ID")
            associated = [switch for switch in switches if switch.route_table_id == table_id]
            if not associated:
                raise CloudNetworkError(
                    f"route table {table_id} is not associated with a vSwitch in accepter VPC "
                    f"{connection.accepter.vpc_id}"
                )
            if local_ip is None:
                return table_id, None
            matches = _match_addresses_to_vswitches(
                (_local_ipv4_address(local_ip),),
                associated,
            )
            if not matches:
                raise CloudNetworkError(
                    f"local IP {local_ip} does not belong to a vSwitch using route table {table_id}"
                )
            return table_id, matches[0]

        addresses = (
            (_local_ipv4_address(local_ip),) if local_ip is not None else tuple(self._local_ip_provider())
        )
        matches = _match_addresses_to_vswitches(addresses, switches)
        by_table: dict[str, RouteMatch] = {}
        for match in matches:
            by_table.setdefault(match.vswitch.route_table_id, match)
        if len(by_table) == 1:
            table_id, match = next(iter(by_table.items()))
            return table_id, match
        if not by_table:
            addresses_text = ", ".join(address.address for address in addresses) or "<none>"
            raise CloudNetworkError(
                "could not match a local IPv4 address to an accepter VPC vSwitch "
                f"(local addresses: {addresses_text}); specify --local-ip or --route-table-id"
            )
        choices = ", ".join(
            f"{match.local_address.address}->{table_id}" for table_id, match in by_table.items()
        )
        raise CloudNetworkError(
            f"multiple route tables match local interfaces ({choices}); "
            "specify --local-ip or --route-table-id"
        )

    def _ensure_environment(self) -> None:
        if not self._environment_checked:
            self.check_environment()


def inspect_connection(
    identity: CloudIdentity,
    connection: PeerConnection,
    *,
    selected_region: str,
    expected_vpc_id: str | None = None,
) -> ConnectionInspection:
    if expected_vpc_id is not None:
        expected_vpc_id = _required_id(expected_vpc_id, "expected VPC ID")

    checks = [
        _check(
            "account",
            identity.account_id == connection.accepter.account_id,
            (
                f"authenticated account {identity.account_id} matches accepter account"
                if identity.account_id == connection.accepter.account_id
                else f"authenticated account {identity.account_id} does not match accepter account "
                f"{connection.accepter.account_id}"
            ),
        ),
        _check(
            "region",
            selected_region == connection.accepter.region,
            (
                f"selected region {selected_region} matches accepter region"
                if selected_region == connection.accepter.region
                else f"selected region {selected_region} does not match accepter region "
                f"{connection.accepter.region}"
            ),
        ),
        _check(
            "business_status",
            connection.business_status == "Normal",
            (
                "connection business status is Normal"
                if connection.business_status == "Normal"
                else f"connection business status is {connection.business_status or '<unknown>'}"
            ),
        ),
    ]
    if expected_vpc_id is not None:
        checks.append(
            _check(
                "accepter_vpc",
                expected_vpc_id == connection.accepter.vpc_id,
                (
                    f"connection targets expected accepter VPC {expected_vpc_id}"
                    if expected_vpc_id == connection.accepter.vpc_id
                    else f"connection targets accepter VPC {connection.accepter.vpc_id}, "
                    f"not expected VPC {expected_vpc_id}"
                ),
            )
        )

    if connection.status in {"Accepting", "Updating", "Activated"}:
        checks.append(NetworkCheck("connection_status", "ok", f"connection is {connection.status}"))
    else:
        checks.append(
            NetworkCheck(
                "connection_status",
                "error",
                f"connection cannot be used in state {connection.status or '<unknown>'}",
            )
        )

    if connection.expires_at:
        expires_at = _parse_timestamp(connection.expires_at)
        if expires_at is None:
            checks.append(
                NetworkCheck(
                    "expiration",
                    "error",
                    f"connection returned an invalid expiration time: {connection.expires_at}",
                )
            )
        elif expires_at <= datetime.now(timezone.utc):
            checks.append(
                NetworkCheck(
                    "expiration",
                    "error",
                    f"connection expired at {connection.expires_at}",
                )
            )
        else:
            checks.append(
                NetworkCheck(
                    "expiration",
                    "ok",
                    f"connection expires at {connection.expires_at}",
                )
            )

    requester_cidrs_valid = _valid_ipv4_cidrs(connection.requester.ipv4_cidrs)
    accepter_cidrs_valid = _valid_ipv4_cidrs(connection.accepter.ipv4_cidrs)
    checks.extend(
        (
            _check(
                "requester_cidrs",
                requester_cidrs_valid,
                (
                    "requester VPC returned valid IPv4 CIDRs"
                    if requester_cidrs_valid
                    else "requester VPC did not return a complete valid IPv4 CIDR list"
                ),
            ),
            _check(
                "accepter_cidrs",
                accepter_cidrs_valid,
                (
                    "accepter VPC returned valid IPv4 CIDRs"
                    if accepter_cidrs_valid
                    else "accepter VPC did not return a complete valid IPv4 CIDR list"
                ),
            ),
        )
    )
    overlap = _first_overlap(connection.requester.ipv4_cidrs, connection.accepter.ipv4_cidrs)
    cidrs_checkable = requester_cidrs_valid and accepter_cidrs_valid
    checks.append(
        _check(
            "cidr_overlap",
            cidrs_checkable and overlap is None,
            (
                "requester and accepter IPv4 CIDRs do not overlap"
                if cidrs_checkable and overlap is None
                else (
                    f"requester and accepter IPv4 CIDRs overlap: {overlap[0]} and {overlap[1]}"
                    if overlap is not None
                    else "CIDR overlap cannot be checked because a VPC CIDR list is invalid"
                )
            ),
        )
    )
    return ConnectionInspection(identity=identity, connection=connection, checks=tuple(checks))


def validate_destination(connection: PeerConnection, destination: str) -> str:
    canonical = normalize_ipv4(destination)
    destination_network = ipaddress.ip_network(canonical)
    remote_networks = _networks(connection.requester.ipv4_cidrs)
    local_networks = _networks(connection.accepter.ipv4_cidrs)
    if not remote_networks:
        raise CloudNetworkError("the peer response does not contain a requester IPv4 CIDR")
    if not any(destination_network.subnet_of(remote) for remote in remote_networks):
        values = ", ".join(str(network) for network in remote_networks)
        raise CloudNetworkError(
            f"destination {canonical} is not contained in requester VPC CIDR(s): {values}"
        )
    for local in local_networks:
        if destination_network.overlaps(local):
            raise CloudNetworkError(f"destination {canonical} overlaps accepter VPC CIDR {local}")
    return canonical


def _parse_peer_connection(payload: dict[str, Any]) -> PeerConnection:
    requester = payload.get("Vpc")
    accepter = payload.get("AcceptingVpc")
    if not isinstance(requester, dict) or not isinstance(accepter, dict):
        raise CloudNetworkError("GetVpcPeerConnectionAttribute did not return both VPC objects")
    connection_id = _required_string(payload, "InstanceId", "connection ID")
    return PeerConnection(
        connection_id=connection_id,
        name=str(payload.get("Name", "")),
        status=_required_string(payload, "Status", "connection status"),
        business_status=str(payload.get("BizStatus", "")),
        expires_at=(str(payload["GmtExpired"]) if payload.get("GmtExpired") else None),
        requester=VpcPeerSide(
            account_id=_required_string(payload, "OwnerId", "requester account ID"),
            region=_required_string(payload, "RegionId", "requester region"),
            vpc_id=_required_string(requester, "VpcId", "requester VPC ID"),
            ipv4_cidrs=_string_tuple(requester.get("Ipv4Cidrs")),
        ),
        accepter=VpcPeerSide(
            account_id=_required_string(payload, "AcceptingOwnerUid", "accepter account ID"),
            region=_required_string(payload, "AcceptingRegionId", "accepter region"),
            vpc_id=_required_string(accepter, "VpcId", "accepter VPC ID"),
            ipv4_cidrs=_string_tuple(accepter.get("Ipv4Cidrs")),
        ),
    )


def _parse_vswitch(payload: dict[str, Any]) -> VSwitch:
    route_table = payload.get("RouteTable")
    if not isinstance(route_table, dict):
        route_table = {}
    return VSwitch(
        vswitch_id=_required_string(payload, "VSwitchId", "vSwitch ID"),
        vpc_id=_required_string(payload, "VpcId", "vSwitch VPC ID"),
        name=str(payload.get("VSwitchName", "")),
        cidr=_required_string(payload, "CidrBlock", "vSwitch CIDR"),
        status=str(payload.get("Status", "")),
        route_table_id=str(route_table.get("RouteTableId", "")),
        route_table_type=str(route_table.get("RouteTableType", "")),
    )


def _parse_vpc(payload: dict[str, Any]) -> Vpc:
    primary = _required_string(payload, "CidrBlock", "VPC IPv4 CIDR")
    secondary = _string_values(
        payload.get("SecondaryCidrBlocks"),
        "SecondaryCidrBlock",
    )
    raw_cidrs = (primary, *secondary)
    if not _valid_ipv4_cidrs(raw_cidrs):
        raise CloudNetworkError("DescribeVpcAttribute did not return a complete valid IPv4 CIDR list")
    cidrs = _canonical_cidrs(raw_cidrs)
    return Vpc(
        vpc_id=_required_string(payload, "VpcId", "VPC ID"),
        account_id=_required_string(payload, "OwnerId", "VPC owner account ID"),
        name=str(payload.get("VpcName", "")),
        region=_required_string(payload, "RegionId", "VPC region"),
        status=_required_string(payload, "Status", "VPC status"),
        ipv4_cidrs=cidrs,
    )


def _matching_peer_route(
    entries: Iterable[dict[str, Any]],
    connection_id: str,
) -> dict[str, Any] | None:
    exact_entries = tuple(entries)
    if len(exact_entries) != 1:
        return None
    entry = exact_entries[0]
    next_hops = _extract_list(entry.get("NextHops"), "NextHop")
    if not next_hops:
        next_hops = [entry]
    if len(next_hops) != 1:
        return None
    next_hop = next_hops[0]
    next_hop_id = next_hop.get("NextHopId") or next_hop.get("InstanceId")
    next_hop_type = next_hop.get("NextHopType") or entry.get("NextHopType")
    if str(next_hop_id) != connection_id or str(next_hop_type) != "VpcPeer":
        return None
    return entry


def _match_addresses_to_vswitches(
    addresses: Sequence[LocalIpv4Address],
    switches: Sequence[VSwitch],
) -> tuple[RouteMatch, ...]:
    matches: list[RouteMatch] = []
    for address in addresses:
        try:
            parsed_address = ipaddress.ip_address(address.address)
        except ValueError:
            continue
        if not isinstance(parsed_address, ipaddress.IPv4Address):
            continue
        for vswitch in switches:
            if vswitch.status != "Available" or not vswitch.route_table_id:
                continue
            try:
                network = ipaddress.ip_network(vswitch.cidr, strict=False)
            except ValueError:
                continue
            if isinstance(network, ipaddress.IPv4Network) and parsed_address in network:
                matches.append(RouteMatch(local_address=address, vswitch=vswitch))
    return tuple(matches)


def _extract_list(value: Any, nested_key: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get(nested_key, [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_values(value: Any, nested_key: str) -> tuple[str, ...]:
    if isinstance(value, dict):
        value = value.get(nested_key, [])
    if not isinstance(value, list):
        return ()
    result: list[str] = []
    for item in value:
        if isinstance(item, dict):
            nested = item.get(nested_key)
            if isinstance(nested, list):
                result.extend(str(entry) for entry in nested if str(entry))
            continue
        text = str(item)
        if text:
            result.append(text)
    return tuple(result)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if str(item))


def _required_string(payload: dict[str, Any], key: str, label: str) -> str:
    value = str(payload.get(key, ""))
    if not value:
        raise CloudNetworkError(f"cloud response did not contain {label}")
    return value


def _required_id(value: str, label: str) -> str:
    result = value.strip()
    if not result:
        raise CloudNetworkError(f"{label} must not be empty")
    return result


def _string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    return str(value) if value is not None and str(value) else None


def _integer(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _networks(values: Iterable[str]) -> tuple[ipaddress.IPv4Network, ...]:
    result: list[ipaddress.IPv4Network] = []
    for value in values:
        try:
            network = ipaddress.ip_network(value, strict=False)
        except ValueError:
            continue
        if isinstance(network, ipaddress.IPv4Network):
            result.append(network)
    return tuple(result)


def _canonical_cidrs(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(network) for network in _networks(values)}))


def _valid_ipv4_cidrs(values: Iterable[str]) -> bool:
    raw = tuple(values)
    return bool(raw) and len(_networks(raw)) == len(raw)


def _first_overlap(
    requester_values: Iterable[str],
    accepter_values: Iterable[str],
) -> tuple[str, str] | None:
    requester = _networks(requester_values)
    accepter = _networks(accepter_values)
    for left in requester:
        for right in accepter:
            if left.overlaps(right):
                return str(left), str(right)
    return None


def _parse_timestamp(value: str) -> datetime | None:
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def _check(name: str, condition: bool, detail: str) -> NetworkCheck:
    return NetworkCheck(name=name, status="ok" if condition else "error", detail=detail)


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
