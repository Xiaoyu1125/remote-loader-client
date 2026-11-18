"""Retained Aliyun network commands, not registered by the public CLI.

This module has no console-script or __main__ entry point.
"""

from __future__ import annotations

import argparse
import json
import sys

from remote_loader.beanfleet import BeanfleetClient, BeanfleetPeering, BeanfleetPeeringStatus
from remote_loader.client_config import ClientConfig
from remote_loader.cloud_network import AliyunNetworkClient, LocalNetwork
from remote_loader.peering_workflow import PeeringWorkflow
from remote_loader.telemetry.logging import configure_logging


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="remote-loader")
    sub = root.add_subparsers(dest="command", required=True)

    network = sub.add_parser("network", help="manage a cloud private-network connection")
    network_sub = network.add_subparsers(dest="network_command", required=True)

    def add_network_context(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--provider",
            choices=("aliyun",),
            default="aliyun",
            help="cloud provider (default: aliyun)",
        )
        command.add_argument("--profile", required=True, help="Alibaba Cloud CLI profile")
        command.add_argument("--region", required=True, help="local VPC region")

    def add_local_selector(
        command: argparse.ArgumentParser,
        *,
        require_vpc_id: bool = False,
    ) -> None:
        command.add_argument(
            "--local-ip",
            help="local instance IPv4 address when multiple interfaces match",
        )
        command.add_argument(
            "--vpc-id",
            required=require_vpc_id,
            help="expected local VPC ID or a discovered candidate to select",
        )

    def add_connection_context(
        command: argparse.ArgumentParser,
        *,
        require_vpc_id: bool = False,
    ) -> None:
        add_network_context(command)
        add_local_selector(command, require_vpc_id=require_vpc_id)
        command.add_argument("--connection-id", required=True, help="VPC peering instance ID")

    network_auth = network_sub.add_parser(
        "auth",
        help="check Alibaba Cloud CLI, plugins, and profile identity",
    )
    add_network_context(network_auth)

    network_info = network_sub.add_parser(
        "info",
        help="discover local VPC and route-table information",
    )
    add_network_context(network_info)
    add_local_selector(network_info)

    network_connect = network_sub.add_parser(
        "connect",
        help="create or resume a complete beanfleet VPC peering workflow",
    )
    add_network_context(network_connect)
    add_local_selector(network_connect, require_vpc_id=True)
    network_connect.add_argument(
        "--connection-id",
        help="existing VPC peering ID to resume instead of creating a new connection",
    )
    network_connect.add_argument(
        "--yes",
        action="store_true",
        help="pre-approve both displayed mutation plans",
    )

    network_inspect = network_sub.add_parser(
        "inspect",
        help="preview and validate a supplied VPC peering connection",
    )
    add_connection_context(network_inspect)

    network_accept = network_sub.add_parser(
        "accept",
        help="accept a validated VPC peering connection",
    )
    add_connection_context(network_accept, require_vpc_id=True)
    network_accept.add_argument("--yes", action="store_true", help="execute the displayed plan")

    network_route = network_sub.add_parser(
        "route",
        help="discover the local route table and create the return route",
    )
    add_connection_context(network_route, require_vpc_id=True)
    network_route.add_argument(
        "--destination",
        required=True,
        help="requester CIDR or a narrower host route",
    )
    network_route.add_argument(
        "--route-table-id",
        help="explicit route table ID when automatic discovery is ambiguous",
    )
    network_route.add_argument("--yes", action="store_true", help="execute the displayed plan")

    network_status = network_sub.add_parser(
        "status",
        help="query connection status and optionally inspect a return route",
    )
    add_connection_context(network_status)
    network_status.add_argument("--destination", help="return-route destination to inspect")
    network_status.add_argument("--route-table-id", help="explicit route table ID")
    return root


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parser().parse_args(argv)
    try:
        return _manage_network(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


def _manage_network(args: argparse.Namespace) -> int:
    client = _network_client(args)
    if args.network_command == "connect":
        return _connect_network(args, client)
    if args.network_command == "auth":
        version = client.check_environment()
        _print_json(
            {
                "provider": args.provider,
                "region": args.region,
                "profile": args.profile,
                "aliyun_cli_version": version,
                "identity": client.identity().to_dict(),
            }
        )
        return 0
    if args.network_command == "info":
        network = client.discover_local_network(
            local_ip=args.local_ip,
            vpc_id=args.vpc_id,
        )
        _print_json(
            {
                "provider": args.provider,
                "region": args.region,
                "profile": args.profile,
                "local_network": network.to_dict(),
            }
        )
        return 0

    if args.network_command == "route":
        plan = client.plan_route(
            args.connection_id,
            args.destination,
            local_ip=args.local_ip,
            route_table_id=args.route_table_id,
            expected_vpc_id=args.vpc_id,
        )
        _print_network_preview("route", plan.to_dict())
        if not args.yes:
            print("route was not changed; review the preview and re-run with --yes", file=sys.stderr)
            return 2
        result = client.apply_route(plan)
        _print_json({"route": result.to_dict()})
        return 0

    inspection = client.inspect_connection(
        args.connection_id,
        expected_vpc_id=args.vpc_id,
        local_ip=args.local_ip,
        verify_local_network=not (
            args.network_command == "status"
            and args.destination is not None
            and args.route_table_id is not None
            and args.local_ip is None
        ),
    )
    if args.network_command == "inspect":
        _print_json(inspection.to_dict())
        return 1 if inspection.has_errors else 0
    if args.network_command == "accept":
        inspection.require_actionable()
        if inspection.connection.status in {"Updating", "Activated"}:
            _print_json(
                {
                    "connection": inspection.connection.to_dict(),
                    "checks": [check.to_dict() for check in inspection.checks],
                }
            )
            return 0
        _print_network_preview("accept", inspection.to_dict())
        if not args.yes:
            raise ValueError("accept was not executed; review the preview and re-run with --yes")
        connection = client.accept_connection(
            args.connection_id,
            expected_vpc_id=args.vpc_id,
            local_ip=args.local_ip,
        )
        _print_json({"connection": connection.to_dict()})
        return 0
    if args.network_command == "status":
        payload: dict[str, object] = {"connection": inspection.to_dict()}
        if args.destination:
            payload["route"] = client.plan_route(
                args.connection_id,
                args.destination,
                local_ip=args.local_ip,
                route_table_id=args.route_table_id,
                expected_vpc_id=args.vpc_id,
            ).to_dict()
        elif args.route_table_id:
            raise ValueError("--route-table-id requires --destination")
        _print_json(payload)
        return 1 if inspection.has_errors else 0
    raise ValueError(f"unknown network command {args.network_command!r}")


def _network_client(args: argparse.Namespace) -> AliyunNetworkClient:
    if args.provider != "aliyun":
        raise ValueError(f"unsupported cloud provider {args.provider!r}")
    return AliyunNetworkClient(profile=args.profile, region=args.region)


def _beanfleet_client(config: ClientConfig) -> BeanfleetClient:
    if config.beanfleet_endpoint is None:
        raise ValueError(
            "beanfleet_endpoint is not configured; run 'remote-loader config edit "
            "--beanfleet-endpoint URL' first"
        )
    return BeanfleetClient(
        config.beanfleet_endpoint,
        verify=config.verify_beanfleet_tls,
    )


def _connect_network(args: argparse.Namespace, client: AliyunNetworkClient) -> int:
    config = ClientConfig.load()
    beanfleet = _beanfleet_client(config)
    local_network: LocalNetwork | None = None
    peering_id = args.connection_id
    try:
        local_network = client.discover_local_network(
            local_ip=args.local_ip,
            vpc_id=args.vpc_id,
        )
        workflow = PeeringWorkflow(
            network_client=client,
            beanfleet_client=beanfleet,
            region_id=args.region,
            progress=_print_peering_progress,
        )
        if peering_id is None:
            creation = workflow.creation_plan(local_network)
            _print_network_preview("beanfleet create", creation.to_dict())
            if not _approve_network_action("create the peering connection", args.yes):
                print("peering connection was not created", file=sys.stderr)
                return 2
            peering = workflow.create(creation)
            peering_id = peering.peering_id
            print(
                f"created peering {peering_id}; keep this ID to resume after interruption",
                file=sys.stderr,
            )

        completion = workflow.prepare_completion(local_network, peering_id)
        _print_network_preview("peering completion", completion.to_dict())
        if completion.peering.status == BeanfleetPeeringStatus.ACTIVATED:
            print(
                "WARNING: beanfleet is still Activated; /accepted may already be running, "
                "and repeating it is not guaranteed idempotent.",
                file=sys.stderr,
            )
        if not _approve_network_action(
            "accept the connection, configure beanfleet, and ensure every reviewed route exists",
            args.yes,
        ):
            print(_resume_network_command(args, local_network, peering_id), file=sys.stderr)
            return 2
        result = workflow.complete(completion)
        result_payload = result.to_dict()
        if result_payload.get("ready") is not True:
            raise ValueError(f"peering {peering_id} finished without all readiness conditions")
        _print_json(result_payload)
        return 0
    except Exception:
        if peering_id is not None and local_network is not None:
            print(_resume_network_command(args, local_network, peering_id), file=sys.stderr)
        raise
    finally:
        beanfleet.close()


def _approve_network_action(action: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    print(f"Proceed to {action}? [y/N] ", file=sys.stderr, end="", flush=True)
    return sys.stdin.readline().strip().lower() in {"y", "yes"}


def _resume_network_command(args: argparse.Namespace, local_network: LocalNetwork, peering_id: str) -> str:
    return (
        "connection left in place; inspect beanfleet status before recovery because "
        "repeating /accepted while Activated is not guaranteed idempotent; "
        "candidate command: remote-loader network connect "
        f"--profile {args.profile} --region {args.region} "
        f"--vpc-id {local_network.vpc.vpc_id} "
        f"--local-ip {local_network.local_address.address} "
        f"--connection-id {peering_id}"
    )


def _print_peering_progress(phase: str, peering: BeanfleetPeering) -> None:
    print(
        f"peering {peering.peering_id}: {phase}: {peering.status.value}",
        file=sys.stderr,
    )


def _print_network_preview(action: str, payload: dict[str, object]) -> None:
    print(f"{action} preview:", file=sys.stderr)
    print(json.dumps(payload, indent=2, ensure_ascii=False), file=sys.stderr)


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))
