from __future__ import annotations

import argparse
import getpass
import json
import math
import sys
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import urlparse

from remote_loader.auth import TokenStore
from remote_loader.cli.diagnostics import (
    DiagnosticReport,
    collect_connection_diagnostics,
    collect_environment_diagnostics,
)
from remote_loader.cli.statistics import add_statistics_parser, run_statistics
from remote_loader.client_config import (
    ClientConfig,
    ConfigStore,
    PlacementConfig,
    RuntimeConfig,
    normalize_beanfleet_endpoint,
    normalize_control_endpoint,
)
from remote_loader.config_migration import is_schema6_config, migrate_schema6
from remote_loader.exceptions import ConfigurationError
from remote_loader.loader import RemoteDataLoader
from remote_loader.telemetry.logging import configure_logging
from remote_loader.transfer_state import TransferStateStore

INTERACTIVE_INSTALL = "pip install 'modalink-remote-loader-client[interactive]'"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="remote-loader")
    sub = root.add_subparsers(dest="command")

    sub.add_parser("interactive", help="open the guided terminal client")

    receive = sub.add_parser(
        "receive", help="consume raw VLA samples or VQA batches and print JSON summaries"
    )
    receive.add_argument("batch_spec", metavar="BATCH_SPEC", help="BatchSpec YAML filepath")

    config = sub.add_parser("config", help="manage client connection configuration")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    config_sub.add_parser("show", help="print the effective client configuration")
    config_migrate = config_sub.add_parser(
        "migrate", help="explicitly upgrade a saved schema 6 configuration to mTLS schema 7"
    )
    config_migrate.add_argument("--control-endpoint", required=True, help="Loader control HTTPS URL")
    config_migrate.add_argument("--tls-ca-file", help="path to your Loader CA bundle (not bundled)")
    config_migrate.add_argument("--tls-client-cert-file", help="path to your GPU client certificate (not bundled)")
    config_migrate.add_argument("--tls-client-key-file", help="path to your GPU client private key (not bundled)")
    config_edit = config_sub.add_parser(
        "edit",
        help="update connection configuration from command-line options",
    )
    config_edit.add_argument("--provider", help="service provider id, for example aliyun")
    config_edit.add_argument("--region", help="service region id, for example cn-beijing")
    config_edit.add_argument("--zone", help="availability zone id, for example cn-beijing-f")
    config_edit.add_argument("--control-endpoint", help="control service HTTPS URL")
    config_edit.add_argument("--tls-ca-file", help="CA bundle used to verify loader certificates")
    config_edit.add_argument("--tls-client-cert-file", help="GPU client certificate PEM file")
    config_edit.add_argument("--tls-client-key-file", help="GPU client private key PEM file")
    # Keep legacy settings parseable without an empty hidden mutually exclusive group.
    beanfleet_endpoint = config_edit.add_argument_group()
    beanfleet_endpoint.add_argument("--beanfleet-endpoint", help=argparse.SUPPRESS)
    beanfleet_endpoint.add_argument(
        "--clear-beanfleet-endpoint",
        action="store_true",
        default=None,
        help=argparse.SUPPRESS,
    )
    config_edit.add_argument(
        "--verify-beanfleet-tls",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    config_edit.add_argument(
        "--prefetch-batches",
        type=int,
        help="prefetch capacity in HC frames (VLA samples or VQA batches)",
    )
    config_edit.add_argument("--training-id", help="preassigned training ID shared by all ranks of this run")
    config_edit.add_argument("--scheduler-endpoint", help="scheduler HTTP(S) URL")
    config_edit.add_argument(
        "--node-control-port",
        type=int,
        help="control-service port on assigned nodes",
    )
    config_edit.add_argument(
        "--node-control-scheme",
        choices=("https",),
        help="control-service URL scheme on assigned nodes (HTTPS required)",
    )
    config_edit.add_argument(
        "--verify-scheduler-tls",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="verify scheduler HTTPS certificates",
    )
    config_edit.add_argument(
        "--clear-scheduler",
        action="store_true",
        default=None,
        help="disable scheduler discovery",
    )

    token = sub.add_parser("token", help="manage the pre-issued token")
    token_sub = token.add_subparsers(dest="token_command", required=True)
    token_save = token_sub.add_parser("save", help="store a pre-issued token without network access")
    token_save.add_argument(
        "--token-file",
        help="read the token from a file instead of the hidden interactive prompt",
    )

    transfer = sub.add_parser("transfer", help="inspect local transfer state")
    transfer_sub = transfer.add_subparsers(dest="transfer_command", required=True)
    transfer_sub.add_parser("current", help="show the current local transfer")
    transfer_sub.add_parser("last", help="show the most recent local transfer result")

    diagnose = sub.add_parser("diagnose", help="run client diagnostics")
    diagnose_sub = diagnose.add_subparsers(dest="diagnose_command", required=True)
    diagnose_sub.add_parser("environment", help="check the local Python and PyTorch environment")
    diagnose_sub.add_parser(
        "connection", aliases=["scheduler"], help="check scheduler allocation and VLA/VQA node connections"
    )

    training = sub.add_parser(
        "training", help="register, inspect, and release training data-service resources"
    )
    training_sub = training.add_subparsers(dest="training_command", required=True)
    register = training_sub.add_parser("register", help="register using the saved training parameters")
    register.add_argument("training_config", nargs="?", help="optional training YAML/JSON filepath")
    status = training_sub.add_parser("status", help="show the assigned VLA/VQA resource states")
    release = training_sub.add_parser("release-rank", help="release one local GPU's VLA and VQA resources")
    release.add_argument("--local-rank", type=int, required=True, help="GPU index on this host to release")
    end = training_sub.add_parser("end", help="release all ranks and end the scheduler registration")
    for command in (register, status, release, end):
        command.add_argument("--scheduler-endpoint", help="override the saved scheduler HTTP(S) URL")
        command.add_argument(
            "--verify-scheduler-tls",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="verify scheduler HTTPS certificates",
        )
        if command is not register:
            command.add_argument("--training-id", help="override the saved training ID")

    add_statistics_parser(sub)
    return root


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    root = parser()
    try:
        if not raw_args:
            if _is_interactive_terminal():
                return _run_interactive()
            root.print_help()
            return 0

        args = root.parse_args(raw_args)
        if args.command == "interactive":
            return _run_interactive()
        if args.command == "receive":
            return _receive(args.batch_spec)
        if args.command == "config":
            if args.config_command == "show":
                return _show_config()
            if args.config_command == "edit":
                return _edit_config(args)
            if args.config_command == "migrate":
                return _migrate_config(args)
        if args.command == "token" and args.token_command == "save":
            return _save_token(args.token_file)
        if args.command == "transfer":
            return _show_transfer(args.transfer_command)
        if args.command == "diagnose":
            if args.diagnose_command == "environment":
                return _diagnose_environment()
            if args.diagnose_command in {"connection", "scheduler"}:
                return _diagnose_connection()
        if args.command == "statistics":
            return run_statistics(args)
        if args.command == "training":
            return _manage_training(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 2


def _manage_training(args: argparse.Namespace) -> int:
    from remote_loader.cli.training import TrainingCommands, load_training_config

    commands = TrainingCommands()
    options = {
        "scheduler_endpoint": args.scheduler_endpoint,
        "verify_scheduler_tls": args.verify_scheduler_tls,
    }
    if args.training_command == "register":
        training = load_training_config(args.training_config) if args.training_config else None
        result = commands.register(training, **options)
    else:
        options["training_id"] = args.training_id
        if args.training_command == "status":
            result = commands.status(**options)
        elif args.training_command == "release-rank":
            result = commands.release_rank(local_rank=args.local_rank, **options)
        else:
            result = commands.end(**options)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _preview_scalar(value: float | int) -> float | int | str:
    if not isinstance(value, float) or math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _receive(batch_spec: str) -> int:
    def fields(item, prefix=""):
        for name, value in item.items():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(value, dict):
                yield from fields(value, path)
            else:
                yield path, value

    try:
        with RemoteDataLoader.from_yaml(batch_spec) as loader:
            counter = "batch_index" if loader.spec.is_vqa else "sample_index"
            for item_index, item in enumerate(loader, start=1):
                leaves = list(fields(item))
                if item_index == 1:
                    schema = {
                        name: (
                            {"shape": list(value.shape), "dtype": str(value.dtype)}
                            if hasattr(value, "shape") and hasattr(value, "dtype")
                            else {"type": type(value).__name__}
                        )
                        for name, value in leaves
                    }
                    print(json.dumps({"schema": schema}, allow_nan=False), flush=True)
                values = {
                    name: [_preview_scalar(element) for element in value.reshape(-1)[:6].tolist()]
                    for name, value in leaves
                    if hasattr(value, "shape") and hasattr(value, "dtype")
                }
                print(json.dumps({counter: item_index, "values": values}, allow_nan=False), flush=True)
    except KeyboardInterrupt:
        return 130
    return 0


def _read_token(token_file: str | None) -> str:
    if token_file is None:
        return getpass.getpass("Token: ").strip()
    try:
        return Path(token_file).expanduser().read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError("could not read token file") from exc


def _show_config() -> int:
    try:
        config = ClientConfig.load()
    except ConfigurationError as exc:
        if is_schema6_config(ConfigStore()):
            raise ConfigurationError("schema 6 configuration requires remote-loader config migrate") from exc
        raise
    payload = config.to_dict()
    if config.scheduler is not None:
        payload["scheduler"]["endpoint"] = config.scheduler.endpoint
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _migrate_config(args: argparse.Namespace) -> int:
    store = ConfigStore()
    config, backup = migrate_schema6(
        store,
        control_endpoint=args.control_endpoint,
        tls_ca_file=args.tls_ca_file,
        tls_client_cert_file=args.tls_client_cert_file,
        tls_client_key_file=args.tls_client_key_file,
    )
    print(f"Migrated schema 6 to schema {config.schema_version} at {store.path}. Backup: {backup}.")
    return 0


def _edit_config(args: argparse.Namespace) -> int:
    if args.beanfleet_endpoint is not None and args.clear_beanfleet_endpoint:
        raise ValueError("--beanfleet-endpoint cannot be combined with --clear-beanfleet-endpoint")
    option_names = (
        "provider",
        "region",
        "zone",
        "control_endpoint",
        "tls_ca_file",
        "tls_client_cert_file",
        "tls_client_key_file",
        "beanfleet_endpoint",
        "clear_beanfleet_endpoint",
        "verify_beanfleet_tls",
        "prefetch_batches",
        "training_id",
        "scheduler_endpoint",
        "node_control_port",
        "node_control_scheme",
        "verify_scheduler_tls",
        "clear_scheduler",
    )
    if all(getattr(args, name) is None for name in option_names):
        raise ValueError("config edit requires at least one option; use --help to list them")

    store = ConfigStore()
    try:
        current = store.load_optional() or ClientConfig.default()
    except ConfigurationError as exc:
        if is_schema6_config(store):
            raise ConfigurationError("schema 6 configuration requires remote-loader config migrate") from exc
        raise
    endpoint_changed = args.control_endpoint is not None
    endpoint = (
        current.control_endpoint
        if not endpoint_changed
        else normalize_control_endpoint(args.control_endpoint)
    )
    if not endpoint.startswith("https://"):
        raise ValueError("control endpoint must use HTTPS")

    beanfleet_endpoint_changed = args.beanfleet_endpoint is not None or args.clear_beanfleet_endpoint is True
    if args.clear_beanfleet_endpoint:
        beanfleet_endpoint = None
    elif args.beanfleet_endpoint is not None:
        beanfleet_endpoint = normalize_beanfleet_endpoint(args.beanfleet_endpoint)
    else:
        beanfleet_endpoint = current.beanfleet_endpoint
    if beanfleet_endpoint is None and args.verify_beanfleet_tls is not None:
        raise ValueError("--verify-beanfleet-tls requires a configured --beanfleet-endpoint")
    if (
        beanfleet_endpoint_changed
        and beanfleet_endpoint is not None
        and urlparse(beanfleet_endpoint).scheme.lower() == "http"
        and args.verify_beanfleet_tls is not False
    ):
        raise ValueError("plain HTTP requires --no-verify-beanfleet-tls")

    verify_beanfleet_tls = (
        args.verify_beanfleet_tls
        if args.verify_beanfleet_tls is not None
        else (True if beanfleet_endpoint_changed else current.verify_beanfleet_tls)
    )
    placement = _updated_placement(current.placement, args)
    runtime = RuntimeConfig(
        prefetch_batches=(
            current.runtime.prefetch_batches if args.prefetch_batches is None else args.prefetch_batches
        )
    )
    training = current.training
    if args.training_id is not None:
        training = replace(training, training_id=args.training_id)
    scheduler_option_names = (
        "scheduler_endpoint",
        "node_control_port",
        "node_control_scheme",
        "verify_scheduler_tls",
    )
    scheduler_values_set = any(getattr(args, name) is not None for name in scheduler_option_names)
    if args.clear_scheduler and (scheduler_values_set or args.training_id is not None):
        raise ValueError("--clear-scheduler cannot be combined with scheduler setting options")
    if args.clear_scheduler:
        scheduler = None
    elif scheduler_values_set or args.training_id is not None:
        current_scheduler = current.resolve_scheduler(args.scheduler_endpoint)
        scheduler = replace(
            current_scheduler,
            control_scheme=(
                args.node_control_scheme
                if args.node_control_scheme is not None
                else current_scheduler.control_scheme
            ),
            control_port=(
                args.node_control_port
                if args.node_control_port is not None
                else current_scheduler.control_port
            ),
            verify_tls=(
                args.verify_scheduler_tls
                if args.verify_scheduler_tls is not None
                else current_scheduler.verify_tls
            ),
        )
    else:
        scheduler = current.scheduler
    updated = replace(
        current,
        control_endpoint=endpoint,
        beanfleet_endpoint=beanfleet_endpoint,
        tls_ca_file=(args.tls_ca_file if args.tls_ca_file is not None else current.tls_ca_file),
        tls_client_cert_file=(
            args.tls_client_cert_file
            if args.tls_client_cert_file is not None
            else current.tls_client_cert_file
        ),
        tls_client_key_file=(
            args.tls_client_key_file if args.tls_client_key_file is not None else current.tls_client_key_file
        ),
        verify_beanfleet_tls=verify_beanfleet_tls,
        placement=placement,
        runtime=runtime,
        scheduler=scheduler,
        training=training,
    )
    store.save(updated)
    print(f"Configuration saved to {store.path}.")
    return 0


def _updated_placement(
    current: PlacementConfig | None,
    args: argparse.Namespace,
) -> PlacementConfig | None:
    if args.provider is None and args.region is None and args.zone is None:
        return current

    values = {
        "provider": args.provider if args.provider is not None else getattr(current, "provider", None),
        "region": args.region if args.region is not None else getattr(current, "region", None),
        "zone": args.zone if args.zone is not None else getattr(current, "zone", None),
    }
    missing = [f"--{name}" for name, value in values.items() if value is None]
    if missing:
        raise ValueError("placement update requires " + ", ".join(missing) + " when no saved value exists")
    return PlacementConfig(**values)


def _save_token(token_file: str | None) -> int:
    store = TokenStore()
    store.save(_read_token(token_file))
    print(f"Token stored in {store.path}.")
    return 0


def _show_transfer(which: str) -> int:
    state = TransferStateStore().load()
    snapshot = state.current_transfer if which == "current" else state.last_transfer
    payload = (
        None
        if snapshot is None
        else snapshot.model_dump(
            mode="json",
            exclude={"session_id", "transfer_id"},
        )
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def _print_diagnostics(report: DiagnosticReport) -> int:
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str))
    return report.exit_code


def _diagnose_environment() -> int:
    return _print_diagnostics(collect_environment_diagnostics())


def _diagnose_connection() -> int:
    return _print_diagnostics(collect_connection_diagnostics())


def _is_interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _interactive_dependencies_available() -> bool:
    missing = [package for package in ("questionary", "rich") if find_spec(package) is None]
    if not missing:
        return True
    print(
        "Interactive mode requires the optional 'interactive' dependencies "
        f"({', '.join(missing)}). Install them with:\n  {INTERACTIVE_INSTALL}",
        file=sys.stderr,
    )
    return False


def _run_interactive() -> int:
    if not _interactive_dependencies_available():
        return 2
    from remote_loader.cli.interactive import run_interactive

    return run_interactive()


if __name__ == "__main__":
    raise SystemExit(main())
