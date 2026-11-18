"""Explicit statistics commands; no training/configuration mutation."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from remote_loader.client_config import ClientConfig
from remote_loader.exceptions import ConfigurationError
from remote_loader.statistics import StatisticsClient
from remote_loader.statistics_spec import StatisticsSpec


def add_statistics_parser(sub: Any) -> None:
    statistics = sub.add_parser("statistics", help="compute finite, fixed-snapshot tensor statistics")
    commands = statistics.add_subparsers(dest="statistics_command", required=True)
    catalog = commands.add_parser("catalog", help="inspect the ordered statistics catalog")
    submit = commands.add_parser("submit", help="submit a statistics YAML without starting training")
    submit.add_argument("spec", metavar="YAML")
    submit.add_argument("--wait", action="store_true", help="wait for a successful result")
    submit.add_argument(
        "--timeout", type=float, default=300.0, help="wait deadline in seconds (default: 300)"
    )
    submit.add_argument(
        "--output", metavar="PATH", help="save result JSON; requires --wait; never overwrites"
    )
    submit.add_argument("--idempotency-key", help="reuse a submission key only for the identical request")
    status = commands.add_parser("status", help="read job state")
    status.add_argument("job_id", metavar="ID")
    result = commands.add_parser("result", help="retrieve a completed result")
    result.add_argument("job_id", metavar="ID")
    result.add_argument("--output", metavar="PATH", help="save result JSON without overwriting")
    cancel = commands.add_parser("cancel", help="cancel a job, not its training allocation")
    cancel.add_argument("job_id", metavar="ID")
    for command in (catalog, submit, status, result, cancel):
        command.add_argument(
            "--control-endpoint", help="use this one HTTPS endpoint; bypass scheduler discovery"
        )
        command.add_argument(
            "--local-rank", type=int, default=0,
            help="accepted for compatibility; statistics discovery covers the whole training",
        )
        command.add_argument("--training-id", help="override training ID for read-only allocation discovery")
        command.add_argument("--scheduler-endpoint", help="override scheduler URL for read-only discovery")
        command.add_argument("--verify-scheduler-tls", action=argparse.BooleanOptionalAction, default=None)


def write_result(path: str | Path, value: dict[str, Any]) -> None:
    """Publish complete JSON atomically, refusing existing files and symlinks."""
    destination = Path(path).expanduser()
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _client(args: argparse.Namespace) -> StatisticsClient:
    if args.control_endpoint is not None:
        if (
            args.scheduler_endpoint is not None
            or args.training_id is not None
            or args.verify_scheduler_tls is not None
        ):
            raise ConfigurationError("explicit control endpoint cannot be combined with scheduler overrides")
        return StatisticsClient(
            args.control_endpoint,
            client_config=ClientConfig.load(),
        )
    config = ClientConfig.load()
    scheduler = config.scheduler
    if args.scheduler_endpoint is not None:
        scheduler = config.resolve_scheduler(args.scheduler_endpoint)
    if args.verify_scheduler_tls is not None:
        if scheduler is None:
            raise ConfigurationError("--verify-scheduler-tls requires scheduler discovery")
        scheduler = replace(scheduler, verify_tls=args.verify_scheduler_tls)
    training = config.training
    if args.training_id is not None:
        if scheduler is None:
            raise ConfigurationError("--training-id requires scheduler discovery")
        training = replace(training, training_id=args.training_id)
    config = replace(
        config,
        scheduler=scheduler,
        training=training,
    )
    return StatisticsClient.from_config(config, local_rank=args.local_rank)


def run_statistics(args: argparse.Namespace) -> int:
    output = getattr(args, "output", None)
    spec = None
    if args.statistics_command == "submit":
        if output is not None and not args.wait:
            raise ConfigurationError("submit --output requires --wait")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ConfigurationError("--timeout must be finite and positive")
        spec = StatisticsSpec.from_yaml(args.spec)
    if output is not None and os.path.lexists(Path(output).expanduser()):
        raise ConfigurationError("output path already exists; refusing to overwrite")
    with _client(args) as client:
        if args.statistics_command == "catalog":
            value = client.catalog()
        elif args.statistics_command == "submit":
            job = client.submit(spec, idempotency_key=args.idempotency_key)
            job_output = {**job, "control_endpoint": client.endpoint}
            if args.wait:
                # Keep the ID visible even if the wait times out or the job fails.
                print(json.dumps(job_output, ensure_ascii=False, allow_nan=False), file=sys.stderr)
                value = client.wait(job["job_id"], timeout=args.timeout)
            else:
                value = job_output
        elif args.statistics_command == "status":
            value = {**client.status(args.job_id), "control_endpoint": client.endpoint}
        elif args.statistics_command == "result":
            value = client.result(args.job_id)
        else:
            value = {**client.cancel(args.job_id), "control_endpoint": client.endpoint}
    if output is not None:
        write_result(output, value)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    return 0
