from __future__ import annotations

import os
import platform
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from ipaddress import ip_address
from typing import Any, Literal

from remote_loader.auth import TokenStore
from remote_loader.client_config import ClientConfig
from remote_loader.control import ControlClient
from remote_loader.scheduler import SchedulerClient

DiagnosticStatus = Literal["ok", "warning", "error", "skipped"]


@dataclass(frozen=True)
class DiagnosticCheck:
    name: str
    status: DiagnosticStatus
    detail: object


@dataclass(frozen=True)
class DiagnosticReport:
    checks: tuple[DiagnosticCheck, ...]

    @property
    def exit_code(self) -> int:
        return 1 if any(check.status == "error" for check in self.checks) else 0

    def to_dict(self) -> dict[str, dict[str, object]]:
        return {
            check.name: {
                "status": check.status,
                "detail": check.detail,
            }
            for check in self.checks
        }


def collect_environment_diagnostics(
    *,
    torch_version_loader: Callable[[], str] | None = None,
) -> DiagnosticReport:
    torch_version_loader = torch_version_loader or _torch_version
    checks = [
        DiagnosticCheck("python", "ok", platform.python_version()),
        DiagnosticCheck("platform", "ok", platform.platform()),
    ]

    try:
        checks.append(DiagnosticCheck("torch", "ok", torch_version_loader()))
    except ImportError:
        checks.append(
            DiagnosticCheck(
                "torch",
                "warning",
                "missing (install PyTorch in your training environment)",
            )
        )
    except Exception as exc:
        checks.append(DiagnosticCheck("torch", "error", f"unavailable: {exc}"))
    return DiagnosticReport(tuple(checks))


def collect_connection_diagnostics(
    *,
    token_store: Any | None = None,
    config_loader: Callable[[], ClientConfig] | None = None,
    control_client_factory: Callable[..., Any] | None = None,
    scheduler_client_factory: Callable[..., Any] | None = None,
    local_rank: int | None = None,
) -> DiagnosticReport:
    token_store = token_store or TokenStore()
    config_loader = config_loader or ClientConfig.load
    control_client_factory = control_client_factory or ControlClient
    scheduler_client_factory = scheduler_client_factory or SchedulerClient
    checks: list[DiagnosticCheck] = []

    try:
        token = token_store.load()
        checks.append(DiagnosticCheck("credentials", "ok", "available"))
    except Exception as exc:
        token = None
        checks.append(DiagnosticCheck("credentials", "error", f"unavailable: {exc}"))

    try:
        config = config_loader()
        checks.append(DiagnosticCheck("service_configuration", "ok", "available"))
    except Exception as exc:
        config = None
        checks.append(
            DiagnosticCheck(
                "service_configuration",
                "error",
                f"unavailable: {exc}",
            )
        )

    if config is None:
        checks.append(
            DiagnosticCheck(
                "control_service",
                "skipped",
                "skipped: credentials or service configuration unavailable",
            )
        )
        return DiagnosticReport(tuple(checks))

    aggregate = config.scheduler is not None and local_rank is None
    if aggregate and token is None:
        checks.append(DiagnosticCheck("node_connections", "skipped", "credentials unavailable"))
        return DiagnosticReport(tuple(checks))

    endpoints: list[tuple[str, str]] = []
    assignment_complete = True
    if config.scheduler is None:
        endpoints.append(("control_service", config.control_endpoint))
    elif aggregate:
        assert token is not None
        assignment, assignment_complete, error = _training_assignment(
            config,
            scheduler_client_factory=scheduler_client_factory,
            token=token,
        )
        if error is not None:
            checks.append(DiagnosticCheck("scheduler_assignment", "error", error))
            checks.append(DiagnosticCheck("node_connections", "skipped", "assignment unavailable"))
            return DiagnosticReport(tuple(checks))
        assert assignment is not None
        detail = (
            "all training nodes assigned and ready"
            if assignment_complete
            else ("no nodes assigned" if not assignment else "training assignment is incomplete or not ready")
        )
        checks.append(
            DiagnosticCheck("scheduler_assignment", "ok" if assignment_complete else "error", detail)
        )
        endpoints.extend(
            (
                "node_connections",
                _node_control_endpoint(
                    node,
                    scheme=config.scheduler.control_scheme,
                    port=config.scheduler.control_port,
                ),
            )
            for node in assignment
        )
    else:
        for vqa in (False, True) if config.training.vqa else (False,):
            check_name = "scheduler_vqa_assignment" if vqa else "scheduler_assignment"
            assignment, assignment_error = _scheduler_assignment(
                config,
                scheduler_client_factory=scheduler_client_factory,
                local_rank=local_rank,
                vqa=vqa,
                token=token,
            )
            if assignment_error is not None:
                checks.append(DiagnosticCheck(check_name, "error", assignment_error))
                continue
            assert assignment is not None
            checks.append(DiagnosticCheck(check_name, "ok", list(assignment)))
            prefix = "vqa_node_control" if vqa else "node_control"
            endpoints.extend(
                (
                    f"{prefix}:{node}",
                    _node_control_endpoint(
                        node,
                        scheme=config.scheduler.control_scheme,
                        port=config.scheduler.control_port,
                    ),
                )
                for node in assignment
            )

    if token is None:
        for name, _endpoint in endpoints:
            checks.append(
                DiagnosticCheck(
                    name,
                    "skipped",
                    "skipped: credentials unavailable",
                )
            )
        return DiagnosticReport(tuple(checks))

    try:
        ssl_context = config.create_tls_context()
    except Exception as exc:
        checks.append(DiagnosticCheck("client_mtls", "error", f"unavailable: {exc}"))
        return DiagnosticReport(tuple(checks))

    connected = 0
    for name, endpoint in endpoints:
        client = None
        control_status: DiagnosticStatus = "ok"
        control_detail: object
        try:
            client = control_client_factory(
                endpoint,
                token,
                ssl_context=ssl_context,
            )
            control_detail = client.health()
        except Exception as exc:
            control_status = "error"
            control_detail = f"unavailable: {exc}"
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception as exc:
                    if control_status == "ok":
                        control_status = "error"
                        control_detail = f"unavailable: failed to close control client: {exc}"

        if control_status == "ok":
            connected += 1
        if not aggregate:
            checks.append(DiagnosticCheck(name, control_status, control_detail))
    if aggregate:
        checks.append(
            DiagnosticCheck(
                "node_connections",
                "ok" if assignment_complete and connected == len(endpoints) else "error",
                {"total": len(endpoints), "connected": connected},
            )
        )
    return DiagnosticReport(tuple(checks))


def collect_scheduler_diagnostics(
    *,
    config_loader: Callable[[], ClientConfig] | None = None,
    scheduler_client_factory: Callable[..., Any] | None = None,
) -> DiagnosticReport:
    config_loader = config_loader or ClientConfig.load
    scheduler_client_factory = scheduler_client_factory or SchedulerClient
    try:
        config = config_loader()
    except Exception as exc:
        return DiagnosticReport(
            (
                DiagnosticCheck(
                    "scheduler_configuration",
                    "error",
                    f"unavailable: {exc}",
                ),
                DiagnosticCheck(
                    "scheduler_assignment",
                    "skipped",
                    "skipped: scheduler configuration unavailable",
                ),
            )
        )

    if config.scheduler is None:
        return DiagnosticReport(
            (
                DiagnosticCheck(
                    "scheduler_configuration",
                    "error",
                    "scheduler mode is not configured",
                ),
                DiagnosticCheck(
                    "scheduler_assignment",
                    "skipped",
                    "skipped: scheduler mode is not configured",
                ),
            )
        )

    checks = [DiagnosticCheck("scheduler_configuration", "ok", "available")]
    for vqa in (False, True) if config.training.vqa else (False,):
        check_name = "scheduler_vqa_assignment" if vqa else "scheduler_assignment"
        assignment, error = _scheduler_assignment(
            config,
            scheduler_client_factory=scheduler_client_factory,
            vqa=vqa,
        )
        if error is not None:
            checks.append(DiagnosticCheck(check_name, "error", error))
        else:
            checks.append(DiagnosticCheck(check_name, "ok", list(assignment or ())))
    return DiagnosticReport(tuple(checks))


def _training_assignment(
    config: ClientConfig,
    *,
    scheduler_client_factory: Callable[..., Any],
    token: str,
) -> tuple[tuple[str, ...] | None, bool, str | None]:
    """Read the whole training without discovering or allocating a local rank."""
    assert config.scheduler is not None
    client = None
    try:
        client = scheduler_client_factory(
            config.scheduler.endpoint,
            token=token,
            verify=config.scheduler.verify_tls,
        )
        resources = client.training_status(config.training.training_id)
        gpus = resources.get("gpus")
        if not isinstance(gpus, list):
            raise ValueError("scheduler status must contain a GPU list")
        nodes: dict[str, None] = {}
        gpu_ids: set[str] = set()
        complete = True
        for gpu in gpus:
            if not isinstance(gpu, dict) or not isinstance(gpu.get("training_id"), str):
                raise ValueError("scheduler status contains an invalid GPU record")
            if gpu["training_id"] != config.training.training_id:
                continue
            identity = gpu.get("gpu_id")
            if not isinstance(identity, str) or not identity:
                raise ValueError("scheduler status contains an unidentified GPU")
            if identity in gpu_ids:
                complete = False
            gpu_ids.add(identity)
            for group in ("cpu_nodes", "vqa_nodes") if config.training.vqa else ("cpu_nodes",):
                assigned = gpu.get(group)
                if not isinstance(assigned, list):
                    raise ValueError("scheduler status contains an invalid node list")
                group_nodes = set()
                if not assigned:
                    complete = False
                for node in assigned:
                    if not isinstance(node, dict):
                        raise ValueError("scheduler status contains an invalid node")
                    address = node.get("ip")
                    if not isinstance(address, str) or "%" in address:
                        raise ValueError("scheduler status contains an invalid node IP")
                    try:
                        address = str(ip_address(address))
                    except ValueError:
                        raise ValueError("scheduler status contains an invalid node IP") from None
                    group_nodes.add(address)
                    nodes[address] = None
                    if node.get("state") != "ready" or node.get("error"):
                        complete = False
                if group == "cpu_nodes" and config.training.cpu_per_gpu is not None:
                    complete = complete and len(group_nodes) >= config.training.cpu_per_gpu
        training = config.training
        if training.num_nodes is not None and training.gpus_per_node is not None:
            complete = complete and len(gpu_ids) == training.num_nodes * training.gpus_per_node
        return tuple(nodes), complete and bool(gpu_ids) and bool(nodes), None
    except Exception as exc:
        return None, False, f"unavailable: {exc}"
    finally:
        if client is not None:
            with suppress(Exception):
                client.close()


def _scheduler_assignment(
    config: ClientConfig,
    *,
    scheduler_client_factory: Callable[..., Any],
    local_rank: int | None = None,
    vqa: bool = False,
    token: str | None = None,
) -> tuple[tuple[str, ...] | None, str | None]:
    scheduler_config = config.scheduler
    if scheduler_config is None:
        return None, "unavailable: scheduler mode is not configured"
    client = None
    try:
        client = scheduler_client_factory(
            scheduler_config.endpoint,
            **({"token": token} if token is not None else {}),
            verify=scheduler_config.verify_tls,
        )
        return client.get_nodes(
            config.training.training_id,
            int(os.environ.get("LOCAL_RANK", "0")) if local_rank is None else local_rank,
            **({"vqa": True} if vqa else {}),
        ), None
    except Exception as exc:
        return None, f"unavailable: {exc}"
    finally:
        if client is not None:
            with suppress(Exception):
                client.close()


def _node_control_endpoint(node: str, *, scheme: str, port: int) -> str:
    host = f"[{node}]" if ":" in node else node
    return f"{scheme}://{host}:{port}"


def _torch_version() -> str:
    import torch

    return str(torch.__version__)
