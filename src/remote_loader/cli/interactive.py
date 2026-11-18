from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any

import questionary
from questionary import Choice
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from remote_loader.auth import TokenStore
from remote_loader.batch_spec import BatchSpec, CanonicalSampleSpec, VQABatchSpec
from remote_loader.beanfleet import (
    BeanfleetClient,
    BeanfleetPeering,
    BeanfleetPeeringStatus,
)
from remote_loader.cli.diagnostics import (
    DiagnosticCheck,
    DiagnosticReport,
    collect_connection_diagnostics,
    collect_environment_diagnostics,
)
from remote_loader.client_config import (
    ClientConfig,
    ConfigStore,
    TrainingConfig,
    TrainingDataset,
)
from remote_loader.cloud_detection import CloudEnvironment, detect_cloud_environment
from remote_loader.cloud_network import (
    ALIYUN_CLI_INSTALL_URL,
    ALIYUN_PLUGIN_INSTALL,
    AliyunNetworkClient,
    ConnectionInspection,
    LocalNetwork,
    PeerConnection,
    RoutePlan,
)
from remote_loader.config_migration import is_schema6_config
from remote_loader.control import ControlClient
from remote_loader.exceptions import ConfigurationError
from remote_loader.peering_workflow import (
    PeeringCompletionPlan,
    PeeringCreationPlan,
    PeeringWorkflow,
    PeeringWorkflowResult,
)
from remote_loader.resource_catalog import PROVIDERS, find_provider, find_region
from remote_loader.scheduler import SchedulerClient
from remote_loader.transfer_state import TransferSnapshot, TransferStateStore, utc_now


class PromptCancelled(Exception):
    pass


class QuestionaryPrompts:
    def select(
        self,
        message: str,
        choices: list[tuple[str, str]],
        *,
        default: str | None = None,
    ) -> str | None:
        return questionary.select(
            message,
            choices=[Choice(title=title, value=value) for title, value in choices],
            default=default,
        ).ask()

    def text(self, message: str, *, default: str = "") -> str | None:
        return questionary.text(message, default=default).ask()

    def password(self, message: str) -> str | None:
        return questionary.password(message).ask()

    def confirm(self, message: str, *, default: bool = False) -> bool | None:
        return questionary.confirm(message, default=default).ask()


class InteractiveClient:
    def __init__(
        self,
        *,
        config_store: ConfigStore | None = None,
        token_store: TokenStore | None = None,
        transfer_state_store: TransferStateStore | None = None,
        console: Console | None = None,
        prompts: Any | None = None,
        control_client_factory: Callable[..., Any] = ControlClient,
        network_client_factory: Callable[..., Any] = AliyunNetworkClient,
        beanfleet_client_factory: Callable[..., Any] = BeanfleetClient,
        peering_workflow_factory: Callable[..., Any] = PeeringWorkflow,
        cloud_detector: Callable[[], CloudEnvironment] = detect_cloud_environment,
    ) -> None:
        self.config_store = config_store or ConfigStore()
        self.token_store = token_store or TokenStore()
        self.transfer_state_store = transfer_state_store or TransferStateStore()
        self.console = console or Console()
        self.prompts = prompts or QuestionaryPrompts()
        self.control_client_factory = control_client_factory
        self.network_client_factory = network_client_factory
        self.beanfleet_client_factory = beanfleet_client_factory
        self.peering_workflow_factory = peering_workflow_factory
        self.cloud_detector = cloud_detector

    def run(self) -> int:
        self.console.print(Panel.fit("[bold]Modalink Remote Loader[/bold]", border_style="cyan"))
        try:
            self._ensure_connection_config()
            self._run_menu(
                "请选择操作",
                [
                    ("保存 Token", "token-save", self.save_token),
                    ("启动训练", "training", self.start_training),
                    ("诊断", "diagnostics", self.show_diagnostics),
                    ("查看", "view", self.show_views),
                ],
                exit_label="退出",
                exit_key="exit",
            )
            return 0
        except ConfigurationError as exc:
            self.console.print(Text.assemble(("ERROR: ", "bold red"), str(exc)))
            return 2
        except KeyboardInterrupt:
            self.console.print("\n[yellow]已取消。[/yellow]")
            return 130

    def _ensure_connection_config(self) -> ClientConfig:
        try:
            current = self.config_store.load_optional()
        except ConfigurationError as exc:
            if is_schema6_config(self.config_store):
                raise ConfigurationError(
                    "现有 schema 6 配置需要先运行 remote-loader config migrate；原文件未修改。"
                ) from exc
            raise
        if current is not None:
            return current
        detected = self.cloud_detector()
        config = self.config_store.load_or_detect(detector=lambda: detected, persist=True)
        self.console.print(f"[green]已识别{detected.label}；Scheduler：{detected.scheduler_endpoint}[/green]")
        self.console.print(
            "云环境自动识别不会提供 mTLS 凭据；请运行 remote-loader config edit，"
            "通过 --tls-ca-file、--tls-client-cert-file 和 --tls-client-key-file 配置您自己的凭据。"
        )
        return config

    def _run_menu(
        self,
        title: str,
        actions: list[tuple[str, str, Callable[[], None]]],
        *,
        exit_label: str = "返回",
        exit_key: str = "back",
    ) -> None:
        handlers = {key: handler for _label, key, handler in actions}
        while True:
            action = self.prompts.select(
                title,
                [(label, key) for label, key, _handler in actions] + [(exit_label, exit_key)],
            )
            if action in {None, exit_key}:
                return
            try:
                handlers[action]()
            except PromptCancelled:
                self.console.print("[yellow]已取消当前操作。[/yellow]")
            except Exception as exc:
                self.console.print(Text.assemble(("ERROR: ", "bold red"), str(exc)))

    def show_diagnostics(self) -> None:
        self._run_menu(
            "诊断",
            [
                ("环境诊断", "environment", self.show_environment_diagnostics),
                ("连接诊断", "connection", self.show_connection_diagnostics),
            ],
        )

    def show_views(self) -> None:
        self._run_menu(
            "查看",
            [
                ("查看训练设置", "training-settings", self.show_training_config),
                ("查看连接配置", "show-config", self.show_config),
                ("查看当前传输", "current-transfer", self.show_current_transfer),
                ("查看最近传输结果", "last-transfer", self.show_last_transfer),
            ],
        )

    def start_training(self) -> None:
        from remote_loader.cli.training import TrainingCommands

        config = self.config_store.load()
        scheduler = config.scheduler
        if scheduler is None:
            raise ConfigurationError("未生成 Scheduler 配置；请检查云厂商自动识别或显式 CLI 配置。")
        training = replace(
            config.training,
            num_nodes=config.training.num_nodes or 1,
            gpus_per_node=config.training.gpus_per_node or 1,
        )
        field_labels = {"num_nodes": "节点总数", "gpus_per_node": "节点GPU数"}
        self.console.print(Text(f"Scheduler：{scheduler.endpoint}"))
        self.console.print("[dim]修改后自动保存；version 默认 2，weight 和 cpu_per_gpu 可留空。[/dim]")

        def save_training(updated: TrainingConfig) -> None:
            nonlocal config, training
            saved = replace(config, training=updated)
            self.config_store.save(saved)
            self.console.print(Text(f"配置已保存到 {self.config_store.path}。", style="green"))
            config, training = saved, updated

        while True:
            choices = []
            for key in (
                "training_id",
                "num_nodes",
                "gpus_per_node",
                "datasets",
                "cpu_per_gpu",
                "vqa",
                "vqa_datasets",
            ):
                if key == "vqa_datasets" and not training.vqa:
                    continue
                value = getattr(training, key)
                if key == "vqa":
                    rendered = "Y" if value else "N"
                elif key in {"datasets", "vqa_datasets"}:
                    rendered = f"{len(value)} 个数据集（进入管理）"
                elif key == "cpu_per_gpu" and value is None:
                    rendered = "调度器默认值"
                else:
                    rendered = str(value) if value is not None else "未填写"
                choices.append((f"{field_labels.get(key, key):<18}{rendered}", key))
            action = self.prompts.select(
                "启动训练：请选择操作",
                [
                    *choices,
                    ("开启服务", "submit"),
                    ("关闭服务（释放全部 dataloader 资源）", "end"),
                    ("返回主菜单", "back"),
                ],
            )
            if action in {None, "back"}:
                return
            if action == "end":
                self.end_training()
                continue
            if action != "submit":
                try:
                    if action == "vqa":
                        enabled = self._required(self.prompts.confirm("vqa", default=training.vqa))
                        save_training(replace(training, vqa=enabled))
                        if not enabled:
                            continue
                        action = "vqa_datasets"
                    if action == "training_id":
                        value = self._required(
                            self.prompts.text("training_id", default=training.training_id or "")
                        ).strip()
                        if not value:
                            raise ConfigurationError("training_id 不能为空。")
                        save_training(replace(training, training_id=value))
                    elif action in {"num_nodes", "gpus_per_node", "cpu_per_gpu"}:
                        previous = getattr(training, action)
                        value = self._required(
                            self.prompts.text(
                                field_labels.get(action, action),
                                default="" if previous is None else str(previous),
                            )
                        ).strip()
                        count = (
                            None
                            if action == "cpu_per_gpu" and not value
                            else self._positive_int(value, field_labels.get(action, action))
                        )
                        save_training(replace(training, **{action: count}))
                    elif action in {"datasets", "vqa_datasets"}:
                        self._edit_training_datasets(
                            getattr(training, action),
                            title=action,
                            on_change=lambda entries, field=action: save_training(
                                replace(training, **{field: entries})
                            ),
                        )
                except PromptCancelled:
                    self.console.print("[yellow]已取消本项修改。[/yellow]")
                except ConfigurationError as exc:
                    self.console.print(Text.assemble(("ERROR: ", "bold red"), str(exc)))
                continue
            try:
                request = training.to_request()
            except ConfigurationError as exc:
                self.console.print(Text.assemble(("ERROR: ", "bold red"), str(exc)))
                continue
            self.console.print(
                Panel(
                    Text(json.dumps(request, ensure_ascii=False, indent=2)), title="dataloader 服务启动申请"
                )
            )
            if not self._required(self.prompts.confirm("确认开启 dataloader 服务", default=False)):
                self.console.print("[yellow]已取消开启服务，设置已保留。[/yellow]")
                continue
            self.console.print("[yellow]正在启动 dataloader 服务[/yellow]")
            commands = TrainingCommands(
                config_store=self.config_store,
                token_store=self.token_store,
                client_factory=SchedulerClient,
            )
            commands.register(
                training, scheduler_endpoint=scheduler.endpoint, verify_scheduler_tls=scheduler.verify_tls
            )
            self._wait_for_training(replace(config, training=training))

    def end_training(self) -> None:
        from remote_loader.cli.training import TrainingCommands

        config = self.config_store.load()
        scheduler = config.scheduler
        if scheduler is None:
            raise ConfigurationError("未生成 Scheduler 配置；请检查云厂商自动识别或显式 CLI 配置。")
        training_id = (config.training.training_id or "").strip()
        if not training_id:
            raise ConfigurationError("请先填写要关闭服务的 training_id。")
        self.console.print(
            Panel(
                Text(f"Training ID：{training_id}\nScheduler：{scheduler.endpoint}"),
                title="关闭 dataloader 服务",
            )
        )
        self.console.print("将释放该训练所有 rank 的 dataloader 资源。请先停止数据读取并关闭 loader。")
        if not self._required(self.prompts.confirm("确认关闭服务并释放全部 dataloader 资源", default=False)):
            self.console.print("[yellow]已取消关闭服务。[/yellow]")
            return
        commands = TrainingCommands(
            config_store=self.config_store,
            token_store=self.token_store,
            client_factory=SchedulerClient,
        )
        result = commands.end(
            training_id=training_id,
            scheduler_endpoint=scheduler.endpoint,
            verify_scheduler_tls=scheduler.verify_tls,
        )
        self.console.print(
            Panel(Text(json.dumps(result, ensure_ascii=False, indent=2)), title="服务关闭结果")
        )
        self.console.print("[dim]本地设置已保留，可继续修改或再次启动服务。[/dim]")

    def _edit_training_datasets(
        self,
        entries: tuple[TrainingDataset, ...],
        *,
        title: str,
        on_change: Callable[[tuple[TrainingDataset, ...]], None],
    ) -> tuple[TrainingDataset, ...]:
        draft = list(entries)
        add_first = not draft
        while True:
            if add_first:
                action = "add"
                add_first = False
            else:
                self._render_training_datasets(tuple(draft), title=title)
                choices = [
                    (
                        f"{index + 1}. {entry.id}  "
                        f"weight={entry.weight if entry.weight is not None else '默认'}  "
                        f"version={entry.version if entry.version is not None else '待填写'}",
                        f"edit:{index}",
                    )
                    for index, entry in enumerate(draft)
                ]
                action = self.prompts.select(
                    f"{title}：管理数据集",
                    [*choices, ("添加数据集", "add"), ("返回训练设置", "back")],
                )
            if action in {None, "back"}:
                return tuple(draft)
            try:
                if action == "add":
                    dataset_id = self._prompt_training_dataset_field("id")
                    weight = self._prompt_training_dataset_field("weight")
                    entry = TrainingDataset(id=dataset_id, weight=weight)
                    version = self._prompt_training_dataset_field("version", default="2")
                    updated = (*draft, replace(entry, version=version))
                    on_change(updated)
                    draft = list(updated)
                elif action.startswith("edit:"):
                    index = int(action.split(":", 1)[1])

                    def save_entry(
                        entry: TrainingDataset | None,
                        *,
                        index: int = index,
                        draft: list[TrainingDataset] = draft,
                    ) -> None:
                        updated = list(draft)
                        if entry is None:
                            del updated[index]
                        else:
                            updated[index] = entry
                        on_change(tuple(updated))
                        draft[:] = updated

                    self._edit_training_dataset(draft[index], on_change=save_entry)
            except PromptCancelled:
                self.console.print("[yellow]已取消本项修改。[/yellow]")
            except ConfigurationError as exc:
                self.console.print(Text.assemble(("ERROR: ", "bold red"), str(exc)))

    def _edit_training_dataset(
        self,
        entry: TrainingDataset,
        *,
        on_change: Callable[[TrainingDataset | None], None],
    ) -> TrainingDataset | None:
        while True:
            self._render_training_datasets((entry,), title="数据集详情")
            action = self.prompts.select(
                "数据集：选择要修改的字段",
                [
                    (f"id: {entry.id}", "id"),
                    (f"weight: {entry.weight if entry.weight is not None else '默认'}", "weight"),
                    (f"version: {entry.version if entry.version is not None else '待填写'}", "version"),
                    ("删除数据集", "delete"),
                    ("返回数据集列表", "back"),
                ],
            )
            if action in {None, "back"}:
                return entry
            try:
                if action == "delete":
                    if self._required(self.prompts.confirm("确认删除此数据集？", default=False)):
                        on_change(None)
                        return None
                elif action in {"id", "weight", "version"}:
                    previous = getattr(entry, action)
                    value = self._prompt_training_dataset_field(
                        action,
                        default=str(previous)
                        if previous is not None
                        else ("2" if action == "version" else ""),
                    )
                    updated = replace(entry, **{action: value})
                    on_change(updated)
                    entry = updated
            except PromptCancelled:
                self.console.print("[yellow]已取消本项修改。[/yellow]")
            except ConfigurationError as exc:
                self.console.print(Text.assemble(("ERROR: ", "bold red"), str(exc)))

    def _prompt_training_dataset_field(self, field: str, *, default: str = "") -> str | float | int | None:
        value = self._required(self.prompts.text(field, default=default)).strip()
        if field == "id":
            if not value:
                raise ConfigurationError("id 不能为空。")
            return value
        if field == "weight" and not value:
            return None
        try:
            return float(value) if field == "weight" else self._positive_int(value or "2", "version")
        except ValueError as exc:
            message = "weight 必须是数字。" if field == "weight" else "version 必须是整数。"
            raise ConfigurationError(message) from exc

    def _render_training_datasets(self, entries: tuple[TrainingDataset, ...], *, title: str) -> None:
        table = Table(title=title)
        table.add_column("#")
        table.add_column("id", overflow="fold")
        table.add_column("weight")
        table.add_column("version")
        for index, entry in enumerate(entries):
            table.add_row(
                str(index + 1),
                Text(entry.id),
                str(entry.weight) if entry.weight is not None else "默认",
                str(entry.version) if entry.version is not None else "待填写",
            )
        self.console.print(table)

    def _wait_for_training(self, config: ClientConfig) -> None:
        assert config.training.gpus_per_node is not None
        last_report = None
        self.console.print("仅检查本机各 GPU 的节点；其他训练机器由各自的进程或 TUI 独立发现和检查节点。")
        try:
            with self.console.status("正在等待本机 dataloader 节点就绪（Ctrl+C 可停止等待）"):
                while True:
                    for local_rank in range(config.training.gpus_per_node):
                        report = collect_connection_diagnostics(
                            token_store=self.token_store,
                            config_loader=lambda: config,
                            control_client_factory=self.control_client_factory,
                            local_rank=local_rank,
                        )
                        assigned_nodes = next(
                            (
                                check.detail
                                for check in report.checks
                                if check.name == "scheduler_assignment" and check.status == "ok"
                            ),
                            None,
                        )
                        if (
                            isinstance(assigned_nodes, list)
                            and config.training.cpu_per_gpu is not None
                            and len(assigned_nodes) < config.training.cpu_per_gpu
                        ):
                            # A healthy first node does not mean allocation is complete.
                            report = DiagnosticReport(
                                (
                                    *report.checks,
                                    DiagnosticCheck(
                                        "cpu_assignment_count",
                                        "error",
                                        {
                                            "local_rank": local_rank,
                                            "assigned": len(assigned_nodes),
                                            "requested": config.training.cpu_per_gpu,
                                        },
                                    ),
                                )
                            )
                        if not report.checks or any(check.status != "ok" for check in report.checks):
                            last_report = report
                            if any(
                                check.name in {"credentials", "service_configuration"}
                                and check.status == "error"
                                for check in report.checks
                            ):
                                self._render_diagnostics(report, title="本机连接诊断")
                                self.console.print(
                                    "[yellow]申请已提交；请修复凭据或配置后重新检查连接。[/yellow]"
                                )
                                return
                            break
                    else:
                        break
                    time.sleep(2)
        except KeyboardInterrupt:
            self.console.print(
                "[yellow]已停止等待本机节点；服务申请已提交，远端继续准备 dataloader 节点。[/yellow]"
            )
            if last_report is not None:
                self._render_diagnostics(last_report, title="最近一次本机连接诊断")
            return
        self.console.print("[green]本机所有 GPU 的 dataloader 节点已就绪；不代表其他训练机器已就绪。[/green]")

    @staticmethod
    def _connection_settings(config: ClientConfig) -> list[tuple[str, str, str, str]]:
        scheduler = config.scheduler
        return [
            (
                "Scheduler",
                "Scheduler 地址",
                "未配置" if scheduler is None else scheduler.endpoint,
                "scheduler-endpoint",
            )
        ]

    def save_token(self) -> None:
        self.console.print(f"Token 保存位置：{self.token_store.path}")
        if self.token_store.path.exists():
            overwrite = self._required(
                self.prompts.confirm(
                    "已有 Token，是否覆盖？",
                    default=False,
                )
            )
            if not overwrite:
                raise PromptCancelled
        token = self._required(self.prompts.password("Token")).strip()
        if not token:
            raise ConfigurationError("Token must not be empty")
        self.token_store.save(token)
        self.console.print("[green]Token 已保存。[/green]")

    def manage_private_network(self) -> None:
        """Retained for future restoration; not exposed by the public menus."""
        self.console.print(
            Panel.fit(
                "[bold]私网连接管理[/bold]",
                border_style="cyan",
            )
        )
        try:
            current = self.config_store.load()
        except ConfigurationError:
            current = ClientConfig.default()
        placement = current.placement
        current_provider = placement.provider if placement is not None else None
        provider_id = self._required(
            self.prompts.select(
                "云厂商",
                [(f"{provider.label} ({provider.id})", provider.id) for provider in PROVIDERS],
                default=current_provider,
            )
        )
        if provider_id != "aliyun":
            raise ConfigurationError(f"private-network management does not support {provider_id!r}")
        provider = find_provider(provider_id)
        if provider is None:
            raise ConfigurationError(f"unknown service provider {provider_id!r}")

        current_region = (
            placement.region if placement is not None and placement.provider == provider_id else None
        )
        region_id = self._required(
            self.prompts.select(
                "地域",
                [(f"{region.id} - {region.label}", region.id) for region in provider.regions],
                default=current_region,
            )
        )
        if find_region(provider, region_id) is None:
            raise ConfigurationError(f"unknown region {region_id!r}")
        profile = self._required(self.prompts.text("阿里云 CLI Profile", default="gpu-peer")).strip()
        if not profile:
            raise ConfigurationError("Alibaba Cloud CLI profile must not be empty")

        self.console.print(
            Panel(
                "\n".join(
                    (
                        "请先在当前机器的另一个 shell 中完成身份配置；本客户端不会接收云凭据。",
                        f"CLI 安装文档：{ALIYUN_CLI_INSTALL_URL}",
                        f"安装所需插件：{ALIYUN_PLUGIN_INSTALL}",
                        f"检查身份： aliyun --profile {profile} --region {region_id} sts get-caller-identity",
                    )
                ),
                title="外部身份认证",
            )
        )
        client = self.network_client_factory(profile=profile, region=region_id)
        version = client.check_environment()
        local_network = self._select_local_network(client.local_network_candidates())
        self._render_local_network(
            local_network,
            provider=provider_id,
            region=region_id,
            profile=profile,
            cli_version=version,
        )

        if current.beanfleet_endpoint is not None:
            mode = self._required(
                self.prompts.select(
                    "私网连接方式",
                    [
                        ("创建新的自动连接", "create-auto"),
                        ("从已有连接 ID 继续自动流程", "resume-auto"),
                        ("使用手工排障流程", "manual"),
                        ("退出私网连接管理", "back"),
                    ],
                )
            )
            if mode == "back":
                return
            if mode in {"create-auto", "resume-auto"}:
                connection_id: str | None = None
                if mode == "resume-auto":
                    connection_id = self._required(self.prompts.text("已有对等连接 ID")).strip()
                    if not connection_id:
                        raise ConfigurationError("peering connection ID must not be empty")
                self._automate_private_connection(
                    client,
                    local_network,
                    config=current,
                    profile=profile,
                    region_id=region_id,
                    connection_id=connection_id,
                )
                return
        else:
            self.console.print(
                "[yellow]尚未配置 Beanfleet 地址；将进入手工排障流程。"
                "可先使用 remote-loader config edit 设置地址。[/yellow]"
            )

        raw_connection_id = self._required(self.prompts.text("已有对等连接 ID（仅用于手工排障；留空退出）"))
        connection_id = raw_connection_id.strip()
        if not connection_id:
            return

        inspection = client.inspect_connection(
            connection_id,
            expected_vpc_id=local_network.vpc.vpc_id,
            local_ip=local_network.local_address.address,
        )
        self._render_connection_inspection(inspection, title="接收前预览和检查")
        route_context: tuple[str, str, str] | None = None
        while True:
            choices = [("刷新连接状态", "status")]
            if inspection.connection.status == "Accepting":
                choices.append(("接收连接", "accept"))
            if route_context is not None and inspection.connection.status == "Activated":
                choices.append(("查看返回路由状态", "route-status"))
            if inspection.connection.status == "Activated":
                choices.append(("检查或配置返回路由", "route"))
            choices.append(("退出私网连接管理", "back"))

            action = self._required(
                self.prompts.select(
                    "私网连接操作",
                    choices,
                )
            )
            if action == "back":
                return
            try:
                if action == "status":
                    inspection = client.inspect_connection(
                        connection_id,
                        expected_vpc_id=local_network.vpc.vpc_id,
                        local_ip=local_network.local_address.address,
                    )
                    self._render_connection_status(inspection, title="连接状态")
                elif action == "accept":
                    inspection = self._accept_private_connection(
                        client,
                        connection_id,
                        local_network,
                    )
                    self._render_connection_status(inspection, title="接收连接状态")
                elif action == "route":
                    route_context = self._configure_private_route(
                        client,
                        connection_id,
                        local_network,
                        inspection.connection,
                    )
                elif action == "route-status" and route_context is not None:
                    destination, local_ip, route_table_id = route_context
                    plan = client.plan_route(
                        connection_id,
                        destination,
                        local_ip=local_ip,
                        route_table_id=route_table_id,
                        expected_vpc_id=local_network.vpc.vpc_id,
                    )
                    self._render_route_plan(plan, title="返回路由状态")
            except PromptCancelled:
                self.console.print("[yellow]已取消当前操作。[/yellow]")
            except Exception as exc:
                self.console.print(Text.assemble(("ERROR: ", "bold red"), str(exc)))

    def _automate_private_connection(
        self,
        network_client: Any,
        local_network: LocalNetwork,
        *,
        config: ClientConfig,
        profile: str,
        region_id: str,
        connection_id: str | None,
    ) -> None:
        endpoint = config.beanfleet_endpoint
        if endpoint is None:
            raise ConfigurationError(
                "beanfleet endpoint is not configured; update the client configuration first"
            )
        beanfleet_client = self.beanfleet_client_factory(
            endpoint,
            verify=config.verify_beanfleet_tls,
        )
        workflow = self.peering_workflow_factory(
            network_client=network_client,
            beanfleet_client=beanfleet_client,
            region_id=region_id,
            progress=self._render_peering_progress,
        )
        try:
            if connection_id is None:
                creation_plan = workflow.creation_plan(local_network)
                self._render_peering_creation_plan(creation_plan)
                confirmed = self._required(
                    self.prompts.confirm(
                        "确认向 Beanfleet 提交以上信息并创建对等连接？",
                        default=False,
                    )
                )
                if not confirmed:
                    raise PromptCancelled
                peering = workflow.create(creation_plan)
                connection_id = peering.peering_id
                self.console.print(
                    f"[green]Beanfleet 已创建连接 {connection_id}；请保存该 ID，"
                    "后续可用它检查状态或恢复。[/green]"
                )

            completion_plan = workflow.prepare_completion(local_network, connection_id)
            self._render_peering_completion_plan(completion_plan)
            if completion_plan.peering.status == BeanfleetPeeringStatus.ACTIVATED:
                self.console.print(
                    "[bold yellow]Beanfleet 当前仍为 Activated；/accepted 可能已经在执行，"
                    "重复通知尚未保证幂等。[/bold yellow]"
                )
            confirmed = self._required(
                self.prompts.confirm(
                    "确认接收连接、通知 Beanfleet，并确保以上全部本地路由存在？",
                    default=False,
                )
            )
            if not confirmed:
                self.console.print(
                    "[yellow]连接已保留，未执行本次后续操作。续跑命令：[/yellow]\n"
                    f"remote-loader network connect --profile {profile} "
                    f"--region {region_id} --vpc-id {local_network.vpc.vpc_id} "
                    f"--local-ip {local_network.local_address.address} "
                    f"--connection-id {connection_id}"
                )
                return

            result = workflow.complete(completion_plan)
            self._render_peering_result(result)
        except Exception:
            if connection_id is not None:
                self.console.print(
                    "[yellow]连接已保留；恢复前请先确认 Beanfleet 状态。"
                    "如果已调用 /accepted 但状态仍为 Activated，重复通知尚未保证幂等。"
                    "候选恢复命令：[/yellow]\n"
                    f"remote-loader network connect --profile {profile} "
                    f"--region {region_id} --vpc-id {local_network.vpc.vpc_id} "
                    f"--local-ip {local_network.local_address.address} "
                    f"--connection-id {connection_id}"
                )
            raise
        finally:
            close = getattr(beanfleet_client, "close", None)
            if close is not None:
                close()

    def _render_peering_creation_plan(self, plan: PeeringCreationPlan) -> None:
        payload = plan.to_dict()
        self.console.print(
            Panel(
                json.dumps(payload["request"], ensure_ascii=False, indent=2),
                title=f"Beanfleet 创建请求 · {payload['beanfleet_endpoint']}",
                border_style="yellow",
            )
        )
        self.console.print("[dim]不会上传阿里云凭据、ARN、实例 IP、vSwitch 或路由表信息。[/dim]")

    def _render_peering_completion_plan(self, plan: PeeringCompletionPlan) -> None:
        self._render_connection_inspection(plan.inspection, title="接收前完整检查")
        routes = Table(title="本地路由只读计划")
        routes.add_column("目标")
        routes.add_column("路由表")
        routes.add_column("现有状态")
        routes.add_column("动作")
        for route in plan.routes:
            routes.add_row(
                route.destination,
                route.route_table_id,
                route.existing_status or "不存在",
                "创建" if route.create_required else "复用",
            )
        self.console.print(routes)
        self.console.print(
            "[yellow]Beanfleet Ready 仅表示平台侧就绪；上述本地路由全部 Available 后流程才会成功。[/yellow]"
        )
        self.console.print(
            "[dim]本次确认授权确保这些固定目标存在；若预览为“复用”的路由随后被删除，"
            "client 可以重新创建。CIDR、VPC、路由表或 next hop 变化仍会中止。[/dim]"
        )

    def _render_peering_progress(self, phase: str, peering: BeanfleetPeering) -> None:
        self.console.print(f"[dim]{phase}: {peering.peering_id} -> {peering.status.value}[/dim]")

    def _render_peering_result(self, result: PeeringWorkflowResult) -> None:
        payload = result.to_dict()
        table = Table(title="自动私网连接结果")
        table.add_column("项目")
        table.add_column("值")
        table.add_row("连接 ID", result.peering.peering_id)
        table.add_row("Beanfleet", result.peering.status.value)
        table.add_row("阿里云连接", result.connection.status)
        for route in result.routes:
            table.add_row(f"本地路由 {route.destination}", route.existing_status or "—")
        self.console.print(table)
        if payload["ready"]:
            self.console.print("[green]平台和本地路由均已就绪。[/green]")
        else:
            raise ConfigurationError("automatic peering workflow finished without all readiness conditions")

    def _accept_private_connection(
        self,
        client: Any,
        connection_id: str,
        local_network: LocalNetwork,
    ) -> ConnectionInspection:
        inspection = client.inspect_connection(
            connection_id,
            expected_vpc_id=local_network.vpc.vpc_id,
            local_ip=local_network.local_address.address,
        )
        inspection.require_actionable()
        if inspection.connection.status in {"Activated", "Updating"}:
            return inspection

        confirmed = self._required(
            self.prompts.confirm(
                f"确认接收连接 {inspection.connection.connection_id}？",
                default=False,
            )
        )
        if not confirmed:
            raise PromptCancelled
        client.accept_connection(
            connection_id,
            expected_vpc_id=local_network.vpc.vpc_id,
            local_ip=local_network.local_address.address,
        )
        return client.inspect_connection(
            connection_id,
            expected_vpc_id=local_network.vpc.vpc_id,
            local_ip=local_network.local_address.address,
        )

    def _configure_private_route(
        self,
        client: Any,
        connection_id: str,
        local_network: LocalNetwork,
        connection: PeerConnection,
    ) -> tuple[str, str, str]:
        local_ip = local_network.local_address.address
        route_table_id = local_network.vswitch.route_table_id
        if connection.status != "Activated":
            raise ConfigurationError(
                f"connection must be Activated before configuring routes; got {connection.status}"
            )

        self.console.print(
            "[yellow]对等连接 API 只提供 CPU 侧 VPC 网段，不提供具体实例私网 IP。"
            "自动流程不需要实例 IP；仅在手工 /32 高级排障时自行填写 CPU 私网 IP。"
            "保留默认值则为整个发起端 VPC 配置路由。[/yellow]"
        )
        default_destination = connection.requester.ipv4_cidrs[0] if connection.requester.ipv4_cidrs else ""
        destination = self._required(
            self.prompts.text(
                "CPU 侧路由目标（VPC 网段，或自行确认的实例私网 IP）",
                default=default_destination,
            )
        ).strip()
        if not destination:
            raise ConfigurationError("route destination must not be empty")

        plan = client.plan_route(
            connection_id,
            destination,
            local_ip=local_ip,
            route_table_id=route_table_id,
            expected_vpc_id=local_network.vpc.vpc_id,
        )
        self._render_route_plan(plan, title="返回路由检查")
        if not plan.create_required:
            self.console.print("[green]目标路由已经指向该对等连接，无需修改。[/green]")
            return plan.destination, local_ip, route_table_id

        confirmed = self._required(
            self.prompts.confirm(
                f"确认在路由表 {plan.route_table_id} 中创建该路由？",
                default=False,
            )
        )
        if not confirmed:
            raise PromptCancelled
        result = client.apply_route(plan)
        self.console.print(
            "[green]路由创建请求已提交；"
            f"条目 ID：{result.existing_route_id or '尚未返回'}，"
            f"状态：{result.existing_status or 'Submitted'}。"
            "后续可从菜单单独刷新路由状态。[/green]"
        )
        return result.destination, local_ip, route_table_id

    def _select_local_network(
        self,
        candidates: tuple[LocalNetwork, ...],
    ) -> LocalNetwork:
        if not candidates:
            raise ConfigurationError(
                "未能根据本机 IPv4 地址匹配当前地域中可用的交换机；请检查地域、Profile 和实例网络。"
            )
        if len(candidates) == 1:
            network = candidates[0]
            self.console.print(
                "[green]已根据本机 IP "
                f"{network.local_address.address} 匹配 VPC {network.vpc.vpc_id}、"
                f"交换机 {network.vswitch.vswitch_id} 和路由表 "
                f"{network.vswitch.route_table_id}。[/green]"
            )
            return network

        selected = self._required(
            self.prompts.select(
                "发现多个本机网络候选，请选择当前实例使用的网络",
                [
                    (
                        f"{network.local_address.interface}: {network.local_address.address}"
                        f" -> {network.vpc.vpc_id}"
                        f" -> {network.vswitch.vswitch_id}"
                        f" -> {network.vswitch.route_table_id}",
                        str(index),
                    )
                    for index, network in enumerate(candidates)
                ],
            )
        )
        return candidates[int(selected)]

    def _render_local_network(
        self,
        network: LocalNetwork,
        *,
        provider: str,
        region: str,
        profile: str,
        cli_version: str,
    ) -> None:
        table = Table(title="身份与本机网络")
        table.add_column("项目")
        table.add_column("值")
        table.add_column("用途")
        local_only = "仅本机"
        share = "发送给 Beanfleet"
        table.add_row("云厂商", provider, local_only)
        table.add_row("地域", region, share)
        table.add_row("Profile", profile, local_only)
        table.add_row("阿里云 CLI", cli_version, local_only)
        table.add_row("账号 UID", network.identity.account_id, share)
        table.add_row("身份类型", network.identity.identity_type or "—", local_only)
        table.add_row("身份 ARN", Text(network.identity.arn or "—"), local_only)
        table.add_row("VPC ID", network.vpc.vpc_id, share)
        table.add_row("VPC IPv4 网段", ", ".join(network.vpc.ipv4_cidrs), share)
        table.add_row("实例私网 IP", network.local_address.address, local_only)
        table.add_row("网卡", network.local_address.interface, local_only)
        table.add_row("交换机 ID", network.vswitch.vswitch_id, local_only)
        table.add_row("交换机网段", network.vswitch.cidr, local_only)
        table.add_row("路由表 ID", network.vswitch.route_table_id, local_only)
        table.add_row("路由表类型", network.vswitch.route_table_type or "—", local_only)
        self.console.print(table)
        self.console.print(
            "[yellow]自动流程只向 Beanfleet 上传 region_id、vpc_id、account_id "
            "和 cidr_blocks；不会上传阿里云凭据、ARN、实例 IP、vSwitch 或路由表信息。"
            "手工排障流程仍由用户自行传递所需信息。[/yellow]"
        )

    def _render_connection_inspection(
        self,
        inspection: ConnectionInspection,
        *,
        title: str,
    ) -> None:
        connection = inspection.connection
        summary = Table(title=title)
        summary.add_column("项目")
        summary.add_column("值")
        summary.add_row("连接 ID", connection.connection_id)
        summary.add_row("连接名称", connection.name or "—")
        summary.add_row("连接状态", connection.status)
        summary.add_row("业务状态", connection.business_status or "—")
        summary.add_row("失效时间", connection.expires_at or "—")
        summary.add_row(
            "发起端",
            f"{connection.requester.account_id} / {connection.requester.region}"
            f" / {connection.requester.vpc_id}",
        )
        summary.add_row("发起端网段", ", ".join(connection.requester.ipv4_cidrs) or "—")
        summary.add_row(
            "发起端实例私网 IP",
            "自动流程不需要；仅手工 /32 高级排障时自行取得",
        )
        summary.add_row(
            "接收端",
            f"{connection.accepter.account_id} / {connection.accepter.region} / {connection.accepter.vpc_id}",
        )
        summary.add_row("接收端网段", ", ".join(connection.accepter.ipv4_cidrs) or "—")
        self.console.print(summary)

        checks = Table(title="检查结果")
        checks.add_column("检查")
        checks.add_column("状态")
        checks.add_column("详情")
        labels = {"ok": "[green]正常[/green]", "error": "[red]失败[/red]"}
        for check in inspection.checks:
            checks.add_row(check.name, labels.get(check.status, check.status), check.detail)
        self.console.print(checks)

    def _render_connection_status(
        self,
        inspection: ConnectionInspection,
        *,
        title: str,
    ) -> None:
        connection = inspection.connection
        table = Table(title=title)
        table.add_column("项目")
        table.add_column("值")
        table.add_row("连接 ID", connection.connection_id)
        table.add_row("连接状态", connection.status)
        table.add_row("业务状态", connection.business_status or "—")
        table.add_row("失效时间", connection.expires_at or "—")
        errors = [check.detail for check in inspection.checks if check.status == "error"]
        check_result = Text("\n".join(errors), style="red") if errors else Text("通过", style="green")
        table.add_row("安全检查", check_result)
        self.console.print(table)

    def _render_route_plan(self, plan: RoutePlan, *, title: str) -> None:
        table = Table(title=title)
        table.add_column("项目")
        table.add_column("值")
        table.add_row("连接 ID", plan.connection_id)
        table.add_row("目标网段", plan.destination)
        table.add_row("路由表 ID", plan.route_table_id)
        table.add_row("本机 IP", plan.local_ip or "手动指定路由表")
        table.add_row("交换机 ID", plan.vswitch_id or "—")
        table.add_row("动作", "创建路由" if plan.create_required else "无需创建")
        table.add_row("路由条目 ID", plan.existing_route_id or "—")
        table.add_row("路由状态", plan.existing_status or "尚未创建")
        self.console.print(table)

    def show_current_transfer(self) -> None:
        snapshot = self.transfer_state_store.load().current_transfer
        if snapshot is None:
            self.console.print("[dim]当前没有正在进行的传输。[/dim]")
            return
        self._render_transfer(snapshot, title="当前传输")

    def show_last_transfer(self) -> None:
        snapshot = self.transfer_state_store.load().last_transfer
        if snapshot is None:
            self.console.print("[dim]暂无传输记录。[/dim]")
            return
        self._render_transfer(snapshot, title="最近传输结果")

    def show_environment_diagnostics(self) -> None:
        self._render_diagnostics(
            collect_environment_diagnostics(),
            title="环境诊断",
        )

    def show_connection_diagnostics(self) -> None:
        report = collect_connection_diagnostics(
            token_store=self.token_store,
            config_loader=self.config_store.load,
            control_client_factory=self.control_client_factory,
        )
        self._render_diagnostics(report, title="连接诊断")

    def show_training_config(self) -> None:
        config = self.config_store.load()
        table = Table(title="训练设置")
        table.add_column("项目")
        table.add_column("当前值")
        for key, value in config.training.to_dict().items():
            if value is None:
                rendered = "调度器默认值" if key == "cpu_per_gpu" else "未填写"
            else:
                rendered = json.dumps(value, ensure_ascii=False)
            table.add_row(key, Text(rendered))
        self.console.print(table)

    def _render_diagnostics(self, report: DiagnosticReport, *, title: str) -> None:
        table = Table(title=title)
        table.add_column("检查")
        table.add_column("状态")
        table.add_column("详情")
        status_labels = {
            "ok": "[green]正常[/green]",
            "warning": "[yellow]提示[/yellow]",
            "error": "[red]失败[/red]",
            "skipped": "[dim]跳过[/dim]",
        }
        check_labels = {
            "python": "Python",
            "platform": "平台",
            "torch": "PyTorch",
            "credentials": "Token",
            "service_configuration": "连接配置",
            "control_service": "控制服务",
            "scheduler_configuration": "调度器配置",
            "scheduler_assignment": "节点分配",
            "node_connections": "全部节点连接",
        }
        for check in report.checks:
            if check.name == "node_connections" and isinstance(check.detail, dict):
                total, connected = check.detail["total"], check.detail["connected"]
                if not total:
                    detail = "尚无已分配节点"
                elif check.status == "ok":
                    detail = f"全部节点连接成功（{connected}/{total}）"
                else:
                    detail = f"未确认全部节点连接就绪（连接检查成功 {connected}/{total}）"
            elif isinstance(check.detail, dict | list):
                detail = json.dumps(check.detail, ensure_ascii=False, default=str)
            else:
                detail = {
                    "all training nodes assigned and ready": "当前训练节点已全部分配并就绪",
                    "no nodes assigned": "尚无已分配节点",
                    "training assignment is incomplete or not ready": "当前训练节点分配不完整或尚未就绪",
                }.get(str(check.detail), str(check.detail))
            table.add_row(
                check_labels.get(check.name, check.name),
                status_labels[check.status],
                detail,
            )
        self.console.print(table)

    def show_config(self) -> None:
        config = self.config_store.load()
        table = Table(title="连接配置")
        table.add_column("分组", style="cyan", no_wrap=True)
        table.add_column("项目", no_wrap=True)
        table.add_column("当前值")
        rows = self._connection_settings(config)
        previous_group = None
        for index, (group, label, value, _key) in enumerate(rows):
            table.add_row(
                group if group != previous_group else "",
                label,
                Text(value),
                end_section=index + 1 < len(rows) and rows[index + 1][0] != group,
            )
            previous_group = group
        self.console.print(table)

    def _render_transfer(self, snapshot: TransferSnapshot, *, title: str) -> None:
        status_labels = {
            "STARTING": "启动中",
            "RUNNING": "传输中",
            "PAUSED": "已暂停",
            "COMPLETED": "已完成",
            "STOPPED": "已结束",
            "FAILED": "失败",
        }
        table = Table(title=title)
        table.add_column("项目")
        table.add_column("值")
        table.add_row("状态", status_labels[snapshot.status])
        unit = "Batch" if snapshot.spec.is_vqa else "Sample"
        table.add_row(f"已接收 {unit}", str(snapshot.batches_received))
        table.add_row(f"已交付 {unit}", str(snapshot.batches_consumed))
        table.add_row("已接收数据", self._format_bytes(snapshot.bytes_received))
        table.add_row("开始时间", self._format_time(snapshot.started_at))
        table.add_row("最近交付时间", self._format_time(snapshot.last_batch_at))
        if snapshot.finished_at is not None:
            table.add_row("结束时间", self._format_time(snapshot.finished_at))
        table.add_row(
            "运行时长",
            self._format_duration(snapshot.started_at, snapshot.finished_at or utc_now()),
        )
        table.add_row("最后更新时间", self._format_time(snapshot.updated_at))
        self.console.print(table)
        self._render_batch_spec(snapshot.spec)

    def _render_batch_spec(self, spec: BatchSpec | CanonicalSampleSpec | VQABatchSpec) -> None:
        summary = Table(title="BatchSpec 摘要")
        summary.add_column("项目")
        summary.add_column("值")
        if spec.is_vqa:
            summary.add_row("batch.size", str(spec.batch.size))
        else:
            summary.add_row("交付单位", "单个 canonical sample / HC frame")
        summary.add_row("estimated tensor bytes", str(spec.estimated_batch_bytes))
        if spec.estimated_metadata_bytes:
            summary.add_row("JSON metadata budget", str(spec.estimated_metadata_bytes))
        summary.add_row("schema hash", spec.schema_hash)
        self.console.print(summary)

        output = Table(title="传输字段")
        output.add_column("名称")
        output.add_column("Shape / Rows")
        output.add_column("Dtype / Items")
        for name, schema in sorted(spec.output.items()):
            if not isinstance(schema, Mapping):
                schema = schema.model_dump(mode="json")
            if "shape" in schema:
                shape = (
                    " × ".join("N" if value is None else str(value) for value in schema["shape"]) or "scalar"
                )
                output.add_row(name, shape, schema["dtype"])
            elif schema["type"] == "list":
                output.add_row(name, str(schema["length"]), schema["items"])
            else:
                output.add_row(name, "—", str(schema.get("value", schema["type"])))
        self.console.print(output)

    @staticmethod
    def _format_time(value: datetime | None) -> str:
        if value is None:
            return "—"
        return value.astimezone().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _format_duration(started_at: datetime, ended_at: datetime) -> str:
        total_seconds = max(int((ended_at - started_at).total_seconds()), 0)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours} 小时 {minutes} 分 {seconds} 秒"
        if minutes:
            return f"{minutes} 分 {seconds} 秒"
        return f"{seconds} 秒"

    @staticmethod
    def _format_bytes(value: int) -> str:
        amount = float(value)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if amount < 1024 or unit == "TiB":
                return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
            amount /= 1024
        raise AssertionError("unreachable")

    @staticmethod
    def _required(value: Any | None) -> Any:
        if value is None:
            raise PromptCancelled
        return value

    @staticmethod
    def _positive_int(value: str, label: str, *, maximum: int | None = None) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ConfigurationError(f"{label} must be an integer") from exc
        if parsed <= 0:
            raise ConfigurationError(f"{label} must be positive")
        if maximum is not None and parsed > maximum:
            raise ConfigurationError(f"{label} must not exceed {maximum}")
        return parsed


def run_interactive() -> int:
    return InteractiveClient().run()
