# 声明式张量统计

统计是一项独立且有限的数据准备任务。它从服务已授权、固定版本的 `catalog` 中读取实际的规范张量（canonical tensor）。它**不会**获取 LeRobot 预计算的均值/标准差元数据，也不会只采样某个训练 rank、消费训练调度器 IOUnit、应用训练混合权重或改变 `RemoteDataLoader` 的采样行为。解析 YAML 不需要 PyTorch、Token 或服务连接。

## 离线声明

本文的 `statistics.yaml` 表示用户自行编写或通过私有渠道取得的统计输入文件，公开仓库和 wheel 均不提供该文件；调用时请替换为实际路径。统计声明独立于训练 sample YAML，提交前须根据实际 `catalog` 核对字段、窗口和预算。例如，对于 20 Hz、float32 类型且宽度为 7 的 `action`、宽度为 8 的 `state`，可以声明 `action` 窗口偏移 0–31、`observation` 窗口 `[0, 4, 8, 12, 16, 20, 24, 28, 32]`；每个 episode 不丢弃开头的 anchor，丢弃末尾 32 个 **anchor**，`stride` 为 1，并显式声明 2147483648 bytes（2 GiB）的统计工作区预算。这一参数说明不请求或解码图像字段，不代表所有数据集的实际 schema 或推荐配置。

```python
from remote_loader import StatisticsSpec

spec = StatisticsSpec.from_yaml("statistics.yaml")
request = spec.to_request()
print(spec.request_hash)
# A Python mapping uses the same strict validation:
spec = StatisticsSpec.from_dict(request)
```

所有请求对象都拒绝未知字段。统计 YAML 中不要加入 `sampling`、存储 locator、任意变换逻辑或 Python 表达式。必须提供 `scope`、`view`、非空的具名 `statistics` 和 `execution.max_memory_bytes`。格式默认为 `{type: tensor_statistics, version: 1}`；扫描的首尾丢弃数默认为 0，`stride` 默认为 1，`non_finite` 默认为 `error`。`scope.grouping` 必须显式设为 `pooled` 或 `per_dataset`；`dataset_indices` 必须是非空、唯一、非负的索引，并会按 `catalog` 顺序归一化。`pooled` 统计中的元素等权，不使用训练混合权重。

`view` 复用 canonical v2 的 `batch_format`、`require_metadata`、`windows`、`fields` 和 `image_output`。字段使用完整的 `target` 路径数组标识，包括 `[observation.state]` 与 `[observation, state]` 的区别。`shape` 是单样本形状，不包含隐式 batch 轴。统计目标支持数值张量；字符串、JSON 和 bool 目标会被拒绝。标量张量的 `shape` 为 `[]`。

### 轴、掩码与精确统计指标

每个统计查询声明 `field`、可选 `mask`、`reduce_axes`、可选 `non_finite`，以及非空的具名 `metrics`：

- `reduce_axes` 必须且只能包含一次 `sample`；额外的唯一非负整数轴指向**单样本**张量。对于 `[time, feature]`，`[sample]` 产生 `[time, feature]`，`[sample, 0]` 产生 `[feature]`，`[sample, 0, 1]` 产生标量。未归约轴保持原顺序。
- `mask.field` 必须引用已声明的 bool 张量。`mask.axes` 将掩码的每个维度映射到唯一的单样本张量轴；维度必须与目标维度相等，或可通过单元素维度广播。时间掩码 `[T]` 使用 `axes: [0]`，标量掩码使用 `axes: []`。`temporal_mask` 中的 `True` 表示有效；`is_pad` 中的 `True` 表示填充，二者含义相反。
- 先应用掩码，再处理非有限值。`non_finite: error` 拒绝有效位置中的 NaN/Inf；`exclude` 会排除这些值。没有任何有效元素的输出位置会失败，而不是生成伪造的 0 或 NaN。
- `count` 返回整数计数。`min` 和 `max` 保持数值精度。`mean` 使用 float64 累加并输出。`std` 必须显式指定 `ddof: 0` 或 `ddof: 1`，使用 float64 矩，并在 count 不大于 ddof 时失败。
- `quantile` 要求 `q` 是 `[0, 1]` 范围内的有限数值，并显式指定 `method: lower` 或 `method: linear`。两种方法都是精确算法：`lower` 选择排序后索引 `floor(q*(n-1))` 并保持整数精度；`linear` 在相邻顺序统计量之间插值。不提供隐式近似算法。
- 任意统计指标都可以增加 `post_reduce: {op: min, axes: [0]}` 或 `op: max`。其中非空且唯一的轴指向**第一阶段统计结果**，不是原始单样本张量。

沿用上述参数说明，若将 `action` 的 `count/min/max/mean/std/q01/q99` 按时间步保留，结果为 `[32, 7]`；通过 `ddof: 1` 可显式使用样本标准差。若将 state 输出命名为 `proprio`，其 q01/q99 可先在九个 `observation` 时间步上计算为 `[9, 8]`，再沿时间轴分别执行 min/max，得到 `[8]`。相应的 `action` 和 `proprio` 有效性掩码的 `shape` 分别为 `[32]` 和 `[9]`。这是通用的两阶段包络，不是针对 state/action 的特殊实现开关。先计算 quantile 再执行 min/max，与先合并时间维再计算一次 quantile 并不等价；只有显式 post-reduction 才会移除 `observation` 时间轴。

## 固定版本、覆盖范围与身份

编写 `scope.dataset_indices` 前，先调用 `StatisticsClient.catalog()` 或 `remote-loader statistics catalog`。每个索引都是返回的 `catalog_id` 所属有序 `scheduler_datasets` 中对应条目的零基位置；应将检查得到的映射与 YAML 一同保存。提交时，SDK 读取同一元数据并绑定不透明的 `catalog_id`。服务在扫描前验证该身份，固定实际 manifest version，并以有限、不打乱、不分片的语义读取每个选中数据集一次。过期的 catalog ID 会返回 HTTP 409；客户端不会静默改用其他 catalog 提交。

首尾丢弃规则会排除每个 episode 边界处的 anchor，但这些行仍可作为窗口上下文。完成修剪后，`stride` 会独立应用到各数据集按升序排列的符合条件 anchor。`stride: 1` 选择所有符合条件 anchor；更大的 `stride` 表示显式稀疏后的统计，不能表示成步长为 1 的统计。`coverage` 包含各数据集的 `eligible_anchors`、`scanned_anchors`、`selected_anchors` 及 `complete: true`；读取错误、覆盖范围缺失、取消或预算失败都不能发布部分成功结果。

成功结果包含：

- `statistics_format: {type: tensor_statistics_result, version: 1}`；
- `job_id`、`catalog_id`、归一化声明的 `request_hash`，以及 `algorithm: tensor-statistics-v1`；
- 选中的 `datasets`，包含 catalog 索引、不透明的数据集键和实际 manifest version，不含私有 locator 或凭证；
- `coverage` 和 `results`，其中每个结果分组包含 `dataset_indices` 和具名 `statistics`；
- 每个统计指标的 `shape`、`dtype` 和标量或嵌套 `values`，并保留单元素维度和整数值。

请求指纹使用 UTF-8、按 key 排序的紧凑 JSON，排除 `None`、禁止 NaN，并包含规范化请求值。`image_output.antialias` 的省略值和显式 `true` 都按默认值规范化为省略，`false` 保留并改变请求 hash。`pooled` 请求产生一个分组；`per_dataset` 请求按 catalog 顺序产生多个分组。应保存完整结果，而不是只保存裸 mean/std 数组：catalog、version、spec 和 algorithm identity 都是判断结果能否复用的必要信息。

## SDK：单一独立协调者

在训练开始前使用一个协调客户端，或使用一个已存在且空闲的独立服务。不要从每个 rank/worker 提交。统计任务和训练数据加载不能同时占用同一个 CPU 服务工作槽；冲突会立即返回 HTTP 409。该功能不会分配新服务，也不会释放现有资源分配。

```python
from remote_loader import StatisticsClient, StatisticsSpec
from remote_loader.exceptions import (
    StatisticsCancelledError,
    StatisticsFailedError,
    StatisticsTimeoutError,
)

spec = StatisticsSpec.from_yaml("statistics.yaml")
with StatisticsClient.from_config() as client:
    job = client.submit(spec, idempotency_key="libero-normalization-run-001")
    print(client.endpoint, job["job_id"])
    try:
        result = client.wait(job["job_id"], timeout=3600)
    except StatisticsTimeoutError:
        # Waiting stopped; the remote job remains active. Inspect or explicitly cancel.
        state = client.status(job["job_id"])
        raise
    except (StatisticsFailedError, StatisticsCancelledError):
        # No partial result exists.
        raise
```

`from_config(config: ClientConfig | None = None, *, local_rank: int = 0, token: str | None = None)` 在省略配置时加载已保存配置。调度器模式通过 `/status?training_id=…` 读取整场训练的已有分配，按响应顺序选择第一个 `ready` 且无错误的普通 CPU 节点；没有就绪节点时明确报错。GPU 的 `host_ip` 可以为空。`local_rank` 保留以兼容已有调用，统计节点选择与本机 rank 无关。直连模式使用已配置的 endpoint。SDK 不会登记、启动、结束或释放训练资源，也不会广播请求或回退到其他节点。以后重新连接时，应将 `client.endpoint` 与 job ID 一同记录。新建客户端必须指向原 endpoint，尤其是在资源分配已变化后。

直接构造时，将 `LOADER_PRIVATE_IP` 设为分配给作业的 CPU 私网 IP，并确认私有渠道取得的服务端证书覆盖该 IP。先按 [deployment.md](deployment.md) 配置实际 Scheduler、全部三项绝对 TLS 路径和 Modalink Token；公开仓库和包不含这些凭据。

```python
import os

from remote_loader import ClientConfig, StatisticsClient

config = ClientConfig.load()
loader_ip = os.environ["LOADER_PRIVATE_IP"]
with StatisticsClient(
    f"https://{loader_ip}:8080",
    token=issued_token,
    client_config=config,
    connect_timeout=5.0,
    read_timeout=30.0,
    max_retries=3,
) as client:
    job = client.submit(spec)
```

`token=None` 使用现有 `TokenStore`。直接构造与 `from_config()` 都要求 HTTPS，使用 schema 7 `ClientConfig` 中完整配置的 Loader CA、客户端证书和客户端私钥。省略 `client_config` 时自动加载已有配置；自动发现 Scheduler 不会提供凭据。TLS 握手完成并验证对端后才会发送 token。构造函数也接受现有 `httpx` 的 `transport` 注入，以支持受控集成。

方法：

- `submit(spec, *, idempotency_key=None) -> dict`：校验并冻结归一化声明，绑定 catalog identity，并返回任务。省略 key 时只生成一次，并在 HTTP 重试之间复用。显式 key 若用于不同请求正文，会以 409 失败。
- `status(job_id) -> dict`：返回 `QUEUED`、`RUNNING`、`SUCCEEDED`、`FAILED` 或 `CANCELLED`，以及身份字段和 `error`。
- `result(job_id) -> dict`：只获取成功结果；获取未完成或未成功的结果会返回 HTTP 409，并映射为 `RemoteLoaderError`。
- `cancel(job_id) -> dict`：幂等取消。清理可能仍在进行；服务会继续持有工作槽，直到 reader/workspace 完成释放。
- `wait(job_id, *, timeout=300.0, poll_interval=0.25) -> dict`：轮询并返回结果，而不是任务。失败、取消和超时会抛出对应的类型化异常。有限正数的调用方截止时间会约束轮询等待、重试退避和单次请求的 HTTP 超时预算，但不会改变共享客户端的默认 HTTP 超时。
- `close()` 和上下文管理器退出只关闭 HTTP 资源，不会取消任务或终止训练资源。

认证、权限、请求格式错误和服务不可用使用现有 `RemoteLoaderError` 层级。每次响应都会校验已知任务身份，因此变化后的任务、声明或 catalog 不能静默替代其他结果。提交客户端会保留分离的归一化声明，并据此校验结果的 `dataset` 范围、`grouping`、`coverage`/`stride`、查询和指标名称、`shape` 与 `dtype`。每个结果都必须通过完整的数据集、分组和覆盖范围一致性检查，以及递归的形状与有限数值检查；即使由没有该声明的新连接客户端获取也一样。整数值必须适合声明的 `dtype`；bool 不是数值。格式错误的结果抛出 `ProtocolError`。在调用方截止时间前耗尽传输重试时抛出 `ServiceUnavailableError`；只有调用方截止时间已到才抛出 `StatisticsTimeoutError`。

## CLI 与显式 JSON 文件

使用正常保存的 `TokenStore` 和连接配置。命令输出 JSON。不带 `--wait` 时，`submit` 只返回任务信息，不下载结果：

```bash
remote-loader statistics catalog
remote-loader statistics submit statistics.yaml
remote-loader statistics status JOB_ID --control-endpoint CONTROL_ENDPOINT
remote-loader statistics result JOB_ID --control-endpoint CONTROL_ENDPOINT --output ./libero-statistics.json
remote-loader statistics cancel JOB_ID --control-endpoint CONTROL_ENDPOINT

remote-loader statistics submit statistics.yaml \
  --wait --timeout 3600 --output ./libero-statistics.json \
  --idempotency-key libero-normalization-run-001
```

每个子命令都接受 `--control-endpoint HTTPS_URL`，以及只读发现参数 `--training-id ID`、`--scheduler-endpoint URL` 和 `--verify-scheduler-tls` / `--no-verify-scheduler-tls`。`--local-rank N` 默认 0，仅为兼容已有命令而保留，不参与统计节点选择。

显式指定 `--control-endpoint` 会跳过调度器发现，与调度器覆盖参数互斥。连接使用 schema 7 配置中显式指定的三项 mTLS 凭据绝对路径，不依赖包内身份。服务端证书必须覆盖所选地址；Scheduler TLS 与此 Loader mTLS 配置独立。

`submit`、`status` 和 `cancel` 输出的任务 JSON 包含 `control_endpoint`。由于发现顺序可能变化，后续处理同一任务的进程必须显式传入相同的 `control_endpoint`。

`submit --output` 必须与 `--wait` 一起使用；不会隐式选择等待行为或文件路径。等待开始前，包含 endpoint 的已提交任务会打印到 stderr，因此超时或失败后仍能保留 job ID 和 endpoint。成功结果保持服务端结构：打印到 stdout，并且仅在显式请求时保存为 JSON，不增加 CLI 外层包装。输出以 0600 权限模式原子发布，并拒绝任何已存在目标，包括符号链接；不提供覆盖开关。文件错误返回非零退出状态，但仍可使用 job ID 和原 endpoint 获取任务。任何命令都不会修改 YAML、Token、已保存连接设置或训练配置。

## 资源限制与结果生命周期

精确 quantile 会保留输入值，并需要额外计算工作区。兼容指标共享缓冲区；增加预算不会把近似方法变成精确方法，因为两种受支持的方法本身都是精确的。必须提供 `execution.max_memory_bytes`，其范围覆盖统计累加器、精确 quantile 存储和工作区，但不限制进程总 RSS、读取器缓冲区或编解码器。服务还会限制预算、活跃任务数、终态缓存数和序列化结果大小；超过限制时任务失败，不返回部分统计。

默认服务限制为：一个活跃任务与训练共享一个工作槽、最大请求统计工作区为 2 GiB、最多缓存 32 个终态任务、序列化缓存总量 64 MiB、每个序列化结果最多 8 MiB。更小的显式请求预算会被遵守。最旧的终态任务会先被淘汰，其幂等条目也会一同淘汰；因此复用已淘汰的 key 可能启动新的扫描，而不是获取历史任务。

任务和结果归属于认证身份及当前运行中的服务。内存终态缓存有界，旧任务可能被淘汰；服务重启不会保留任务或结果，也不会恢复扫描。释放或重启服务前应显式下载成功结果，再将同一个不可变结果分发给各训练 rank。取消和服务关闭会停止并清理 reader；取消或失败的任务都不存在有效的部分结果。普通训练加载行为不变。
