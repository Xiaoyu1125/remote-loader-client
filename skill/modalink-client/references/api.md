# 用户接入文档与数据读取

本页是常用调用与数据检查索引；完整训练用法见 [用户指南](../../../user-guide.md)，YAML 参数见 [batch-spec.md](batch-spec.md)。示例中的 `batch.yaml` 是用户自行编写或通过私有渠道取得的输入文件，请替换为实际路径；公开仓库和 wheel 均不提供 batch 请求 YAML。wheel 也不含辅助脚本；安装、连接、登记和释放资源统一见 [deployment.md](deployment.md)。

公开仓库和包不含证书、私钥或 Token。连接前按部署指南通过私有渠道取得有效凭据，用 `remote-loader config edit` 配置实际 Scheduler 及 Loader CA、客户端证书、客户端私钥的全部三项绝对路径，再保存 Modalink Token；SDK 沿用该配置。float32 相机需要服务支持经过处理的 uint8 请求，仅支持 HC1 不足以保证兼容；worker 在用户 preprocess/collate 前恢复 float32，显式 uint8 请求保持原样。

## 按请求选择入口

| 需要 | 入口 |
| --- | --- |
| 离线检查声明 | `BatchSpec.from_yaml(path)`，直接查看 `spec.output` |
| 读取 VLA 样本或 VQA batch | `RemoteDataLoader.from_yaml(path)`、`with`、`next` 或迭代 |
| 修改声明 | 编辑 YAML，或修改 `spec.to_request()` 的结果后用 `BatchSpec.from_dict()` 构建新声明，再新建 loader |
| PyTorch 接入 | `to_torch_dataloader()`；VLA 按需设置本地 batch size 和回调 |
| 图像输出 | 在 YAML 中配置 `image_output` 与相机 shape/dtype，见 [batch-spec.md](batch-spec.md)；不要重复已完成的图像预处理 |
| 搬运设备 | `loader.to_device(item, device)`，不改变 dtype 或数值范围 |
| 检查数据集与读取状态 | 启动后查看 `loader.dataset_metadata`；调用 `loader.metrics()` |
| 暂停与结束 | `pause()` / `resume()`；退出 `with` 或调用 `close()` |

## 离线检查与真实读取

离线解析不连接服务，也不消耗数据：

```python
from remote_loader import BatchSpec

spec = BatchSpec.from_yaml("batch.yaml")
print(spec.output)
print("tensor 字节估算:", spec.estimated_batch_bytes)
```

`spec.output` 适用于 canonical VLA 和 VQA；VLA 的文本/JSON 字段没有 tensor 的 shape/dtype，不要对每个输出都读取这两个属性。`spec.to_request()` 返回完整请求，但 canonical 请求没有 `output` 成员。

完成资源登记和自动连接配置后，有限读取一个 item：

```python
from remote_loader import RemoteDataLoader

with RemoteDataLoader.from_yaml("batch.yaml") as loader:
    item = next(loader)
    for name, value in item.items():
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            print(name, tuple(value.shape), value.dtype)
        else:
            print(name, type(value).__name__)
    print(loader.metrics())
```

`with` 启动并关闭读取。若不用 `with`，应先 `start()`，并在 `finally` 中 `close()`。新建后不能直接 `next()`；重复迭代继续当前位置，不会重置 epoch，关闭后不能重新启动同一对象。

VLA 每次返回一个样本，保留声明的键、嵌套路径、shape 和 dtype，包括标量 `[]` 和长度为 1 的时间轴；VQA 每次返回完整 batch。上面的循环只概览顶层字段，不能代替嵌套内容检查。需要确认 task、mask、null、空容器或具体 JSON 值时，直接按实际键访问 `item`，不要根据诊断摘要中的点分路径推断嵌套结构。

`loader.dataset_metadata` 启动前为 `{}`，启动后为 `{control_endpoint: 元信息字典}`，单节点也不展平。按节点保留上下文并处理缺失键；它不保证每个样本含同一套任务或模型信息。

使用源码中或独立交付的检查脚本时，`SKILL_DIR` 指向 skill 目录，`BATCH_SPEC` 指向用户提供的 YAML：

```bash
python "$SKILL_DIR/scripts/inspect_batch.py" --spec "$BATCH_SPEC" --receive --items 1
```

不加 `--receive` 只离线检查；`--values` 可在真实读取时预览数值。脚本输出是诊断摘要，不是完整样本。`remote-loader receive batch.yaml` 则会持续读取，不提供有限条数选项。

### 等待与预取

- `prefetch_batches` 继承已保存的配置，默认 8，范围 1–1024；VLA 单位是样本，VQA 单位是完整 batch，不是训练步数。`prefetch_bytes=None` 自动估算；显式字节上限必须为正整数且至少容纳一个 item，不包括训练侧额外分配。
- 首项可有最长 300 秒预热等待；已有首项后，`batch_timeout` 默认 10 秒，必须为有限正数。检查脚本的稳态等待默认 60 秒，不能把它当作首次等待上限。
- `BatchTimeoutError` 表示等待数据超时，本次不消费数据，也不自动关闭 loader，可以按训练策略重试；它不限制模型计算时间。
- `session_timeout` 默认 300 秒，用于等待新会话就绪；不是整个初始化或训练的总时限，超时可抛出 Python 内置 `TimeoutError`。

## 接入训练要点

PyTorch helper 默认不预处理、不组批，`loader.to_torch_dataloader()` 直接迭代单个 VLA 样本。需要本地组批且字段适合 PyTorch 默认 collation 时：

```python
from remote_loader import RemoteDataLoader
from remote_loader.pytorch import close_torch_dataloader

with RemoteDataLoader.from_yaml("batch.yaml") as loader:
    dataloader = loader.to_torch_dataloader(batch_size=32)
    try:
        batch = next(iter(dataloader))
    finally:
        close_torch_dataloader(dataloader)
```

| helper 参数 | 默认值与约束 |
| --- | --- |
| `batch_size` | `None`；VLA 可设为正整数，在本地组批 |
| `preprocess` | `None`；可选 callable，逐样本执行，即使不组批也会执行 |
| `collate_fn` | `None`；可选 callable，提供时必须同时指定正整数 batch size；组批时省略则使用 PyTorch 默认 collation |
| `num_workers` | `0`；正整数启用原生 worker，其余 DataLoader 参数按 PyTorch 规则透传 |
| `pin_memory` / `pin_memory_device` | `False` / `""`；按训练环境选择 |

回调由训练项目按模型需要提供，不要求任何特定 processor。helper 接受 `num_workers`（默认 0），其他参数透传给原生 PyTorch DataLoader；worker 预取、持久化和启动方式只在 `num_workers>0` 时配置。也可以将 `remote_loader.pytorch.RemoteIterableDataset(loader, preprocess=...)` 交给用户自己的 DataLoader，由用户指定 worker 数量、collate 和其他选项。

`close_torch_dataloader(dataloader, *, iterator=None, close_source=True)` 先取消取数等待，再回收 worker。helper 返回原生 `DataLoader` 的子类，用弱引用记录所有活动迭代器，因此两种 `persistent_workers` 配置都可直接关闭。若用户自行创建原生 DataLoader，提前退出时需保存并传入 `iterator`；存在多个活动迭代器时分别清理。关闭过程中数据源报错仍会执行 worker 回收。helper 创建的 DataLoader 关闭后禁止重新迭代。

创建进程独占远端会话和接收 buffer；helper 的 0-worker 路径直接读取，正整数 worker 通过独立的认证 loopback IPC 取样，不复制网络 loader，preprocess/collate 在原生 worker 内执行。IPC 有 pickle 序列化复制成本；总内存还包含 DataLoader 预取和处理中的 batch。PyTorch 管理组批、顺序、每 worker 尾批及提前退出语义，不宣称零拷贝、自动 GPU overlap 或固定加速比例。推荐 `spawn` 与模块顶层 callback。正常结束或提前结束时，先停止迭代，再按上面的方式关闭，最后退出 loader context。该调用可重复执行；默认同时关闭关联的 `RemoteDataLoader`，仅当后续仍要使用该 `RemoteDataLoader` 时传 `close_source=False`。调用后该 PyTorch DataLoader 不可继续使用。详细示例和本地 IPC 信任边界见[用户接口文档](../../../user-guide.md#训练框架要求-pytorch-dataloader-时)。

不要增加 `DistributedSampler`、再次按 worker/rank 切片，或通过训练框架再次分发这一数据流。每个训练进程使用自己的 loader，进程与本机设备设置见 [deployment.md](deployment.md)。

VQA 已经是完整六字段 batch；helper 必须保持 `batch_size=None`、`preprocess=None`、`collate_fn=None`。其数据格式不等于任意模型的最终输入，仍需模型自己的文本/输入适配。图像模式和形状约束见 [batch-spec.md](batch-spec.md)，不要重复服务已完成的图像预处理。

### 指标、暂停与读取状态

`loader.metrics()` 可在启动前后调用。常用键包括：

| 指标 | 用途 |
| --- | --- |
| `state` | 当前读取状态 |
| `batches_received` / `batches_consumed` | 已接收/已读取的 item 数：VLA 样本或 VQA 完整 batch |
| `receive_batches_per_s` / `consume_batches_per_s` | 接收与消费速率，不是模型训练步速率 |
| `queue_size` / `queue_capacity` / `queued_bytes` | 排队 item 数、容量和所计入的数据字节，不是进程总内存 |
| `batch_wait_seconds` / `consumer_data_stall_ratio` | 等待数据的耗时与停顿情况 |
| `reconnect_count` | 重连次数 |

`available_batches` 是剩余容量，不是已就绪的 item 数。scheduler 模式还会提供各节点指标；这些指标不能代替 GPU 或模型训练性能测量。

`pause()` 用于运行中的 loader，暂停后应先 `resume()` 再继续读取；暂停保留排队数据，已开始的交付可能完成。它不保证会话无限有效，也不是关闭后重启。`close()` 可重复调用，已返回的数据仍可使用。

`state_dict()` 可导出读取位置，但位置在 item 离开 client 接收队列时就已推进，可能领先于 worker 处理和训练消费；之后的预处理、组批或训练失败不会回滚读取。**当前 scheduler 服务不支持关闭后从读取 checkpoint 精确续读。**不要将保存读取状态当作断点续训保证；有效会话内的 `pause()` / `resume()` 不属于关闭后恢复。只有明确支持恢复的服务才可使用 `load_state_dict()`，且必须在启动前调用，请求与训练拓扑须匹配。

退出 `with` 或调用 `close()` **不释放训练登记的资源**。停止读取后按 [deployment.md](deployment.md) 释放，自动化启停见 [training-scripts.md](training-scripts.md)。

## 常见异常

SDK 异常均从 `remote_loader.exceptions` 导入，例如 `from remote_loader.exceptions import BatchTimeoutError, ConfigurationError, SessionStateError`。

- `ConfigurationError`：YAML 或连接配置不合法；按错误指出的字段修正。
- `AuthenticationError` / `PermissionDeniedError`：检查登录凭证与访问授权，不在日志中打印 Token。
- `BatchTimeoutError`：等待数据超时，可按策略重试。
- `SessionStateError` / `SessionExpiredError`：检查是否暂停、关闭或会话已失效，不把重建 loader 当作精确续读。
- `ServiceUnavailableError` / `SchemaMismatchError`：检查连接或实际数据与声明是否一致；诊断步骤见 [deployment.md](deployment.md)。

并非所有参数错误都是 SDK 异常：构造参数可能抛出 `ValueError`，错误回调可能抛出 `TypeError`；迭代正常结束使用内置 `StopIteration`。
