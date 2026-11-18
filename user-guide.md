# Modalink client 用户接口文档

训练代码主要使用 `RemoteDataLoader` 读取 VLA 单样本或 VQA batch；需要查看或调整请求时，再使用 `BatchSpec`。

本文假定已在训练环境安装 client 和匹配的 PyTorch，完成连接、Token 与训练登记，并在各训练机器确认本机节点就绪。公开仓库和包不含证书、私钥或 Token；先通过私有渠道取得有效凭据，用 `remote-loader config edit` 配置实际 Scheduler 和全部三项绝对 TLS 文件路径，再使用 TUI/SDK。完整准备步骤见 [README](README.md)。示例中的 YAML 来自源码仓库或单独交付文件，不包含在 wheel 中；路径与源字段须按实际数据集确认。

| 操作 | 接口 |
| --- | --- |
| 创建与读取 | `RemoteDataLoader.from_yaml()`、`with`、`next()`、迭代 |
| 查看或构建请求 | `BatchSpec.from_yaml()/from_dict()` |
| 搬到训练设备 | `loader.to_device()` |
| 查看供给情况 | `loader.metrics()`、`loader.dataset_metadata` |
| 暂停与继续 | `loader.pause()/resume()` |
| 可选的 PyTorch 接入 | `loader.to_torch_dataloader()` |

## 1. 读取边界：VLA sample 与 VQA batch

### 取一个 VLA canonical sample

```python
from remote_loader import RemoteDataLoader

with RemoteDataLoader.from_yaml("batch.yaml") as loader:
    sample = next(loader)
    print(sample)
```

VLA 每次返回一个完整样本，不自动组训练 batch。返回的 key、嵌套结构、shape 和 dtype 由请求与实际数据决定；不要把时间维或 JSON 列表长度当作 batch 维，也不要无条件取 `[0]` 或 squeeze。

[batch.yaml](batch.yaml) 是有源能力要求的模板，不保证任意数据集提供其中的相机、task 或 mask。模型所需的任务、机器人配置、频率和掩码必须来自真实数据或训练配置，不能用占位值绕过缺失字段。

### 持续读取

```python
from remote_loader import RemoteDataLoader

with RemoteDataLoader.from_yaml("batch.yaml") as loader:
    for sample in loader:
        print(sample)
```

流结束时循环自然结束，手动 `next()` 则抛出 `StopIteration`。提前退出或异常时，`with` 仍会关闭 loader。

也可先 `start()` 或 `iter(loader)`，再调用 `next()`，并在 `finally` 中 `close()`。刚构造的 loader 不能直接 `next()`。再次迭代同一个 loader 会继续当前位置，不会重置 epoch；关闭后不能重启，需要创建新对象。

## 2. 查看实际数据与请求

### 查看实际收到的数据

直接检查返回样本的字段。例如以下代码查看顶层字段；嵌套字典可按实际 key 继续展开：

```python
from remote_loader import RemoteDataLoader

with RemoteDataLoader.from_yaml("batch.yaml") as loader:
    sample = next(loader)
    for name, value in sample.items():
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            print(name, tuple(value.shape), value.dtype)
        else:
            print(name, type(value).__name__, value)
```

这会消费一个 VLA 样本或一个 VQA batch。没有不推进位置的 peek、`loader[0]` 或 `len(loader)`；已经返回的数据在关闭 loader 后仍可使用。

### 只看请求定义

```python
from remote_loader import BatchSpec

spec = BatchSpec.from_yaml("batch.yaml")
for name, field in spec.output.items():
    print(name, field)
print("估算 tensor 字节数:", spec.estimated_batch_bytes)
```

解析 YAML 不连接服务、不消费数据，也不需要 PyTorch。它只验证请求格式，不能证明实际数据集具有所需字段。`estimated_batch_bytes` 不包括训练侧处理、组批和其他缓存，不能视为进程总内存。

### VQA：由训练登记选择数据集

启用 VQA 并登记 `vqa_datasets` 后，使用以下 `vqa.yaml`；数据集 ID 和版本不写在该 YAML 中：

```yaml
batch:
  size: 32
batch_format:
  type: vlm_vqa
  version: 2
```

```python
from remote_loader import RemoteDataLoader

with RemoteDataLoader.from_yaml("vqa.yaml") as loader:
    batch = next(loader)
    print(batch["pixel_values"].shape, batch["samples"])
```

也可用 `VQABatchSpec(batch={"size": 32})` 构建请求。`batch.size` 是每次交付的样本数，必须为正整数。每次返回完整六字段 batch：

| 字段 | 内容 |
| --- | --- |
| `pixel_values` | float32 `[B, N, 3, 256, 256]`，像素范围 `[-1, 1]`；`N` 可变且至少为 1 |
| `samples` | 长度为 B 的文本/图像槽位列表；包含模板、问题、答案，可含 thinking；`image0/image1/...` 的尺寸值为 `(256, 256)` 元组 |
| `action` | `None` |
| `action_is_pad` | `None` |
| `action_dim_is_pad` | `None` |
| `is_vlm_batch` | `True` |

`output` 通常省略；需要显式声明时，必须匹配上述完整格式，见 [BatchSpec 参考](skill/modalink-client/references/batch-spec.md)。VQA 不使用 VLA 的 `fields/windows/image_output` 设置。

不要再本地组批或重复图像缩放、归一化。完整数据 batch 不等于任意模型的直接输入：文本处理和其他模型适配仍由训练项目决定。读取无需先查询 metadata；连接与 Token 沿用当前训练配置，未分配 VQA 资源时不会改读 VLA。

### 混合读取 VLA 和 VQA

同时配置 VLA、VQA 数据集并开启服务后，可在源码仓库根目录运行[混合读取示例](examples/mixed_remote_batches.py)：

```bash
python examples/mixed_remote_batches.py --vla-batch-size 8 --vqa-batch-size 32 --batches 100 --seed 42
```

每次以 80% 的概率返回一组本地收集的 VLA samples，或以 20% 的概率返回一个完整 VQA batch。比例和 `--batches` 均按输出组计数；`--seed` 只控制组类型选择，不控制数据集采样。该示例演示读取，不代替模型所需的预处理与组批。

## 3. 修改 canonical v2 声明：按需使用

VLA 使用 `batch_format: {type: canonical_sample, version: 2}`。`BatchSpec.from_yaml/from_dict` 自动识别格式，也可直接使用 `CanonicalSampleSpec`。详细约束以 [BatchSpec 参考](skill/modalink-client/references/batch-spec.md)为准。

| YAML 部分 | 决定什么 |
| --- | --- |
| `batch_format` | `canonical_sample`，`version: 2` |
| `sampling` | 随机种子、首尾丢帧与每个 episode 的最少锚点数 |
| `require_metadata` | 可选 FPS 要求，不是重采样指令 |
| `windows` | 具名整数帧窗口：`offsets`，或完整的 `start/step/count` |
| `fields` | 输出路径、真实来源、窗口与单样本类型/shape |
| `image_output` | 所选相机的图像输出方式 |

数据集与版本在训练登记中设置，不放入 VLA YAML。每个 field 在 `tensor` 和 `value_type: string/json` 中二选一：

- `source.column` 是源列的字面名称；`target` 是输出路径数组。`["observation.state"]` 是含点号的单个 key，`["observation", "state"]` 则是嵌套路径。
- `tensor.shape` 是精确单样本形状，不含 B；标量为 `[]`。普通字段不会隐式 reshape。
- 使用窗口时，shape 第一维必须等于窗口点数；`offsets: [0]` 也保留 T=1。不设窗口则不额外添加 T 维。
- `add_feature_axis: true` 显式增加末尾长度为 1 的特征轴，shape 也需包含该维度。
- 按数据需要选用普通列、任务查表、常量、`temporal_mask`、`is_pad` 或 `nonempty` 来源。注意有效掩码和 padding 掩码的含义不同，不能只改名替代。

`sampling.min_samples_per_episode` 默认为 0，范围 0–1000000；0 表示不补锚点。服务端先保留首尾裁剪后的锚点，不足时按 episode 内的帧顺序补入尚未保留的锚点，最多保留该 episode 的全部帧，不重复锚点。补入锚点的窗口仍按 episode 边界填充，训练侧必须处理对应 padding mask。此参数需要匹配的服务端支持。

### 会话级图像输出

`image_output` 是配置对象，默认开启。选择模式时同步确认图像字段的 dtype 和 shape：

| 选择 | 图像字段与输出 |
| --- | --- |
| 省略配置，或 `enabled: true` | 声明 `float32`；按目标尺寸处理，并将像素缩放到浮点范围；可追加 mean/std 标准化 |
| `enabled: false` | 声明 `uint8` 和源尺寸，保留原始尺寸与整数像素 |
| `resize_uint8: true` | resize 后保留整数像素精度，再转为 float32；仅开启处理时可用，默认 false |
| `antialias` | 默认 `true`；设为 `false` 时使用 PyTorch/ATen `align_corners=false` non-antialias bilinear，仅适用于 `resize_uint8: false` 的 float32 路径 |
| `padding` | 默认 `[0, 0, 0, 0]`；按 `[left, right, top, bottom]` 指定四个非负整数边距，在 mean/std 标准化之前补零 |
| `mean` / `std` | 同时提供三个有限通道值，std 均不能为 0；省略则不做 mean/std 标准化 |

省略 `antialias` 与显式 `true` 会规范化为相同请求和 hash；`false` 会保留在请求中并改变 hash，但不改变输出 schema。

处理后的最终画布 H/W 取 `tensor.shape` 最后两维，包含 padding，所有选中相机必须使用同一最终 H/W；相机为三通道，无窗口时使用 CHW，有窗口时使用 TCHW。实际 resize 内容尺寸为 `H - top - bottom`、`W - left - right`，两者必须大于 0。源分辨率可以不同，不根据尺寸差异自动推断补边。`resize_uint8: true` 与 `antialias: false` 不能同时使用。关闭处理时，shape 必须匹配各相机源尺寸，且不能再设置 mean/std、`resize_uint8: true`、`antialias: false` 或非零 padding。

边框在标准化前为零；未设置 mean/std 时交付 0，设置后每通道为 `(0 - mean[c]) / std[c]`。以下示例的最终画布为 448×224，内容 resize 到 224×224，顶部补 224 行；mean/std 均为 0.5，因此交付的顶部边框为 -1。SDK 按逻辑请求交付 float32 像素，不添加 batch 轴，也不修改源 metadata。相机身份由启动时各节点 metadata 的 `camera_keys` 与源 `column` 对应确定，不根据 target 名称或形状推断。

以下示例要求数据集真实存在 `observation.images.head_rgb`。尺寸和 mean/std 仅供演示，需按模型要求调整：

```yaml
batch_format: {type: canonical_sample, version: 2}
image_output:
  enabled: true
  resize_uint8: true
  padding: [0, 0, 224, 0]
  mean: [0.5, 0.5, 0.5]
  std: [0.5, 0.5, 0.5]
windows:
  observation: {offsets: [0]}
fields:
  - target: [images, front]
    source: {column: observation.images.head_rgb}
    window: observation
    tensor: {shape: [1, 3, 448, 224], dtype: float32}
```

完整示例：

- [libero_canonical.yaml](examples/libero_canonical.yaml)：原尺寸 uint8。
- [libero_image_output.yaml](examples/libero_image_output.yaml)：两路相机的 256×320 float32 输出，mean/std 为 0.5；不是通用 LIBERO 训练配方。
- [g05.yaml](g05.yaml)：G05 LIBERO 输入，要求 20 FPS、9 帧观测窗口和 32 帧动作窗口。
- [pi05.yaml](pi05.yaml)：PI0.5 LIBERO 输入，要求 20 FPS、当前帧观测和 50 帧动作窗口；服务完成 224×224 non-antialias resize，SDK 交付逻辑 float32 图像。

这些配置只声明 client 输出。使用前核对实际数据集的字段、shape、帧率和图像范围；模型所需的归一化、提示词处理及 loss masking 按训练项目要求执行。PI0.5 的 224×224 resize 已由服务完成，训练侧不要重复；训练前还需确认数据集提供所需归一化统计，并确认 loss 使用 `action_is_pad` 过滤 padding action。

若使用 [libero_canonical.py](examples/libero_canonical.py) helper 读取并保存样本，必须显式传入当前用户可写、尚不存在的输出文件，例如 `python examples/libero_canonical.py --output "$HOME/modalink-samples/libero_sample_100.pt"`。该 helper 会实际读取数据，应先完成凭据、训练登记及 rank 配置；不要依赖默认 `/output` 目录可写。

**不要在训练端重复已配置的 resize、图像 padding、`/255` 或 mean/std 标准化。** 对声明为 `float32` 的相机，SDK 派生独立的 `uint8` 服务请求，保留 resize、antialias、padding 和目标尺寸，服务端不执行 mean/std；浮点 resize 的结果在传输前量化为 uint8，因此不保证与旧的全浮点路径逐位一致。接收、预取及共享内存 IPC 均保留 uint8；DataLoader worker 在用户 preprocess/collate 之前恢复 float32、除以 255 并应用 YAML mean/std。直接迭代或 `num_workers=0` 时在调用线程恢复。显式 `uint8` 请求保持原样，不除以 255、不归一化；state/action 等非图像值不变。公开 spec 和 checkpoint 仍绑定原始 dtype、mean/std，wire schema 和 dtype 校验保持严格，协议仍为 HC v1。 其他模型专用 crop、动作维度 padding、动作转换或提示词处理按实际要求执行一次。修改图像配置后创建新的 loader，并确认返回的 shape、dtype 和数值范围符合模型要求。

服务端须支持经过处理的 uint8 相机请求；只支持 HC v1（HC1）不足以保证兼容。部署前向服务方确认能力，保留原 float32 YAML 与相同的训练窗口、采样和模型工作负载，不为适配传输而修改逻辑输出要求。

连接地址和 TLS 仍使用 client 连接配置；rank、world size 与预取仍使用 loader 的相应参数，不加入单样本 YAML。

### 文本与 JSON 字段

```python
from remote_loader import CanonicalSampleSpec

spec = CanonicalSampleSpec.from_dict({
    "batch_format": {"type": "canonical_sample", "version": 2},
    "fields": [
        {"target": ["task"], "source": {"column": "task"}, "value_type": "string"},
        {"target": ["events"], "source": {"column": "event_index"}, "value_type": "json"},
    ],
})
```

此例要求源提供 task 与 event_index。`string` 不把编号或 null 转为文字；`json` 保留列表、嵌套对象、null 和空容器等 JSON 值，不接受 NaN/Infinity。

物理文本或 JSON 列需要时间窗口时，使用 `value_type: json` 加 `window`，按窗口偏移顺序返回 T 个源值组成的列表；`value_type: string` 只表示单个字符串。`is_pad` 和 `temporal_mask` 可引用有窗口的 tensor 或 JSON 列，但被引用字段的偏移序列必须完全一致，不能只比较长度；mask 自身仍为 `[T]` bool tensor。

如需程序化修改，使用 `spec.to_request()` 得到声明，修改后再用 `BatchSpec.from_dict()` 构建；`spec.output` 只用于查看，不放回 YAML。已创建 loader 的请求不会随外部 spec 修改而变化。

## 4. 接入训练

直接读取即可用于已有训练流程，PyTorch DataLoader 适配是可选项。SDK 不要求特定 processor 类；`preprocess` 与 `collate_fn` 均按模型需要选择，也可以都不提供。

### 训练框架要求 PyTorch DataLoader 时

不传参数时，仍逐个返回未经本地处理的 VLA 样本：

```python
from remote_loader import RemoteDataLoader
from remote_loader.pytorch import close_torch_dataloader

with RemoteDataLoader.from_yaml("batch.yaml") as loader:
    dataloader = loader.to_torch_dataloader()
    try:
        sample = next(iter(dataloader))
    finally:
        close_torch_dataloader(dataloader)
```

需要本地组批时，设置正整数 `batch_size`：

```python
from remote_loader import RemoteDataLoader
from remote_loader.pytorch import close_torch_dataloader

# 前提：所选字段适合 PyTorch 默认 collation。
with RemoteDataLoader.from_yaml("batch.yaml") as loader:
    dataloader = loader.to_torch_dataloader(batch_size=8)
    try:
        batch = next(iter(dataloader))
    finally:
        close_torch_dataloader(dataloader)
```

未传 `collate_fn` 时使用 PyTorch 默认 collation；它不一定适合包含 null、可变长度 JSON 或模型专用结构的样本。请按实际字段和模型选择自定义 collation，不要把该例当作通用训练输入配方。

| 参数 | 用法 |
| --- | --- |
| `batch_size=None` | 不本地组批；正整数才组批 |
| `preprocess=None` | 可选逐样本 callable；即使不组批也会执行 |
| `collate_fn=None` | 可选的样本列表组批 callable；提供时必须同时设置正整数 `batch_size` |
| `num_workers=0` | 必须是非负整数，不接受布尔值；0 在创建进程处理，正整数使用原生 PyTorch worker |
| `pin_memory=False` / `pin_memory_device=""` | 按训练环境需要选用 |

若同时使用 preprocess 与本地组批，先逐样本处理，再 collate。VQA 已是完整 batch，`batch_size/preprocess/collate_fn` 必须均保持 `None`。

`num_workers=0` 是默认值，helper 在创建进程直接读取且不启动本地 IPC；正整数启用原生 DataLoader worker。helper 的其他参数直接传给 PyTorch，例如 `prefetch_factor`、`persistent_workers`、`multiprocessing_context`、`worker_init_fn`、`timeout` 和 `drop_last`。仅在 `num_workers>0` 时设置 worker 预取、持久化和启动方式；透传参数由 PyTorch 校验。helper 构造失败或本地 bridge 线程创建/启动失败时，会立即释放已创建的本地资源，不关闭调用方持有的远端 source。

helper 返回 PyTorch `DataLoader` 的子类，自动以弱引用跟踪它创建的迭代器，兼容 `persistent_workers=False` 和 `True`。停止迭代后调用 `close_torch_dataloader(dataloader)`，先取消取数等待，再回收所有仍存活的迭代器所属 worker；创建过程中遇到关闭的迭代器也会被清理。若关闭时 persistent iterator 正在重置，会向其实际消费队列发送原生重置确认，再回收 worker；这只发生在关闭路径，不增加锁或改变正常取样流程。该调用可重复执行，关闭后再次 `iter(dataloader)` 会报错。默认同时关闭关联的 `RemoteDataLoader`；仅当后续仍要使用该 `RemoteDataLoader` 时传 `close_source=False`。

也可以完全由训练项目创建原生 DataLoader。提前结束时保存实际迭代器，并传给 `close_torch_dataloader(dataloader, iterator=iterator)`；若自行创建了多个活动迭代器，需分别传入清理。以下示例只读取有限 batch，不运行模型：

```python
from torch.utils.data import DataLoader
from remote_loader import RemoteDataLoader
from remote_loader.pytorch import RemoteIterableDataset, close_torch_dataloader


def main():
    with RemoteDataLoader.from_yaml("batch.yaml") as remote:
        dataset = RemoteIterableDataset(remote)
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            num_workers=8,
            prefetch_factor=2,
            persistent_workers=True,
            multiprocessing_context="spawn",
        )
        iterator = None
        try:
            iterator = iter(dataloader)
            for index, batch in enumerate(iterator, start=1):
                print(f"batch={index} fields={list(batch)}")
                if index == 10:
                    break
        finally:
            close_torch_dataloader(dataloader, iterator=iterator)


if __name__ == "__main__":
    main()
```

需要模型专用 CPU 处理时，向 `RemoteIterableDataset(remote, preprocess=...)` 传入模块顶层 callable；`collate_fn` 仍传给原生 DataLoader。完整命令行示例见 [examples/pytorch_multiworker.py](examples/pytorch_multiworker.py)。使用 helper 时同样可以传 `num_workers=8, prefetch_factor=2, multiprocessing_context="spawn"`。

- 网络接收、buffer、统计和读取位置留在创建 loader 的进程；worker 不继承或序列化这些对象用于取样，也不另建远端会话。SDK 不创建 CPU 处理进程池。
- worker 经各自独立、带随机密钥认证的 loopback IPC 连接取样；创建进程仅串行取出 sample，传输编码与复制不持有取样锁，CPU 处理与组批仍在原生 worker 中进行。控制通道使用 pickle 传递有大小上限的元数据与描述符，不再用 pickle 搬运 CPU tensor 存储。网络 JSON 限额保持不变；本地 pickle 预算根据该限额及 tensor 数量、维数另行推导编码开销，由写端、读端和控制帧统一采用，不把两种编码的字节数直接等同，也不增加用户配置项。
- 每个活动 worker 连接独占一个可复用共享内存槽；创建进程写入 tensor 存储，worker 先复制到自身独占的 CPU 存储，再执行 preprocess/collate。后续取样或关闭不会覆盖已返回的 sample 或其视图。这是两段显式存储复制，不是零拷贝；组批和设备搬运仍可能另外分配/复制。
- worker 在下一次请求中确认上一项已完成私有复制；槽需要扩容时先释放 worker 旧映射并确认，再替换槽及代次。固定大小数据复用容量，不维护旧槽池，不增加逐样本跨进程锁。worker 只关闭映射，创建进程负责解除共享内存名称；异常、提前退出及 `close_source=False` 也通过关闭函数回收槽。
- 本地 IPC 面向同一可信训练作业，不是多租户服务；目前没有认证前连接配额和完整握手超时隔离，不应把认证机制当成本机恶意连接的资源隔离。
- 推荐 `spawn`；入口需有 `__main__` 保护，preprocess、collate、worker_init_fn 及其状态必须可序列化。CPU worker 不应携带 CUDA 模型。`fork` 的多线程和 CUDA 限制遵循 PyTorch 与 Python，本机 Python 3.12 会给出多线程 fork 警告。
- POSIX 下，创建进程在启动 worker 前准备标准共享内存 tracker，让 worker 沿用同一 tracker；fork 子进程通过 CPython 内部接口重置继承的 tracker 锁，避免继承已消失线程的持锁状态。该处理不增加 fork 屏障锁或逐样本锁，不改变 sample 的 index、分配逻辑或槽位交接协议，也不消除其他线程库及 CUDA 的 fork 限制。
- 接收 buffer 的预取预算、共享槽容量与原生 DataLoader 的预取预算分别计算。每个活动连接保留其最大已用槽容量直到关闭，多组活动迭代器会增加连接与槽数；tensor 存储仍受原协议限额约束，本地元数据使用上述编码预算；两者都不是按上限预分配。DataLoader 还可能保留约 `num_workers * prefetch_factor` 个 batch，以及正在处理的 worker 私有 sample；`prefetch_bytes` 不是整个训练进程的 RSS 或系统共享内存上限。
- 保留原生 IterableDataset 的组批、排序和 `drop_last` 语义：尾批按 worker 处理，worker 竞争取样，不承诺切换 worker 数量后仍保持完全相同的全局顺序。提前结束可能丢弃已预取但未训练的样本。
- `num_workers`、`prefetch_factor`、`persistent_workers` 和 batch size 会影响进程数、内存占用和提前停止时尚未消费的数据量；按实际训练环境测量，不假设固定加速比例。
- 保持 loader context 存活到 DataLoader 使用结束；正常结束或提前退出时先停止迭代，再在 `finally` 中调用关闭函数。SDK helper 自动跟踪迭代器；手工创建的原生 DataLoader 使用上面的 `iterator` 参数。重新迭代不表示开始新的数据集 epoch。
- 数据已经属于当前 rank，不添加 `DistributedSampler`，也不再次按 worker ID 或 `world_size` 分片。VLA 只选一处本地组批；VQA 不再次组批。

### 搬运设备与选择 dtype

`loader.to_device(value, device, non_blocking=True)` 递归搬运字典、列表和元组中的 tensor，不改变 dtype、数值范围或组批方式，也不会自动选择本机 CUDA 设备。可在训练代码中对 sample 或已准备好的模型 batch 调用它。

数值目标 dtype 在 `fields[].tensor.dtype` 中声明。根据模型需求修改后创建新 loader；较窄类型可能丢失精度或溢出，不能只为减少内存而忽略字段含义。

## 5. 训练中的可选操作

### 调整预取和等待时间

```python
from remote_loader import RemoteDataLoader

with RemoteDataLoader.from_yaml(
    "batch.yaml",
    prefetch_batches=2,
    batch_timeout=60.0,
) as loader:
    sample = next(loader)
```

| 参数 | 默认值与含义 |
| --- | --- |
| `prefetch_batches` | 读取已有配置，默认 8，范围 1–1024；VLA 按样本、VQA 按完整 batch 计数，不是模型 optimizer batch 数 |
| `prefetch_bytes` | 省略或 `None` 时，启动后按派生 wire 请求估算（图像按 uint8），并更新公开属性；显式正整数预算不变，单项超过该上限会失败。worker 恢复的 float32 内存不包含在此预算内 |
| `batch_timeout` | 默认 10 秒，必须为有限正数；首项到达之后，每次等待读取的超时 |

首项有独立的最长 300 秒预热等待，不受普通 `batch_timeout` 的十秒默认值限制。等待超时会抛出 `BatchTimeoutError`，不会因此自动关闭 loader，也不会消费数据，可以重试；它不限制模型计算时间。

### 查看数据供给与元信息

```python
from remote_loader import RemoteDataLoader

with RemoteDataLoader.from_yaml("batch.yaml") as loader:
    sample = next(loader)
    stats = loader.metrics()
    print(stats["batches_received"], stats["batches_consumed"])
    print(stats["queue_size"], stats["queued_bytes"])
    for endpoint, metadata in loader.dataset_metadata.items():
        print(endpoint, metadata)
```

`metrics()` 提供接收/消费数量、队列、等待和重连指标；VLA 按样本计数，VQA 按 batch 计数。这些不是模型或 GPU 性能指标。`queue_size` 是待消费数量，`available_batches` 则是剩余容量，不要混淆。

`dataset_metadata` 总是 `{control_endpoint: 元信息字典}`，单节点也相同，启动前为 `{}`。读取所得快照的修改不会影响 loader。元信息内容依数据源而异，请检查缺失 key，并保留节点与数据集上下文；不要把任意第一个节点的统计量当作所有数据的统计量。读取样本不需要额外调用 `ControlClient.get_metadata()`。

### 暂停与继续

```python
from remote_loader import RemoteDataLoader

with RemoteDataLoader.from_yaml("batch.yaml") as loader:
    sample = next(loader)
    loader.pause()
    # 暂停期间不调用 next(loader)。
    loader.resume()
    following_sample = next(loader)
```

`pause()` 用于正在运行的 loader，保留待消费数据；继续读取前先 `resume()`，且原会话必须仍有效。`close()` 可重复调用，但关闭后的 loader 不能重启，也不能通过 resume 恢复。

### 读取位置与训练 checkpoint

**当前 scheduler 的 VLA 和 VQA 服务均不支持关闭、会话过期或重启后的跨会话精确续读。** 不要把 `load_state_dict()` 当作当前部署的断点续训方案；同一有效会话内的 pause/resume 不等于跨会话恢复。

`state_dict()` 可用于记录读取位置，但保存不代表能够恢复。sample 从 client 接收队列取出时读取位置即已推进，不等待 optimizer 更新成功；多 worker 模式下，尚在 IPC 或 DataLoader 预取中的 sample 也可能已经计入。preprocess、collate 或训练失败不会回滚。不要把失败后的读取位置与更早的模型状态当作同一步成功 checkpoint。

### 多进程训练

每个进程创建自己的 loader，同一训练的所有机器使用一致的配置。训练框架负责启动进程和设置环境，SDK 不创建训练进程组或选择模型设备。

| 参数 | 取值顺序与要求 |
| --- | --- |
| `rank` | 显式参数 → `RANK` → 0；满足 `0 <= rank < world_size` |
| `world_size` | 显式参数 → `WORLD_SIZE` → 已保存的 `num_nodes * gpus_per_node` → 1；已保存训练规模时必须与其一致 |
| `local_rank` | 显式参数 → `LOCAL_RANK` → 0；非负整数，用于本机 GPU 资源发现 |

`LOCAL_RANK` 不从全局 rank 推导。例如两机各两卡，第二台机器的全局 rank 是 `2/3`，本机 local rank 仍是 `0/1`。各机器都要确认本机节点就绪。

使用 Accelerate 时，可继续包装模型、优化器等，但不要将此 DataLoader 传给 `accelerator.prepare()` 或 `prepare_data_loader()`，避免再次分配数据。普通 DDP 同样不添加 sampler。各 rank 的训练步数和有限流结束策略需统一，不能让一个 rank 提前结束而其他 rank 继续等待梯度同步。

## 6. 读取时的常见问题

| 现象 | 处理 |
| --- | --- |
| 刚创建 loader 就 next 报错 | 使用 with，或先 start/iter |
| 暂停后 next 报错 | 先 resume；关闭的 loader 则需新建 |
| `StopIteration` | 本次流结束或 loader 已关闭；需要新一轮读取时创建新对象 |
| `BatchTimeoutError` | 结合队列、等待指标和节点状态排查；可继续重试，不等于 loader 已关闭 |
| `ConfigurationError` | 检查 YAML 字段、shape、dtype 和连接配置；本地解析通过不保证源字段存在 |
| `SchemaMismatchError` | 核对请求与实际数据的字段、类型和形状 |
| 认证/连接失败 | 检查实际环境中的 Token、训练登记、本机 LOCAL_RANK 和网络；操作见 README |
| 默认 collation 报错 | 检查 null、可变长度 JSON 等字段，按模型需要选用 preprocess/collate_fn |
| 模型输入报错 | 核对 key、shape、dtype、文本处理及图像模式，避免重复预处理 |

公开异常从 `remote_loader.exceptions` 导入；暂停等状态问题可按 `SessionStateError` 处理。构造参数或 callback 类型错误也可能是 `ValueError`、`TypeError`，并非全部归为 `ConfigurationError`。

## 7. 训练资源管理

`with`、`start()` 和 `close()` 只管理数据读取会话，**不登记或释放训练资源**。训练参数与数据集选择在 client 配置中管理，单样本输出格式另写 YAML。

### 在 TUI 中设置与启停服务

“启动训练”中编辑训练参数后，选择“开启服务”登记并等待本机就绪；取消确认不回滚已保存设置，Ctrl+C 停止等待也不释放资源。修改本地配置不会改变已登记的训练。

整项训练结束时，先停止所有读取并关闭 loader，再选择“关闭服务（释放全部 dataloader 资源）”并确认。该操作释放整个训练的数据服务资源，不只释放本机一个 GPU。

### 使用 CLI

各进程退出后，在各自所在机器对每个本机 GPU 显式释放，例如 GPU 0：

```bash
remote-loader training release-rank --local-rank 0
```

必须覆盖**各主机各 GPU**，不能从一台机器用全局 rank 释放另一台机器的资源。命令不会自动读取 `LOCAL_RANK` 作为参数；脚本应显式传 `--local-rank "$LOCAL_RANK"`。

作业管理流程也可在所有读取已停止后整体结束训练：

```bash
remote-loader training end
```

CLI 释放命令立即执行，请先核对当前训练 ID；`end` 不应放在每个 rank 的普通退出钩子中。完整登记、状态查询、连接覆盖参数见[安装与接入](skill/modalink-client/references/deployment.md)，自动化模板见[训练脚本](skill/modalink-client/references/training-scripts.md)。
