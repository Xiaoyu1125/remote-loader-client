# 编写和修改 BatchSpec

根据实际数据集的字段、形状、相机和模型要求自行编写 YAML，或使用通过私有渠道取得的配置。公开仓库和 wheel 均不提供 batch 请求 YAML；本文的 `batch.yaml` 表示用户提供的输入文件，请使用其实际路径。wheel 也不含辅助脚本。安装与数据集登记见 [deployment.md](deployment.md)，SDK 调用见 [api.md](api.md)。

## Canonical VLA 声明

当前 VLA 使用下面的声明；每次读取返回一个样本，形状不含训练 batch 轴。数据集及版本在训练登记中选择，不写入本 YAML。

| 顶层参数 | 填写方法 |
| --- | --- |
| `batch_format` | 固定为 `{type: canonical_sample, version: 2}` |
| `sampling` | 可省略。`shuffle_seed` 为整数，默认 42；`drop_n_first_frames`、`drop_n_last_frames` 为首尾丢帧数，默认均为 0，范围 0–1000000 |
| `require_metadata` | 可省略；`fps` 为有限正数，要求数据集满足该帧率，不执行重采样 |
| `windows` | 具名时间窗口，定义相对当前帧的整数偏移 |
| `fields` | 1–256 个输出字段，每项指定 `target`、`source`，并在 `tensor` 与 `value_type` 中二选一 |
| `image_output` | 所选 RGB 相机共用的图像输出设置，默认开启；见下文 |

不要添加 `batch`、`dataset` 或请求级 `output`。`spec.output` 是解析后供检查的输出视图，不是此 YAML 的配置部分。

### 字段与来源

- `target` 是由非空字符串组成的输出路径，最多 16 段，每段最多 512 字符。`[observation.state]` 表示字面键 `"observation.state"`，`[observation, state]` 表示嵌套键；二者不同，但不能在同一声明中同时使用这类点连接后相同的路径。目标不能重复，也不能同时定义某个键及其子路径。
- `source.column` 是真实源列的完整字面名称，点号不表示拆分源列。改 `target` 不会改变读取哪一列。
- `window` 引用已定义的窗口名，只用于 `column` 来源；不填时读取当前行，不加时间轴。
- `tensor` 只填写 `shape` 和 `dtype`；非 tensor 使用 `value_type: string` 或 `value_type: json`。
- `add_feature_axis` 默认 `false`。标量源需要末尾特征轴时显式设为 `true`，同时将目标 shape 写为末尾含 1 的形状，例如单步窗口的标量从 `[1]` 变为 `[1, 1]`；仅适用于 `column` tensor。

每个 `source` 选择一种形式：

| 来源 | 写法与前提 |
| --- | --- |
| 真实列 | `{column: action}`；源列必须存在且符合声明的类型和形状 |
| 任务查表 | `{index_column: task_index, lookup: {table: tasks, key_column: task_index, value_column: task}}`；使用该数据集自己的任务表，输出必须为 `value_type: string` |
| 常量 | `{constant: 20}`；值必须是有限 JSON。字符串输出必须给字符串，tensor 常量必须是与 shape 一致的数值或布尔值；只能用于已确认的固定值，不能代替缺失的任务或 mask |
| 有效时间位置 | `{temporal_mask: {fields: [[action]], reduction: equal}}`；`true` 表示有效位置 |
| 填充时间位置 | `{is_pad: {fields: [[action]], reduction: equal}}`；`true` 表示填充位置，不是有效位置 |
| 文本是否非空 | `{nonempty: [task]}`；引用已声明的字符串目标，输出用标量 `bool` tensor 或 `value_type: json` |

两种时间 mask 的 `fields` 引用的是 **target 路径**，不是源列名；必须引用已声明、带窗口的列 tensor，所有被引用字段的偏移序列必须一致。mask 自身不填 `window`，shape 为 `[T]`，dtype 为 `bool`。`reduction` 可选 `all`、`any`、`equal`，多字段应按所需语义选择；要求各字段时间有效性一致时使用 `equal`，不要用它掩盖不一致的数据。直接读取数据集已有 mask 时仍用 `source.column`，并确认其真假含义与窗口；声明不会让不存在的源列自动出现。

## 时间窗口与 shape

窗口名为 1–512 字符；每个窗口选以下一种写法，不能混用：

- `offsets: [-1, 0, 1]`：1–4096 个不重复的整数偏移。
- `{start: 0, step: 1, count: 20}`：三个整数都必填，`step` 非零，`count` 为 1–4096；本例取当前帧起连续 20 帧。

这是帧偏移，不是秒数或重采样频率。带窗口的 tensor 第一维必须等于窗口点数，只有一个点也保留 `T=1`。shape 的各维是 1–1000000 的具体整数，最多 15 维；标量使用 `[]`。SDK 不自动拆分动作向量、补轴或去掉长度为 1 的维度。

下面是一份完整的 20 帧动作声明，前提是数据集提供每帧 7 维的 `action`、8 维的 `observation.state` 和标量 `index`：

```yaml
batch_format: {type: canonical_sample, version: 2}
image_output: {enabled: false}
sampling:
  shuffle_seed: 42
  drop_n_first_frames: 0
  drop_n_last_frames: 0
windows:
  observation: {offsets: [0]}
  action: {start: 0, step: 1, count: 20}
fields:
  - target: [action]
    source: {column: action}
    window: action
    tensor: {shape: [20, 7], dtype: float32}
  - target: [observation.state]
    source: {column: observation.state}
    window: observation
    tensor: {shape: [1, 8], dtype: float32}
  - target: [index]
    source: {column: index}
    tensor: {shape: [], dtype: int64}
  - target: [action_is_pad]
    source: {is_pad: {fields: [[action]], reduction: equal}}
    tensor: {shape: [20], dtype: bool}
```

修改动作长度时同时修改窗口 `count`、动作 shape 第一维和对应 mask 的长度。首尾丢帧与越界填充不是同一设置，训练时应按真实 mask 排除填充位置。

## 每个字段的 dtype

`fields[].tensor.dtype` 声明该字段的目标类型，支持 `bool`、`uint8`、`int8`、`int16`、`uint16`、`int32`、`uint32`、`int64`、`uint64`、`float16`、`bfloat16`、`float32`、`float64`。

按字段含义选择：动作和状态通常按模型要求使用浮点类型，索引与 mask 使用对应整数或布尔类型。非图像类型转换不等于数值归一化，窄类型可能舍入、截断或溢出。修改某字段的 `tensor.dtype` 后检查实际返回值，不使用全局 dtype 覆盖。

## 会话级图像输出

`image_output` 必须是对象，对所选 RGB 相机共同生效；未选图像字段时不影响其他字段。支持以下参数：

| 参数 | 默认值与含义 |
| --- | --- |
| `enabled` | `true`：按目标尺寸输出 `float32`，像素缩放到 0–1 后可继续标准化。设为 `false` 时保持源尺寸 `uint8` |
| `resize_uint8` | `false`；设为 `true` 表示 resize 后保留整数像素精度，再转为 float32 并缩放；根据模型原有处理方式选择 |
| `antialias` | 默认 `true`；设为 `false` 时使用 PyTorch/ATen `align_corners=false` non-antialias bilinear，仅适用于 `resize_uint8: false` 的 float32 路径 |
| `padding` | 默认 `[0, 0, 0, 0]`；顺序为 `[left, right, top, bottom]`，必须是四个非负整数，不接受 bool、浮点数、字符串或 null；在 mean/std 标准化前补零 |
| `mean` / `std` | 默认均不设置；必须同时提供，各为三个数值，转为 float32 后仍须有限，`std` 每项转后仍须非零，逐通道执行 `(x - mean) / std` |

`resize_uint8` 与 mean/std 独立选择：仅启用 `resize_uint8` 不要求 mean/std，仅设置 mean/std 也不要求启用 `resize_uint8`；不设置 mean/std 时输出保持 0–1 范围。`resize_uint8: true` 与 `antialias: false` 不能同时使用。

省略 `antialias` 与显式 `true` 会规范化为相同请求和 hash；`false` 会保留在请求中并改变 hash，但不改变输出 schema。

开启时相机字段声明 `dtype: float32`，shape 为无窗口的 `[3, H, W]` 或有窗口的 `[T, 3, H, W]`；最后两维指定含 padding 的最终画布 H/W，所有选中相机使用同一最终 H/W。内容尺寸为 `H - top - bottom`、`W - left - right`，必须都为正。关闭时设置 `image_output: {enabled: false}`，相机声明源尺寸及 `uint8`，不能再设置 mean/std、启用 `resize_uint8`、设置 `antialias: false` 或指定非零 padding。

边框未标准化时为 0，标准化后各通道为 `(0 - mean[c]) / std[c]`。例如最终 shape `[1, 3, 448, 224]` 配合 `padding: [0, 0, 224, 0]`，内容为 224×224，顶部补 224 行；mean/std 均为 `[0.5, 0.5, 0.5]` 时顶部边框为 -1，首维仍为 T。

所有尺寸、窗口和归一化参数都必须与实际数据及训练项目一致。例如，用户声明 224×224 non-antialias resize 时，由服务完成 resize，SDK 恢复逻辑 float32 图像；模型所需的归一化统计和 `action_is_pad` loss masking 仍须准备。**训练代码不要重复执行已经配置的 resize、图像 padding、像素缩放或标准化。**

float32 相机声明保持不变：SDK 派生经过处理的 uint8 请求，接收、预取与 IPC 保持 uint8，在 DataLoader worker 的用户 preprocess/collate 前恢复 float32、`/255` 和 mean/std；直接迭代及 `num_workers=0` 在调用线程恢复。显式 uint8 请求不变，训练窗口和采样语义不变。服务端必须支持此类经过处理的 uint8 相机请求；仅支持 HC v1（HC1）传输不足以保证兼容，须向服务方确认。

## 文本和 JSON 字段

用 `value_type` 声明非 tensor 数据。下面是独立的完整例子，前提是数据集提供真实 `task` 字符串和 `events` JSON 列：

```yaml
batch_format: {type: canonical_sample, version: 2}
fields:
  - target: [task]
    source: {column: task}
    value_type: string
  - target: [events]
    source: {column: events}
    value_type: json
  - target: [has_task]
    source: {nonempty: [task]}
    tensor: {shape: [], dtype: bool}
```

`string` 不把数字或 null 转成文本。`json` 保留对象、列表、数字、字符串、布尔值、null 和空容器，不允许 tensor、NaN 或 Infinity；无需额外的 length/items 包装。只有 `task_index` 时，确认数据集存在任务表再使用上表的查表来源，不要填空任务代替缺失数据。

## 调整训练 batch size

VLA YAML 只声明单样本，训练 batch size 不进入 YAML，也不加入 `tensor.shape`。按需调用 `loader.to_torch_dataloader(batch_size=32)` 在本地组批；`preprocess`、`collate_fn` 可选，默认 collation 和约束见 [api.md](api.md)。不要再次按 rank 分片。

VQA 使用单独的完整声明：

```yaml
batch_format: {type: vlm_vqa, version: 2}
batch: {size: 32}
```

`batch.size` 必须为正整数，并受图像数据大小限制（当前默认每批 2 GiB）。单图样本最多 2730 个；图像槽位更多时需进一步减小。每次读取已经是完整六字段 batch：`pixel_values` 为 float32 `[B,N,3,256,256]`、像素范围 [-1,1]；`samples` 包含 B 条文本/图像槽位数据；`action`、`action_is_pad`、`action_dim_is_pad` 均为 null；`is_vlm_batch` 为 true。N 是正整数且可变化。这不代表任何模型都能直接消费，仍须按模型适配文本和输入结构；不要在 helper 中再次组批或预处理图像。

## 校验并使用

先用实际安装的 SDK 离线解析保存后的 YAML：

```python
from remote_loader import BatchSpec

spec = BatchSpec.from_yaml("batch.yaml")
print(spec.output)
print("tensor 字节估算:", spec.estimated_batch_bytes)
```

这一步不连接服务。`spec.output` 便于检查类型与形状；完整请求用 `spec.to_request()` 查看，不要将派生输出写回 YAML。字节估算不是训练进程内存：VLA 只估算单样本 tensor，VQA 因相机数可变使用保守上限。

本地解析成功不证明源列、任务表、帧率或实际数据符合要求。完成接入后按 [api.md](api.md) 有限读取，直接检查返回对象的 shape、dtype、mask、图像范围及文本/JSON 内容。修改声明后创建新 loader；当前 scheduler 服务不支持关闭后从读取 checkpoint 精确续读，不能以保存读取状态替代这一限制。
