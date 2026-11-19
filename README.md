# Modalink 数据加载器客户端

remote-loader-client 用于从 Modalink 数据服务读取 VLA 单样本或 VQA batch。通过 YAML 选择字段与输出格式，再用 Python SDK 接入训练；CLI 配置连接与 mTLS 证书，TUI 提供训练资源管理和诊断。

## 安装

SDK、CLI 和 TUI 的最低 Python 版本为 3.10。先激活实际训练环境，克隆公开仓库的 `main` 分支并安装：

```bash
git clone --branch main https://github.com/Xiaoyu1125/remote-loader-client
cd remote-loader-client
python -m pip install ".[interactive]"
```

`interactive` 提供终端菜单。接收数据还需要该环境中已安装与训练项目匹配的 PyTorch，client 不会代为安装。完整准备步骤见[安装与接入](skill/modalink-client/references/deployment.md)。
CRC32 校验使用随 client 一起安装的 `zlib-ng`，无需额外配置。

wheel 不包含本仓库的 YAML、skill 或辅助脚本；使用它们时需取得源码仓库或单独交付文件，并确认实际路径。更新源码不会更新另一环境已安装的包；安装新版本后，重启仍加载旧版本的训练进程。

确认当前训练 Python 加载的版本与位置：

```bash
python -c "import remote_loader; print(remote_loader.__version__); print(remote_loader.__file__)"
remote-loader --help
```

## 快速开始

公开仓库及其构建的包不含证书、私钥或 Token。先通过服务方的私有渠道取得有效的 Loader 服务端 CA、客户端证书、客户端私钥和 Modalink Token，以及实际 Scheduler 地址、数据集 ID/版本。确认训练机器能访问服务所在网络；未接通私网时，先看下文“私网连接”。

**首次使用 TUI 或 SDK 前必须配置完整的三项绝对 TLS 文件路径及实际 Scheduler。** 将下列路径替换为私有凭据在本机的绝对路径，将 `SCHEDULER_ENDPOINT` 设为服务方提供的实际地址：

```bash
remote-loader config edit \
  --scheduler-endpoint "$SCHEDULER_ENDPOINT" \
  --node-control-scheme https \
  --node-control-port 8080 \
  --tls-ca-file /etc/modalink/server-ca.crt \
  --tls-client-cert-file /etc/modalink/gpu-client.crt \
  --tls-client-key-file /etc/modalink/gpu-client.key
remote-loader token save
remote-loader config show
```

控制端口也应按实际服务调整。三项路径必须同时设置，且训练进程可读；不要将凭据放入源码目录。已有完整自定义配置会被使用，无需重新获取包内资源。Loader 控制 HTTPS 和数据 TLS 均使用这组 mTLS 凭据；Scheduler 的 HTTP/HTTPS 及 TLS 校验设置独立，详见[部署指南](skill/modalink-client/references/deployment.md)。

### 使用 TUI

```bash
remote-loader interactive
```

完成上述配置后，在交互终端直接运行 `remote-loader` 也会进入菜单。client 支持识别训练机器所在的云；未显式指定 Scheduler 时，自动模式按该云选择下表中的地址。这只选择 Scheduler，不提供任何凭据，不能替代前面的私有凭据配置。

| 云厂商 | 自动 Scheduler |
| --- | --- |
| 腾讯云 | `http://10.110.0.2:8080` |
| 火山引擎 | `http://10.100.0.2:8080` |
| 百度智能云 | `http://10.100.0.2:8080` |

训练容器缺少云标识时使用配置里上次识别的云。显式填写的 Scheduler 地址和完整的自定义证书路径优先；跨云或更换服务时须核对实际 Scheduler 及其 Loader 信任链，不会自动获得新的证书或身份。

#### 保存 Token

选择“保存 Token”，输入服务端提供的 Token。默认保存到 `~/.config/remote-loader/credentials.json`，不要放入代码库或 batch YAML。

#### 启动训练

在“启动训练”中填写：

| 设置 | 含义 |
| --- | --- |
| `training_id` | 本次训练 ID，不与同一 scheduler 上其他训练重复 |
| `num_nodes` / `gpus_per_node` | 训练机器总数 / 每台机器的 GPU 数；乘积等于 `WORLD_SIZE` |
| `datasets` | 普通数据集的 `id`、`version` 和可选 `weight` |
| `cpu_per_gpu` | 可选的每 GPU 普通数据服务 CPU 数量，留空由 scheduler 决定 |
| `vqa` / `vqa_datasets` | 是否启用 VQA，以及对应数据集 |

新增数据集的版本默认 `2`，请按实际版本确认。普通数据集权重留空按 `1`；VQA 权重必须全部填写或全部留空。编辑会立即保存，取消最终登记不会撤销已保存设置。

选择“开启服务”并确认后，TUI 登记资源并等待**本机各 GPU** 的数据节点就绪，不代表其他机器也已就绪。各训练机器需分别检查。Ctrl+C 只停止等待，不释放已申请资源；模型训练仍由你的训练脚本启动。

使用结束后，先停止所有读取并关闭 loader，再选择“关闭服务（释放全部 dataloader 资源）”。确认后释放整项训练的数据服务资源，本地配置保留。

### 使用 CLI

CLI、TUI 和 Python SDK 使用相同的已保存连接配置与凭据。三个 `tls_*` 路径为空不能连接本公开发行版的 Loader；必须使用上面的完整自定义配置。业务请求还须提供 Modalink Token。

迁移已有 schema 6 配置时，将 `LOADER_PRIVATE_IP` 设为分配给作业的 CPU Loader 私网 IP，并确认服务端证书覆盖该 IP。以下三项路径替换为本机有效凭据的绝对路径：

```bash
remote-loader config migrate \
  --control-endpoint "https://${LOADER_PRIVATE_IP}:8080" \
  --tls-ca-file /etc/modalink/server-ca.crt \
  --tls-client-cert-file /etc/modalink/gpu-client.crt \
  --tls-client-key-file /etc/modalink/gpu-client.key
```

命令在写入 schema 7 前把原文件原样备份为目标配置路径旁的 `client.json.schema6.bak`，不会覆盖已有备份。它保留训练、放置、预取及 Scheduler 地址/端口，节点控制协议改为 HTTPS，TLS 校验恢复开启；旧 HTTP Beanfleet 地址需迁移后手动重配。失败时原配置不变。完整备份和恢复说明见 [部署指南](skill/modalink-client/references/deployment.md)。
```bash
remote-loader token save
remote-loader config show
```

迁移后核对实际 Scheduler；需要修改时仍同时保留完整的三项绝对 TLS 路径。以下命令使用服务方提供的 `SCHEDULER_ENDPOINT`：

```bash
remote-loader config edit \
  --scheduler-endpoint "$SCHEDULER_ENDPOINT" \
  --node-control-scheme https \
  --node-control-port 8080 \
  --tls-ca-file /etc/modalink/server-ca.crt \
  --tls-client-cert-file /etc/modalink/gpu-client.crt \
  --tls-client-key-file /etc/modalink/gpu-client.key
```

`token save` 默认隐藏输入，也支持 `--token-file PATH`。准备 `training.yaml`，将 ID、数量和数据集替换为实际值：

```yaml
training_id: your-training-id
num_nodes: 1
gpus_per_node: 1
datasets:
  - id: your-dataset-id
    version: 2
```

```bash
remote-loader training register training.yaml
remote-loader training status
remote-loader diagnose connection
```

登记成功后保存这组训练设置，但 `register` 不等待节点就绪，`status` 也不是连接检查。连接诊断读取当前训练的全部节点分配，从当前机器检查各 rank、各训练主机对应的普通节点及启用的 VQA 节点，不再由 `LOCAL_RANK` 限定范围；多机训练应在每台训练机器分别执行。该命令无需设置 `RANK`、`WORLD_SIZE` 或 `LOCAL_RANK`。登记结果不明确时，先用原训练 ID 查询状态，不要直接换 ID 重复申请。

## 数据接口

数据集 ID、版本和权重在训练登记中选择；YAML 描述需要读取的字段和输出格式。完整字段参考见 [BatchSpec](skill/modalink-client/references/batch-spec.md)，SDK 用法见[用户接口文档](user-guide.md)。

本文中的 `batch.yaml` 指用户自行创建或通过私有渠道取得的配置文件；仓库和 wheel 均不提供示例 YAML。请依据上述字段参考、实际数据集能力和模型输入要求准备配置，并将代码中的路径替换为实际文件路径。

### canonical v2 与图像输出

VLA 使用 `batch_format: {type: canonical_sample, version: 2}`，通过 `fields`、`windows` 等声明单样本；tensor 的 shape 不含 batch 维。配置中的 task、mask 和相机列必须由实际数据集提供，不能假定任意数据集都具备这些字段。

按模型要求设置观测与动作窗口、帧率、图像尺寸及归一化。需要原尺寸 uint8 时显式设置 `image_output: {enabled: false}`；需要处理后的 float32 图像时，按模型设置目标尺寸和 mean/std。训练前确认数据集提供所需归一化统计，并按模型要求使用 `action_is_pad` 过滤 padding action。

图像处理默认开启，`image_output.antialias` 默认 `true`；float32 路径可设置 `false` 以匹配 PyTorch/ATen non-antialias bilinear。图像 `tensor.shape` 是含补边的最终画布尺寸。完整 dtype、padding、归一化和 hash 规则见[用户接口文档](user-guide.md)。**不要在训练端重复已配置的 resize、padding、`/255` 或 mean/std 标准化。** YAML 中相机的 `float32` 是逻辑输出：SDK 依据节点 metadata 的 `camera_keys` 派生 uint8 wire 请求，接收、预取和共享内存 IPC 保持 uint8，在 DataLoader worker 的用户 preprocess/collate 之前恢复 float32、`/255` 和 mean/std（直接迭代及 `num_workers=0` 在调用线程恢复）。服务保留原 resize/antialias/padding，浮点 resize 后量化为 uint8；显式 uint8 请求及非图像字段不变。公开 spec/checkpoint 保留原 dtype 与归一化，传输仍严格使用 HC v1。

服务端必须支持**经过处理的 uint8 相机请求**，仅支持 HC v1（HC1）传输协议不足以保证兼容。部署前向服务方确认该能力；无需改变原 YAML 的 float32 声明、训练窗口或模型工作负载，显式 uint8 请求保持原语义。

### 取一个 VLA canonical sample

完成登记并确认节点就绪后，在能够找到对应 YAML 的目录运行：

```python
from remote_loader import RemoteDataLoader

with RemoteDataLoader.from_yaml("batch.yaml") as loader:
    sample = next(loader)
    print(sample)
```

每次 `next(loader)` 返回一个样本；持续读取可用 `for sample in loader`。`with` 自动启动并关闭 loader，但**不会释放训练资源**。

可以直接使用返回数据；模型需要额外处理或本地组批时，再选用 `to_torch_dataloader()`。`preprocess` 和 `collate_fn` 均可选，具体用法与限制见[用户接口文档](user-guide.md)。

`to_torch_dataloader()` 支持原生 PyTorch 的 `num_workers`、`prefetch_factor` 和 `persistent_workers` 参数，并自动跟踪创建的迭代器。`num_workers=0` 默认在创建进程直接读取，不开启本地 IPC；正整数 worker 通过认证 loopback 控制通道与各连接独占的可复用共享内存槽取样，并执行 CPU preprocess/collate，远端连接和接收 buffer 仍由每个训练 rank 的创建进程独占。正常结束或提前结束时，先停止迭代，再在退出 loader context 前调用 `remote_loader.pytorch.close_torch_dataloader(dataloader)`；仅当后续仍要使用同一个 `RemoteDataLoader` 时传 `close_source=False`。调用后该 PyTorch DataLoader 不可继续使用。手工创建原生 DataLoader 时，关闭函数需传入 `iterator=iterator`，用法见[用户接口文档](user-guide.md)。控制通道仅传元数据与确认消息；CPU tensor 存储经共享槽复制到 worker 私有内存后才进入 preprocess/collate。仍有存储复制、元数据序列化及原生组批开销，不保证 GPU overlap 或固定加速比例。

### VQA：读取完整 batch

训练登记时启用 VQA 并填写 `vqa_datasets`，然后使用：

```python
from remote_loader import RemoteDataLoader, VQABatchSpec

with RemoteDataLoader(VQABatchSpec(batch={"size": 32})) as loader:
    batch = next(loader)
    print(batch["pixel_values"].shape, batch["samples"])
```

每次返回完整六字段 batch，图像为 float32 `[B, N, 3, 256, 256]`，像素范围 `[-1, 1]`。不要再本地组批或重复图像预处理；文本及其他模型输入仍需按模型要求接入，并非任意模型都能直接消费。完整格式见[用户接口文档](user-guide.md)。

### 查看实际数据

```bash
remote-loader receive batch.yaml
```

该命令会实际读取数据，显示字段类型、tensor shape/dtype 和数值摘要，按 Ctrl+C 停止。摘要不等于完整样本；需要检查 JSON、空容器或嵌套结构时，直接查看 SDK 返回值。

## 连接数据服务器

可用 `REMOTE_LOADER_CONFIG` 指向另一份 client 配置；它不改变 Token 的默认保存路径。不同训练使用独立配置与 ID，同一训练的各机器保持参数一致。完整安装、连接与 TLS 设置见[安装与接入](skill/modalink-client/references/deployment.md)。

多进程训练中，每个进程创建自己的 loader，使用全局 `RANK/WORLD_SIZE` 和本机 `LOCAL_RANK`。例如两机各两卡，第二台机器的全局 rank 是 `2/3`，本机 `LOCAL_RANK` 仍是 `0/1`；它用于本机资源发现，不能用全局 rank 代替。不要对已分配的数据再次分片。

每个进程停止读取并关闭 loader 后，需在**所在机器**释放对应本机 GPU 的资源；必须覆盖各主机的各个 GPU：

```bash
remote-loader training release-rank --local-rank 0
```

若要结束整项训练，停止所有读取后执行以下命令，或使用 TUI 的“关闭服务”：

```bash
remote-loader training end
```

`end` 释放全部训练数据服务资源，不是每个进程的常规退出命令。自动化启停模板见[训练脚本](skill/modalink-client/references/training-scripts.md)。当前 scheduler 服务不支持关闭、会话过期或重启后从读取 checkpoint 精确续读；同一有效会话内的暂停/继续是另一回事。

## 私网连接

跨 VPC 访问前，需确认双方网段不冲突、路由双向可达，并允许实际使用的服务端口。火山引擎同地域、跨账号接入由双方手动配置：

1. Modalink 提供账号 ID、CEN ID、服务子网和访问地址；客户提供账号 ID、地域、VPC ID 和实际访问服务的子网。
2. 客户在 VPC 控制台授权指定 CEN，Modalink 将客户 VPC 接入 CEN 并配置服务侧访问。
3. 双方检查路由和安全组，从实际训练环境验证服务端口，再运行 client 连接诊断和有限读取。

详细步骤见[私网接入说明](skill/modalink-client/references/network.md)。

## 查询及诊断

TUI 的“查看”提供配置、当前传输和最近传输结果；“诊断”提供环境与连接检查。对应 CLI：

```bash
remote-loader config show
remote-loader training status
remote-loader transfer current
remote-loader transfer last
remote-loader diagnose environment
remote-loader diagnose connection
```

传输记录属于本机当前系统用户，多机训练需分别查看。连接诊断对当前训练已分配的节点去重后，从当前机器检查控制服务健康状态；仅显示汇总结果，不列出逐机器连接状态。CLI JSON 中的 `node_connections` 包含总节点数和连接成功数。无节点、分配信息不完整、节点尚未就绪或任一连接失败时，不报告全部成功。检查不申请资源、不读取样本，也不能替代其他训练主机自身的网络与数据读取验证。TUI 启动后的等待仍只检查本机各 GPU，SDK 仍按本机 `LOCAL_RANK` 发现节点。实际读取、超时与模型输入问题见[用户接口文档](user-guide.md)。
