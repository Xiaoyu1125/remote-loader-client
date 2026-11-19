---
name: modalink-client
description: Use when installing or configuring Modalink client, inspecting VLA samples or VQA v2 batches, writing BatchSpec YAML, integrating the SDK with training, diagnosing private-network access, or preparing training lifecycle scripts.
---

# Modalink client

使用训练环境中的 Python 3.10 或以上版本，从 https://github.com/Xiaoyu1125/remote-loader-client 的 `main` 分支克隆，并在仓库根目录执行 `python -m pip install ".[interactive]"`。匹配训练项目的 PyTorch 由训练环境提供。wheel 不包含本 skill、辅助脚本或 YAML；保留源码目录布局，`SKILL_DIR` 指向本文件所在目录，`BATCH_SPEC` 指向实际 YAML。

公开仓库和包不含证书、私钥或 Token。用户须通过私有渠道取得有效 Loader CA、客户端证书、客户端私钥和 Modalink Token；在使用 TUI/SDK 前按 [deployment.md](references/deployment.md) 用 `remote-loader config edit` 配置实际 Scheduler 及全部三项**绝对 TLS 路径**，再保存 Token。已有完整自定义配置会被使用。client 的腾讯云、火山引擎和百度智能云发现仅选择 Scheduler，不提供凭据。Scheduler TLS 与 Loader 控制/数据 mTLS 独立。火山引擎同地域、跨账号 CEN 仍需手动接入。命令在实际训练环境执行。若只能指导用户操作，提供当前步骤的命令或控制台操作，确认结果后再继续。

## 按任务读取

| 任务 | 使用的文档 |
| --- | --- |
| 安装、Token、自动连接配置、训练资源启停 | [deployment.md](references/deployment.md) |
| 编写启动和关闭脚本 | [training-scripts.md](references/training-scripts.md) |
| 修改字段、时间窗口、图像处理和类型 | [batch-spec.md](references/batch-spec.md) |
| 固定版本全量统计、独立 coordinator、统计 SDK/CLI | [statistics.md](references/statistics.md) |
| 检查数据、使用 SDK、接入训练 | 交付的 `user-guide.md`；[api.md](references/api.md) |
| 火山引擎同地域、跨账号 CEN 私网接入 | [network.md](references/network.md) |

## 配置并读取数据

1. 核对训练 Python、安装包和用户自行编写或通过私有渠道取得的 YAML，按安装说明检查实际使用的 SDK 与 PyTorch。公开仓库和 wheel 不提供 batch 或 statistics 请求 YAML；文档中的文件名仅表示用户提供的输入路径。离线检查 YAML 不需要 Token 或服务地址。
2. 核对实际 Scheduler、完整的三项绝对 TLS 路径和已保存 Token，并确认服务支持经过处理的 uint8 相机请求（仅支持 HC1 不够）。火山 CEN 尚未接通时，客户按网络指南完成 VPC 授权，Modalink 完成服务侧接入，再检查双方网段、路由和访问控制。不要输出凭证内容。
3. 在 TUI“启动训练”中设置 `training_id`、`num_nodes`、`gpus_per_node` 和数据集；需要 VQA 时同时配置 VQA 数据集。数据集版本和权重按实际训练填写。各机器使用一致的训练配置，`WORLD_SIZE` 等于 `num_nodes * gpus_per_node`。
4. 选择“开启服务”并确认。编辑会立即保存，取消开启不撤销已保存设置。TUI 只确认本机 GPU 就绪，其他机器需各自检查。CLI 自动化使用 `training register` 和 `training status`，具体步骤见安装说明；登记数据服务不会启动模型训练。
5. 先离线检查 YAML，再从每台实际训练主机或训练容器各运行一次连接诊断；该命令汇总当前训练全部已分配节点，不按 GPU 或 `LOCAL_RANK` 重复运行，也不能代替其他训练主机自身的网络检查。诊断无需设置 rank 环境变量；有限读取仍使用训练框架提供的 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`，且不要与同一本机 GPU 的训练同时读取。
6. 每个进程结束并关闭 loader 后，在其所在机器释放本机对应 GPU：`remote-loader training release-rank --local-rank "$LOCAL_RANK"`。只有结束整个训练时才使用 `training end` 或 TUI“关闭服务”；关闭 loader 本身不释放训练资源。

## 数据与训练用法

- VLA 每次读取一个 sample；当前 YAML 使用 `batch_format: {type: canonical_sample, version: 2}`，通过 `fields` 和 `windows` 声明字段与时间窗口。数据集在训练配置中选择，训练 batch size 在训练代码中设置。
- `target` 定义输出路径，`tensor.shape` 是不含 batch 维的单样本形状；标量为 `[]`，单步时间维也保留。字段和任务、掩码应来自真实数据或支持的派生声明，不填占位值来绕过缺失数据。
- `image_output` 默认开启，图像声明为目标尺寸的 `float32`；`antialias` 默认 `true`，float32 路径可设置 `false` 以匹配 PyTorch/ATen non-antialias bilinear；保留原尺寸 `uint8` 时设置 `image_output: {enabled: false}`。模型代码不要重复已配置的 resize、`/255` 或 mean/std 标准化。
- 用户的模型配置使用前须核对数据字段、shape、帧率、图像范围和训练项目要求，并准备模型所需的归一化统计和 `action_is_pad` loss masking。已配置的 resize 由服务完成，SDK 恢复逻辑 float32 图像，训练侧不要重复 resize。
- float32 相机 YAML 保持原声明；SDK 派生保留 resize/antialias/padding 的 uint8 请求，接收、预取与 IPC 保持 uint8，在 DataLoader worker 的用户 preprocess/collate 前恢复 float32、`/255` 与 mean/std（直接迭代和 `num_workers=0` 在调用线程恢复）。显式 uint8 请求和非图像字段不变，不改变窗口、采样或模型工作负载。
- `to_torch_dataloader()` 默认逐个返回样本。VLA 可通过正整数 `batch_size` 本地组批；`preprocess`、`collate_fn` 按模型需要选用，未指定 `collate_fn` 时使用 PyTorch 默认组批。自定义 `collate_fn` 须同时设置正整数 `batch_size`。
- helper 的 `num_workers=0` 默认在创建进程直接读取，不开启本地 IPC；正整数 worker 经本地认证 IPC 取样后运行 CPU preprocess/collate，创建进程仍独占接收 buffer，不另建远端 session，也不按 worker ID 再次分片。推荐 `spawn` 和模块顶层 callback；IPC 有复制成本，提前退出与每 worker 尾批遵循原生 IterableDataset 语义。helper 自动跟踪迭代器，正常结束或提前结束时调用 `remote_loader.pytorch.close_torch_dataloader(dataloader)`，然后退出 loader context。用户自行创建原生 DataLoader 时保存实际迭代器，关闭时传入 `iterator=iterator`。
- VQA 每次返回完整 batch，不再通过 helper 设置 `batch_size`、`preprocess` 或 `collate_fn`，也不要重复图像归一化；文本等模型输入仍按训练项目处理。
- 每个训练进程使用自己的 loader，不再增加 `DistributedSampler` 或让训练框架重复分片。设备搬运使用 `to_device()`，不会改变 dtype 或数值范围。
- 当前服务不支持关闭后通过读取 checkpoint 精确续读；暂停后继续使用同一个有效 loader 与重新启动训练不同。模型 checkpoint 仍由训练代码管理。

## 附带脚本

使用目标训练 Python；`SKILL_DIR` 指向源码中的 `skill/modalink-client` 或独立交付的 skill 目录，`BATCH_SPEC` 指向用户自行编写或通过私有渠道取得的 YAML 文件：

```bash
# 离线检查，不连接服务
python "$SKILL_DIR/scripts/inspect_batch.py" --spec "$BATCH_SPEC"

# 服务就绪后读取一个返回项，预览少量 tensor 数值
python "$SKILL_DIR/scripts/inspect_batch.py" --spec "$BATCH_SPEC" --receive --items 1 --values 6
```

`--items` 计数 VLA samples 或 VQA batches；摘要不是完整数据转储，核对文本、JSON 或嵌套结构时直接检查 SDK 返回值。脚本继承预取配置，新配置默认 8 个返回项。首项预热最多等待 300 秒；后续等待由 `--batch-timeout` 控制，脚本默认 60 秒。

`remote-loader diagnose connection` 从当前机器汇总检查当前训练的全部已分配节点；从每台训练主机或容器各运行一次，不按本机 GPU 重复。诊断不读取样本，通过后仍需实际读取数据。操作结束时说明已完成的检查和读取结果，不把连接诊断当作模型训练成功。
