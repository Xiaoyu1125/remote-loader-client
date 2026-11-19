# 安装、配置与分层验证

从公开仓库 https://github.com/Xiaoyu1125/remote-loader-client 的 `main` 分支安装 client。公开仓库和构建出的包不含证书、私钥或 Token；必须通过服务方私有渠道取得凭据并配置后才能连接服务。云发现只选择 Scheduler，不提供身份。已有可用私网时可直接安装，火山 CEN 手动私网接入见 [network.md](network.md)。

## 1. 核对交付和已有信息

| 项目 | 需要确认的内容 |
| --- | --- |
| 安装包与文档 | 公开仓库 `main` 分支源码及同一快照内的文档 |
| 数据请求 | 用户自行编写或通过私有渠道取得的 batch YAML 路径、可用的数据集 ID/version 和字段；公开仓库和 wheel 不提供请求 YAML |
| 运行环境 | Python 3.10 或以上的实际训练环境，以及已有 PyTorch/CUDA |
| Scheduler | 服务方确认的实际 URL/TLS 设置、本次 training_id、num_nodes、gpus_per_node |
| 节点网络 | 节点 control HTTPS 端口、数据 TLS 端口及放行范围 |
| 凭证 | 私有渠道取得的有效 Loader CA、客户端证书、客户端私钥和 Modalink Token；三项 TLS 文件均须配置为绝对路径 |

所有训练机器使用一致的训练参数，`WORLD_SIZE = num_nodes * gpus_per_node`。`RANK` 是全局进程编号；`LOCAL_RANK` 是当前主机的 GPU 索引，用于本机资源发现和释放，不能用全局编号替代。

由作业管理流程协调一次训练登记，再让各主机检查自己的资源；不要让每个进程各自重复登记。登记数据服务不会启动模型训练，也不表示节点已经就绪。数据集通过训练配置选择，batch YAML 用于描述输出，格式见 [batch-spec.md](batch-spec.md)。

兼容服务必须支持经过 resize/antialias/padding 处理的 uint8 相机请求；仅支持 HC v1（HC1）协议不足以保证兼容。向服务方确认此能力，不要通过改写原 float32 YAML 或训练工作负载绕过。SDK 在 worker 的 preprocess/collate 前恢复逻辑 float32；显式 uint8 请求不变，详见 [用户指南](../../../user-guide.md)。

## 2. 下载与安装

SDK、CLI 和 TUI 均要求 Python 3.10 或以上。先进入实际训练环境，复用已有 PyTorch/CUDA。下面使用 Bash；变量均须按交付信息设置。

~~~bash
python -c "import sys; print(sys.executable); print(sys.version)"
~~~

在实际训练环境克隆并安装，保留 `interactive` extra 以使用 TUI：

~~~bash
git clone --branch main https://github.com/Xiaoyu1125/remote-loader-client
cd remote-loader-client
python -m pip install ".[interactive]"
~~~

PyTorch 必须由训练环境提供，并与训练项目及 CUDA 匹配；client 不会代为安装。完全离线环境须事先准备源码和对应环境的依赖包。若自行构建 wheel，它同样不包含凭据；不要将私有凭据添加到源码或包中。

安装后检查实际加载的版本和位置：

~~~bash
python -c "import importlib.metadata as m; import remote_loader; print(m.version('modalink-remote-loader-client')); print(remote_loader.__file__)"
python -m remote_loader.cli.main diagnose environment
~~~

源码目录不等于已安装的包；更新安装后重启仍使用旧版本的训练进程。下文 `remote-loader` 应来自同一训练环境，不能确认时统一改用 `python -m remote_loader.cli.main`。公开仓库和 wheel 不提供 batch 或 statistics 请求 YAML，运行时 wheel 也不包含辅助脚本：`SKILL_DIR` 指向源码仓库中的 `skill/modalink-client` 或独立交付的 skill 目录，`BATCH_SPEC` 指向用户自行编写或通过私有渠道取得的 YAML。

## 3. 配置私有凭据与实际服务

在启动 TUI、SDK 或连接诊断前，先完成下方“按交付信息配置 mTLS 和端口”的 `remote-loader config edit` 命令：同时设置实际 Scheduler 和三项绝对 TLS 文件路径。已有完整自定义配置直接沿用；schema 6 配置先按迁移段落升级，再核对实际 Scheduler。随后保存 Token：

~~~bash
remote-loader token save
remote-loader config show
~~~

`token save` 隐藏输入 Modalink Token，也可使用 `token save --token-file PATH`。Token 与云账号凭证、HF Token 不能互换；保存成功不代表远程权限已验证。凭证默认保存在 `~/.config/remote-loader/credentials.json`，不要放入代码库或 batch YAML。

TUI、CLI 和 Python SDK 在配置缺失时通过下表中的实例标识选择 Scheduler。

| 云厂商 | 元数据标识 | Scheduler |
| --- | --- | --- |
| 腾讯云 | `ins-...` | `http://10.110.0.2:8080` |
| 火山引擎 | `i-...` | `http://10.100.0.2:8080` |
| 百度智能云 | `/sys/class/dmi/id/sys_vendor` 的值为 `Baidu Cloud` | `http://10.100.0.2:8080` |

腾讯和火山通过实例元数据识别，百度通过本机 DMI 厂商标识识别。自动配置省略 `scheduler.endpoint` 时按云选择 Scheduler；配置中的 `last_detected_cloud` 供训练容器读不到云标识时使用。识别到多家云会报错。`config show` 显示本次实际选择的地址。自动发现不会下载或提供 CA、客户端证书、私钥或 Token。

显式填写的 `scheduler.endpoint` 和完整自定义 TLS 路径会保留。公开发行版应先按服务方资料配置实际地址及三项路径；换云或换服务时重新核对地址与 Loader 信任链。需要回退到 `0.3.1` 时请恢复升级前的配置，旧客户端不识别 `last_detected_cloud`。

### 从 schema 6 升级

在训练环境中确认需要迁移的 schema 6 配置。设置了 `REMOTE_LOADER_CONFIG` 时使用该路径，其次为用户配置，最后为系统配置 `/etc/remote-loader/client.json`。将 `LOADER_PRIVATE_IP` 设为作业分配到的 CPU Loader 私网 IP。

~~~bash
remote-loader config migrate \
  --control-endpoint "https://${LOADER_PRIVATE_IP}:8080" \
  --tls-ca-file /etc/modalink/server-ca.crt \
  --tls-client-cert-file /etc/modalink/gpu-client.crt \
  --tls-client-key-file /etc/modalink/gpu-client.key
remote-loader config show
~~~

迁移命令先验证旧配置和本地证书，再把旧文件原文备份为目标配置旁的 `client.json.schema6.bak`（权限 `0600`），最后原子写入 schema 7。已有同名备份时命令会停止，不会覆盖。旧配置中的训练、放置、预取及 Scheduler 地址/端口会保留；节点 control 协议改为 HTTPS，Scheduler 和 HTTPS Beanfleet 的 TLS 校验恢复开启，HTTP Beanfleet 地址不会自动迁移。Scheduler 自身的 HTTP 地址目前仍可保留；它与 Loader 的两条 mTLS 连接是不同链路。检查交付的 Scheduler 地址与证书后再启动训练。

若验证或备份失败，原配置不会被修改，可修正证书路径或备份文件后重试。若写入失败，原配置仍在，备份已保留；重试前需安全地移动或归档该备份。要回退到旧客户端版本，可从备份恢复旧配置；schema 6 文件无法供新版客户端直接使用。若旧配置来自系统路径，命令不修改系统文件，而把备份和新配置写入用户配置目录；回退时移走用户配置即可重新使用原系统文件。不要把包含证书路径或训练信息的备份放入代码库。

### 按交付信息配置 mTLS 和端口

通过私有渠道取得有效凭据，将下列示例路径替换为本机文件的**绝对路径**，确保训练账号可读并限制私钥访问。公开包没有默认身份；必须同时提供 `--tls-ca-file`、`--tls-client-cert-file` 和 `--tls-client-key-file`。将 `SCHEDULER_ENDPOINT`、`NODE_CONTROL_PORT` 和 `MODALINK_TRAINING_ID` 设为本次服务的实际值，再执行：

~~~bash
remote-loader config edit \
  --training-id "$MODALINK_TRAINING_ID" \
  --scheduler-endpoint "$SCHEDULER_ENDPOINT" \
  --verify-scheduler-tls \
  --node-control-port "$NODE_CONTROL_PORT" \
  --node-control-scheme https \
  --tls-ca-file /etc/modalink/server-ca.crt \
  --tls-client-cert-file /etc/modalink/gpu-client.crt \
  --tls-client-key-file /etc/modalink/gpu-client.key
~~~

`MODALINK_TRAINING_ID` 只是上述命令使用的 shell 变量，不会被 SDK 自动读取。`--training-id` 保存训练 ID，不完成资源登记。节点地址由 client 自动获取，无需手写数据 endpoint。

| 通道 | TLS 配置 |
| --- | --- |
| Scheduler HTTPS | 使用完整 HTTPS URL 和 `--verify-scheduler-tls` |
| 节点 control HTTPS | 使用显式 `--tls-ca-file` 校验服务端证书，并使用配置的客户端证书和私钥进行 mTLS |
| 数据 TLS | session 必须返回 `tls://` endpoint；用同一组 mTLS 证书建立连接 |

Scheduler 的 TLS 设置与两条 Loader 连接独立。证书错误应核对主机名、证书链和信任配置。

### 预取配置

默认预取容量为 **8** 个返回项（VLA sample / VQA batch），允许 1–1024。连接配置修改会保留已有预取值。需要调整时使用 `remote-loader config edit --prefetch-batches N`；内存、等待时间和 Python 临时参数见 [用户指南](../../../user-guide.md)。

## 4. 按层验证

训练登记步骤见下文。节点准备完成后，从每台实际训练主机或训练容器各运行一次汇总连接诊断；不要按本机 GPU 重复运行。

~~~bash
remote-loader config show
remote-loader diagnose environment
remote-loader diagnose connection
~~~

该诊断从当前机器汇总检查当前训练全部已分配节点；每台训练主机或容器各执行一次，以覆盖主机侧网络差异。无需设置 `RANK`、`WORLD_SIZE` 或 `LOCAL_RANK`，也不要按 GPU 重复。`diagnose scheduler` 是同一连接诊断的别名，无需重复运行。诊断不读取样本，也不证明全部训练资源或模型输入已经可用。

| 检查 | 确认内容 |
| --- | --- |
| YAML 离线解析 | 本地配置有效，不要求 Token 或连接服务 |
| 环境诊断 | Python 和 PyTorch 环境，关注 warning/error |
| connection 诊断 | 当前训练全部已分配的普通节点、已启用 VQA 节点可从本主机或容器连接 |
| 有限读取 | 实际取得一个 VLA sample / VQA batch，检查字段、shape、dtype 和数值 |
| 模型训练 | 用真实训练项目确认数据符合模型输入要求 |

有限读取前，设置实际的 `RANK`、`LOCAL_RANK`、`WORLD_SIZE`，不要与同一 GPU 上正在运行的 loader 同时试读。未显式设置时，两个 rank 默认为 0；`WORLD_SIZE` 未设置时使用已保存的完整 `num_nodes * gpus_per_node`，拓扑不完整时才回退为 1。SDK 显式传入的 rank/world_size/local_rank 优先于环境变量；完整拓扑与实际 world_size 不一致会报错。

~~~bash
python "$SKILL_DIR/scripts/inspect_batch.py" --spec "$BATCH_SPEC"
python "$SKILL_DIR/scripts/inspect_batch.py" --spec "$BATCH_SPEC" --receive --items 1 --values 6
~~~

第一条仅离线检查，第二条实际读取。首次读取有最长 300 秒预热等待；后续 SDK 的 `batch_timeout` 默认 10 秒，helper 默认 60 秒。helper 输出是诊断摘要，不是完整样本转储。具体数据、图像处理和 PyTorch 接入见 [用户指南](../../../user-guide.md)，不要重复已完成的图像预处理或数据分片。

## 5. 故障定位与记录

| 问题 | 检查方向 |
| --- | --- |
| import/版本错误 | 实际训练 Python、安装版本与导入位置 |
| 配置错误 | YAML、有效配置路径、地址、协议和 TLS |
| 未登记、无节点或未就绪 | training_id、节点数和每节点 GPU 数、当前主机及 LOCAL_RANK；联系 Modalink 核对准备状态 |
| 401/403 | 实际报错入口、Token 与访问权限 |
| 读取超时 | 从训练环境检查交付的数据通道和服务状态；`BatchTimeoutError` 不会自动关闭 loader |
| 数据格式或校验错误 | client 与服务的配套版本、实际 YAML 和数据；不要跳过校验 |
| 读取成功但模型失败或等待过长 | 返回值是否符合模型要求、是否重复预处理、预取与数据等待情况 |

排障时提供版本、YAML 路径、scheduler 地址、训练 ID、本机 GPU 索引、诊断与读取错误；不要附带 Token 或云密钥。

## 6. 登记与释放训练

TUI 的“启动训练”中填写 `training_id`、`num_nodes`、`gpus_per_node`、datasets、可选 `cpu_per_gpu` 和 VQA 数据集。数据集版本按实际值确认；新增条目默认版本 2 不是通用版本保证。普通 weight 可省略（默认 1）；VQA weight 必须全填或全留空。旧配置迁移后仍未知的 version 必须先补齐。

编辑会自动保存，取消最终确认不会撤销已保存设置。选择“开启服务”并确认后登记并等待**本机**所有 GPU 的普通及启用的 VQA 节点就绪，不代表其他主机就绪。Ctrl+C 只停止等待，已申请的资源仍保留。由作业管理流程协调登记与各主机检查，不让各进程重复申请。

CLI 自动化使用 JSON/YAML 训练配置，完整示例见 [training-scripts.md](training-scripts.md)，它不是 batch YAML：

~~~bash
remote-loader training register "$TRAINING_CONFIG"
remote-loader training status
~~~

省略文件时使用已保存的完整训练参数。`register` 成功后保存配置，但不等待就绪；`status` 仅查询资源。登记结果不明或远端登记成功而本地保存失败时，先用原 ID 和入口查询，不换 ID 重新申请，也不盲目重试：

~~~bash
remote-loader training status \
  --training-id "$MODALINK_TRAINING_ID" \
  --scheduler-endpoint "$SCHEDULER_ENDPOINT"
~~~

训练命令可按交付要求指定 `--verify-scheduler-tls` / `--no-verify-scheduler-tls`。只有登记成功会保存 scheduler 覆盖值，其他训练命令仅在本次调用使用覆盖值。

每个进程停止读取并关闭 loader 后，在**它所在的主机**释放本机 GPU 的普通及 VQA 资源：

~~~bash
remote-loader training release-rank \
  --training-id "$MODALINK_TRAINING_ID" \
  --scheduler-endpoint "$SCHEDULER_ENDPOINT" \
  --local-rank "$LOCAL_RANK"
~~~

各主机各 GPU 都需执行对应释放；scheduler 根据来源主机与本机索引识别资源，不能从一台主机用全局 RANK 释放另一台主机的资源。最后一个 rank 释放后 scheduler 清理训练记录。关闭 loader、退出试读或清除连接配置均不释放训练资源。

停止所有进程并关闭 loader 后，才可整体释放：

~~~bash
remote-loader training end \
  --training-id "$MODALINK_TRAINING_ID" \
  --scheduler-endpoint "$SCHEDULER_ENDPOINT"
~~~

TUI“关闭服务（释放全部 dataloader 资源）”也是整体释放，确认前核对训练 ID；本地配置会保留。CLI 释放命令直接执行，无额外确认，`end` 不应作为每个进程的常规退出钩子。自动化和失败处理见 [training-scripts.md](training-scripts.md)。

当前 scheduler 服务不支持关闭 loader 后从读取 checkpoint 精确续读。模型 checkpoint 仍由训练代码负责，不能把保存读取状态当作已实现断点续读。
