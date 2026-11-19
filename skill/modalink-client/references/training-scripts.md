# 训练登记与关闭脚本模板

交互操作可使用 TUI“启动训练”；需要自动化时，将下方模板保存到自己的作业目录。模板仅登记或释放数据服务，不启动模型训练，也不会自动执行。安装、Token 与自动连接配置统一见 [deployment.md](deployment.md)。

## 前提与训练配置

先确认资源操作授权、实际 scheduler、TLS 设置、本次唯一的 training_id、节点数、每节点 GPU 数及数据集版本。公开仓库和包不含证书、私钥或 Token；须通过私有渠道取得有效凭据，按 [deployment.md](deployment.md) 用 `remote-loader config edit` 保存实际 Scheduler 及全部三项绝对 TLS 路径，再保存 Modalink Token，之后才能使用 TUI/SDK 或这些脚本。自动云发现不提供凭据。文档和脚本模板本身不构成执行登记或释放的授权。所有训练主机使用一致的训练参数和连接设置；独立配置通过 `REMOTE_LOADER_CONFIG` 指定，脚本与训练进程使用同一配置。`PYTHON` 指实际训练环境的解释器。

下面是可保存为 `training-request.json` 的完整训练配置示例：两台主机、每台四个 GPU，整个训练的 `WORLD_SIZE=8`。替换 ID、版本和资源数量后使用，不要把 batch YAML 作为登记文件。数据读取所需的 batch YAML 由用户自行编写或通过私有渠道取得，公开仓库和 wheel 均不提供该输入文件。

~~~json
{
  "training_id": "your-training-id",
  "num_nodes": 2,
  "gpus_per_node": 4,
  "datasets": [
    {"id": "your-dataset-id", "version": 2, "weight": 1.0}
  ],
  "cpu_per_gpu": 2,
  "vqa": false,
  "vqa_datasets": []
}
~~~

- `num_nodes`、`gpus_per_node` 必须为正整数，乘积等于 `WORLD_SIZE`。不要从全局 `RANK` 推导本机 GPU 索引，由 launcher 提供实际 `LOCAL_RANK`。
- `cpu_per_gpu` 可省略，由 scheduler 决定；填写时必须为正整数。
- 普通 datasets 至少一条，version 必须是实际可用的正整数。默认版本 2 不表示所有数据集都用版本 2；旧配置迁移后未知的 version 仍须补齐。weight 可省略（默认 1），显式值须有限且非负，至少一条有效正权重。
- 开启 VQA 时将 `vqa` 改为 `true`，在 `vqa_datasets` 中添加实际 `id/version` 条目。VQA weight 必须全部填写或全部省略，全部填写时至少一条为正数。关闭 VQA 时保留空数组。
- `sub_epoch_size` 固定为 128，不在输入文件中另设；使用训练 CLI 生成登记请求。
- 图像、窗口与输出字段在 batch YAML 中设置，见 [batch-spec.md](batch-spec.md)；训练配置只使用上述训练字段。

## 启动脚本模板

由作业管理流程在一个协调入口执行**一次**登记，不要让每个 rank 各自执行。先在有效连接配置中核对 scheduler 和 TLS；`REQUEST_FILE` 为本次训练配置，`RUN_FILE` 为尚不存在的登记结果记录文件，避免覆盖先前记录。

~~~bash
#!/usr/bin/env bash
set -euo pipefail
set -o noclobber
: "${PYTHON:?请设置训练 Python 可执行路径}"
: "${REQUEST_FILE:?请设置训练配置 JSON/YAML 路径}"
: "${RUN_FILE:?请设置本次 CLI 登记记录路径}"
command -v "$PYTHON" >/dev/null

if "$PYTHON" -m remote_loader.cli.main training register "$REQUEST_FILE" > "$RUN_FILE"; then
  printf '登记已确认且本地配置已保存；尚未启动模型训练。记录：%s\n' "$RUN_FILE"
else
  exit_code=$?
  printf '%s\n' '登记结果或本地保存未确认，请保留记录和错误，先用原 ID/入口查询 status，不要换 ID 重新申请。' >&2
  exit "$exit_code"
fi

# status 仅查询资源，不等待节点就绪。
"$PYTHON" -m remote_loader.cli.main training status
printf '%s\n' '请从每台训练主机或容器各运行一次汇总连接诊断，再按实际 rank 有限试读；确认后由 launcher 启动模型。'
~~~

保存为 `start-training.sh` 后的调用模板：

~~~bash
PYTHON='/实际训练环境/bin/python' \
REQUEST_FILE='./training-request.json' \
RUN_FILE='./training-run.json' \
bash start-training.sh
~~~

脚本不自动重试、等待全部节点或清理资源。登记成功只保存执行机器上的配置；启动训练前，将一致的非敏感连接和完整训练配置准备到其他主机，各主机按 [deployment.md](deployment.md) 私下准备自己的有效 Loader CA、客户端证书、客户端私钥和 Modalink Token，并核对本机三项绝对路径。

登记超时、结果不明、本地保存失败或后续 status 失败，都不能当作“资源未创建”。保留原 ID 和参数，使用已提交的真实目标查询：

~~~bash
"$PYTHON" -m remote_loader.cli.main training status \
  --training-id "$MODALINK_TRAINING_ID" \
  --scheduler-endpoint "$SCHEDULER_ENDPOINT"
~~~

TLS 设置继承有效配置，必要时按交付要求显式传 `--verify-scheduler-tls` / `--no-verify-scheduler-tls`。相同 ID、相同参数的重复登记是幂等的；结果不明时仍先查询，确认已有资源状态后再决定是否重试或恢复本地配置，不更换 ID 或参数盲目重试。

从每台训练主机或容器各运行一次连接诊断；该命令汇总当前训练全部已分配节点，不按 GPU 或 `LOCAL_RANK` 重复，且无需设置 rank 环境变量。有限试读仍需正确的 `RANK/LOCAL_RANK/WORLD_SIZE`，且不能与正在训练的 loader 同时占用同一 GPU 的数据资源。具体命令、首次等待和 helper 路径见 [deployment.md](deployment.md)。TUI“开启服务”也只确认执行主机的 GPU 就绪，不能代替多机检查。

## 按本机 GPU 关闭脚本模板

每个进程停止读取、关闭 loader 后，由其所在主机的作业管理流程执行。`MODALINK_TRAINING_ID` 与 `SCHEDULER_ENDPOINT` 显式指定本次目标，避免误用其他训练的已保存配置。`LOCAL_RANK` 必须是该主机的 GPU 索引，不能使用全局 `RANK`。

~~~bash
#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON:?请设置训练 Python 可执行路径}"
: "${SCHEDULER_ENDPOINT:?请设置本次 scheduler 地址}"
: "${MODALINK_TRAINING_ID:?请设置本次已登记训练 ID}"
: "${LOCAL_RANK:?请设置本机 GPU 索引}"
command -v "$PYTHON" >/dev/null
case "$LOCAL_RANK" in
  ''|*[!0-9]*) printf '%s\n' 'LOCAL_RANK 必须是非负整数' >&2; exit 2 ;;
esac

"$PYTHON" -m remote_loader.cli.main training release-rank \
  --training-id "$MODALINK_TRAINING_ID" \
  --scheduler-endpoint "$SCHEDULER_ENDPOINT" \
  --local-rank "$LOCAL_RANK"
printf '本机 GPU %s 的释放请求已完成，请检查上方返回结果。\n' "$LOCAL_RANK"
~~~

保存为 `stop-training.sh` 后，在每台训练主机上对每个本机 GPU 分别执行。以下仅演示当前主机的 GPU 0：

~~~bash
PYTHON='/实际训练环境/bin/python' \
SCHEDULER_ENDPOINT='https://实际-scheduler-地址:端口' \
MODALINK_TRAINING_ID='本次已登记训练ID' \
LOCAL_RANK=0 \
bash stop-training.sh
~~~

脚本失败会非零退出，不打印完成信息。launcher 应分别保留训练退出码和释放错误：训练已经失败时，最终保留原训练失败，不让清理错误掩盖它；训练成功但释放失败时，应单独报告释放失败。多机异常退出与重试由作业管理流程协调，不让单个进程自动结束整项训练。

只有所有进程都已停止读取并关闭 loader，且确实需要整体终止时，才执行：

~~~bash
"$PYTHON" -m remote_loader.cli.main training end \
  --training-id "$MODALINK_TRAINING_ID" \
  --scheduler-endpoint "$SCHEDULER_ENDPOINT"
~~~

`end` 直接释放全训练资源，不是当前主机或当前 GPU 的局部释放，没有额外确认提示，不应放入各进程的常规退出钩子。TUI“关闭服务”也释放全训练，但会再次确认目标。

`loader.close()`、退出试读及清除连接配置都不释放训练资源。当前 scheduler 服务不支持关闭后从读取 checkpoint 精确续读；重新开启不等于恢复旧读取位置。解除私网前先停止数据访问并完成训练释放，再按对应网络文档协调处理，不影响其他业务。
