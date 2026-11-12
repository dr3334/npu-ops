# 并行 Agent 工作守则（worktree + 设备串行）

> 每个 agent 在**自己的 git worktree** 里干活，分支已建好；**不要**在 main 或别人的 worktree 里改文件。

## 1. 环境
- 仓库根：你的 worktree 目录（如 `D:\deepseek_ workspace\mtk\wt-elem`）。
- 工具链在 WSL：编译/生成/比对用 `wsl.exe -e bash -lc '...'`；adb 用 Git Bash + `MSYS_NO_PATHCONV=1`。
- **设备是共享的**：任何一次板端操作（push/run/pull）都必须持锁：
  ```bash
  NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock" ./tools/devlock.sh run -- <命令>
  ```
  （`tools/devlock.sh` 已支持 `NPU_LOCK` 环境变量；所有 worktree 必须用同一个锁路径。）

## 2. 允许改什么
- 只改自己范围内的文件：`ops/<你的算子>*/`、`tools/build_<你的算子>.py`、`docs/OP-<你的范围>.md`。
- **不要**改：`tools/e2e.sh`、`tools/compare.py`、`tools/run_all.sh`、`tools/healthcheck.sh`、`tools/devlock.sh`、`probe/`、`docs/EXPERIMENTS.md`、`PLAN.md`、`README.md`。
  需要公共工具的行为变化时，在报告里说明，由 Lead 统一改。

## 3. 每个算子必须交付的证据
1. `ops/<name>/model.tflite`（`tools/build_*.py` 可复现生成）+ `model.json`（形状/dtype/量化/黄金来源）；
2. `ops/<name>/model.compile.log`（`tools/compile.sh` 输出，含 exit / target / sha256）；
3. 板端数值结果 `compare.txt`（跑 `tools/e2e.sh`，必须 PASS 或明确记录失败原因）；
4. 一条实验记录（你自己的 `docs/OP-<范围>.md`，按"假设/预测/做法/结果/适用范围/剩余疑问"写）。

## 4. 硬规则（来自 AGENTS.md）
- `--disallow-bridge` 恒开；bridge = 没上 NPU = 失败。
- 先写预测再跑；没有预测的观察不算实验。
- 数值：整数档优先 `codes=small`（逐位），全量程 requant 允许 `int8_tol=1`；浮点先登记容差。
- 失败保留，不许删；不猜寄存器、不 fuzz。
- 每完成一个算子就 `git add -A && git commit`（消息前缀：`feat(<范围>):`）。

## 5. 报告格式（结束时回复）
- 算子清单：名称 / 形状 / 编译结果 / 板端结果 / 数值结论（逐位或误差）；
- 失败与原因（分编译期/运行期）；
- 你自己的 commit 列表；
- 需要 Lead 处理的事项（例如公共工具要改）。
