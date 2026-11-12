# S0 · 工具链与板端环境基线

采集时间：2026-11-12 19:2x–19:4x（本机）。

## 1. 编译/转换工具链（主机侧 WSL）

| 项 | 值 |
|---|---|
| WSL | Ubuntu-22.04，`/mnt/d` 挂载工作区 |
| ncc-tflite | `neuropilot-sdk-basic-7.0.8-build20240807/neuron_sdk/host/bin/ncc-tflite`，`Version: 7.3.15`（x86-64 ELF，需 `LD_LIBRARY_PATH=.../host/lib`） |
| 目标 arch | `mdla5.0`；`--num-mdla=2`；`--l1-size-kb=3072`（取自 GAI `compile_prompt_qwen2.5_0.5B_7B.sh` 的 MT6897 分支） |
| 编译开关 | `--opt=3 --disallow-bridge --show-exec-plan --show-memory-summary`（bridge 出现即判失败） |
| 转换链 | `/mnt/d/ai/myenv/bin/python`：`mtk_converter 7.16.0+release`、TF 2.8.0 |
| tflite 生成 | `/usr/bin/python3`（有 `tflite` schema + flatbuffers 包），单算子模型由 `tools/build_ops.py` 直接构造 |

## 2. 目标设备

| 项 | 值 |
|---|---|
| adb 序列号 | `0123456789ABCDEF` |
| 型号 / 品牌 | `Titan 1` / HEADWOLF |
| 平台 | `ro.board.platform=mt6897`，`ro.hardware=mt8792` |
| 系统 | Android 16（SDK 36），内核 `6.1.145-android14-11`，`arm64-v8a` |
| 权限 | `adb root` 生效，shell `uid=0(root)`，SELinux `Enforcing` |
| NPU 节点 | `/dev/apusys`（system:camera，0660） |
| 厂商 runtime | `/vendor/lib64/libneuron_runtime.so`、`libneuron_runtime.7.so`、`libneuron_adapter*`、`libneuron_graph_delegate.mtk.so`、`libmvpu_*`（mdla 引擎/编译器） |
| 设备侧工具 | `neuron_sdk/mt6897/bin/{neuronrt,ncc-tflite,dla-packer,readdlb,extract-shared}` + `mt6897/lib/*.so`（7.3.15） |

## 3. 设备侧执行方式（已跑通）

`neuronrt`（aarch64，来自 SDK `mt6897/bin`）推送到 `/data/local/tmp/npuops/bin`，模型和数据放 `/data/local/tmp/npuops/dla`：

```bash
cd /data/local/tmp/npuops
LD_LIBRARY_PATH=$PWD/lib ./bin/neuronrt -m hw -a dla/<name>.dla -i dla/<name>.in.bin -o dla/<name>.out.bin -c 20
```

- `-m hw` 走真实硬件；`-m null/cmodel` 可用于对照。
- `-d` 打印 IO 映射表；`-c N` 重复推理并给出总时长与 FPS。
- 实测它使用 `/vendor/lib64` 的运行时即可工作（我们 push 的 `lib/` 目前为空，保留以便以后固定版本覆盖）。

## 4. 踩过的坑（已固化到脚本）

1. **MSYS 路径转换**：Git Bash 会把 `adb push ... /data/local/tmp/...` 的目标改写成 Windows 路径，必须 `MSYS_NO_PATHCONV=1`，同时把源路径写成 `D:/...` 形式。
2. **ncc-tflite 只能在 WSL 跑**：`tools/e2e.sh` 里 Linux 侧（build/compile/compare）统一走 `wsl.exe -e bash -lc`，adb 侧走 Git Bash。
3. **TFLite 模型输入张量不能带数据 buffer**：若输入 tensor 指向有 `data` 的 buffer，ncc 会把它当常量折叠成 `IdentityLayer on NVM`（第一次就踩到）。`build_ops.py` 已固定为 buffer 0 = 空占位。

## 5. 复现命令

```bash
# 主机（Git Bash）
cd "/d/deepseek_ workspace/mtk/npu-ops"
MSYS_NO_PATHCONV=1 ./tools/e2e.sh ops/fc_1x64x64_int8
```

产物：`ops/<name>/{model.tflite, model.json, model.dla, model.compile.log, golden/, out_device.bin, compare.txt, e2e.json}`。
