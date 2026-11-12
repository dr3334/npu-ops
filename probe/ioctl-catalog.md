# ioctl 探测目录 —— 一次 `.dla` 执行到底提交了什么

采集：2026-11-12 · 板 Titan 1 / MT6897(MT8792) · `fc_1x64x64_int8.dla`（7545 B）· `neuronrt -m hw -c 1`。

## 0. 方法（三层采集，外加一个关键教训）

| 层 | 手段 | 证据文件 |
|---|---|---|
| A 应用/运行时 | `strace -f -tt -y -e trace=ioctl,openat,close` | `trace_ioctl.txt` |
| A' payload | 自写 `LD_PRELOAD` shim（NDK aarch64），**调用前后各 dump 一次** | `iolog.c` → `io.log` |
| C 内核消费/完成 | `/sys/kernel/tracing/events/{mdw_rv_events,apusys_rv_events}` | `trace_kernel.txt` |

**教训（差点得出错误结论）**：第一版 shim 只在调用**之后** dump 参数 buffer，于是看到 `op=6,7,8,9,10,11`，误以为这块 BSP 把 `mdw_mem_ioctl_op` 扩展到了 6–11。
实际原因是驱动在部分路径上会把 in-struct 清零再写 out（反汇编 `mdw_mem_ioctl` 可见 `stp xzr,xzr,[x21]`），**POST 内容不是调用者传的 op**。
改成 PRE（调用前）+ POST（调用后）分别记录后，真实 op 只有 **0/1/2/3**，与 6.1 参考头 `mdw_ioctl.h` 完全一致。

> 这条正好印证 `docs/05` 的要求：证据不足时标 UNKNOWN，不能凭一次快照下结论。

## 1. 请求表（一次推理）

| type | nr | `_IOC` size | 次数 | 名称 | ABI 依据 |
|---|---|---|---|---|---|
| 0x41 | 0 | 0x28 | 1 | 未知（→ ENOTTY） | 6.1 头里没有，忽略 |
| 0x41 | 32 | 0x28 | 11 | `APU_MDW_IOCTL_HANDSHAKE` | `_IOWR('A',32,union mdw_hs_args)` |
| 0x41 | 33 | 0x30 | 13 | `APU_MDW_IOCTL_MEM` | `_IOWR('A',33,union mdw_mem_args)` |
| 0x41 | 34 | 0x98 | 2 | `APU_MDW_IOCTL_CMD` | `_IOWR('A',34,union mdw_cmd_args)` |
| 0x62 | 1 | 0x8 | 7 | `DMA_BUF_IOCTL_SYNC` | dma-buf 标准 |

`/dev/apusys` 以 `O_RDWR|O_SYNC` 打开 → fd=4；全部 `'A'` ioctl 打在这个 fd 上。

## 2. MEM 全解码（PRE = 调用者输入，POST = 驱动输出）

结构按 `union mdw_mem_args.in {u32 op; u64 flags; union{...}}`（`op`@0、`flags`@8、union@16）：

| # | 调用 | 输入（解出） | 输出 |
|---|---|---|---|
| 1 | ALLOC | type=0(MAIN) size=400 align=0x100 flags=0x5 | handle=6 |
| 2 | ALLOC | type=MAIN size=8192 align=0x1000 flags=0x0 | handle=7 |
| 3 | MAP | handle=7 size=0x2000 flags=0x2 | type=0, **device_va=0x25c78000** |
| 4 | ALLOC | type=MAIN size=32 align=0x100 flags=0x5 | handle=8 |
| 5 | ALLOC | type=MAIN size=16 align=0x100 flags=0x5 | handle=9 |
| 6 | ALLOC | type=MAIN size=76 align=0x0 flags=0x5 | handle=0xa |
| 7 | ALLOC | type=MAIN size=4096 align=0x100 flags=0x1 | handle=3 |
| 8 | MAP | handle=3 size=0x1000 flags=0x2 | device_va=**0x25c5f000** |
| 9 | ALLOC | type=MAIN size=4096 align=0x100 flags=0x1 | handle=0xb |
| 10 | MAP | handle=0xb size=0x1000 flags=0x2 | device_va=**0x25c7b000** |
| 11–13 | UNMAP | handle=0xb / 3 / 7 | 0 |

由此可以读出的**执行前内存预算**：8 KiB 的 cmdbuf 区 + 3 张 4 KiB 页 + 400/76/32/16 B 的元数据；
MAP 之后才拿到 IOVA（0x25cxxxxx 段），**没 MAP 的 handle 不能给设备用**。
这正好对应 `docs/05 §6` 的"分配/导入 → 建立设备可见地址 → 提交"。

## 3. CMD（与 6.1 `mdw_cmd_in` 布局一致，并有内核侧交叉验证）

| # | op | 关键字段（u64 词） |
|---|---|---|
| 1 | 0 = `RUN` | w3=`uid`(指针型 token)、**w4=`priority`=0xa**、w7 低 32 位 `power_dtime`=0x7d0、后续 `subcmd_infos`/`fence`/`exec_infos` 等指针 |
| 2 | 2 = `DEL` | w2=`id`=1 |

交叉验证：内核 trace 同一时刻打印 `mdw_rv_cmd: ... priority=10, num_subcmds=1` → **priority=10 与 w4=0xa 对上**，说明字段偏移解读成立（不是巧合）。

## 4. HANDSHAKE（板级拓扑，可直接进 P0 设备契约）

| 查询 | 返回 |
|---|---|
| `op=0 BASIC` ×2 | `version=0x4`、`dev_bitmask=0xd6`、`mem_bitmask=0x36`、`meta_size=32` |
| `op=1 DEV` sub=1 | type=1, num=2, meta=`"0x15556"` |
| `op=1 DEV` sub=2/4/6/7 | type=2 num=2 / type=4 num=2 / type=6 num=1 / type=7 num=1 |
| `op=2 MEM` sub=1 | type=1(VLM) start=0x2000000 size=0x600000（**6 MiB**） |
| `op=2 MEM` sub=2/4/5 | size=0x3000 / 0x3000 / 0x1000 |

## 5. 内核侧：一次推理的完整生命周期

```
mdw_rv_cmd(0, pid, num_subcmds=1, priority=10, pwr_plcy=0x10100cb)
apusys_rv_ipi_send(id=4, len=72, csum=…)
mdw_rv_cmd(1) / mdw_rv_cmd(2)          ← 由 irq/548-apu_ipi 线程处理
mdw_rv_subcmd(executed_core_bmp=0x1, history_iptime=9, ipstart/ipend ts)
mdw_rv_cmd_deque(enter_complt=693, pb_put_time=7692, handle_cmd_result_time=37847, ret=0)
apusys_rv_ipi_handle(latency=76077, t_handler=84000)
```

确认：一次推理 = **1 cmd + 1 subcmd + 1 次 RV IPI**；`executed_core_bmp=0x1`（只用 1 个 MDLA core）；
提交与完成不在同一线程 → **submit 返回 ≠ 完成**。

## 6. 确定 / 未确定

**确定**：请求码与 size、fd 与路径、13 次 MEM 的完整 op/参数/句柄/IOVA、CMD RUN/DEL 的 op 与关键字段（priority 已交叉验证）、HANDSHAKE 拓扑、内核命令生命周期、DMA-BUF 同步 7 次。

**未确定**：
- `nr=0`（size 0x28，ENOTTY）的用途；
- CMD 里 `uid` 那类"指针型 token"的确切语义（是用户地址还是驱动 token，需要看 `mdw_cmd_ioctl_run` 实现）；
- MEM `flags=0x5 / 0x1 / 0x2` 的位定义（`F_MDW_MEM_*` 只覆盖 CACHEABLE/32BIT/HIGHADDR，实际值位图还没对完）。

## 7. 复现

```bash
# 用户态顺序 + fd + 内核侧
adb shell 'cd /data/local/tmp/npuops && strace -f -tt -y -e trace=ioctl -o probe/trace_ioctl.txt \
  ./bin/neuronrt -m hw -a dla/fc_1x64x64_int8.dla -i dla/fc_1x64x64_int8.in.bin -o dla/x.bin -c 1'

# payload（PRE/POST）
adb push probe/libiolog.so /data/local/tmp/npuops/probe/
adb shell 'cd /data/local/tmp/npuops && LD_PRELOAD=$PWD/probe/libiolog.so IOLOG=$PWD/probe/io.log \
  LD_LIBRARY_PATH=$PWD/lib ./bin/neuronrt -m hw -a dla/fc_1x64x64_int8.dla -i dla/fc_1x64x64_int8.in.bin -o dla/y.bin -c 1'
/usr/bin/python3 probe/parse_io.py probe/io.log

# 反汇编佐证（驱动会把 in-struct 清零）
llvm-objdump -d --start-address=0x42e8 --stop-address=0x4a3c probe/apusys.ko | head -60
```

## 8. 对下一步（自己提交任务）的意义

现在已具备 L3 的完整知识：`HANDSHAKE`（拿拓扑）→ `MEM ALLOC/MAP`（拿 handle + IOVA）→ `CMD RUN`（subcmd + cmdbuf + fence）。
仍缺的一块是 **cmdbuf 内容**：目前由 `.dla` 携带（`libneuron_runtime` 解析后提交），我们自己还没有生成器。
下一步：把 `.dla` 里被写进 8 KiB cmdbuf 的字节抓出来（hook `MEM MAP` + 在 `CMD RUN` 前 dump 该 IOVA 对应 buffer），与 `docs/05` 的 L3→L4 门槛对齐。
