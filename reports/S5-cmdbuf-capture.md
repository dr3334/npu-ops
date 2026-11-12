# S5 · 命令缓冲（cmdbuf）抓取：一次推理用到的每个 dma-buf

时间：2026-11-12 · 板 Titan 1 / MT6897(MT8792) · 模型 `fc_1x64x64_int8.dla`（7545 B）。

## 1. 为什么做这一步

`probe/ioctl-catalog.md` 已经把 `HANDSHAKE → MEM(ALLOC/MAP) → CMD(RUN)` 走通，但命令内容仍由 `.dla`/`libneuron_runtime` 生成。
要往 L4（自己生成任务）走，必须先把**设备真正读到的字节**抓出来：哪块 buffer 是输入、哪块是输出、哪块是命令/参数。

## 2. 方法（`probe/iolog2.c`）

在 v1（PRE/POST ioctl 记录）基础上加三件事：

1. **mmap 跟踪**：记录 `(host_va, len, fd)`，把 dma-buf 映射和 MEM handle（= fd 编号）对上；
2. **触发点**：
   - `CMD RUN` 之前（PRE）→ 抓输入/静态区；
   - `MEM UNMAP` 之前（POST）→ 抓推理完成后的输出（UNMAP 是收尾前的最后时机）；
3. **dma-buf 读取**：这类映射既没有 page 支撑（`process_vm_readv` 报 EFAULT），也不走 `/proc/self/mem`（EIO），
   只能用**带 SIGSEGV/SIGBUS 保护的 memcpy**（`guard_handler` + `sigsetjmp`）按 4 KiB 分块快照。

构建：NDK r26d，`aarch64-linux-android31-clang -shared -fPIC`。

## 3. 抓到的 buffer 与角色（有证据，不是猜）

| mmap | 大小 | fd=MEM handle | IOVA（来自 MAP 返回） | 角色 | 证据 |
|---|---|---|---|---|---|
| map13 | 4096 | 3 | 0x25c5f000 | **输入激活** | 前 64 B 与 `golden/input.bin` **逐字节相同** |
| map14 | 4096 | 0xb | 0x25c7b000 | **输出** | 推理后前 64 B 与板端输出 **逐字节相同**（`OUTPUT MATCH`） |
| map10 | 400 | 6 | — | 命令/寄存器描述区（静态） | 前后内容 0 字节变化；含 0xa5xxxxxx 类命令字与 0x025c3xxx 设备地址 |
| map11 | 8192 | 7 | 0x25c78000 | 网络/权重参数区（静态） | 前后 0 字节变化；头部是 int16 型系数表 |
| map12 | 76 | 0xa | — | 小结构（subcmd/exec info 候选） | 未解码 |
| map09 | 7545 | fd=4（未确认路径） | — | 疑似 `.dla` 文件映射 | 大小恰等于 `.dla`；但内容与磁盘 `.dla` 有 5290/7545 字节不同、头部被清零 → **UNKNOWN** |

一次推理的 dma-buf 预算（与 ioctl 目录里的 ALLOC 完全对应）：
**输入 4 KiB + 输出 4 KiB + 命令 400 B + 参数 8 KiB + 小结构 76 B + 32/16 B 两个元数据块**。

## 4. 已验证的闭环

```
主机 input.bin ──(MEM MAP 的 host VA)──► map13 ──► MDLA ──► map14 ──► (runtime 拷回) ──► 输出文件
       前 64B 相同 ✅                                          前 64B 相同 ✅
```

也就是说：**我们现在能在提交瞬间"看见"设备将读到的每一个字节，以及它写回的每一个字节"** —— 这是后续自己构造任务的前提条件。

## 5. 还没解决的（明确列为 UNKNOWN）

1. **map09 的身份**：7545 B 与 `.dla` 同尺寸但内容不同（头部 64 B 全 0，差异 5290 B）。需要 hook `open/openat` 确认 fd=4 指向哪个文件，以及是不是运行时把 `.dla` 加载后做了重定位/清零。
2. **map10 的命令编码**：400 B 里能看到 `0xa500000d`、`0xa5001089`、`0xa500211e`、`0xa5004011`、`0xa5004499` 这类字，以及 `0x025c3800/0x025c3810/0x025c2f00/0x025c2e00/0x025c1000` 这类设备地址；但**没有 MDLA5.0 的 regcmd 编码表**，不能凭形状猜"这就是寄存器写"。
3. **0x025c3xxx 与 MAP 返回的 0x25c5f000/0x25c78000 不是同一段**：说明 map10 里引用的可能是别的 buffer（或低 32 位压缩地址），需要与 `.dla` 的 relocation 表对照。
4. `map12`（76 B）的字段含义未解。

## 6. 下一步（三条可选路线，按成本排序）

| 路线 | 做法 | 产出 |
|---|---|---|
| A（最省） | `readdlb`/`ncc-tflite --show-exec-plan` 的 dump 与 map10 逐字节对照 | 命令区是否等于编译器的 exec plan |
| B | 用**两个不同 shape** 的同族模型（如 1×64×64 vs 1×128×128）抓 map10 做 diff | 哪些字段随 shape 变化 → 定位 descriptor 字段 |
| C | 自己写 `apusctl`（NDK），用已知 handle/IOVA 复现 `CMD RUN`，只改一个字段做 A/B | 单字段语义（真正的 L4 入口） |

## 7. 复现

```bash
adb push probe/libiolog2.so /data/local/tmp/npuops/probe/
adb shell 'cd /data/local/tmp/npuops && mkdir -p probe/buf && \
  LD_PRELOAD=$PWD/probe/libiolog2.so IOLOG=$PWD/probe/io2.log IODIR=$PWD/probe/buf \
  LD_LIBRARY_PATH=$PWD/lib ./bin/neuronrt -m hw -a dla/fc_1x64x64_int8.dla \
    -i dla/fc_1x64x64_int8.in.bin -o dla/out.bin -c 1'
adb pull /data/local/tmp/npuops/probe/buf probe/buf
# 期望：map13 前 64B == golden/input.bin；map14 前 64B == 板端输出
```
