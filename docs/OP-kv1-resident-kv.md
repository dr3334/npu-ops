# OP-kv1 —— KV cache 上设备：RuntimeAPI 能绑用户 dma-buf，但不能就地更新

> 分支 `impl/kv-resident`，worktree `wt-kv1`；前驱 docs/OP-q06-prefill-decode.md（票 06）、
> docs/OP-vendor-runtime.md、docs/adr/0006-execution-form-shared-weights.md。
> 设备 mt6897 / Android 16（0123456789ABCDEF），板端全部持 `/d/deepseek_ workspace/mtk/.npu-board.lock`，
> 每个板端块前后各一次 `healthcheck --quick`。公共工具与 `.scratch/` 一行未改；
> 新增 `tools/kv1_runner.c`、`tools/build_kv1.sh`、`tools/e2e_kv1.sh`。

---

## 0. TL;DR（先说结论）

**目标（KV 常驻设备、每 token 只写新行）在当前 RuntimeAPI 上做不到**，但"做不到"的位置很具体，
不是猜的：

| 问题 | 实测答案 | 证据 |
|---|---|---|
| `setInput` 能绑**用户自己的** dma-buf 吗 | **能**。`ionFd` 传 fd，rc=0，图正常装载与推理 | §2.2 |
| 绑上之后，host **往同一块 buffer 里写新行**，下一次 invoke 看得到吗 | **看不到**。写成 20000 与写成 0，下一步输出 **逐字节相同** | §4.1 |
| 那**重新 setInput**（换一块 dma-buf、换 fd）呢 | **同样看不到**，输出仍逐字节相同 | §4.2 |
| 那是不是我们的行映射写错了 | **不是**。同一份 cache 预先写进**文件**再起进程，runner 与 CLI 的 step 1 **逐字节一致** | §4.3 |
| 绑 dma-buf 有没有换来别的好处 | **有**。同一 runner、同一图：KV 在 dma-buf `49.0 ms/invoke`，KV 在匿名内存 `52.9`，CLI `≈60.3` | §5 |
| 8 个 token 逐步对拍 | step 0 三条张量全同；**step 1 起全不同**（|Δcos| 从 8.1e-5 掉到 1.07） | §6 |
| 400 次 invoke 稳定性 | median **49.25 ms**、p10 47.17 / p90 53.69、`rchar` 冻结、400 次输出 == 1 次输出 | §5.3 |

一句话：**"绑一次、反复用"这条语义在 RuntimeAPI 里只对 weights 成立，对 KV（图输入）不成立——
图输入在第一次推理时就被吃掉了**；要换内容只能重建网络实例（进程级，150 ms 量级）。
下一步唯一的候选是 `RuntimeV2` 的**每请求 IOBuffer(fd+offset)**（§8），本票没有测。

---

## 1. 先读的三条 API 事实（带行号）

| 事实 | 位置 |
|---|---|
| 用户提供输入 buffer：`NeuronRuntime_setInput(runtime, handle, buffer, length, BufferAttribute)` | `RuntimeAPI.h:144` |
| 用户 buffer + **偏移窗口**：`setOffsetedInput(..., size_t offset)`（"Reading ION buffer from start addr + offset"） | `RuntimeAPI.h:159` |
| `BufferAttribute` 只有一个字段 `int ionFd`；`ionFd == -1` = 非 ION；注释原文："Android device implementations **may** benefit from this information to eliminate unnecessary data copy" | `Types.h:201-209` |
| RuntimeV2 才是"每次请求重新给 buffer"的 API：`IOBuffer{buffer, length, int fd /*"fill with imported buffer FD"*/, int offset}`，每请求传给 `NeuronRuntimeV2_run` | `RuntimeV2.h:23-37`、`64-72`、`174` |
| 没有"取回运行时内部输入 buffer 指针"的 API：能拿到的只有 size / paddedSize / rank / paddedDims | `RuntimeAPI.h:252-424` |

两条推论直接决定了本票的做法：
1. 想让 NPU 直接读我们的内存，唯一入口就是 `ionFd`（或 V2 的 `fd`）；
2. 想知道"运行时到底有没有另开一份"，只能靠 `/proc/self/maps` 里的 `/dmabuf:` 行（§7）。

设备侧事实：`/dev/dma_heap/mtk_mm-uncached` 存在（正是厂商那份 239 MiB 权重 dma-buf 的 exporter，
见 OP-vendor-runtime §6）；设备 `/vendor/lib64/libneuron_runtime.so` 自报 **7.3.8**，装载我们用 ncc
7.3.15 编的 `.dla` 正常。

---

## 2. 做法：一个最小 aarch64 runner

`tools/kv1_runner.c`（`tools/build_kv1.sh` 用 NDK r26d 编，只依赖 libc/libdl，运行时 `dlopen`
`libneuron_runtime.so`，不钉版本）。它做四件事，和任务卡的 ①②③ 一一对应：

1. `loadNetworkFromFile` 装载 decode `.dla`（+ 外部权重 bin 作为第 9 个输入，与 CLI 同形）；
2. 每个输入一块**常驻** buffer：KV 两块用 `DMA_HEAP_IOCTL_ALLOC` 从 `mtk_mm-uncached` 分配并 `mmap`，
   `setInput(handle, ptr, len, {.ionFd = fd})` **绑一次**；其余 7 个输入用匿名内存；
3. 循环 invoke，**每 token 只把新行**（48 层×2 头 × 64 int16 = 每张 6,144 B）写进 KV 的 dma-buf 的
   第 m 个槽位（`p*C*128 + m*128`，48 段 128 B 的 memcpy），再 invoke 下一次；
4. 每步量时间（`CLOCK_MONOTONIC`）、把输出与 CLI 路径**逐字节**比，并打印 `/proc/self/io`
   （证明循环里没有文件 I/O）与 `/proc/self/maps` 的 dmabuf 行。

### 2.1 IO 表（run 时打印，说明"没有 padding，理论上该零拷贝"）

```
[graph] q06/decode.dla inputs=9 outputs=3
[in  6] size=12582912 padded=12582912 rank=4   <-- KV(k)
[in  7] size=12582912 padded=12582912 rank=4   <-- KV(v)
[in  8] size=16674816 padded=16674816 rank=0   <-- 外部权重 bin
[out 1] size=49152    padded=49152    rank=4   <-- k_out [48,8,64]
```

**9 个输入的 padded == logical**，所以连 `suppressInputConversion` 理论上都不需要。这一点很重要：
它排除了"padding 布局不匹配"这个最常见的失败解释（也单独测过 `--noconv`，见 §4.4）。

### 2.2 绑定本身成功

```
[buf in6] dma-buf heap=mtk_mm-uncached len=12582912 fd=16
[buf in7] dma-buf heap=mtk_mm-uncached len=12582912 fd=17
[setInput  6] rc=0 ionFd=16 0.080 ms
[setInput  7] rc=0 ionFd=17 2.570 ms
[setInput  8] rc=0 ionFd=-1 3.051 ms
```

用户 dma-buf 被接受，没有报错、没有 fallback。

---

## 3. 8 步对拍怎么搭的（可复现）

`tools/e2e_kv1.sh --steps 8`：**同一份 prefill 输出**给两条路径做种子，跑同一条 8 步序列，
两条路径**每一步都逐字节比**（不是只比最后一步）：

```
phase 0  prefill 一次（CLI，未改）           -> 拉 out_hidden/out_k/out_v
phase 1  build_q06.py --seq-init            -> 两条路径同种子、同 step0 输入；同时 clone 状态
phase 2  BASE  8 步：每步 push 8 个输入文件 + neuronrt -c 1（step0 用 -c 20）
phase 3  KVRES 8 步：一个进程、KV 在 dma-buf、每步只写新行
phase 4  逐字节 cmp（8 步 × 3 张量）
phase 5  两条路径各自与同一套 numpy golden 对拍
```

**先给一个可复现性结论**：本次重建的 tflite/.dla/bin 打出来的 BASE 8 步，和票 06 §5.2 的表
**逐位相同**（370/84.56、135/27.76、143/36.90、98/21.68、144/34.31、67/17.93、152/36.43、249/59.76），
说明重新物化产物没有改变数值。顺带补齐了票 06 的 UNKNOWN 8（decode 各步没取到 rel_l2）：
**rel_l2 0.00251–0.01280，门 0.05，PASS**。

---

## 4. 关键否定结果：图输入绑一次之后就"死"了

三个实验，层层排除。全部在同一进程、同一 `.dla`、同一会话。

### 4.1 毒化实验（最直接）

把新行写成 **int16 20000**（而不是图输出的那一行），跑 2 步，然后把 KV 的 dma-buf 内容**读回来**：

| 检查 | 结果 |
|---|---|
| 读回的 cache 与种子逐字节差几个字节 | **正好 6,144 B** = 48×64×2 = 恰好一行/层，位置就是我们算的 `p*C*128 + 16*128` |
| 种子 row16 前几个 int16 | `0 0 0 0` |
| 写后 row16 前几个 int16 | `20000 20000 20000 20000` |
| **step 1 的 hidden（写 0 vs 写 20000）** | **sha256 相同**（`4e748e43bc1b13af…`） |

=> 写确实落到了我们以为的地方、且只落到那里；**设备在下一次 invoke 没有读它**。

对照：写 0 那次读回的 cache 与种子**逐字节相同**（0 处差 0 字节）——因为种子 row16 本来就是 0，
所以"写 0 无效"这件事本身不能当证据；真正的证据是上表最后一行与"20000 那次差 6144 B"。

### 4.2 换一块 dma-buf 重新 setInput（ping-pong）

再分配第二对 dma-buf（fd=30/31），每步把更新后的 cache 整块 memcpy 进去，并用**新的 ptr+fd**
重新 `setInput`：

```
[pingpong] alt buffer for in6 fd=30
[pingpong] alt buffer for in7 fd=31
--- step-1 hidden: pingpong-rebind vs plain ---
IDENTICAL: re-binding a different dma-buf did not change the result
```

=> 连"换 buffer 重新绑"都不影响结果。绑一次之后，后续 `setInput` 对这个 handle 等于空操作。

### 4.3 控制实验：框架本身是对的

把 step 0 的输出行**预先写进种子文件**，起一个新进程只跑 1 次（`--steps 1`），与 CLI 的 step 1 比：

```
SAME: runner==CLI when the cache is pre-seeded (framework correct)
```

=> 行映射、mask、输入文件、比较链全对。**唯一不成立的就是"进程内更新"这一步**。

### 4.4 其它三个可能的解释都排掉了

| 解释 | 测法 | 结果 |
|---|---|---|
| 需要 `DMA_BUF_IOCTL_SYNC`（写合并缓冲没刷） | `--sync 1` 在每次写前后各一次 sync ioctl | step1 起仍 DIFF |
| 输入需要 `suppressInputConversion`（否则运行时转换一次） | `--noconv`（同时关输出转换） | 仍 DIFF（`--noconv --sync 1` 也试了） |
| 我们写错了行/越界 | §4.1 的"差 6144 B、位置正确" | 排除 |

---

## 5. 那 dma-buf 绑上到底买到了什么（性能）

方法按 NOTES §11.5：**同一会话、配对交替 ABCCBA、报比值中位数**；绝对 ms 只作量级。
三个臂，都是 200–400 次 invoke 的同一张 decode 图（C=2048，T=8 padded）：

- **A** = `neuronrt` CLI（票 06 路径，KV 是 host 文件，每进程读一遍）
- **B** = `kv1_runner`，KV 在 **dma-buf**（`ionFd`）
- **C** = `kv1_runner`，KV 在**匿名内存**（`ionFd = -1`，唯一差别就是这一项）

### 5.1 每 invoke（N=100，两轮 ABCCBA；B/C 取 runner 自己的单调时钟中位数）

| 臂 | 每 invoke | 采样 |
|---|---|---|
| A neuronrt CLI | **≈60.3 ms**（wall/100；启动+装载约 1.4 s，用 N=20 与 N=100 的差斜率定） | 4 次：6272/6152/6197/6121 ms |
| B dma-buf KV | **49.0 ms**（中位） | 4 次：48.98 / 50.14 / 48.92 / 49.05 |
| C anon KV | **52.9 ms**（中位） | 4 次：53.23 / 53.36 / 52.35 / 52.39 |

**配对比值（报这个，不报绝对值）**：

| 比值 | 值 | 读法 |
|---|---|---|
| A / B | **1.23** | 常驻进程 + dma-buf KV 比 CLI 快 23%（−11.3 ms/invoke） |
| C / B | **1.08** | 仅"KV 在 dma-buf 而不是匿名内存"这一项 = **−3.9 ms/invoke（−7.3%）** |
| A / C | 1.14 | CLI 自身多出来的部分（每次进程重读 8 个输入文件） |

后一行是本票唯一"能用"的性能结论：**即使不能就地更新，把 KV 放在 dma-buf 里也是有收益的**
（怀疑是运行时对非 ION buffer 每次推理要做一次缓存维护/中转，25 MiB 量级 ≈ 4 ms；未取证，见 §9）。

### 5.2 与票 06 的 68.5 ms/token 的关系

票 06 的 68.5 ms 是 `neuronrt -c 20` 摊薄值；本次 A 臂测得 ≈60.3 ms（同 .dla、同图、同 8 行 padding）。
两者差 −12%，落在 NOTES §11.5 说的 ±15% 漂移里，**不作为性能结论**；本票的结论只用 A/B/C 的配对比值。

### 5.3 400 次 invoke 稳定性（B 臂）

```
[time] n=400 mean=49.878 ms median=49.248 p10=47.172 p90=53.691 min=43.966 max=56.662
[io before] rchar: 42594329  wchar: 105      syscr: 105   read_bytes: 0
[io after]  rchar: 42594430  wchar: 120937   syscr: 107   read_bytes: 0
```

- `rchar` 400 次 invoke 只涨 **101 B**（那 101 B 是读 `/proc/self/io` 自己），`read_bytes = 0`
  => 循环里**没有任何文件 I/O，也没有 25 MB 的搬运**；
- 400 次 invoke 的输出与 1 次 invoke 的输出 **sha256 相同** => 确定性。

### 5.4 每 token 的 host 写入量

| 形态 | 每 token host 写入 | 备注 |
|---|---|---|
| CLI（票 06） | **25,640,088 B**（8 个输入文件全推） | 实测 0.88–1.46 s/步（adb push 段） |
| KVRES（本 runner） | **12,288 B**（K+V 各 6,144 B，写进 dma-buf） | 无 adb、无进程重启 |
| KVRES 的其它小输入 | 493,264 B（x/mask/cos/sin 六个文件） | 与 KV 无关，两种形态都要 |

比值 **25.6 MB → 12 KB（≈2085×）**。这个数字是**形态上成立的**，但它只在**第一步**是对的——
第二步开始设备就不认了（§4）。所以不能把它当成"decode 提速 23%"之外的额外收益。

---

## 6. 8 步逐步对拍（结果）

| step | pos | BASE 码级max | BASE |Δcos| | **KVRES 码级max** | **KVRES |Δcos|** | BASE vs KVRES 逐字节 |
|---|---|---|---|---|---|---|
| 0 | 16 | 370 | 8.13e-5 | 370 | 8.13e-5 | **hidden/k/v 全 IDENTICAL** |
| 1 | 17 | 135 | 8.81e-6 | **39,573** | **1.0719** | 三段全 DIFF |
| 2 | 18 | 143 | 1.34e-5 | 48,174 | 0.9950 | 全 DIFF |
| 3 | 19 | 98 | 5.19e-6 | 41,265 | 1.0330 | 全 DIFF |
| 4 | 20 | 144 | 1.45e-5 | 34,634 | 0.9962 | 全 DIFF |
| 5 | 21 | 67 | 3.13e-6 | 39,577 | 0.9836 | 全 DIFF |
| 6 | 22 | 152 | 1.47e-5 | 37,603 | 1.0640 | 全 DIFF |
| 7 | 23 | 249 | 3.77e-5 | 41,704 | 1.0046 | 全 DIFF |

读法：**step 0 完全一致**（说明绑定与装载都是对的），**step 1 起 KV 路径看到的是"没有更新的缓存"**
（mask 声明 17 个槽位有效，实际只有 16 个有内容，剩下的按 0 参与注意力）→ cos 掉到 0 附近。
BASE 一路 ≤8.2e-5（门 0.003），KVRES 从第 1 步就越门，**这正是本票要暴露的结构性事实**。

BASE 侧的 rel_l2 每步 0.00251–0.01280（门 0.05，PASS），并补齐了票 06 的 UNKNOWN 8。

---

## 7. 运行时到底有没有另开一份 KV 内存？（未定论）

`/proc/self/maps` 里我们进程的 dmabuf 行（`--verbose` 打印）：

```
/dmabuf:APUCB0:<pid>:MDLA    ~7 MB     （NPU 命令缓冲）
/dmabuf:APUBUF0:<pid>:MDLA   ~7.3 MB   （工作缓冲）
+ 我们自己的两块 mtk_mm-uncached 12,582,912 B（KV，fd=16/17）
```

**没有**看到额外的 12.58 MB 级的块（票 06 §7 在 CLI 形态下看到过 K 12,582,912 / V 12,517,376，
那是 CLI 把 host 内存当输入时运行时自己开的）。这反而是个谜：如果运行时用的是我们这块 buffer，
我们的写就该被看到；如果它另开了一块，maps 里应该有那块。两种解释都不完全自洽，
**这是本票最主要的 UNKNOWN（§9.1）**，也说明"图输入在哪儿"需要 V2/驱动侧证据才能定论。

---

## 8. 下一步的三个候选（按性价比排序）

1. **`RuntimeV2`（最该先试）**：`NeuronRuntimeV2_run(runtime, SyncInferenceRequest)` 每次请求都带
   `IOBuffer{buffer, length, fd, offset}`（`RuntimeV2.h:23-37, 64-72, 174`）。这是唯一"每次推理重新给
   buffer"的 API——如果它对每个请求读 fd 的当前内容，KV 常驻 + 每 token 12 KB 就成立。
   本票**没有测**（时间用在了把 V1 的语义钉死上）。代价：V2 要重写 runner 的调用段（~150 行），
   但图的 IO 契约完全一样。
2. **让图自己更新 KV（真正同构厂商）**：厂商的 decode 图 21.6 MB、KV 常驻设备、无拷贝，说明他们的
   KV 不是图输入而是**图内静态张量 + in-place**（他们的 executor 直接给设备指针）。TFLite 是函数式的，
   我们这条路线表达不了"有状态的图"。要走这条，得换执行形态（ncc 侧的 in-place/stateful 支持或
   自己写提交器，后者正是 PLAN.md 里 L3.5 已有的 `apusctl` 方向）。
3. **保持"KV 在 host、单进程常驻"**（今天就能用）：把 25 MB 的搬运从 adb 拿掉，per-token 只剩
   图本身（本票 B 臂 49.0 / C 臂 52.9 ms）。KV 更新的做法是每 token **重建网络实例**，代价 150 ms 量级，
   比不划算；所以这条实际上等价于"再等第 1 条"。

---

## 9. UNKNOWN（不圆）

1. **图输入的驻留形式没有直接证据**：maps 里既没有 12.58 MB 级的第二块，写又看不到效果（§7）。
   可能（a）运行时在**首次推理**时把用户 buffer 拷进自己的 "Input" 段（那块可能不在 maps 的 dmabuf
   行里，例如走 `/dev/ion` 老接口或 `memfd`），也可能（b）它用了我们的 buffer 但 offset/映射属性
   让 NPU 读到的是旧页。要定论需要驱动侧证据（`MemImport`/IOMMU map 的 dump）或 V2 的对照实验。
2. **锁存发生在 setInput 还是首次 invoke**：本票的所有实验都是"bind → 第一次 invoke → 再写"。
   没做"bind A → bind B（都不 invoke）→ invoke"来分离这两个时点。对结论无影响（两种都不支持更新）。
3. **`ionFd` 那 3.9 ms 的来源未取证**：怀疑是"非 ION 输入每次推理要做一次 25 MiB 的缓存维护/中转"，
   但没抓到内存流量证据（`/proc/self/io` 只统计文件 I/O）。
4. **T=1 依旧编不过**（票 06 的硬边界），本票沿用 T=8 只用第 0 行，7/8 的算力浪费仍在。
5. **窗口滚动（m ≥ C）没跑**：8 步离 2048 还很远。
6. **6,144 B 的写到底落没落到 DRAM**：读回（经 mmap）看到 20000，但读回测不出写合并缓冲是否已经
   落盘。不过 ping-pong 那次是 12 MB 整块 memcpy + 换 fd 重新绑，仍然无效，所以"没落盘"解释不了全部。

---

## 10. 复现

```
export NPU_LOCK="/d/deepseek_ workspace/mtk/.npu-board.lock"

# 1) 产物（tflite -> dla -> 抽共享权重）；本票与票 06 用同一条命令
/usr/bin/python3 tools/build_q06.py --mode decode --t 8 --cache 2048 --layers 24 \
  --tab ops/q06_tab.json --out ops/q06_decode_t8_c2048_24l
./tools/compile.sh ops/q06_decode_t8_c2048_24l/model.tflite ops/q06_decode_t8_c2048_24l/model.dla
# （prefill 图同法；extract-shared 出 16,674,816 B 的 bin）

# 2) runner
./tools/build_kv1.sh                     # -> out/kv1/kv1_runner（NDK r26d aarch64）

# 3) 板端
./tools/devlock.sh run -- ./tools/healthcheck.sh --quick
./tools/devlock.sh run -- ./tools/e2e_kv1.sh --steps 8
./tools/devlock.sh run -- ./tools/healthcheck.sh --quick
```

runner 的开关（都是本票的证据工具，不是产品接口）：

| 开关 | 作用 |
|---|---|
| `--kv-heap host` | KV 用匿名内存（`ionFd=-1`），C 臂 |
| `--sync 1` | 每次 host 写前后各一次 `DMA_BUF_IOCTL_SYNC` |
| `--noconv` | `EnvOptions.suppressInputConversion/OutputConversion = true` |
| `--rebind` | 每次写完用同一 ptr 再 `setInput` |
| `--pingpong` | 再分配一对 dma-buf，每次写完换新 ptr+fd 重新 `setInput` |
| `--poison N` | 新行不写图输出，改写常量 N（毒化实验） |
| `--dumpcache F` | 结束时把两块 KV buffer 整块写盘（验证"写落在哪") |
| `--steps S --stepdir DIR --outdir DIR --m0 M` | 多 token 常驻序列（DIR/stepNN/*.bin） |
