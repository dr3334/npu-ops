# npu-ops

MTK MT6897 / MDLA5.0 上把神经网络算子单独编成 `.dla`、在真板执行、再与 CPU 参考对拍的逆向与实证工程。本文件只收本仓库特有的词汇，不收通用编程概念。

## 执行路径

**算子**：
只含单个计算单元的 TFLite 模型，是编译、上板、对拍的最小单位。
_Avoid_: op、模块

**DLA**：
ncc-tflite 把 `model.tflite` 编成的 MDLA5.0 命令与权重包，单文件。
_Avoid_: binary、blob、固件

**厂商 runtime**：
MediaTek 的 `libneuron` / `neuronrt`，按 DLA 执行；在本仓库里充当"算子能力上限"的参照物。
_Avoid_: 官方、SDK、参考实现

**自建提交器**：
不经过厂商 runtime，直接用 `/dev/apusys` 的 ioctl 把命令与 dma-buf 交给驱动执行的路径，文档里也称 L4。其中的命令可以是捕获来的，也可以是自生成的——两者不是一回事，见下面两条。
_Avoid_: own submitter、自研驱动、手写驱动

**命令捕获**：
用厂商 runtime 跑一次、把 cmdbuf 抓下来、再按地址打补丁重放的做法。形状烘死在命令流里，换形状就必须重新捕获。
_Avoid_: 逆向命令、偷命令、录制

**命令自生成**：
按 MDLA5.0 的 ISA 自己构造 cmd stream，不依赖任何捕获产物。这才是 L4 的完整含义，也是本仓库尚未开始的那一步。
_Avoid_: 真自建、纯自研

**L 层级**：
从"能跑通"到"能自己提交"的递进验证档位；L3 指动态权重，L4 指自建提交器。

**常量缓冲**：
编译期固化进 DLA 的权重所占的设备内存。

**动态权重**：
运行时由 host 写入设备缓冲、再被命令引用的权重路径，即 L3。

**IoVA**：
驱动分配给 dma-buf 的设备侧地址。
_Avoid_: 物理地址、PA

**devlock**：
跨 worktree 共享的设备独占锁；任何板端操作都必须持锁。
_Avoid_: 设备锁、mutex

## 证据

**能力矩阵**：
"能力 + 适用范围 + 证据编号"三列的表；没有证据的行不许写进去。`PLAN.md` §1。
_Avoid_: 支持列表、feature list、兼容性表

**对拍**：
把板端输出与 CPU 参考逐字节（整数档）或按容差（浮点档）比较的动作。
_Avoid_: 验证、比对、回归

**golden**：
CPU 侧生成的参考输出，落在 `ops/<name>/golden/`，是对拍的唯一判据。
_Avoid_: 标准答案、baseline、期望值

**适用范围**：
一条能力成立的条件边界（形状、dtype、量化档、K 的大小）；越过边界即视为未验证，不能外推。

**随机固定权重**：
一次性生成、随后冻结的随机初始化权重。板端与 CPU 参考跑同一份权重，用来把"权重从哪来"这个变量隔离在数值对拍之外。
_Avoid_: 假权重、mock 权重、合成权重、测试权重

## 目标模型

**整块**：
一个完整 decoder 层：RMSNorm → 自注意力 → 残差 → RMSNorm → SwiGLU FFN → 残差。
_Avoid_: block、decoder layer、整层、可复用单元

**prefill**：
一次把整段提示词的 T 个 token 一起前向，不保留跨步状态。
_Avoid_: 预填充、批量推理

**增量解码**：
逐 token 前向并复用 KV cache 的阶段；不在本阶段范围内。
_Avoid_: decode、自回归、生成

**GQA（分组查询注意力）**：
query head 数多于 KV head 数，一个 KV head 服务一组 query head 的注意力形式。Qwen2.5-0.5B 是 14 个 query head、2 个 KV head。

**KV head**：
K/V 投影的 head 分组，与 query head 不是同一组，数量也不等；两者不等时不能靠广播补齐。
_Avoid_: kv group、共享头

**head_dim**：
单个 head 的宽度。Qwen2.5-0.5B 的 hidden size 是 896、query head 14 个，head_dim 为 64。
_Avoid_: 头宽、维度
## 量化与精度

**精度档**：一对位宽 `(w, a)`——权重位宽与激活位宽。本仓库一律写成 `w4a8` / `w8a8` / `w4a16` / `f16`；单说「int4」不算精度档，因为分不清是权重还是激活。
_Avoid_: 多精度、混合位数、量化等级

**w4a8**：4 bit 权重 + 8 bit 激活。本仓库已能自建：编译单执行步、板端逐位（int4 解压处 ≤1 码）。

**w4a16**：4 bit 权重 + 16 bit 激活。厂商 LLM runtime 的出厂档（`asym4W_sym16A`）；同权重下比 w4a8 慢约 2 倍，厂商推荐它是为质量不是为速度。

**混合精度图**：同一张图里按子图选不同档位。厂商明确允许「attention 用 w4a16、MLP 用 w4a8」，这是本仓库的目标形态。
_Avoid_: 多精度算子（含糊：可能指给每个算子做多份，也可能指一张图里多档）

**量化契约**：int8 累加后的落码规则——先 `round_half_up(acc·M0/2^31)`，再按 shift 做 round-half-up。这是与设备逐位一致的唯一口径；用 float64 单次舍入的 golden 会系统性差 1 码。
_Avoid_: requant 规则、舍入策略

## 执行形态

**共享权重**：权重不编进 `.dla`，而是打成独立二进制、由多张图共同引用、只装载一次。厂商的 `shared_weights_*.bin` 就是它。
_Avoid_: 外部权重、权重外挂

**装载形态**：一组图在一次运行里怎么进设备内存。**常驻** = 装一次、重复执行；**逐层装载** = 每次执行前换一份 `.dla`。两者代价差一个量级，不能共用一个「加载时间」数字。

**执行计划复用**：已装载的模型重复 `Run()` 时不重新解析、不重新编译图。厂商 runner 的 `SwapModel` 只发生在 prefill ↔ decode 边界，不在每 token。
_Avoid_: 热执行、免重载
