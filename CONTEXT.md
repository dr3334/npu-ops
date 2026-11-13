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
