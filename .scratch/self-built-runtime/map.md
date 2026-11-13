# 地图：自建库跑通 1 层 decoder

> 目的地（默认值，一句话可改）：用我们自己的库（不经 libneuron / neuronrt）在 NPU 上跑通 1 层 decoder —— [T,896] 进、[T,896] 出，同时满足两道验收：①同源等价（与厂商 .dla 路径输出逐字节相同）；②数值正确（与 CPU golden 按契约对拍）。
> 24 层 + decode + KV 自持有是下一张图。

## Destination

自建库跑通 1 层 decoder（prefill T=8/16 一次），全程不经过厂商运行时；验收 = 同源等价 + CPU golden 两道。

## Notes

- 领域文档：CONTEXT.md（自建提交器 / 命令捕获 / 命令自生成 / IoVA）、docs/OWN-SUBMITTER.md（现有覆盖与缺口）、docs/ISSUE-const-fc.md（常量权重 FC 悬案）、docs/adr/0002（自建 = 执行路径）、docs/MAPPING.md（缓冲与地址字布局）。
- 技能：本图用 grilling + domain-modeling；research 类票用 research。
- 边界（默认值，需人确认）：命令流来源允许用厂商 ncc 产物（捕获 + 地址重定位）；不允许调 libneuron / neuronrt；先做 fp16 档（int8/8W16A 迁移单列一票）。
- 判据纪律：①同源等价用逐字节；②数值正确用 CPU golden；两者不得互相替代。板端计时只报配对交替的比值中位数。

## Decisions so far

（暂无 —— 本图刚建立）

## Not yet specified

- 命令自生成（L4）的可行性：取决于票 02（资产参数化边界）—— 如果资产数在 1 层就不可维护，L4 就从「以后再说」变成唯一出路。
- 多图调度与 KV 归属：自建库要不要自己持有 KV，取决于票 01（常量权重）与票 05（精度档）。
- 24 层形态：本图只到 1 层；24 层的调度/权重驻留/编译规模是下一张图的雾。

## Out of scope

- 厂商 runtime 的部署优化（KV 常驻 RuntimeV2、KV 长度扫描、--gno=LTS 等）—— 已归档在 docs/adr/0006、docs/OP-kv1-resident-kv.md、docs/OP-kv-maxlen.md、docs/OP-ncc-flags.md；它们服务于厂商形态那条线，不推进自建。
- 量化档调优（4W/8W/16A 矩阵）—— 已在 quant-runtime 收口。
- 长上下文（C>8192）与窗口滚动 —— 需要时另开一张图。
