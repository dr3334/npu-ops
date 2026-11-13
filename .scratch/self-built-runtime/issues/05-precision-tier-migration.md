# 05: 自建库先做 fp16 还是 8W16A

**Type:** grilling
**Status:** open
**Blocked by:** 03

## Question

现有自建资产全是 int8；模型主档是 8W16A（int8 per-channel 权重 + int16 激活）。

**要拍**：①本图层用哪档（建议 fp16，把调度/权重驻留先隔离）；②若选 8W16A，需要哪些新资产（int16 激活 + per-channel 权重的 requant 走法）。
