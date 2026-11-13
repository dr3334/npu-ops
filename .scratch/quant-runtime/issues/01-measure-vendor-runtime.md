# 01: 量测厂商 runtime 的真实形态（内存 / tok·s⁻¹ / 加载阶段）

**What to build:** 用设备上现成的厂商资产（`/data/local/tmp/llm_sdk/main` + `config_qwen2.5_0.5b_mt6897_2048c_benchmark.yaml`）把「他们到底怎么跑」变成我们自己的数字：一次实跑，三个时点采样。

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] 先只读：`main` 的 strings/版本、config、`profile_qwen25_trace/`、两个 `.atrace`（二进制，能解就解）
- [ ] 实跑一次（持 devlock）：记录 tok/s、prefill 耗时、首 token 延迟
- [ ] 三个时点采 `cat /proc/meminfo`（MemFree/MemAvailable/Cached/SwapFree）、`dumpsys meminfo <pid>`（PSS/private dirty）、`dmabuf_dump -b`（板端有该工具时）
- [ ] 判断「加载发生在哪个阶段」：初始化后 vs decode 期间，DMA-BUF 是否稳定；文件读取是否只在初始化
- [ ] 记录权重常驻的实测形态：250.7 MB 的 `shared_weights_2048c_0.bin` 是否被一次性映射、是否与 `.dla` 分开
- [ ] 跑前备份 `prompt*.txt` 与 config；跑后恢复；把证据（日志/时间序列）落到 `docs/OP-vendor-runtime.md`