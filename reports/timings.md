# 板上计时汇总（自动生成，勿手改）

来源：`reports/timings.json`；生成：`PYTHONPATH=tools /usr/bin/python3 -m npu_lib.stats --render`

口径说明：**中位数**是主指标，MAD 是离散度；跨时间/跨设备比较只认配对交替的比值，
不要拿两次绝对数相除（同一份 `.dla` 会漂 ±15%）。

| 测量 | n | 中位数 ms | MAD | p10 | p90 | 口径 | 备注 |
|---|---|---|---|---|---|---|---|
| `adb_rtt_floor` | 5 | **71.661** | 5.051 | 64.203 | 77.131 | host_wall | adb shell true 的往返时间（一切 adb 操作的地板） |
| `block_1l_fp16/device_submit_once` | 1 | **201.171** | 0.000 | 201.171 | 201.171 | device | npu_run 自报：提交+200ms 等待+读回（不含 adb） |
| `block_1l_fp16/ours_host_wall` | 5 | **294.923** | 5.987 | 289.857 | 309.183 | host_wall | 每次一个新进程 + adb shell；含 usleep(200ms) 盲等 |
| `block_qwen25_1l_fp16/vendor_ms_per_inf` | 1 | **1.039** | 0.000 | 1.039 | 1.039 | vendor_runtime | e2e --repeat 5, 厂商 runtime 自报 |
| `decoder_layer_t8_fp16/vendor_ms_per_inf` | 1 | **2.393** | 0.000 | 2.393 | 2.393 | vendor_runtime | e2e --repeat 2 厂商 runtime 自报；61 个原生层 = 一整层 decoder（D=896, T=8） |
| `npu_lib.matrix/add_1x64_int8/run_alt` | 1 | **725.829** | 0.000 | 725.829 | 725.829 | host_wall | 换一对输入的第二次 g.run() |
| `npu_lib.matrix/add_1x64_int8/run_captured` | 1 | **855.713** | 0.000 | 855.713 | 855.713 | host_wall | g.run() 一次（含 adb 推拉 + npu_run 200ms 等待） |
| `npu_lib.matrix/add_1x64_int8/total_cold` | 1 | **1731.998** | 0.000 | 1731.998 | 1731.998 | host_wall | compile + 捕获 + 两次 run + 两道门 |
| `npu_lib.matrix/add_1x896_int8/run_alt` | 1 | **707.596** | 0.000 | 707.596 | 707.596 | host_wall | 换一对输入的第二次 g.run() |
| `npu_lib.matrix/add_1x896_int8/run_captured` | 1 | **679.974** | 0.000 | 679.974 | 679.974 | host_wall | g.run() 一次（含 adb 推拉 + npu_run 200ms 等待） |
| `npu_lib.matrix/add_1x896_int8/total_cold` | 1 | **1492.068** | 0.000 | 1492.068 | 1492.068 | host_wall | compile + 捕获 + 两次 run + 两道门 |
| `npu_lib.matrix/add_1x96_int8/run_alt` | 1 | **662.860** | 0.000 | 662.860 | 662.860 | host_wall | 换一对输入的第二次 g.run() |
| `npu_lib.matrix/add_1x96_int8/run_captured` | 1 | **703.357** | 0.000 | 703.357 | 703.357 | host_wall | g.run() 一次（含 adb 推拉 + npu_run 200ms 等待） |
| `npu_lib.matrix/add_1x96_int8/total_cold` | 1 | **1480.038** | 0.000 | 1480.038 | 1480.038 | host_wall | compile + 捕获 + 两次 run + 两道门 |
| `npu_lib.matrix/mul_1x64_int8/run_alt` | 1 | **802.011** | 0.000 | 802.011 | 802.011 | host_wall | 换一对输入的第二次 g.run() |
| `npu_lib.matrix/mul_1x64_int8/run_captured` | 1 | **694.680** | 0.000 | 694.680 | 694.680 | host_wall | g.run() 一次（含 adb 推拉 + npu_run 200ms 等待） |
| `npu_lib.matrix/mul_1x64_int8/total_cold` | 1 | **1594.357** | 0.000 | 1594.357 | 1594.357 | host_wall | compile + 捕获 + 两次 run + 两道门 |
| `npu_lib.matrix/mul_1x896_int8/run_alt` | 1 | **690.330** | 0.000 | 690.330 | 690.330 | host_wall | 换一对输入的第二次 g.run() |
| `npu_lib.matrix/mul_1x896_int8/run_captured` | 1 | **692.214** | 0.000 | 692.214 | 692.214 | host_wall | g.run() 一次（含 adb 推拉 + npu_run 200ms 等待） |
| `npu_lib.matrix/mul_1x896_int8/total_cold` | 1 | **1493.812** | 0.000 | 1493.812 | 1493.812 | host_wall | compile + 捕获 + 两次 run + 两道门 |
| `npu_lib.matrix/mul_1x96_int8/run_alt` | 1 | **790.807** | 0.000 | 790.807 | 790.807 | host_wall | 换一对输入的第二次 g.run() |
| `npu_lib.matrix/mul_1x96_int8/run_captured` | 1 | **698.120** | 0.000 | 698.120 | 698.120 | host_wall | g.run() 一次（含 adb 推拉 + npu_run 200ms 等待） |
| `npu_lib.matrix/mul_1x96_int8/total_cold` | 1 | **1581.226** | 0.000 | 1581.226 | 1581.226 | host_wall | compile + 捕获 + 两次 run + 两道门 |
| `npu_lib.matrix/sub_1x64_int8/run_alt` | 1 | **731.612** | 0.000 | 731.612 | 731.612 | host_wall | 换一对输入的第二次 g.run() |
| `npu_lib.matrix/sub_1x64_int8/run_captured` | 1 | **696.515** | 0.000 | 696.515 | 696.515 | host_wall | g.run() 一次（含 adb 推拉 + npu_run 200ms 等待） |
| `npu_lib.matrix/sub_1x64_int8/total_cold` | 1 | **1515.389** | 0.000 | 1515.389 | 1515.389 | host_wall | compile + 捕获 + 两次 run + 两道门 |
| `npu_lib.matrix/sub_1x896_int8/run_alt` | 1 | **766.042** | 0.000 | 766.042 | 766.042 | host_wall | 换一对输入的第二次 g.run() |
| `npu_lib.matrix/sub_1x896_int8/run_captured` | 1 | **775.534** | 0.000 | 775.534 | 775.534 | host_wall | g.run() 一次（含 adb 推拉 + npu_run 200ms 等待） |
| `npu_lib.matrix/sub_1x896_int8/total_cold` | 1 | **1638.914** | 0.000 | 1638.914 | 1638.914 | host_wall | compile + 捕获 + 两次 run + 两道门 |
| `npu_lib.matrix/sub_1x96_int8/run_alt` | 1 | **721.636** | 0.000 | 721.636 | 721.636 | host_wall | 换一对输入的第二次 g.run() |
| `npu_lib.matrix/sub_1x96_int8/run_captured` | 1 | **710.091** | 0.000 | 710.091 | 710.091 | host_wall | g.run() 一次（含 adb 推拉 + npu_run 200ms 等待） |
| `npu_lib.matrix/sub_1x96_int8/total_cold` | 1 | **1531.849** | 0.000 | 1531.849 | 1531.849 | host_wall | compile + 捕获 + 两次 run + 两道门 |
