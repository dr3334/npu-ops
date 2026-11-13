# ops/_t128_bisect —— T=128 编译失败的二分证据（结论：我自己的 bug，不是编译器限制）

这个目录保留的是 2026-11-13 做 T=128 时的**二分过程**，因为它同时是一条教训。

## 当时的现象

T=128 的整块图在 ncc-tflite 7.3.15 上 **EXIT=1**，报错停在 `Fail to process the input model`
（加载/逐层校验期），逐层报的是 5 个 FULLY_CONNECTED：

    ERROR: Invalid FullyConnectedLayer<6>{FULLY_CONNECTED}: FC batch should be equal to input_size / filter_channel
    (还有 FC<16> q/k/v、FC<50>/FC<51> gate/up)

T=8 的同一张图 EXIT=0、61 层、单执行步。

## 二分做了什么（脚本都在本目录）

| 步骤 | 脚本 | 观察 |
|---|---|---|
| 按阶段二分 | `build_decoder_layer.py --stage n1|q3|...` | `n1`（只有 RMSNorm）EXIT=0；`q3`（RMSNorm+FC+RoPE）就 EXIT=1 ⇒ 失败点在「归一化输出喂 FC」这一段 |
| 最小复现 | `probe_fc.py mul_fc` | `x → MUL → FC` 在 T=8 EXIT=0、T=9/16/128 EXIT=1；`x → FC` 在所有 T 都 EXIT=0 ⇒ 当时读成「MUL→FC 在 T>8 被拒」 |
| 换编译选项 | `--opt=0/1/2`、`--fc-to-conv`、`--avoid-reorder` | 全部复现 ⇒ 不是优化 pass |
| 干扰项试验 | `ADD(0)`、`SPLIT(1)`、`RESHAPE` 插在 MUL 与 FC 之间 | 小图上「ADD(0) 有效」、真实图上无效 ⇒ 结论互相矛盾 |
| 看图 | `diff_models.py` 对比失败图与能编过的小图 | **真相**：失败图里 `n1` 的激活张量声明成 `[8,896]`，而输入是 `[128,896]` |

## 真正的根因

`tools/build_decoder_layer.py` 里 `G.mul` / `G.add` 的默认参数写成 `shape=(T, D)`。
Python 的默认参数在**函数定义时**求值，那时 `T` 还是 8；`--T 128` 之后重新绑定的是模块全局，
默认参数不跟着变。于是 T=128 的图里有一批激活声明成 `[8,896]`，ncc 的 FC 校验看到
`input_size / filter_channel = 8` 与实际 batch 128 对不上，就报了那句 `FC batch should be ...`。

改成 `shape=None` + 函数内 `shape = (T, D)` 之后，**同一张图 EXIT=0、61 层、单执行步**，
一行业务逻辑都没改。`fc_entry`/`ADD(0)` 那套「绕开 MUL→FC」的补丁已经全部删掉。

## 教训（与 NOTES 10.4 同一条线）

「编译器的限制」这个结论，只有在**排除自己的图写错之后**才算结论（票 09 的 `MUL(x,1)` 是同一个坑）。
本次最省时间的动作不是换编译选项，而是 `diff_models.py`：把「编不过的那张图」和「能编过的小图」
按张量表逐项对比，第一次就看见了 `[8,896]`。

## 复现

    # 失败的形状（历史事实，需要 --T 默认参数 bug 版本才重现；现在只能用 diff_models 看当前图）
    /usr/bin/python3 tools/build_decoder_layer.py --T 128
    ./tools/compile.sh ops/decoder_layer_t128_fp16/model.tflite /tmp/x.dla
    # 二分小图（对照用，随口可跑）
    /usr/bin/python3 ops/_t128_bisect/probe_fc.py fc_only 128 /tmp/p
    ./tools/compile.sh /tmp/p/model.tflite /tmp/p/model.dla
