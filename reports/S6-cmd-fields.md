# S6 · 命令缓冲里的形状字段：三形状差分

时间：2026-11-12 · 同一板/同一工具链（ncc 7.3.15, mdla5.0）。
抓取方式见 `reports/S5-cmdbuf-capture.md`（`probe/iolog2.c` 在 `CMD RUN` 前快照 400 B 的命令区）。

## 1. 实验

三个形状的 FULLY_CONNECTED，int8，M=1：

| 模型 | K | N | `.dla` |
|---|---|---|---|
| `fc_1x64x64_int8` | 64 | 64 | 7545 B |
| `fc_1x64x128_int8` | 64 | 128 | 11899 B |
| `fc_1x128x128_int8` | 128 | 128 | 20093 B |

命令区（400 B = 100 个 32-bit 字）在三个模型里**大小完全相同**，因此可以逐字对齐做差分。

## 2. 结果：25/100 个字随形状变化

（`K-dependent` = 只在 K 变化时变；`N-dependent` = 只在 N 变化时变；`both` = 两者都影响，通常是地址/步长）

| word | 偏移 | 64×64 | 64×128 | 128×128 | 依赖 |
|---|---|---|---|---|---|
| w5 | 0x014 | 0x06000040 | 0x06000080 | 0x06000080 | K |
| w18 | 0x048 | 0x00400000 | 0x00800000 | 0x00800000 | K |
| w23 | 0x05c | 0x025c3800 | 0x025c2800 | 0x025c3000 | 地址 |
| w28 | 0x070 | 0x40 | 0x40 | 0x80 | **N** |
| w29 | 0x074 | 0x0802001f | 0x0802001f | 0x0803102f | **N** |
| w30 | 0x078 | 0x04000040 | 0x04000040 | 0x04000080 | **N** |
| w39 | 0x09c | 0x401e0005 | 0x401e0005 | 0x402e0005 | **N** |
| w40 | 0x0a0 | 0x025c3810 | 0x025c2820 | 0x025c3020 | 地址 |
| w41 | 0x0a4 | 0x10 | 0x20 | 0x40 | 步长（随 K/N 变） |
| w43 | 0x0ac | 0x40 | 0x40 | 0x80 | **N** |
| w44 | 0x0b0 | 0x40 | 0x80 | 0x80 | K |
| w50–w52 | 0x0c8–0x0d0 | 0x40 | 0x40 | 0x80 | **N** |
| w59 | 0x0ec | 0x10040000 | 0x10080000 | 0x10080000 | K |
| w60 | 0x0f0 | 0x0002001f | 0x0002001f | 0x0003102f | **N** |
| w61 | 0x0f4 | 0x1101e000 | 0x1101e000 | 0x1102e000 | **N** |
| w62–w63 | 0x0f8–0x0fc | 0x40 | 0x40 | 0x80 | **N** |
| w82 | 0x148 | 4 | 8 | 8 | K（=K/16） |
| w83–w84 | 0x14c–0x150 | 0x40 | 0x80 | 0x80 | K |
| w87 | 0x15c | 0x40 | 0x80 | 0x80 | K |
| w95 | 0x17c | 0x40 | 0x80 | 0x80 | K |
| w97 | 0x184 | 0x00400001 | 0x00800001 | 0x00800001 | K |

剩余 75 个字在所有形状下**完全不变** → 属于算子类型/常量/通用配置，而不是 shape。

## 3. 能得出的结论（以及不能得出的）

**可以确定**：
1. 命令区是**定长 400 B**，与 K/N 无关 → "同族算子共享同一份命令模板，只改 25 个字"这一假设成立，L4 的参数化改造量被量化了。
2. 形状以**裸整数**出现在命令字里（0x40/0x80 直接就是 64/128），不是压缩编码 → 生成器可以直接按字段改写。
3. w82 = K/16（64→4，128→8），是**分块数**而不是维度本身；w5/w18/w59/w97 是 K 的直接编码（含高位打包）。
4. 地址类字段（w23/w40）随**参数区大小**变化，说明它们指向 8 KiB/20 KiB 的参数区（map11），不是输入/输出 buffer。

**不能确定（需要 ISA 表，不猜）**：
- 每个字段的寄存器/描述符名字与位域（例如 w29=0x0803102f 里哪几位是 tile K/N、哪几位是 opcode）；
- 0x025c3xxx 地址与 MAP 返回的 0x25c5f000/0x25c78000 不是同一段，说明命令里用的是**另一组 IOVA**（可能是参数区内部偏移或另一块 buffer），需与 `.dla` 的 relocation 表对照；
- M>1 时是否有额外字段（本次只测 M=1）。

## 4. 复现

```bash
# 1) 三个形状各编一份
/usr/bin/python3 tools/build_ops.py fc --m 1 --k 64  --n 64  --dtype int8 --out ops/fc_1x64x64_int8/model.tflite
/usr/bin/python3 tools/build_ops.py fc --m 1 --k 64  --n 128 --dtype int8 --out ops/fc_1x64x128_int8/model.tflite
/usr/bin/python3 tools/build_ops.py fc --m 1 --k 128 --n 128 --dtype int8 --out ops/fc_1x128x128_int8/model.tflite
./tools/compile.sh ...

# 2) 用 S5 的 shim 各跑一次，取 map10（400 B 命令区）
adb shell '... LD_PRELOAD=$PWD/probe/libiolog2.so IODIR=$PWD/probe/buf ... neuronrt -m hw -a dla/xxx.dla ...'

# 3) 差分
/usr/bin/python3 probe/diff_cmd.py probe/buf/map10_64x64.bin probe/buf/map10_k64n128.bin probe/buf/map10_128x128.bin
```

## 5. 下一步

1. **A 路（最省）**：把 `readdlb`/`ncc-tflite --show-exec-plan` 的输出与这 400 B 对齐，给字段起名（有编译器自己的 exec plan 做对照，比猜 ISA 可靠）。
2. **C 路（L4 入口）**：写 `apusctl`，用**已抓到的命令区字节 + 已知 handle/IOVA** 直接复现一次 `CMD RUN`，确认"自己提交"能跑出同样结果；再逐个改 shape 字段做 A/B 验证 §2 的归类。
