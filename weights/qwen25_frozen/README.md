# weights/qwen25_frozen —— 冻结随机权重夹具（Qwen2.5-0.5B 形状，fp16）

**这个目录里唯一入库的文件是 `manifest.json` 与本 README。** 每个张量的 `<name>.npz` 是
可复现产物（约 332 MB），按 `.gitignore` 的 `*.npz` 规则不入库。

## 怎么拿到权重

```bash
# 生成（WSL 的 /usr/bin/python3；约 8 s，332 MB）
wsl.exe -e bash -lc "cd '<你的 worktree>' && /usr/bin/python3 tools/gen_qwen25_weights.py"
```

同一 seed 生成的结果**逐字节相同**（`tools/gen_qwen25_weights.py --verify` 会重新生成并与
manifest 里的 sha256 逐一比对，25/25 张量应全 OK）。

## 怎么用

```python
import sys; sys.path.insert(0, "<repo>/tools")
import qwen25_weights as qw

w = qw.get("layer0.attn.q_proj.weight")   # fp16 [896,896]，只读，加载即校验 sha256
```

目录解析顺序：显式参数 → 环境变量 `QWEN25_WEIGHTS_DIR` → `<repo>/weights/qwen25_frozen`。
**没有自己生成产物的 worktree，把 `QWEN25_WEIGHTS_DIR` 指向已有的产物目录即可**（不要跨机各生成各的，
见 `docs/OP-weights.md` §11）。

## 契约（冻结，下游不要改名）

张量名 / shape / dtype / sha256 自票 02 起冻结；新增只允许在清单尾部追加张量（例如 `layer2.*`），
不得改名，也不得改已生成张量的 seed 派生规则。完整清单、实测统计量与取舍见 `docs/OP-weights.md`。
