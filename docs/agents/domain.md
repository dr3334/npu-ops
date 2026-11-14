# 领域文档（Domain Docs）

技能在探索本仓库代码库时，应当如何消费本仓库的领域文档。

本仓库是**单上下文（single-context）**布局：根目录一份 `CONTEXT.md`，决策记录放 `docs/adr/`。

## 开始探索前先读这些

- 根目录的 **`CONTEXT.md`**，或
- 根目录的 **`CONTEXT-MAP.md`**（若存在）：它指向每个上下文各一份 `CONTEXT.md`，读与当前主题相关的那些。
- **`docs/adr/`**：读与即将改动的区域相关的 ADR。多上下文仓库里还要看 `src/<context>/docs/adr/` 下的上下文级决策。

上述文件不存在时**静默继续**：不要点出缺失，也不要主动提议先创建它们。`/domain-modeling` 技能（经 `/grill-with-docs` 与 `/improve-codebase-architecture` 到达）会在术语或决策真正定下来时才按需创建。

## 文件结构

单上下文仓库（大多数仓库，包括本仓库）：

```text
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-event-sourced-orders.md
│   └── 0002-postgres-for-write-model.md
└── src/
```

多上下文仓库（根目录有 `CONTEXT-MAP.md` 时）：

```text
/
├── CONTEXT-MAP.md                    ← 系统级决策
├── docs/adr/
└── src/
    ├── ordering/
    │   ├── CONTEXT.md
    │   └── docs/adr/                 ← 上下文级决策
    └── billing/
        ├── CONTEXT.md
        └── docs/adr/
```

## 使用术语表里的词汇

产出里点名某个领域概念时（issue 标题、重构提案、假设、测试名），用 `CONTEXT.md` 里定义的词，不要漂到术语表明确避免的同义词上。

需要用的概念还不在术语表里，这本身是信号：要么你在发明项目不用的语言（重新考虑），要么是真的缺口（记下来交给 `/domain-modeling`）。

## 冲突的 ADR 要显式指出

产出与既有 ADR 相矛盾时，明确写出来，而不是悄悄覆盖：

> _与 ADR-0007（event-sourced orders）冲突，但值得重开，因为……_
