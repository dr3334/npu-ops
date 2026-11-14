# Issue 记录位置：本地 Markdown

本仓库的 issue 与 spec 都以 markdown 文件存放在 `.scratch/` 下。

**为什么不用 GitHub Issues**：`origin` 指向公开仓库，而代码与计划目前只留在本地
（见 `docs/adr/0004-local-tracker-until-public.md`）。等整块 PASS、公开时机由人决定后，
再把 spec/issue 一次性迁到 GitHub。

## 约定

- 一个特性一个目录：`.scratch/<feature-slug>/`
- spec 是 `.scratch/<feature-slug>/spec.md`
- 实现票一票一文件：`.scratch/<feature-slug>/issues/<NN>-<slug>.md`，从 `01` 起编号，**不要**写成单个合并的 tickets 文件
- triage 状态写成每个 issue 文件顶部的 `Status:` 行（角色字符串见 `triage-labels.md`）
- 评论与过程记录追加在文件底部 `## Comments` 标题下
- 阻塞关系写成顶部的 `Blocked by: NN, NN` 行；被列出的票全部完成后，本票才算解阻

## 技能说"发布到 issue tracker"时

在 `.scratch/<feature-slug>/` 下新建文件（目录不存在就建）。

## 技能说"取回相关工单"时

读取对应路径的文件。用户通常会直接给出路径或票号。

## Wayfinding 操作

供 `/wayfinder` 使用。**地图**是一个文件，每张子票一个文件。

- **地图**：`.scratch/<effort>/map.md`（Notes / Decisions-so-far / Fog 正文）。
- **子票**：`.scratch/<effort>/issues/NN-<slug>.md`，从 `01` 起编号，问题写在正文里。
  `Type:` 行记类型（`research` / `prototype` / `grilling` / `task`），`Status:` 行记 `claimed` / `resolved`。
- **阻塞**：顶部 `Blocked by: NN, NN` 行。列出的每个文件都是 `resolved` 时，本票解阻。
- **前沿**：扫 `.scratch/<effort>/issues/`，取未关闭、未解阻、未认领的文件，编号最小者胜。
- **认领**：动手前先把 `Status: claimed` 写回去。
- **结案**：正文追加 `## Answer`，置 `Status: resolved`，再把上下文指针（结论摘要 + 链接）追进 `map.md` 的 Decisions-so-far。
