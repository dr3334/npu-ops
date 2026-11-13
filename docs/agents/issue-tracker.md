# Issue 记录位置：GitHub

本仓库的 issue 与 spec 都存在 GitHub Issues 里（`dr3334/npu-ops`）。所有操作走 `gh` CLI。

## 约定

- **建 issue**：`gh issue create --title "..." --body "..."`。正文多行时用 heredoc。
- **读 issue**：`gh issue view <number> --comments`，评论用 `jq` 过滤，同时把 labels 一并取回。
- **列 issue**：`gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`，按需加 `--label` / `--state` 过滤。
- **评论**：`gh issue comment <number> --body "..."`
- **加 / 减标签**：`gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **关闭**：`gh issue close <number> --comment "..."`

在克隆目录内运行时，`gh` 会自己从 `git remote -v` 推断仓库，不用显式指定。

## PR 是否作为需求入口

**PRs as a request surface: no。**（本仓库不把外部 PR 当作需求来源；若以后要收，把这里改成 `yes`，`/triage` 会读到这个开关。）

改成 `yes` 之后，PR 走与 issue 相同的标签与状态，命令换成 `gh pr` 系列：

- **读 PR**：`gh pr view <number> --comments`，diff 用 `gh pr diff <number>`。
- **列外部 PR 待 triage**：`gh pr list --state open --json number,title,body,labels,author,authorAssociation,comments`，只保留 `authorAssociation` 为 `CONTRIBUTOR`、`FIRST_TIME_CONTRIBUTOR`、`NONE` 的（丢掉 `OWNER` / `MEMBER` / `COLLABORATOR`）。
- **评论 / 加标签 / 关闭**：`gh pr comment`、`gh pr edit --add-label` / `--remove-label`、`gh pr close`。

GitHub 的 issue 与 PR 共用一套编号，裸写 `#42` 可能指向任一种：先 `gh pr view 42`，失败再回退 `gh issue view 42`。

## 技能说"发布到 issue tracker"时

建一个 GitHub issue。

## 技能说"取回相关工单"时

执行 `gh issue view <number> --comments`。

## Wayfinding 操作

供 `/wayfinder` 使用。**地图（map）**是一张 issue，**子票（child）**是挂在它下面的 issue。

- **地图**：一张带 `wayfinder:map` 标签的 issue，正文承载 Notes / Decisions-so-far / Fog。`gh issue create --label wayfinder:map`。
- **子票**：以 GitHub sub-issue 关联到地图（`gh api` 打 sub-issues 端点）。未启用 sub-issue 时，把子票加进地图正文的 task list，并在子票正文顶部写 `Part of #<map>`。标签 `wayfinder:<type>`（`research` / `prototype` / `grilling` / `task`）。认领后 assign 给动手的人。
- **阻塞**：用 GitHub 原生的 **issue dependencies**，这是唯一在 UI 上可见、可依赖的表示。加边：`gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`，其中 `<blocker-db-id>` 是阻塞方的**数据库 id**（`gh api repos/<owner>/<repo>/issues/<n> --jq .id`），**不是** `#number`，也不是 `node_id`。GitHub 用 `issue_dependencies_summary.blocked_by` 报未关闭的阻塞方（这才是活的闸门）。dependencies 不可用时，回退成子票正文顶部的 `Blocked by: #<n>, #<n>` 行。所有阻塞方都关闭，票才算解阻。
- **前沿查询**：列出地图下未关闭的子票（`gh issue list --state open`，范围限于该地图的 sub-issues / task list），剔掉仍有未关闭阻塞方的（`issue_dependencies_summary.blocked_by > 0`，或 `Blocked by` 行里还有未关闭 issue）和已有 assignee 的；按地图顺序取第一个。
- **认领**：`gh issue edit <n> --add-assignee @me`，这是本会话的第一次写操作。
- **结案**：`gh issue comment <n> --body "<answer>"`，然后 `gh issue close <n>`，再把上下文指针（结论摘要 + 链接）追加到地图的 Decisions-so-far。
