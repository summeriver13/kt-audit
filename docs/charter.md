# kt-audit charter

> 状态：draft，2026-10-20。目标：4 周 MVP，每周 2-4 小时。

## 一句话

**Before you trust an AUC, audit the evaluation.**

## 问题

知识追踪（KT）论文和库都在报 AUC，但评估协议不一致：

- split 是否按时间？用户是否跨 split？
- 一道题有多个知识点时，训练集和测试集是否共享了同一道题的不同 KC？
- question-level 与 KC-level 的预测是否被混用？
- 报告的 AUC 和用公开预测重算的 AUC 是否一致？

pyKT 是事实上的 KT benchmark，但无 CI、几乎无测试、外部 PR 通道已冷。研究者需要一个**轻量、可机器读取、能放进论文 artifact** 的审计工具。

## 用户

- 准备投稿或放 artifact 的 KT 研究者。
- 接 KT 模型做产品原型的工程师。
- 复现实验、审稿、带学生的研究生。

## 非目标

- 不训练 DKT/AKT，不做模型库。
- 不做 leaderboard，不托管数据集。
- 不证明因果，只标记可疑模式。
- v0.1 不做 LLM-KT 评估。
- 不替代人工判断；报告里的 finding 是证据，不是判决。

## MVP：三个检查

| 检查 | 问题 | 默认严重度 |
| --- | --- | --- |
| temporal split | 同一用户的 train 事件是否晚于 valid/test？用户是否跨 split？ | error / warn |
| multi-KC leakage | 同一道题是否同时含有 train-only 和 test-only 的 KC？ | error |
| prediction alignment | 是否有缺失、重复、越界、常量预测？重算 AUC 是否与报告一致？ | error / warn |

### 1. temporal split

- 同一 user 的 `max(train timestamp) >= min(valid/test timestamp)`：error。
- 同一 user 同时出现在多个 split：warn（可能是随机 split，也可能泄漏）。
- 全局 split 时间区间重叠：warn。
- 没有 `timestamp`：跳过并 warn。

### 2. multi-KC leakage

- 构建 `item_id -> set(kc_ids)`。
- 同一道题的 KC 集合里，既有只在 train 出现的 KC，又有只在 test 出现的 KC：error。
- 同一道题同时出现在 train 和 test：warn。
- 没有 `item_id`：跳过并 warn。
- 这是启发式，不是因果证明；报告必须写清。

### 3. prediction alignment

- `y_pred` 为 NaN/inf/越界：error。
- 主键重复（`user_id + item_id + timestamp`）：error。
- 预测常量（标准差为 0）：warn。
- 可选 `reported_auc`：与重算 AUC 差值超过 tolerance：error。
- 没有 `y_pred`：跳过指标检查。

## 输入契约

一行一个交互/预测。CSV 或 JSONL。

| 列 | 必填 | 说明 |
| --- | :--: | --- |
| `user_id` | 是 | 用户 ID |
| `timestamp` | 是 | ISO8601 或数值；同一用户内可比较 |
| `split` | 是 | `train` / `valid` / `test` |
| `kc_ids` | 是 | 管道分隔或 JSON 数组 |
| `item_id` | 否 | 题目 ID；缺失时跳过 multi-KC 检查 |
| `y_true` | 否 | 0/1；有 `y_pred` 时必填 |
| `y_pred` | 否 | [0,1]；缺失时跳过指标检查 |
| `fold` | 否 | 交叉验证折 |
| `sequence_id` | 否 | 序列 ID |
| `position` | 否 | 序列内位置 |

完整说明见 `docs/data-contract.md`。

## 输出

`kt-audit run events.csv --out report/` 生成：

- `report.json`：机器可读，带 `schema_version`、`findings`、`metrics`、`verdict`。
- `report.md`：人读，先给 verdict，再给证据与限制。

退出码：

- `0`：pass 或 warn。
- `1`：fail（至少一个 error）。
- `2`：输入/配置错误。

## 设计约束

- 运行依赖只加 `numpy`、`pandas`、`scikit-learn`。
- CLI 用标准库 `argparse`，不引入 Typer/Click。
- 配置用 JSON，不引入 YAML/TOML 解析器。
- `src/` 布局；包名 `kt_audit`，命令 `kt-audit`。
- 每个 check 是纯函数：输入 DataFrame + config，输出 Finding 列表。
- 报告排序稳定，示例截断，避免大文件。
- 对外 README/文档英文；内部 charter/notes 中文。

## 4 周成功标准

- `uv run pytest` 绿，三个 check 都有正反例。
- `kt-audit run examples/leaky.csv --out /tmp/report` 能生成 JSON + Markdown。
- 示例报告能抓到三类故意泄漏。
- GitHub Actions 在 Python 3.10-3.13 上绿。
- 一个陌生人按 README 能在 5 分钟内跑出报告。

**不以下载量、stars、外部 PR 数作为早期成功标准。**

## 后续（v0.2+）

- pyKT 输出 adapter。
- bootstrap AUC 置信区间。
- HTML 报告。
- 更多检查：重复交互、冷启动、序列截断、标签泄漏。
- 真实数据集 case study（注意授权）。

## 风险

- 启发式会有假阳性；必须提供 examples 和 limitations。
- 容易膨胀成“又一个 KT 框架”；v0.1 冻结在三个检查。
- 数据集不能随便打包；示例用合成数据。
- 维护时间有限；每周 2-4 小时，不承诺 SLA。

## 许可与治理

- MIT。
- 初始单维护者；PR 欢迎，但不承诺响应时间。
- AI 可参与开发，但每条 finding 的定义和测试必须有人复核。
