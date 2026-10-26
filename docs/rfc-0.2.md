# RFC: kt-audit 0.2

> 状态：draft，2026-10-26。作者：Xia Jiang。审阅：待定。

## 背景

v0.1 回答「协议合法吗」：切分、多 KC 泄漏、预测对齐。2026-10 的两次实测（LongTutor 复现、KT 产线实验台）显示，**协议完全合法，数字仍然可以毫无意义**。四类失效模式各有实例：

1. **可查表**：LongTutor 的诊断标签能被 7 个标量特征的决策树复现到 Acc 82.3 / F1 0.632，远高于最好 LLM 的 42.0 / 39.7。标签由规则决定，模型分数测的是"愿不愿意按规则查表"。
2. **不可区分**：13 个模型的 Acc 全部低于多数类基线 50.6，跨度 19–49 排不出顺序；而 Macro-F1 的跨度有 31.7。
3. **静默降级**：`max_tokens` 3000 时 200 条里 29 条（14.5%）静默解析失败，输出文件照写，T1 少 7 分。
4. **无来源**：产出的 JSON 没有输入哈希与工具版本，重跑覆盖后无法判读。

## 范围

新增两个检查、一个字段、一个子命令：

| 件 | 回答的问题 | 默认严重度 |
| --- | --- | --- |
| `cheap_baseline` | 报告的预测，是否只是用户历史聚合的化身？ | error / warn |
| `degenerate` | 预测分布是否退化到指标不可信？ | warn |
| provenance | 这份报告对应哪份输入、哪个工具版本？ | 字段 |
| `compare` | 两个模型差 3 分，是差异还是噪声？ | 子命令 |

### cheap_baseline

只用**过去事件**的聚合特征，训练 `DecisionTreeClassifier(max_depth=4)`，在 test 上算 AUC，与同一批行上的 `y_pred` 的 AUC 比较。

特征（全部按用户、按时间顺序，只用当前行之前的事件）：`prior_n`、`prior_acc`、`prior_gap`、`same_item_prior_acc`、`same_item_prior_n`、`kc_prior_acc`、`kd_global_acc`（按 KC 的全局累计正确率）、`position`（若有）。

判据：
- `baseline_auc >= model_auc`：error，「报告的模型没有超过只看历史聚合的基线」。
- `baseline_auc >= model_auc - 0.02`：warn。
- 无 `y_pred` 但有 `reported_auc`：与 `reported_auc` 比。
- 训练行 < 50、测试行 < 30、或测试集只有一个类别：跳过并 info。

为什么这是 KT 特有的：KT 的核心主张就是「长历史有额外信息」。如果 8 个聚合特征能追平模型，那个主张在这份数据上不成立。

### degenerate

四个判据，都只看预测分布，不看模型：

| 判据 | 阈值 | 含义 |
| --- | --- | --- |
| 不同取值太少 | `nunique < 10` 且 `n >= 30` | 预测几乎是几个档 |
| 分布过窄 | IQR < 0.02 且 `n >= 30` | 没有区分度，AUC 会贴着 0.5 |
| 样本太小 | `n < 30` | 置信区间宽到无法比较 |
| 少数类太稀 | 少数类占比 < 1% | AUC 不稳定 |

阈值是模块常量，不进 config：v0.2 只暴露开关 `degenerate`。要调的时候改常量并发版，不走配置。

### provenance

`report.json` 增加 `tool_version`，`input.sha256` 记录输入的 SHA-256。`audit_frame` 收可选 `digest` 参数，`audit_path` 负责计算。

### compare

`kt-audit compare a.csv b.csv`：按主键配对，同一批行上做配对 bootstrap（默认 2000 轮、seed 0），输出 ΔAUC 的 95% 区间。

- CI 跨 0：`tie`。
- CI 全为正：`b_better`；全为负：`a_better`。
- 主键不同、或同一主键的 `y_true` 不一致：输入错误，退出码 2。

退出码：compare 只在输入错误时非 0；它描述事实，不做判决。

## 接口变更

- `AuditConfig` 新增 `cheap_baseline`、`degenerate`、`baseline_warn_margin`。
- `Report` 新增 `tool_version`、`baselines`。
- `schema_version` 由 `0.1` 升到 `0.2`；新增字段，不改旧字段语义。
- CLI 新增 `compare` 子命令与 `--no-baseline`、`--no-degenerate`。

## 非目标

- 不训练 KT 模型，不提供 SOTA。
- 不做在线 LLM 判分审计。`judge-audit`（读两份已保存判分，算 κ、MAD、Δ 符号翻转）留到 0.3，且必须离线、确定性。
- 不实现 pyKT adapter 与 HTML 报告；原 roadmap 的这两项推迟到 0.3。
- `cheap_baseline` 是启发式：稀疏数据上树可能赢，报告必须写明限制。

## 验收标准

- 四个新件都有正反例测试；`pytest` 绿、覆盖率不低于 0.1.1。
- `uv run kt-audit run examples/floor.csv` 触发 `cheap_baseline` error。
- `uv run kt-audit run examples/degenerate.csv` 触发 `degenerate` warn。
- `uv run kt-audit compare examples/compare_a.csv examples/compare_b.csv` 输出 `b_better`。
- README、data-contract、CHANGELOG、charter 同步。
- 在 LongTutor 的五档预测文件上跑一次，作为 case study 的原始材料。

## 风险

- 阈值拍脑袋：先用两组真实数据标定，把数字写进 README，后续按反馈调。
- 特征泄漏到未来：所有特征严格用 `shift(1)` 或时间序累计，测试里放一个「未来标签不影响树」的反例。
- 工具膨胀：`compare` 不做模型比较矩阵，只做两两配对。
