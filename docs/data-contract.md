# Data contract

`kt-audit` 接受 **一行一个交互/预测** 的宽表。CSV（UTF-8）或 JSONL；Python API 直接收 `pandas.DataFrame`。

## 列

| 列 | 类型 | 必填 | 约束 |
| --- | --- | :--: | --- |
| `user_id` | string/int | 是 | 非空；同一用户内 timestamp 可比较 |
| `timestamp` | ISO8601 / 数值 | 是 | 可解析；不要混用 naive 与 timezone-aware |
| `split` | string | 是 | `train` / `valid` / `test`；`val`、`validation` 会归一为 `valid` |
| `kc_ids` | string/list | 是 | 管道分隔或 `["kc_a","kc_b"]`；至少一个 |
| `item_id` | string/int | 否 | 缺失时跳过 multi-KC 检查 |
| `y_true` | 0/1 | 否 | 有 `y_pred` 时必填；不能是 NaN |
| `y_pred` | float | 否 | 有限值，且 `0 <= y_pred <= 1` |
| `fold` | int/string | 否 | 有 fold 时按 fold 分别跑检查 |
| `sequence_id` | string/int | 否 | 用于序列级检查（v0.2） |
| `position` | int | 否 | 序列内位置（v0.2） |

## 解析规则

- `split`：转小写并 strip；`val` / `validation` -> `valid`；其他值报错。
- `kc_ids`：以 `[` 开头按 JSON 数组解析；否则按 `|` 分割、strip、去空。
- `timestamp`：优先 `pandas.to_datetime`；数值保持数值。若解析后既有 tz-aware 又有 naive，报错。
- `user_id` / `item_id`：统一转字符串，保留原始值用于报告。
- `y_true`：只接受 0/1（含 `0.0`/`1.0`）。
- `y_pred`：必须是有限浮点，范围 [0,1]。

## 主键

按可用列依次选择：

1. `(fold, user_id, item_id, timestamp)`
2. `(fold, user_id, timestamp, sequence_id, position)`
3. `(fold, user_id, timestamp)`

主键重复记一条 `alignment` error。

## 最小示例

```csv
user_id,item_id,kc_ids,timestamp,split,y_true,y_pred
u1,q1,kc_a|kc_b,2024-01-01T10:00:00,train,1,0.80
u1,q2,kc_b,2024-01-01T10:05:00,train,0,0.30
u2,q1,kc_a|kc_b,2024-01-02T09:00:00,test,1,0.75
u2,q3,kc_c,2024-01-02T09:10:00,test,0,0.20
```

这个例子里 `q1` 同时出现在 train/test，且 `kc_a` 只在 train、`kc_b` 在两边都有——multi-KC 检查会给出 warn/error，具体看配置。

## pyKT 映射（v0.2）

pyKT 的列名不同（`uid`、`questions`、`concepts` 等）。v0.1 不内置 adapter；示例用合成数据，避免数据集授权问题。v0.2 再加 `--adapter pykt`。

## 限制

- 时间戳只用于 split 泄漏检查，不推断因果。
- 没有 `item_id` 时无法检查题目级多 KC 泄漏。
- 没有 `y_pred` 时只做数据/泄漏检查，不算指标。
- 跨 fold 的重复不一定是错；默认按 fold 隔离检查。
