---
name: log_exp
description: 记录实验到 CSV（支持 evaluation 与 training），自动识别类型；无法识别时要求显式指定。
---

# log_exp

用于把当前实验记录到 `experiments/evaluation.csv` 或 `experiments/training.csv`。

## 必要输入
- `exp_id_ref`

## 自动识别规则
- 优先根据 `script_path` 判断：
  - 包含 `/evaluation/` -> `eval`
  - 包含 `/training/` -> `train`
- 其次根据 `csv_path` 文件名判断：
  - `evaluation.csv` -> `eval`
  - `training.csv` -> `train`
- 若仍无法判断：直接报错并要求传 `--mode eval|train`（禁止猜测写入）。

## 默认值
- `mode`: `auto`
- `job_id`: 自动（SLURM_JOB_ID；否则 tmux/local + 时间戳）
- `status`: `submitted`
- `csv_path/script_path`: 按 `mode` 自动选默认

## 执行命令

```bash
bash skills/log_exp/scripts/append_exp_log.sh --exp-id-ref <exp_id_ref>
```

显式指定 eval：

```bash
bash skills/log_exp/scripts/append_exp_log.sh \
  --mode eval \
  --exp-id-ref lvsm_01 \
  --job-id 3175739
```

显式指定 train：

```bash
bash skills/log_exp/scripts/append_exp_log.sh \
  --mode train \
  --exp-id-ref lvsm_02
```

## 输出约定
- 打印 mode、写入文件、行号、完整追加行。

## 安全规则
- 校验 CSV 表头必须匹配 schema。
- 默认禁止重复 job_id（`--force` 才允许）。
- `eval` 的 `test_config` 字段自动加引号。
