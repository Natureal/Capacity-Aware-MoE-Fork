# Residual increment / energy-mass statistics

Script: [`scripts/moe_residual_increment_stats.py`](../../scripts/moe_residual_increment_stats.py)

## Metric (replaces N_eff)

For each token and layer, compute expert-slot contributions matching the MoE block:

`Delta_h[i] = w_i * E_i(x)`, `r_i = ||Delta_h[i]||_2`.

Sort slots by **energy** `r_i^2` (descending). **m@θ** = smallest number of slots such that cumulative energy ≥ θ × total energy (default θ=0.9).

This measures how many experts are needed to explain most of the **actual MoE residual branch energy**, not router softmax shape.

**Cost:** the hook recomputes expert forwards (≈2× expert compute per layer per batch); keep `--max_texts` small.

## Run

```bash
conda activate capacity-moe   # or your torch env
cd /path/to/Capacity-Aware-MoE
CUDA_VISIBLE_DEVICES=<free_gpu> python scripts/moe_residual_increment_stats.py \
  --model ./lm-evaluation-harness/models/OLMoE-1B-7B-0924 \
  --device cuda --max_texts 24 --max_length 64 --mass 0.9 \
  --output_dir ./experiments/residual_increment_outputs
```

### PIQA 请求示例（任务数据）

```bash
python scripts/moe_residual_increment_stats.py \
  --source piqa --piqa_split validation \
  --max_texts 16 --max_length 128 --report_examples 8 \
  --output_dir ./experiments/residual_increment_outputs_piqa
```

每条样本格式：`Goal: ...\\n(A) ...\\n(B) ...`。`residual_increment_summary.json` 里 `per_example` 含每条请求的 `m_needed_mean`、`piqa_meta`（label、goal 预览）及每层 `per_layer_mean_m_needed`。

Outputs: `residual_increment_summary.json`, `residual_increment_report.txt`.

## Legacy

Previous N_eff script `scripts/moe_routing_neff.py` has been removed in favor of this file.
