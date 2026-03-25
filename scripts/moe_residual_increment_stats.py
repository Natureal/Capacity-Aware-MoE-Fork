#!/usr/bin/env python3
"""
MoE 残差支路增量分析（不用 gate 熵 / N_eff）。

对每个 token、每一层、每个 top-k slot：
  Delta_h[i] = w_i * E_i(x)   （与 OlmoeSparseMoeBlock 中一致）
  r_i = ||Delta_h[i]||_2

在「贡献 L2 能量」意义下排序（r_i^2 从大到小），定义：
  m@theta = 最小的 m，使得 sum_{前 m 个} r_(j)^2 >= theta * sum_j r_j^2

由此回答：
  (1) 同一 layer 内，不同 token 的 m@theta 是否差异大；
  (2) 同一 token 位置，不同 layer 的 m@theta 是否差异大。

注意：在 forward hook 里会再算一遍各 expert 前向（与主 forward 重复），仅适合小规模统计。

baseline 路由：expert_capacity=None, strategy=None。
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]


def _register_local_olmoe():
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers import AutoModel, AutoModelForCausalLM

    path = REPO_ROOT / "modeling_hf" / "modeling_olmoe.py"
    if not path.is_file():
        raise FileNotFoundError(f"Local OLMoE not found: {path}")
    spec = importlib.util.spec_from_file_location("capacity_aware_olmoe", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    AutoModel.register(OlmoeConfig, mod.OlmoeModel, exist_ok=True)
    AutoModelForCausalLM.register(OlmoeConfig, mod.OlmoeForCausalLM, exist_ok=True)


def _percentiles(x, qs=(10, 25, 50, 75, 90)):
    if x.numel() == 0:
        return {}
    x = x.float().flatten()
    out = {}
    for q in qs:
        out[f"p{q}"] = float(torch.quantile(x, q / 100.0).item())
    return out


def _contrib_l2_per_slot(module, hidden_states_bsh):
    """
    hidden_states_bsh: [B, S, H] —— 与进入 mlp 的 tensor 一致（已 layernorm）
    返回 slot_norms [T, top_k]，slot_norms[t,s] = || w_{t,s} * E_{t,s}(x) ||_2
    """
    b, s, h = hidden_states_bsh.shape
    flat = hidden_states_bsh.reshape(-1, h)
    logits = module.gate(flat)
    scores = F.softmax(logits.float(), dim=-1)
    w, idx = torch.topk(scores, k=module.top_k, dim=-1, sorted=False)
    t = flat.shape[0]
    k = module.top_k
    device = flat.device
    slot_norms = torch.zeros(t, k, device=device, dtype=torch.float32)

    for expert_idx in range(module.num_experts):
        match = idx == expert_idx
        if not match.any():
            continue
        tok_idx, slot_idx = torch.where(match)
        h_sub = flat[tok_idx]
        out_e = module.experts[expert_idx](h_sub).float()
        w_rel = w[tok_idx, slot_idx].unsqueeze(-1)
        contrib = w_rel * out_e
        slot_norms[tok_idx, slot_idx] = contrib.norm(dim=-1)

    return slot_norms


def _m_at_mass(slot_norms, mass, top_k):
    """
    slot_norms: [T, k] 每个 slot 的 ||Delta h||_2
    按 r^2 从大到小累积，求最小 m 使累积能量 >= mass * 总能量。
    返回 m_needed: [T] long，取值 1..k
    """
    e = (slot_norms.float() ** 2).clamp_min(0.0)
    total = e.sum(dim=-1).clamp_min(1e-20)
    e_sorted, _ = torch.sort(e, dim=-1, descending=True)
    cum = torch.cumsum(e_sorted, dim=-1)
    frac = cum / total.unsqueeze(-1)
    ge = frac >= mass
    # 第一个 True 的下标 +1 => 需要的 expert 个数
    idx_first = ge.long().argmax(dim=-1)
    any_ge = ge.any(dim=-1)
    m = idx_first + 1
    m = torch.where(any_ge, m, torch.full_like(m, top_k))
    return m


def _top1_energy_ratio(slot_norms):
    e = (slot_norms.float() ** 2).clamp_min(0.0)
    total = e.sum(dim=-1).clamp_min(1e-20)
    mx, _ = e.max(dim=-1)
    return mx / total


def _format_piqa_prompt(row) -> str:
    """PIQA：goal + 两选项，整段作为一条请求。"""
    g = (row.get("goal") or "").strip()
    s1 = (row.get("sol1") or "").strip()
    s2 = (row.get("sol2") or "").strip()
    return f"Goal: {g}\n(A) {s1}\n(B) {s2}"


def load_piqa_texts(max_texts, max_chars, split="validation"):
    texts = []
    meta = []
    try:
        from datasets import load_dataset

        ds = load_dataset("piqa", split=split, trust_remote_code=True)
        for i, row in enumerate(ds):
            t = _format_piqa_prompt(row)[:max_chars]
            if len(t) < 15:
                continue
            texts.append(t)
            lab = row.get("label")
            meta.append(
                {
                    "dataset_index": i,
                    "label": int(lab) if lab is not None else None,
                    "goal_preview": (row.get("goal") or "")[:160],
                }
            )
            if len(texts) >= max_texts:
                break
    except Exception as e:
        print(f"[warn] could not load piqa: {e}", file=sys.stderr)
    return texts, meta


def load_sample_texts(max_texts, max_chars):
    texts = []
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
        for row in ds:
            t = (row.get("text") or "").strip()
            if len(t) < 20:
                continue
            texts.append(t[:max_chars])
            if len(texts) >= max_texts:
                break
    except Exception as e:
        print(f"[warn] could not load wikitext: {e}", file=sys.stderr)

    if len(texts) < max_texts:
        fallback = [
            "The mixture of experts architecture routes each token to a small subset of feedforward networks.",
            "Scientists observed that load imbalance during inference causes straggler effects on GPUs.",
            "In natural language, some tokens are function words while others carry most of the semantic content.",
            "Machine learning systems often trade off accuracy against latency under fixed compute budgets.",
            "Neural networks with sparse activation can scale to very large parameter counts efficiently.",
            "Quantization reduces memory footprint while approximate attention lowers quadratic cost.",
            "Large language models are pretrained on diverse corpora and then aligned with human preferences.",
            "Routing networks assign each token to a few experts out of many available experts.",
            "Sparse models activate only a fraction of parameters for each forward pass.",
            "Backpropagation through expert layers requires careful handling of discrete routing.",
            "Distributed training partitions experts across devices to fit huge models in memory.",
            "Latency in MoE depends on the slowest expert in each parallel group.",
            "Function words like articles and prepositions may need fewer specialized experts.",
            "Named entities and rare tokens often exhibit distinct routing patterns.",
            "Punctuation marks are short but can appear in varied syntactic contexts.",
            "Code tokens mix identifiers and operators with structured syntax.",
            "Mathematical expressions use symbols that differ from ordinary prose.",
            "Multilingual models route tokens across many languages with shared experts.",
            "Fine-tuning adapts pretrained routers to downstream task distributions.",
            "Evaluation metrics include accuracy perplexity and task-specific benchmarks.",
            "Hardware accelerators optimize matrix multiplications for transformer layers.",
            "Batching increases throughput when sequences share similar lengths.",
            "Gradient clipping stabilizes training of very deep transformer stacks.",
            "Regularization prevents overfitting when the model has billions of parameters.",
            "Tokenizers split text into subword units that the model consumes.",
            "Residual connections help gradients flow through many layers.",
            "Layer normalization stabilizes activations across depth and batch.",
            "Rotary embeddings encode relative position without absolute positional bias.",
            "Attention maps can be sparse to reduce compute on long sequences.",
            "Caching key values speeds up autoregressive decoding in transformers.",
        ]
        for t in fallback:
            if len(texts) >= max_texts:
                break
            texts.append(t[:max_chars])

    return texts[:max_texts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=str(REPO_ROOT / "lm-evaluation-harness" / "models" / "OLMoE-1B-7B-0924"),
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_texts", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--mass", type=float, default=0.9, help="能量占比阈值，得到 m@mass")
    parser.add_argument(
        "--source",
        type=str,
        choices=("wikitext", "piqa"),
        default="wikitext",
        help="wikitext: 新闻语料（失败则用内置 fallback）；piqa: PIQA validation 上的问答请求",
    )
    parser.add_argument(
        "--piqa_split",
        type=str,
        default="validation",
        help="PIQA 的 HF split，一般为 validation",
    )
    parser.add_argument(
        "--report_examples",
        type=int,
        default=12,
        help="报告中逐条打印前多少条请求的摘要统计",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(REPO_ROOT / "experiments" / "residual_increment_outputs"),
    )
    args = parser.parse_args()

    _register_local_olmoe()
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model path not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.expert_capacity = None
    if hasattr(config, "strategy"):
        config.strategy = None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None,
    )
    model.eval()
    model.to(args.device)

    num_layers = config.num_hidden_layers
    top_k = config.num_experts_per_tok
    mass = args.mass

    per_layer_m: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}
    per_layer_top1: dict[int, list[torch.Tensor]] = {i: [] for i in range(num_layers)}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inputs, _output):
            inp = inputs[0]
            if inp.dim() != 3:
                return
            with torch.inference_mode():
                slot_norms = _contrib_l2_per_slot(module, inp)
                m_need = _m_at_mass(slot_norms, mass, module.top_k)
                t1 = _top1_energy_ratio(slot_norms)
            b, s, _ = inp.shape
            per_layer_m[layer_idx].append(m_need.view(b, s).detach().cpu().float())
            per_layer_top1[layer_idx].append(t1.view(b, s).detach().cpu().float())

        return hook

    for i in range(num_layers):
        hooks.append(model.model.layers[i].mlp.register_forward_hook(make_hook(i)))

    piqa_meta = None
    if args.source == "piqa":
        texts, piqa_meta = load_piqa_texts(args.max_texts, max_chars=256, split=args.piqa_split)
        if not texts:
            print("[error] PIQA 未加载到任何样本（检查网络或 datasets 缓存）", file=sys.stderr)
            sys.exit(1)
    else:
        texts = load_sample_texts(args.max_texts, max_chars=256)

    print(
        f"source={args.source}, n={len(texts)}, max_length={args.max_length}, mass={mass}."
    )

    with torch.no_grad():
        for text in texts:
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            enc = {k: v.to(args.device) for k, v in enc.items()}
            model(**enc)

    for h in hooks:
        h.remove()

    summary = {
        "model": str(model_path),
        "data_source": args.source,
        "piqa_split": args.piqa_split if args.source == "piqa" else None,
        "num_texts": len(texts),
        "max_length": args.max_length,
        "top_k": top_k,
        "mass_threshold": mass,
        "metric": "m_at_mass",
        "definition": (
            "Per token-layer: sort expert slots by ||w*E(x)||_2^2 descending; "
            "m is smallest count reaching cumulative energy >= mass * total energy."
        ),
    }

    # 逐条请求：第 i 次 forward 对应 texts[i]，与 per_layer_m[li][i] 对齐
    per_example = []
    n_fwd = len(texts)
    for bi in range(n_fwd):
        m_flat = torch.cat(
            [per_layer_m[li][bi].flatten().float() for li in range(num_layers)]
        )
        t1_flat = torch.cat(
            [per_layer_top1[li][bi].flatten().float() for li in range(num_layers)]
        )
        ex = {
            "index": bi,
            "seq_len_tokens": int(per_layer_m[0][bi].numel()),
            "m_needed_mean_all_layers_tokens": float(m_flat.mean().item()),
            "m_needed_std_all_layers_tokens": float(m_flat.std(unbiased=False).item()),
            "top1_energy_ratio_mean": float(t1_flat.mean().item()),
            "text_preview": texts[bi][:300],
        }
        if piqa_meta is not None and bi < len(piqa_meta):
            ex["piqa_meta"] = piqa_meta[bi]
        per_layer_mean_m = {
            str(li): float(per_layer_m[li][bi].float().mean().item())
            for li in range(num_layers)
        }
        ex["per_layer_mean_m_needed"] = per_layer_mean_m
        per_example.append(ex)
    summary["per_example"] = per_example

    per_layer_stats = {}
    all_m_concat = []

    for li in range(num_layers):
        if not per_layer_m[li]:
            continue
        cat = torch.cat([t.flatten() for t in per_layer_m[li]])
        all_m_concat.append(cat)
        cat_top1 = torch.cat([t.flatten() for t in per_layer_top1[li]])
        per_layer_stats[li] = {
            "m_needed_mean": float(cat.mean().item()),
            "m_needed_std": float(cat.std(unbiased=False).item()),
            "m_needed_min": float(cat.min().item()),
            "m_needed_max": float(cat.max().item()),
            **_percentiles(cat),
            "n_tokens": int(cat.numel()),
            "top1_energy_ratio_mean": float(cat_top1.mean().item()),
            "top1_energy_ratio_std": float(cat_top1.std(unbiased=False).item()),
        }

    summary["per_layer"] = per_layer_stats

    if all_m_concat:
        glob = torch.cat(all_m_concat)
        summary["global_all_layers_tokens_m_needed"] = {
            "mean": float(glob.mean().item()),
            "std": float(glob.std(unbiased=False).item()),
            **_percentiles(glob),
            "min": float(glob.min().item()),
            "max": float(glob.max().item()),
            "n": int(glob.numel()),
        }

    # Insight (2): one sentence, batch=1
    trace_text = texts[0] if texts else "The mixture of experts architecture routes tokens to experts."
    enc = tokenizer(
        trace_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_length,
    )
    enc = {k: v.to(args.device) for k, v in enc.items()}
    seq_len = enc["input_ids"].shape[1]

    trace_m = {i: [] for i in range(num_layers)}

    def make_trace_hook(layer_idx):
        def hook(module, inputs, _output):
            inp = inputs[0]
            with torch.inference_mode():
                slot_norms = _contrib_l2_per_slot(module, inp)
                m_need = _m_at_mass(slot_norms, mass, module.top_k)
            trace_m[layer_idx].append(m_need.view(1, -1).detach().cpu().float())

        return hook

    trace_hooks = []
    for i in range(num_layers):
        trace_hooks.append(model.model.layers[i].mlp.register_forward_hook(make_trace_hook(i)))

    with torch.no_grad():
        model(**enc)

    for h in trace_hooks:
        h.remove()

    m_lt = torch.stack([trace_m[i][0][0] for i in range(num_layers)], dim=0)
    positions_to_show = [0, min(1, seq_len - 1), min(4, seq_len - 1), min(8, seq_len - 1), seq_len // 2, seq_len - 1]
    positions_to_show = sorted(set(p for p in positions_to_show if 0 <= p < seq_len))

    trace_table = {}
    for p in positions_to_show:
        trace_table[str(p)] = [float(m_lt[li, p].item()) for li in range(num_layers)]

    summary["trace_one_sentence"] = {
        "text_preview": trace_text[:120],
        "seq_len": int(seq_len),
        "positions_m_needed_vs_layer": trace_table,
    }

    layer_stds = [per_layer_stats[i]["m_needed_std"] for i in range(num_layers) if i in per_layer_stats]
    cross_layer_std_per_pos = float(m_lt.float().std(dim=0).mean().item())

    summary["metrics"] = {
        "mean_of_per_layer_std_m_needed": float(sum(layer_stds) / max(len(layer_stds), 1)),
        "mean_over_positions_of_std_m_needed_across_layers": cross_layer_std_per_pos,
        "interpretation": {
            "insight1": "Larger per-layer std(m_needed) => tokens disagree on how many experts are needed (by mass).",
            "insight2": "Larger mean_p std_across_layers(m_needed) => same token position changes needed-expert-count across depth.",
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "residual_increment_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append(
        f"=== MoE residual increment / energy mass (m@{mass:.2f}), top_k={top_k} ===\n"
    )
    lines.append(f"Model: {model_path}\n")
    if summary.get("global_all_layers_tokens_m_needed"):
        g = summary["global_all_layers_tokens_m_needed"]
        lines.append(
            f"Global m_needed: mean={g['mean']:.4f} std={g['std']:.4f} "
            f"p10={g['p10']:.4f} p50={g['p50']:.4f} p90={g['p90']:.4f} (n={g['n']})\n"
        )
    lines.append("\n--- (1) Same layer, across tokens: per-layer std of m_needed ---\n")
    for li in range(num_layers):
        if li not in per_layer_stats:
            continue
        st = per_layer_stats[li]
        lines.append(
            f"  Layer {li:2d}: m_mean={st['m_needed_mean']:.3f} m_std={st['m_needed_std']:.3f} "
            f"top1_E1_ratio_mean={st['top1_energy_ratio_mean']:.3f}\n"
        )

    lines.append("\n--- (2) Same sentence: m_needed(layer) at token positions ---\n")
    lines.append(f"seq_len={seq_len} preview: {trace_text[:100]!r}\n")
    for p in positions_to_show:
        vals = trace_table[str(p)]
        lines.append(
            f"  pos {p:3d}: min={min(vals):.2f} max={max(vals):.2f} mean={sum(vals)/len(vals):.2f}\n"
        )

    lines.append("\n--- Aggregated ---\n")
    m = summary["metrics"]
    lines.append(f"  mean(per-layer std of m_needed): {m['mean_of_per_layer_std_m_needed']:.6f}\n")
    lines.append(
        f"  mean over pos of std(m_needed across layers): {m['mean_over_positions_of_std_m_needed_across_layers']:.6f}\n"
    )

    if per_example:
        lines.append(
            f"\n--- Per-request stats (first {min(args.report_examples, len(per_example))} examples) ---\n"
        )
        for ex in per_example[: args.report_examples]:
            lines.append(f"  [#{ex['index']}] seq_len={ex['seq_len_tokens']}  ")
            lines.append(
                f"m_mean={ex['m_needed_mean_all_layers_tokens']:.3f}  "
                f"m_std={ex['m_needed_std_all_layers_tokens']:.3f}  "
                f"top1_E1_ratio_mean={ex['top1_energy_ratio_mean']:.3f}\n"
            )
            if ex.get("piqa_meta"):
                pm = ex["piqa_meta"]
                lines.append(f"      PIQA label={pm.get('label')}  goal: {pm.get('goal_preview', '')[:100]!r}\n")
            preview = ex.get("text_preview", "").replace("\n", " ")[:140]
            lines.append(f"      text: {preview!r}...\n")

    report_path = out_dir / "residual_increment_report.txt"
    report_text = "".join(lines)
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
