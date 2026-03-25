#!/usr/bin/env python3
"""
Per-layer contribution profiling: for each MoE layer, compute the average
||w_i * E_i(x)||_2 across all token-expert assignments.

Shows how much each layer's experts actually contribute to the residual stream,
revealing which layers are "heavy" vs "light".
"""
import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]


def _register_local_olmoe():
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers import AutoModel, AutoModelForCausalLM

    path = REPO_ROOT / "modeling_hf" / "modeling_olmoe.py"
    spec = importlib.util.spec_from_file_location("capacity_aware_olmoe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    AutoModel.register(OlmoeConfig, mod.OlmoeModel, exist_ok=True)
    AutoModelForCausalLM.register(OlmoeConfig, mod.OlmoeForCausalLM, exist_ok=True)


def load_texts(max_texts, max_chars):
    texts = []
    try:
        from datasets import load_dataset
        ds = load_dataset("piqa", split="validation", trust_remote_code=True)
        for row in ds:
            g = (row.get("goal") or "").strip()
            s1 = (row.get("sol1") or "").strip()
            s2 = (row.get("sol2") or "").strip()
            t = f"Goal: {g}\n(A) {s1}\n(B) {s2}"[:max_chars]
            if len(t) >= 15:
                texts.append(t)
            if len(texts) >= max_texts:
                break
    except Exception as e:
        print(f"[warn] piqa not loaded: {e}", file=sys.stderr)
    if not texts:
        texts = [
            "The mixture of experts architecture routes each token to a subset of feedforward networks.",
            "Large language models are pretrained on diverse corpora and then aligned with human preferences.",
            "Neural networks with sparse activation can scale to very large parameter counts efficiently.",
            "Scientists observed that load imbalance during inference causes straggler effects on GPUs.",
        ]
    return texts[:max_texts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default=str(REPO_ROOT / "lm-evaluation-harness" / "models" / "OLMoE-1B-7B-0924"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_texts", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    _register_local_olmoe()
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.expert_capacity = None
    if hasattr(config, "strategy"):
        config.strategy = None

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=None,
    ).eval().to(args.device)

    num_layers = config.num_hidden_layers
    top_k = config.num_experts_per_tok
    num_experts = config.num_experts

    captured = {}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            inp = inputs[0]
            if inp.dim() == 3:
                captured[layer_idx] = inp.detach()
        return hook

    hooks = []
    for i in range(num_layers):
        hooks.append(model.model.layers[i].mlp.register_forward_hook(make_hook(i)))

    texts = load_texts(args.max_texts, max_chars=256)
    print(f"Loaded {len(texts)} texts, max_length={args.max_length}")
    print(f"Model: {args.model}")
    print(f"Layers: {num_layers}, Experts: {num_experts}, top_k: {top_k}\n")

    layer_stats = {i: {"sum_norm": 0.0, "sum_norm_sq": 0.0,
                        "count": 0, "sum_hidden_norm": 0.0,
                        "sum_score": 0.0, "n_dropped_score": 0,
                        "n_dropped_contrib": 0, "n_total": 0,
                        "sum_max_load": 0.0, "sum_gini": 0.0,
                        "sum_top1_share": 0.0, "sum_n_active": 0.0,
                        "sum_n_overloaded": 0.0, "n_batches": 0}
                   for i in range(num_layers)}

    gamma = 1.0

    with torch.no_grad():
        for ti, text in enumerate(texts):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            enc = {k: v.to(args.device) for k, v in enc.items()}
            captured.clear()
            model(**enc)

            for li in range(num_layers):
                if li not in captured:
                    continue
                inp = captured[li]
                b, s, h = inp.shape
                flat = inp.view(-1, h)
                T = flat.shape[0]
                module = model.model.layers[li].mlp

                expert_capacity = math.ceil(gamma * top_k * T / num_experts)

                logits = module.gate(flat)
                scores = F.softmax(logits.float(), dim=-1)
                topk_w, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)

                contrib_norms = torch.zeros(T, top_k, device=flat.device, dtype=torch.float32)
                for e in range(num_experts):
                    match = topk_idx == e
                    if not match.any():
                        continue
                    tok, slot = torch.where(match)
                    h_sub = flat[tok]
                    e_out = module.experts[e](h_sub).float()
                    w_e = topk_w[tok, slot].float().unsqueeze(-1)
                    contrib_norms[tok, slot] = (w_e * e_out).norm(dim=-1)

                st = layer_stats[li]
                st["sum_norm"] += contrib_norms.sum().item()
                st["sum_norm_sq"] += (contrib_norms ** 2).sum().item()
                st["count"] += contrib_norms.numel()
                st["sum_hidden_norm"] += flat.float().norm(dim=-1).sum().item()
                st["sum_score"] += topk_w.sum().item()

                # Expert load distribution analysis
                mask_init = torch.zeros(T, num_experts, dtype=torch.bool, device=flat.device)
                mask_init.scatter_(-1, topk_idx, True)
                usage = mask_init.sum(dim=0).float()  # [num_experts]

                n_total = T * top_k
                n_active = (usage > 0).sum().item()
                max_load = usage.max().item()
                top1_share = max_load / n_total

                # Gini coefficient of expert load
                sorted_u, _ = usage.sort()
                n_e = float(num_experts)
                cum = sorted_u.cumsum(0)
                gini = 1.0 - 2.0 * cum.sum().item() / (n_e * sorted_u.sum().item() + 1e-12) + 1.0 / n_e

                overloaded = (usage > expert_capacity).nonzero(as_tuple=True)[0]
                n_drop_score = 0
                for c in overloaded:
                    assigned = mask_init[:, c].nonzero(as_tuple=True)[0]
                    n_over = max(0, assigned.numel() - expert_capacity)
                    n_drop_score += n_over

                st["n_dropped_score"] += n_drop_score
                st["n_total"] += n_total
                st["sum_max_load"] += max_load
                st["sum_gini"] += gini
                st["sum_top1_share"] += top1_share
                st["sum_n_active"] += n_active
                st["sum_n_overloaded"] += len(overloaded)
                st["n_batches"] += 1

            if (ti + 1) % 8 == 0:
                print(f"  processed {ti+1}/{len(texts)}")

    for h in hooks:
        h.remove()

    # Print results — Part 1: Contribution
    print(f"\n{'='*100}")
    print(f"Per-Layer MoE Contribution Profile (gamma={gamma}, {len(texts)} PIQA texts)")
    print(f"{'='*100}")
    print(f"{'Layer':>5} | {'mean||wE(x)||':>14} | {'std':>10} | {'mean_score':>10} | "
          f"{'mean||h||':>10} | {'contrib/||h||':>13} | {'drop_rate':>9}")
    print("-" * 90)

    global_sum, global_count = 0.0, 0
    for li in range(num_layers):
        st = layer_stats[li]
        if st["count"] == 0:
            continue
        mean_norm = st["sum_norm"] / st["count"]
        mean_sq = st["sum_norm_sq"] / st["count"]
        std_norm = max(0, mean_sq - mean_norm ** 2) ** 0.5
        mean_score = st["sum_score"] / st["count"]
        mean_hidden = st["sum_hidden_norm"] / (st["count"] / top_k)
        ratio = mean_norm / max(mean_hidden, 1e-12)
        drop_rate = st["n_dropped_score"] / max(st["n_total"], 1) * 100

        print(f"{li:5d} | {mean_norm:14.4f} | {std_norm:10.4f} | {mean_score:10.4f} | "
              f"{mean_hidden:10.2f} | {ratio:12.4%} | {drop_rate:8.1f}%")
        global_sum += st["sum_norm"]
        global_count += st["count"]

    print("-" * 90)
    if global_count > 0:
        print(f"{'AVG':>5} | {global_sum/global_count:14.4f}")

    # Part 2: Expert concentration metrics
    print(f"\n{'='*100}")
    print(f"Per-Layer Expert Load Concentration (capacity = ceil({gamma} * {top_k} * T / {num_experts}))")
    print(f"{'='*100}")
    print(f"{'Layer':>5} | {'Gini':>6} | {'top1_share':>10} | {'max_load':>8} | "
          f"{'#active':>7}/{num_experts} | {'#overload':>9} | {'drop_rate':>9}")
    print("-" * 80)

    for li in range(num_layers):
        st = layer_stats[li]
        nb = st["n_batches"]
        if nb == 0:
            continue
        gini = st["sum_gini"] / nb
        top1 = st["sum_top1_share"] / nb
        max_l = st["sum_max_load"] / nb
        n_act = st["sum_n_active"] / nb
        n_over = st["sum_n_overloaded"] / nb
        drop_rate = st["n_dropped_score"] / max(st["n_total"], 1) * 100

        print(f"{li:5d} | {gini:6.3f} | {top1:9.1%} | {max_l:8.1f} | "
              f"{n_act:5.1f}/{num_experts}  | {n_over:7.1f}   | {drop_rate:8.1f}%")

    print("-" * 80)

    # Part 3: bar chart
    print(f"\n--- Contribution by layer ---")
    means = []
    for li in range(num_layers):
        st = layer_stats[li]
        means.append(st["sum_norm"] / st["count"] if st["count"] > 0 else 0)
    max_mean = max(means) if means else 1
    for li in range(num_layers):
        bar_len = int(means[li] / max_mean * 50)
        print(f"  L{li:2d} | {'█' * bar_len}{'░' * (50 - bar_len)} | {means[li]:.4f}")

    print(f"\n--- Gini coefficient by layer (higher = more concentrated) ---")
    ginis = []
    for li in range(num_layers):
        st = layer_stats[li]
        ginis.append(st["sum_gini"] / st["n_batches"] if st["n_batches"] > 0 else 0)
    max_gini = max(ginis) if ginis else 1
    for li in range(num_layers):
        bar_len = int(ginis[li] / max(max_gini, 1e-6) * 50)
        print(f"  L{li:2d} | {'█' * bar_len}{'░' * (50 - bar_len)} | {ginis[li]:.3f}")


if __name__ == "__main__":
    main()
