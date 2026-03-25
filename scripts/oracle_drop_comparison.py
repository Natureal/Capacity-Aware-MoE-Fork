#!/usr/bin/env python3
"""
Phase 1+2: Oracle drop comparison & correlation analysis.

Phase 1 — Layer-level distortion:
  For each MoE layer, compare three strategies for dropping ~32% of assignments:
    (a) no_drop:     full top-k, no capacity constraint (reference)
    (b) score_drop:  drop by lowest router score (current method)
    (c) contrib_drop: drop by smallest ||w_i * E_i(x)||_2 (oracle)
  Measure ||h_dropped - h_full||_2 / ||h_full||_2 per layer.

Phase 2 — Correlation:
  For every token-expert assignment in overloaded experts, record:
    - router_score (softmax weight)
    - contribution norm ||w * E(x)||_2
    - hidden state norm ||x||_2
    - margin (score at this expert minus (k+1)-th best score)
  Compute Spearman rank-correlation between each signal and contribution.
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _register_local_olmoe():
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers import AutoModel, AutoModelForCausalLM

    path = REPO_ROOT / "modeling_hf" / "modeling_olmoe.py"
    spec = importlib.util.spec_from_file_location("capacity_aware_olmoe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    AutoModel.register(OlmoeConfig, mod.OlmoeModel, exist_ok=True)
    AutoModelForCausalLM.register(OlmoeConfig, mod.OlmoeForCausalLM, exist_ok=True)


def _moe_layer_forward(module, flat_hidden, scores, topk_idx, topk_weight):
    """Replay MoE forward given explicit (topk_idx, topk_weight).
    dropped slots have topk_idx == num_experts; their contribution is 0."""
    T, H = flat_hidden.shape
    num_experts = module.num_experts
    out = torch.zeros(T, H, device=flat_hidden.device, dtype=flat_hidden.dtype)
    expert_mask = F.one_hot(topk_idx, num_classes=num_experts + 1).permute(2, 1, 0)
    for e in range(num_experts):
        idx, top_x = torch.where(expert_mask[e])
        if top_x.numel() == 0:
            continue
        h_sub = flat_hidden[top_x]
        e_out = module.experts[e](h_sub) * topk_weight[top_x, idx, None]
        out.index_add_(0, top_x, e_out.to(out.dtype))
    return out


def _compute_contrib_norms(module, flat_hidden, scores, top_k):
    """For each token's top-k assignments, compute ||w_i * E_i(x)||_2.
    Returns (topk_weight, topk_idx, contrib_norms) all [T, top_k]."""
    T, K = scores.shape
    topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
    contrib_norms = torch.zeros(T, top_k, device=flat_hidden.device, dtype=torch.float32)
    for e in range(module.num_experts):
        match = topk_idx == e
        if not match.any():
            continue
        tok, slot = torch.where(match)
        h_sub = flat_hidden[tok]
        e_out = module.experts[e](h_sub).float()
        w_e = topk_weight[tok, slot].float().unsqueeze(-1)
        contrib_norms[tok, slot] = (w_e * e_out).norm(dim=-1)
    return topk_weight, topk_idx, contrib_norms


def _apply_capacity_with_priority(priority_per_expert, mask_buffer, expert_capacity):
    """Given priority_per_expert [T, K] (higher=keep), enforce capacity on
    overloaded experts in mask_buffer [T, K] (bool). Returns new mask_buffer."""
    mb = mask_buffer.clone()
    usage = mb.sum(dim=0)
    cols = (usage > expert_capacity).nonzero(as_tuple=True)[0]
    if cols.numel() == 0:
        return mb
    for c in cols:
        assigned = mb[:, c].nonzero(as_tuple=True)[0]
        if assigned.numel() <= expert_capacity:
            continue
        prio = priority_per_expert[assigned, c]
        _, keep_order = prio.topk(expert_capacity)
        keep_set = assigned[keep_order]
        mb[:, c] = False
        mb[keep_set, c] = True
    return mb


def _drop_and_gather(scores, topk_idx, mask_buffer, num_experts):
    """Apply mask to topk_idx: unmasked slots -> sentinel."""
    topk_weight = scores.gather(-1, topk_idx)
    top_mask = mask_buffer.gather(-1, topk_idx)
    dropped_idx = topk_idx.masked_fill(~top_mask, num_experts)
    return topk_weight, dropped_idx


def _spearman_corr(a, b):
    """Spearman rank correlation between 1-D tensors."""
    if a.numel() < 3:
        return float('nan')

    def _rank(x):
        _, inv = x.sort()
        ranks = torch.empty_like(x)
        ranks[inv] = torch.arange(len(x), device=x.device, dtype=x.dtype)
        return ranks

    ra = _rank(a.float())
    rb = _rank(b.float())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if denom < 1e-12:
        return float('nan')
    return float((ra * rb).sum() / denom)


def load_piqa_texts(max_texts, max_chars):
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
            "The mixture of experts architecture routes each token to a small subset of feedforward networks.",
            "Scientists observed that load imbalance during inference causes straggler effects on GPUs.",
            "Machine learning systems often trade off accuracy against latency under fixed compute budgets.",
            "Neural networks with sparse activation can scale to very large parameter counts efficiently.",
            "Large language models are pretrained on diverse corpora and then aligned with human preferences.",
        ]
    return texts[:max_texts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default=str(REPO_ROOT / "lm-evaluation-harness" / "models" / "OLMoE-1B-7B-0924"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_texts", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "experiments" / "oracle_drop_comparison"))
    args = parser.parse_args()

    _register_local_olmoe()
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    model_path = Path(args.model)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.expert_capacity = None
    if hasattr(config, "strategy"):
        config.strategy = None

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=None,
    )
    model.eval().to(args.device)

    num_layers = config.num_hidden_layers
    top_k = config.num_experts_per_tok
    num_experts = config.num_experts
    gamma = args.gamma

    captured = {}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            inp = inputs[0]
            if inp.dim() != 3:
                return
            captured[layer_idx] = inp.detach()
        return hook

    hooks = []
    for i in range(num_layers):
        hooks.append(model.model.layers[i].mlp.register_forward_hook(make_hook(i)))

    texts = load_piqa_texts(args.max_texts, max_chars=256)
    print(f"Loaded {len(texts)} texts, gamma={gamma}, max_length={args.max_length}")

    per_layer_distortion_score = {i: [] for i in range(num_layers)}
    per_layer_distortion_contrib = {i: [] for i in range(num_layers)}
    per_layer_drop_rate = {i: [] for i in range(num_layers)}

    corr_score_vs_contrib = {i: [] for i in range(num_layers)}
    corr_hnorm_vs_contrib = {i: [] for i in range(num_layers)}
    corr_margin_vs_contrib = {i: [] for i in range(num_layers)}

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

                topk_weight, topk_idx, contrib_norms = _compute_contrib_norms(
                    module, flat, scores, top_k)

                # --- No-drop reference ---
                h_full = _moe_layer_forward(module, flat, scores, topk_idx, topk_weight)
                h_full_norm = h_full.float().norm().clamp_min(1e-12)

                # Build initial mask
                mask_init = torch.zeros(T, num_experts, dtype=torch.bool, device=flat.device)
                mask_init.scatter_(-1, topk_idx, True)

                # --- Score-based drop ---
                mask_score = _apply_capacity_with_priority(
                    scores, mask_init, expert_capacity)
                tw_s, ti_s = _drop_and_gather(scores, topk_idx, mask_score, num_experts)
                h_score = _moe_layer_forward(module, flat, scores, ti_s, tw_s)

                # --- Contribution-based drop (oracle) ---
                # Build per-expert contribution priority: for each (token, expert)
                # use the contribution norm of that assignment.
                contrib_priority = torch.zeros(T, num_experts, device=flat.device, dtype=torch.float32)
                for slot in range(top_k):
                    tok_range = torch.arange(T, device=flat.device)
                    expert_col = topk_idx[:, slot]
                    valid = expert_col < num_experts
                    contrib_priority[tok_range[valid], expert_col[valid]] = contrib_norms[tok_range[valid], slot]

                mask_contrib = _apply_capacity_with_priority(
                    contrib_priority, mask_init, expert_capacity)
                tw_c, ti_c = _drop_and_gather(scores, topk_idx, mask_contrib, num_experts)
                h_contrib = _moe_layer_forward(module, flat, scores, ti_c, tw_c)

                dist_score = float((h_score.float() - h_full.float()).norm() / h_full_norm)
                dist_contrib = float((h_contrib.float() - h_full.float()).norm() / h_full_norm)
                n_dropped = int((ti_s == num_experts).sum().item())
                n_total = T * top_k
                drop_rate = n_dropped / max(n_total, 1)

                per_layer_distortion_score[li].append(dist_score)
                per_layer_distortion_contrib[li].append(dist_contrib)
                per_layer_drop_rate[li].append(drop_rate)

                # --- Phase 2: Correlation on overloaded-expert assignments ---
                dropped_score = (ti_s == num_experts)
                dropped_contrib = (ti_c == num_experts)
                all_valid = mask_init.gather(-1, topk_idx)

                score_vals = []
                contrib_vals = []
                hnorm_vals = []
                margin_vals = []

                topk_plus1, _ = torch.topk(scores, k=top_k + 1, dim=-1, sorted=True)
                next_best = topk_plus1[:, -1]

                for slot in range(top_k):
                    v = all_valid[:, slot]
                    if not v.any():
                        continue
                    score_vals.append(topk_weight[v, slot])
                    contrib_vals.append(contrib_norms[v, slot])
                    hnorm_vals.append(flat[v].float().norm(dim=-1))
                    margin_vals.append((topk_weight[v, slot] - next_best[v]).float())

                if score_vals:
                    sv = torch.cat(score_vals)
                    cv = torch.cat(contrib_vals)
                    hv = torch.cat(hnorm_vals)
                    mv = torch.cat(margin_vals)

                    corr_score_vs_contrib[li].append(_spearman_corr(sv, cv))
                    corr_hnorm_vs_contrib[li].append(_spearman_corr(hv, cv))
                    corr_margin_vs_contrib[li].append(_spearman_corr(mv, cv))

            if (ti + 1) % 4 == 0:
                print(f"  processed {ti+1}/{len(texts)}")

    for h in hooks:
        h.remove()

    # --- Aggregate results ---
    summary = {
        "model": str(model_path), "gamma": gamma, "top_k": top_k,
        "num_texts": len(texts), "max_length": args.max_length,
    }

    phase1 = {}
    global_score_dists = []
    global_contrib_dists = []
    for li in range(num_layers):
        ds = per_layer_distortion_score[li]
        dc = per_layer_distortion_contrib[li]
        dr = per_layer_drop_rate[li]
        if not ds:
            continue
        ms = sum(ds) / len(ds)
        mc = sum(dc) / len(dc)
        mr = sum(dr) / len(dr)
        gap_pct = (ms - mc) / max(ms, 1e-12) * 100
        phase1[li] = {
            "distortion_score_drop": ms,
            "distortion_contrib_drop": mc,
            "gap_pct": gap_pct,
            "drop_rate": mr,
        }
        global_score_dists.extend(ds)
        global_contrib_dists.extend(dc)
    summary["phase1_per_layer"] = phase1

    if global_score_dists:
        gs = sum(global_score_dists) / len(global_score_dists)
        gc = sum(global_contrib_dists) / len(global_contrib_dists)
        summary["phase1_global"] = {
            "mean_distortion_score_drop": gs,
            "mean_distortion_contrib_drop": gc,
            "gap_pct": (gs - gc) / max(gs, 1e-12) * 100,
        }

    phase2 = {}
    for li in range(num_layers):
        sc = corr_score_vs_contrib[li]
        hc = corr_hnorm_vs_contrib[li]
        mc_list = corr_margin_vs_contrib[li]
        if not sc:
            continue
        valid_sc = [v for v in sc if v == v]
        valid_hc = [v for v in hc if v == v]
        valid_mc = [v for v in mc_list if v == v]
        phase2[li] = {
            "spearman_score_vs_contrib": sum(valid_sc) / max(len(valid_sc), 1) if valid_sc else None,
            "spearman_hnorm_vs_contrib": sum(valid_hc) / max(len(valid_hc), 1) if valid_hc else None,
            "spearman_margin_vs_contrib": sum(valid_mc) / max(len(valid_mc), 1) if valid_mc else None,
        }
    summary["phase2_per_layer"] = phase2

    if phase2:
        vals_sc = [v["spearman_score_vs_contrib"] for v in phase2.values() if v["spearman_score_vs_contrib"] is not None]
        vals_hc = [v["spearman_hnorm_vs_contrib"] for v in phase2.values() if v["spearman_hnorm_vs_contrib"] is not None]
        vals_mc = [v["spearman_margin_vs_contrib"] for v in phase2.values() if v["spearman_margin_vs_contrib"] is not None]
        summary["phase2_global"] = {
            "mean_spearman_score_vs_contrib": sum(vals_sc) / max(len(vals_sc), 1) if vals_sc else None,
            "mean_spearman_hnorm_vs_contrib": sum(vals_hc) / max(len(vals_hc), 1) if vals_hc else None,
            "mean_spearman_margin_vs_contrib": sum(vals_mc) / max(len(vals_mc), 1) if vals_mc else None,
        }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "oracle_drop_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append(f"=== Oracle Drop Comparison (gamma={gamma}, top_k={top_k}) ===\n")
    lines.append(f"Model: {model_path}\n")
    lines.append(f"Texts: {len(texts)}, max_length={args.max_length}\n")

    lines.append("\n--- Phase 1: Layer-level distortion ---\n")
    lines.append(f"{'Layer':>5} | {'dist(score)':>12} | {'dist(contrib)':>14} | {'gap%':>7} | {'drop_rate':>9}\n")
    lines.append("-" * 60 + "\n")
    for li in sorted(phase1.keys()):
        p = phase1[li]
        lines.append(
            f"{li:5d} | {p['distortion_score_drop']:12.6f} | {p['distortion_contrib_drop']:14.6f} "
            f"| {p['gap_pct']:6.1f}% | {p['drop_rate']:8.1%}\n"
        )
    if summary.get("phase1_global"):
        g = summary["phase1_global"]
        lines.append("-" * 60 + "\n")
        lines.append(
            f"{'AVG':>5} | {g['mean_distortion_score_drop']:12.6f} | {g['mean_distortion_contrib_drop']:14.6f} "
            f"| {g['gap_pct']:6.1f}%\n"
        )

    lines.append("\n--- Phase 2: Spearman correlation (signal vs contribution norm) ---\n")
    lines.append(f"{'Layer':>5} | {'score':>8} | {'hnorm':>8} | {'margin':>8}\n")
    lines.append("-" * 42 + "\n")
    for li in sorted(phase2.keys()):
        p = phase2[li]
        def _fmt(v):
            return f"{v:8.4f}" if v is not None else "     N/A"
        lines.append(
            f"{li:5d} | {_fmt(p['spearman_score_vs_contrib'])} | "
            f"{_fmt(p['spearman_hnorm_vs_contrib'])} | {_fmt(p['spearman_margin_vs_contrib'])}\n"
        )
    if summary.get("phase2_global"):
        g2 = summary["phase2_global"]
        def _fmt(v):
            return f"{v:8.4f}" if v is not None else "     N/A"
        lines.append("-" * 42 + "\n")
        lines.append(
            f"{'AVG':>5} | {_fmt(g2['mean_spearman_score_vs_contrib'])} | "
            f"{_fmt(g2['mean_spearman_hnorm_vs_contrib'])} | {_fmt(g2['mean_spearman_margin_vs_contrib'])}\n"
        )

    report = "".join(lines)
    report_path = out_dir / "oracle_drop_report.txt"
    report_path.write_text(report)

    print(report)
    print(f"\nWrote: {json_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
