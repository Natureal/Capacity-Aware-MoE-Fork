#!/usr/bin/env python3
"""
Per-layer gamma allocation: profile per-layer load statistics, compute optimal
gamma allocations under the constraint that total execution time (= sum of
per-layer max expert load) matches uniform gamma=1.0, and optionally evaluate.

Usage:
  # Profile only (computes gammas)
  CUDA_VISIBLE_DEVICES=0 python scripts/layerwise_gamma.py --phase profile --max_texts 200

  # Evaluate a specific allocation
  CUDA_VISIBLE_DEVICES=0 python scripts/layerwise_gamma.py --phase eval --alloc contrib --tasks piqa arc_challenge hellaswag
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_HARNESS = REPO_ROOT / "lm-evaluation-harness"
MODEL_DIR = EVAL_HARNESS / "models" / "OLMoE-1B-7B-0924"


def _register_local_olmoe():
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers import AutoModel, AutoModelForCausalLM
    path = REPO_ROOT / "modeling_hf" / "modeling_olmoe.py"
    spec = importlib.util.spec_from_file_location("capacity_aware_olmoe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    AutoModel.register(OlmoeConfig, mod.OlmoeModel, exist_ok=True)
    AutoModelForCausalLM.register(OlmoeConfig, mod.OlmoeForCausalLM, exist_ok=True)


def load_texts(max_texts, max_chars=256):
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
    return texts


# ---------------------------------------------------------------------------
# Phase 1: Profile
# ---------------------------------------------------------------------------
def profile(args):
    _register_local_olmoe()
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    config = AutoConfig.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    config.expert_capacity = None
    if hasattr(config, "strategy"):
        config.strategy = None

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), config=config, trust_remote_code=True,
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

    texts = load_texts(args.max_texts)
    print(f"Profiling with {len(texts)} texts on {args.device}")

    gamma_ref = 1.0

    layer_stats = {i: {
        "sum_contrib": 0.0, "count": 0,
        "sum_max_load": 0.0, "sum_gini": 0.0,
        "sum_natural_max": 0.0,
        "n_dropped_score": 0, "n_total": 0,
        "sum_damage": 0.0,
        "n_batches": 0,
    } for i in range(num_layers)}

    with torch.no_grad():
        for ti, text in enumerate(texts):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            enc = {k: v.to(args.device) for k, v in enc.items()}
            captured.clear()
            model(**enc)

            for li in range(num_layers):
                if li not in captured:
                    continue
                inp = captured[li]
                flat = inp.view(-1, inp.shape[-1])
                T = flat.shape[0]
                module = model.model.layers[li].mlp

                expert_capacity = math.ceil(gamma_ref * top_k * T / num_experts)

                logits = module.gate(flat)
                scores = F.softmax(logits.float(), dim=-1)
                topk_w, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)

                # Compute contribution norms
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

                # Expert load distribution (before dropping)
                usage = torch.zeros(num_experts, device=flat.device)
                for k_slot in range(top_k):
                    usage.scatter_add_(0, topk_idx[:, k_slot], torch.ones(T, device=flat.device))

                natural_max = usage.max().item()

                # Gini coefficient
                sorted_u, _ = usage.sort()
                n_e = float(num_experts)
                cum = sorted_u.cumsum(0)
                gini = 1.0 - 2.0 * cum.sum().item() / (n_e * sorted_u.sum().item() + 1e-12) + 1.0 / n_e

                # Simulate score dropping and compute damage
                overloaded_mask = usage > expert_capacity
                n_drop = 0
                damage = 0.0
                for eid in overloaded_mask.nonzero(as_tuple=True)[0]:
                    eid_val = eid.item()
                    match = topk_idx == eid_val
                    tok_slots = match.nonzero(as_tuple=False)  # [N, 2]
                    if tok_slots.shape[0] <= expert_capacity:
                        continue
                    # score-based drop: drop lowest-scoring assignments
                    slot_scores = topk_w[tok_slots[:, 0], tok_slots[:, 1]]
                    _, sort_idx = slot_scores.sort()
                    n_over = tok_slots.shape[0] - expert_capacity
                    dropped_idx = sort_idx[:n_over]
                    drop_contribs = contrib_norms[tok_slots[dropped_idx, 0], tok_slots[dropped_idx, 1]]
                    damage += drop_contribs.sum().item()
                    n_drop += n_over

                st = layer_stats[li]
                st["sum_contrib"] += contrib_norms.sum().item()
                st["count"] += contrib_norms.numel()
                st["sum_max_load"] += natural_max
                st["sum_natural_max"] += natural_max
                st["sum_gini"] += gini
                st["n_dropped_score"] += n_drop
                st["n_total"] += T * top_k
                st["sum_damage"] += damage
                st["n_batches"] += 1

            if (ti + 1) % 20 == 0:
                print(f"  processed {ti+1}/{len(texts)}")

    for h in hooks:
        h.remove()

    # Compute per-layer summary
    summary = {}
    for li in range(num_layers):
        st = layer_stats[li]
        nb = st["n_batches"]
        if nb == 0:
            continue
        summary[li] = {
            "mean_contrib": st["sum_contrib"] / max(st["count"], 1),
            "mean_natural_max": st["sum_natural_max"] / nb,
            "mean_gini": st["sum_gini"] / nb,
            "drop_rate": st["n_dropped_score"] / max(st["n_total"], 1),
            "mean_damage": st["sum_damage"] / nb,
        }

    # Print profile
    print(f"\n{'='*90}")
    print(f"Per-Layer Profile ({len(texts)} PIQA, gamma_ref={gamma_ref})")
    print(f"{'='*90}")
    print(f"{'Layer':>5} | {'mean_contrib':>12} | {'natural_max':>11} | {'gini':>6} | "
          f"{'drop_rate':>9} | {'mean_damage':>11}")
    print("-" * 72)
    for li in sorted(summary.keys()):
        s = summary[li]
        print(f"{li:5d} | {s['mean_contrib']:12.4f} | {s['mean_natural_max']:11.1f} | "
              f"{s['mean_gini']:6.3f} | {s['drop_rate']:8.1%} | {s['mean_damage']:11.2f}")

    # Compute gamma allocations
    L = num_layers
    allocations = compute_allocations(summary, L)

    # Print allocations
    print(f"\n{'='*90}")
    print(f"Gamma Allocations (constraint: sum = {L})")
    print(f"{'='*90}")
    for name, gammas in allocations.items():
        gstr = " ".join(f"{g:.3f}" for g in gammas)
        print(f"\n{name}:")
        print(f"  gammas: [{gstr}]")
        print(f"  sum={sum(gammas):.2f}, min={min(gammas):.3f}, max={max(gammas):.3f}")
        colon_str = ":".join(f"{g:.4f}" for g in gammas)
        print(f"  model_args string: layer_gammas={colon_str}")

    # Save
    out_dir = REPO_ROOT / "experiments" / "layerwise_gamma"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "profile": {str(k): v for k, v in summary.items()},
        "allocations": {name: gammas for name, gammas in allocations.items()},
        "config": {"num_layers": L, "num_experts": num_experts, "top_k": top_k,
                    "gamma_ref": gamma_ref, "n_texts": len(texts)},
    }
    with open(out_dir / "profile_and_gammas.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_dir / 'profile_and_gammas.json'}")

    return allocations


def compute_allocations(summary, L):
    """Compute several per-layer gamma allocations, all summing to L."""
    layers = sorted(summary.keys())
    assert len(layers) == L

    contribs = [summary[l]["mean_contrib"] for l in layers]
    ginis = [summary[l]["mean_gini"] for l in layers]
    drop_rates = [summary[l]["drop_rate"] for l in layers]
    damages = [summary[l]["mean_damage"] for l in layers]

    def _normalize(weights, target_sum=L, clip_min=0.5, clip_max=2.0):
        """Normalize weights so they sum to target, with clipping."""
        s = sum(weights)
        if s < 1e-12:
            return [1.0] * L
        gammas = [w / s * target_sum for w in weights]
        # Iterative clipping with redistribution
        for _ in range(20):
            excess = 0.0
            n_free = 0
            for i in range(L):
                if gammas[i] < clip_min:
                    excess += clip_min - gammas[i]
                    gammas[i] = clip_min
                elif gammas[i] > clip_max:
                    excess -= gammas[i] - clip_max
                    gammas[i] = clip_max
                else:
                    n_free += 1
            if abs(excess) < 1e-6 or n_free == 0:
                break
            adjust = excess / n_free
            for i in range(L):
                if clip_min < gammas[i] < clip_max:
                    gammas[i] -= adjust
        # Final normalization to exact sum
        s = sum(gammas)
        gammas = [g / s * target_sum for g in gammas]
        return gammas

    allocations = {}

    # 1. Uniform baseline
    allocations["uniform"] = [1.0] * L

    # 2. Contribution-proportional: high-contribution layers get more capacity
    allocations["contrib"] = _normalize(contribs)

    # 3. Damage-proportional: layers with more expected damage get more capacity
    allocations["damage"] = _normalize([max(d, 1e-8) for d in damages])

    # 4. Gini-proportional: concentrated layers get more capacity
    allocations["gini"] = _normalize(ginis)

    # 5. Combined: contrib * gini
    allocations["contrib_x_gini"] = _normalize(
        [c * g for c, g in zip(contribs, ginis)])

    # 6. Drop-rate proportional
    allocations["drop_rate"] = _normalize(
        [max(d, 0.01) for d in drop_rates])

    # 7. Sqrt-damage (moderate reallocation)
    allocations["sqrt_damage"] = _normalize(
        [max(d, 1e-8) ** 0.5 for d in damages])

    # 8. Damage-mild: tighter bounds to avoid starving early layers
    allocations["damage_mild"] = _normalize(
        [max(d, 1e-8) for d in damages], clip_min=0.7, clip_max=1.5)

    # 9. Log-damage: log-scale smoothing of damage signal
    import math as _m
    allocations["log_damage"] = _normalize(
        [_m.log1p(max(d, 1e-8)) for d in damages])

    # 10. Compound-weighted damage: accounts for error propagation through later layers
    # Each layer's distortion propagates through (L-l) remaining layers
    compound_weights = [damages[l] * (L - l) for l in range(L)]
    allocations["compound"] = _normalize(compound_weights, clip_min=0.7, clip_max=1.5)

    # 11. Very narrow damage allocation (0.9-1.1): minimal perturbation
    allocations["damage_narrow"] = _normalize(
        [max(d, 1e-8) for d in damages], clip_min=0.9, clip_max=1.1)

    # 12. Inverse early boost: give MORE capacity to early layers (opposite direction)
    # Weight by (l+1) to boost early layers
    early_boost = [damages[l] / (l + 1) for l in range(L)]
    allocations["early_boost"] = _normalize(early_boost, clip_min=0.8, clip_max=1.3)

    return allocations


# ---------------------------------------------------------------------------
# Phase 2: Evaluate
# ---------------------------------------------------------------------------
TASK_FEWSHOTS = {
    "openbookqa": 0, "piqa": 0, "rte": 0, "winogrande": 5,
    "boolq": 0, "arc_challenge": 25, "hellaswag": 10,
}

def evaluate(args):
    """Run lm_eval with a specific per-layer gamma allocation."""
    # Load saved allocations
    profile_path = REPO_ROOT / "experiments" / "layerwise_gamma" / "profile_and_gammas.json"
    if not profile_path.exists():
        print(f"Profile not found at {profile_path}. Run --phase profile first.")
        sys.exit(1)

    with open(profile_path) as f:
        data = json.load(f)

    alloc_name = args.alloc
    if alloc_name not in data["allocations"]:
        print(f"Unknown allocation '{alloc_name}'. Available: {list(data['allocations'].keys())}")
        sys.exit(1)

    gammas = data["allocations"][alloc_name]
    gamma_str = ":".join(f"{g:.4f}" for g in gammas)
    print(f"Using allocation '{alloc_name}': {[f'{g:.3f}' for g in gammas]}")
    print(f"Sum = {sum(gammas):.4f}")

    tasks = args.tasks
    output_base = MODEL_DIR / "expert_capacity-1.0" / f"layergamma_{alloc_name}"
    output_base.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        fewshot = TASK_FEWSHOTS.get(task, 0)
        task_json = output_base / f"{task}.json"
        task_log = output_base / f"{task}.out"

        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", (
                f"pretrained={MODEL_DIR},"
                f"expert_capacity=1.0,"
                f"strategy=score,"
                f"trust_remote_code=True,dtype=bfloat16"
            ),
            "--tasks", task,
            "--num_fewshot", str(fewshot),
            "--batch_size", "auto",
            "--output_path", str(task_json),
        ]

        env = dict(os.environ, LAYER_GAMMAS=gamma_str)

        print(f"\n{'='*60}")
        print(f"Task: {task} (fewshot={fewshot})")
        print(f"Output: {task_json}")
        print(f"LAYER_GAMMAS={gamma_str}")
        print(f"{'='*60}")

        with open(task_log, "w") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                  cwd=str(EVAL_HARNESS), env=env)
        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode}), check {task_log}")
        else:
            print(f"  DONE")

    # Collect results
    print(f"\n{'='*60}")
    print(f"Results for allocation '{alloc_name}'")
    print(f"{'='*60}")
    for task in tasks:
        rdir = output_base / f"{task}.json"
        rfiles = sorted(rdir.glob("results_*.json")) if rdir.is_dir() else []
        if not rfiles:
            print(f"  {task}: no results")
            continue
        with open(rfiles[-1]) as f:
            rd = json.load(f)
        results = rd.get("results", {})
        for k, v in results.items():
            acc = v.get("acc,none", "N/A")
            accn = v.get("acc_norm,none", "N/A")
            print(f"  {task}: acc={acc}, acc_norm={accn}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["profile", "eval", "both"], default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_texts", type=int, default=200)
    parser.add_argument("--alloc", default="damage",
                        help="Allocation to evaluate (for eval phase)")
    parser.add_argument("--tasks", nargs="+",
                        default=["piqa", "arc_challenge", "hellaswag"])
    args = parser.parse_args()

    if args.phase in ("profile", "both"):
        allocations = profile(args)

    if args.phase in ("eval", "both"):
        evaluate(args)


if __name__ == "__main__":
    main()
