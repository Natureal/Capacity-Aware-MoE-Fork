#!/usr/bin/env python3
"""
MoE 专家贡献度分析（OLMoE top-k=8）:
  C = ||y_full - y_top1|| / (||y_full|| + eps)
其中 y_full 为 gate 选出的 top-k 专家加权和，y_top1 仅保留 router 分数最高的那一项专家与其权重。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

EPS = 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get(
            "OLMOE_MODEL_PATH",
            str(
                Path(__file__).resolve().parent
                / "models"
                / "OLMoE-1B-7B-0924"
            ),
        ),
        help="本地 OLMoE 目录（含 config、权重、tokenizer）",
    )
    p.add_argument("--output-dir", type=str, default=".", help="图像与日志输出目录")
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--min-chars", type=int, default=100)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument(
        "--text-file",
        type=str,
        default=None,
        help="可选：每段一篇文本（空行分段），用于离线环境替代 WikiText-103",
    )
    return p.parse_args()


def get_dtype(name: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _chunks_from_wikitext_file(path: Path, num_samples: int, min_chars: int) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"\n\s*\n+", raw)
    texts: list[str] = []
    for p in parts:
        t = p.strip()
        if len(t) > min_chars:
            texts.append(t)
        if len(texts) >= num_samples:
            break
    return texts


def _download_wikitext2_valid(cache_path: Path) -> None:
    url = (
        "https://raw.githubusercontent.com/pytorch/examples/main/"
        "word_language_model/data/wikitext-2/wiki.valid.txt"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, cache_path)


def _synthetic_long_texts(num_samples: int, min_chars: int) -> list[str]:
    base = (
        "Mixture of Experts routes each token to a subset of feed-forward networks. "
        "The router produces a probability distribution over experts. "
    ) * 40
    return [base + f" DocId={i}." for i in range(num_samples)]


def load_wikitext_samples(
    num_samples: int, min_chars: int, text_file: str | None
) -> list[str]:
    if text_file:
        p = Path(text_file).expanduser().resolve()
        texts = _chunks_from_wikitext_file(p, num_samples, min_chars)
        if len(texts) < num_samples:
            raise RuntimeError(
                f"--text-file 中仅得到 {len(texts)} 条长度 > {min_chars} 的段落"
            )
        return texts

    try:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        texts: list[str] = []
        for row in ds:
            t = (row.get("text") or "").strip()
            if len(t) > min_chars:
                texts.append(t)
            if len(texts) >= num_samples:
                break
        if len(texts) >= num_samples:
            return texts
    except Exception as e:
        print(f"[warn] HuggingFace 加载 WikiText-103 失败 ({e})，尝试镜像文本 …")

    cache = Path(os.environ.get("TMPDIR", "/tmp")) / "wiki.valid.txt"
    try:
        if not cache.is_file():
            print(f"[info] 下载 WikiText-2 valid 到 {cache} …")
            _download_wikitext2_valid(cache)
        texts = _chunks_from_wikitext_file(cache, num_samples, min_chars)
        if len(texts) >= num_samples:
            print(f"[info] 使用 WikiText-2 valid 段落 {num_samples} 条")
            return texts[:num_samples]
    except Exception as e:
        print(f"[warn] WikiText-2 镜像失败 ({e})，使用内置长文本 …")

    texts = _synthetic_long_texts(num_samples + 5, min_chars)
    texts = [t for t in texts if len(t) > min_chars][:num_samples]
    if len(texts) < num_samples:
        raise RuntimeError(f"无法准备 {num_samples} 条长度 > {min_chars} 的文本")
    print(f"[info] 使用内置合成文本 {num_samples} 条（非 WikiText-103）")
    return texts


def input_device_for_model(model: torch.nn.Module) -> torch.device:
    """与模型首层权重同设备（含 device_map=auto 分片）。"""
    return next(model.parameters()).device


def build_moe_hooks(model: torch.nn.Module, num_layers: int):
    """
    在 OlmoeSparseMoeBlock 上注册 hook，计算每层每个 token 的 C。
    使用与 checkpoint 一致的 gate 输出（含 norm_topk_prob 行为）。
    """
    storage: list[torch.Tensor | None] = [None] * num_layers

    def make_hook(layer_idx: int):
        def hook_fn(module, inputs, output):
            hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs
            if hidden_states is None:
                return
            batch_size, seq_len, hidden_dim = hidden_states.shape
            flat = hidden_states.reshape(-1, hidden_dim)
            with torch.no_grad():
                _, top_k_weights, top_k_index = module.gate(flat)
                y_full = output.reshape(-1, hidden_dim)
                y_top1 = module.experts(
                    flat,
                    top_k_index[:, :1].contiguous(),
                    top_k_weights[:, :1].contiguous(),
                )
                num = torch.linalg.vector_norm(y_full - y_top1, dim=-1)
                den = torch.linalg.vector_norm(y_full, dim=-1) + EPS
                c = (num / den).view(batch_size, seq_len)
            storage[layer_idx] = c.detach().cpu()

        return hook_fn

    hooks = []
    for i in range(num_layers):
        h = model.model.layers[i].mlp.register_forward_hook(make_hook(i))
        hooks.append(h)
    return hooks, storage


def remove_hooks(hooks: list):
    for h in hooks:
        h.remove()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir():
        print(f"模型路径不存在: {model_path}", file=sys.stderr)
        return 1

    dtype = get_dtype(args.dtype)

    print(f"加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载文本数据（优先 WikiText-103）…")
    texts = load_wikitext_samples(args.num_samples, args.min_chars, args.text_file)

    use_cuda = torch.cuda.is_available() and (args.device is None or str(args.device).startswith("cuda"))
    dev_s = args.device or ("cuda" if use_cuda else "cpu")
    print(f"加载模型: {model_path} (dtype={args.dtype}, device={dev_s})")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map="auto" if use_cuda else None,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"device_map='auto' 失败: {e}，尝试单卡加载", file=sys.stderr)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map=None,
            trust_remote_code=True,
        ).to(dev_s)

    model.eval()
    in_dev = input_device_for_model(model)
    num_layers = model.config.num_hidden_layers

    all_mats: list[torch.Tensor] = []
    total_valid_tokens = 0

    for si, text in enumerate(texts):
        hooks, storage = build_moe_hooks(model, num_layers)
        enc = tokenizer(
            text,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = enc["input_ids"].to(in_dev)
        attn = enc["attention_mask"].to(in_dev)

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn)

        remove_hooks(hooks)

        if any(x is None for x in storage):
            raise RuntimeError("部分 MoE 层未写入贡献度，请检查 hook")

        mat = torch.stack(storage, dim=0).float()  # [L, S]
        valid_len = int(attn.sum().item())
        total_valid_tokens += valid_len
        mask = attn.bool().squeeze(0).cpu()

        all_mats.append(mat[:, mask])

        if (si + 1) % 10 == 0:
            print(f"  已完成 {si + 1}/{args.num_samples} 条 …")

    # ----- 图1: 热力图（第一个样本，仅非 pad 位置）-----
    plot_h = all_mats[0].numpy()
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    im = ax1.imshow(plot_h, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax1.set_xlabel("token position (non-pad, sample 0)")
    ax1.set_ylabel("layer")
    ax1.set_title("Non-top1 relative contribution (top-8 MoE), sample 0")
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    fig1.set_dpi(150)
    fig1.tight_layout()
    p1 = out_dir / "heatmap_contribution.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"已保存 {p1}")

    # ----- 汇总所有 (layer, token) 用于统计 -----
    flat_layers: list[np.ndarray] = []
    for m in all_mats:
        flat_layers.append(m.numpy())
    L = num_layers
    all_concat = np.concatenate([x.reshape(L, -1) for x in flat_layers], axis=1)

    # ----- 图2: 每层 boxplot -----
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    data_per_layer = [all_concat[li, :].tolist() for li in range(L)]
    ax2.boxplot(data_per_layer, positions=range(L), showfliers=False)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("contribution C")
    ax2.set_title("Insight 1: distribution of C within each layer (all tokens)")
    fig2.set_dpi(150)
    fig2.tight_layout()
    p2 = out_dir / "insight1_boxplot.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"已保存 {p2}")

    layer_means = []
    layer_stds = []
    layer_cvs = []
    for li in range(L):
        v = all_concat[li, :]
        mu = float(np.mean(v))
        sd = float(np.std(v))
        layer_means.append(mu)
        layer_stds.append(sd)
        layer_cvs.append(sd / (mu + EPS))
    print("\n--- Insight 1: 每层 mean / std / CV ---")
    for li in range(L):
        print(
            f"  layer {li:2d}: mean={layer_means[li]:.6f}  std={layer_stds[li]:.6f}  CV={layer_cvs[li]:.6f}"
        )
    mean_cv_layers = float(np.mean(layer_cvs))
    insight1_ok = mean_cv_layers > 0.5

    # ----- 图3: 选若干 token 位置（样本0，有效区）-----
    first = flat_layers[0]
    seq_len = first.shape[1]
    positions = sorted(
        set(
            int(p)
            for p in [
                0,
                seq_len // 8,
                seq_len // 4,
                seq_len // 2,
                (3 * seq_len) // 4,
                seq_len - seq_len // 8 - 1,
                seq_len - 1,
            ]
            if 0 <= p < seq_len
        )
    )
    if len(positions) < 5:
        positions = list(range(min(seq_len, 8)))

    curves = np.stack([first[:, p] for p in positions], axis=1)
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    for j, p in enumerate(positions):
        ax3.plot(range(L), curves[:, j], marker="o", ms=3, label=f"pos {p}")
    ax3.set_xlabel("layer")
    ax3.set_ylabel("C")
    ax3.set_title("Insight 2: C vs layer for selected token positions (sample 0)")
    ax3.legend(fontsize=8, ncol=2)
    fig3.set_dpi(150)
    fig3.tight_layout()
    p3 = out_dir / "insight2_lineplot.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"已保存 {p3}")

    token_cross_cvs = []
    for j in range(curves.shape[1]):
        vec = curves[:, j]
        mu = float(np.mean(vec))
        if mu > 1e-8:
            token_cross_cvs.append(float(np.std(vec) / mu))
    cross_mean = float(np.mean(token_cross_cvs)) if token_cross_cvs else 0.0
    cross_med = float(np.median(token_cross_cvs)) if token_cross_cvs else 0.0
    print("\n--- Insight 2: 所选 token 跨层 CV ---")
    print(f"  mean={cross_mean:.6f}  median={cross_med:.6f}")
    insight2_ok = cross_mean > 0.5 or cross_med > 0.5

    # ----- 图4: prunable ratio -----
    thresholds = [0.01, 0.05, 0.10, 0.15, 0.20]
    flat_all = all_concat.flatten()
    ratios = []
    for t in thresholds:
        ratios.append(100.0 * float(np.mean(flat_all < t)))
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.bar([str(t) for t in thresholds], ratios, color="steelblue")
    ax4.set_xlabel("threshold")
    ax4.set_ylabel("prunable ratio (%)")
    ax4.set_title("(token, layer) pairs with C < threshold")
    fig4.set_dpi(150)
    fig4.tight_layout()
    p4 = out_dir / "prunable_ratio.png"
    fig4.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig4)
    print(f"已保存 {p4}")

    pruned_005 = 100.0 * float(np.mean(flat_all < 0.05))

    print("\n===== 实验总结 =====")
    print(f"模型: {model_path}")
    print(f"样本数: {args.num_samples}, 总有效 token 数: {total_valid_tokens}")
    print(
        f"Insight 1 (同层 token 间差异): 各层平均 CV = {mean_cv_layers:.6f} → "
        f"{'成立' if insight1_ok else '不成立'}"
    )
    print(
        f"Insight 2 (同 token 跨层差异): token 平均跨层 CV = {cross_mean:.6f} "
        f"(median={cross_med:.6f}) → {'成立' if insight2_ok else '不成立'}"
    )
    print(
        f"可裁剪空间: contribution < 0.05 时, {pruned_005:.2f}% 的 (token, layer) pair 可降为 top-1"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
