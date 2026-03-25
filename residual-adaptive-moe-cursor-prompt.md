# Residual-Aware Adaptive Expert Routing

## 核心思想

在每层 MoE 路由之前，利用当前层已有的信息（残差范数变化率 + attention entropy）计算每个 token 的"重要性分数"，重要性高的 token 激活更多专家，重要性低的 token 少激活专家甚至跳过 MoE 层。训练无关，纯推理时优化。

## 算法流程（每层）

```
输入: x_l (进入当前 block 的隐状态), x_after_attn (经过 attention sublayer 后的隐状态), attn_weights (attention 概率矩阵)

1. 计算 token 重要性:
   importance(i) = α · residual_score(i) + (1-α) · attention_score(i)

2. 路由决策:
   if importance(i) > θ_high:
       k(i) = k_max  (如 top-2, 走完整路由)
   elif importance(i) > θ_low:
       k(i) = 1      (只走 top-1 专家)
   else:
       k(i) = 0      (跳过 MoE，直接走残差连接)

3. 按 k 值分桶，批量执行专家计算
```

## 重要性计算

### 信号 1: 残差范数变化率

衡量 attention sublayer 对当前 token 的改变程度。改变大 → token 还在"演化"中 → 需要更多专家处理。

```python
delta = x_after_attn - x_l
residual_score = torch.norm(delta, dim=-1) / (torch.norm(x_l, dim=-1) + 1e-8)
# shape: [batch, seq], 值域 [0, +∞), 越大越重要
```

### 信号 2: Attention Entropy

衡量 token 作为 query 时注意力的集中程度。注意力越集中（低熵）→ 有明确的信息需求 → 更重要。

```python
# attn_weights: [batch, heads, seq, seq]
entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1)  # [batch, heads, seq]
avg_entropy = entropy.mean(dim=1)  # [batch, seq], 对 head 平均
attention_score = 1.0 - avg_entropy / torch.log(torch.tensor(seq_len, dtype=torch.float))
# shape: [batch, seq], 归一化到 [0, 1], 越大越重要
```

### 综合

```python
# 两个信号都归一化到 [0, 1] 后加权
residual_score_norm = torch.sigmoid((residual_score - residual_score.mean()) / (residual_score.std() + 1e-8))
importance = alpha * residual_score_norm + (1 - alpha) * attention_score
```

alpha、θ_high、θ_low 在校准集上用少量数据 grid search 确定。

## 关键约束

- 因果性: 所有信号来自当前层已完成的计算，不依赖未来信息
- 跳过 MoE (k=0) 时直接用残差 x_after_attn 作为输出，等价于 AdaMoE 的 null expert 但无需训练
