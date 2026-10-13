# 第14讲 Mask机制：Padding Mask与Causal Mask

> 📌 **本讲问题**：为什么 Decoder 的 Self-Attention 要把未来位置遮掉？Padding 是怎么不污染注意力的？这两个 Mask 分别长什么样？为什么必须加在 Softmax 之前？

## 1. 数学工具/预备知识

**ch04 的 Softmax。** 把 logits 变成概率；给 logits 加 $-\infty$ 会使对应项 $\exp(-\infty)=0$。

**上三角/下三角矩阵。** 下三角矩阵 $M_{ij}=1$ 当 $j\leq i$，否则 0。

**序列填充（padding）。** batch 里句子长度不一，短的补 `<pad>` 对齐到最长长度。

## 2. 完整推导

### 2.1 两种 Mask 的动机

**Padding Mask**：一个 batch 里句子长度不同，短句要补 `<pad>` 对齐。这些 `<pad>` 位置没有实际意义，不能让注意力关注到它们。

**Causal Mask（因果掩码）**：Decoder 自回归生成时，位置 $t$ 只能看到 $1,\dots,t$，不能看到 $t+1,\dots$（否则偷看答案）。

### 2.2 Softmax 前注入 -∞

Mask 的实现：在 Softmax 之前，把要屏蔽位置的 score 设成 $-\infty$（工程上用大负数如 $-10^9$）：

$$\text{MaskedScores}_{ij}=\begin{cases}S_{ij}, & \text{允许关注}\\ -10^9, & \text{禁止关注}\end{cases}$$

经过 Softmax：

$$\alpha_{ij}=\frac{\exp(\text{MaskedScores}_{ij})}{\sum_k\exp(\text{MaskedScores}_{ik})}$$

被屏蔽位置 $\exp(-10^9)\approx 0$，权重趋近 0。数学上等于完全不关注。关键：必须在 Softmax **之前**加，因为 Softmax 会归一化；之后乘 0 会破坏归一化（权重和不再为 1）。

### 2.3 Causal Mask 的下三角结构

对序列长度 $n$，Causal Mask 是 $n\times n$ 矩阵，下三角为 1、严格上三角为 0：

$$\text{Mask}_{ij}=\begin{cases}1, & j\leq i\\ 0, & j>i\end{cases}$$

位置 $i$ 可以看到所有 $j\leq i$，看不到 $j>i$。

### 2.4 Padding Mask 的来源

batch 中两句长度 3 和 5，pad 后统一长度 5。Padding Mask 形状 `(batch, 1, seq_len)`，标记哪些是真实 token：

$$\text{PaddingMask}_{b,j}=\begin{cases}1, & \text{位置 }j\text{ 真实}\\ 0, & \text{位置 }j\text{ 是 padding}\end{cases}$$

注意力打分中，padding 位置作为 key 时被屏蔽（任何 query 都不能关注 pad）。

### 2.5 交叉注意力的 Mask

Cross-Attention 中 Q 来自 Decoder，K/V 来自 Encoder。Encoder 侧可能有 padding，所以 Cross-Attention 的 Mask 来自 Encoder 侧 Padding Mask——Decoder 不能关注 Encoder 的 pad 位置。

## 3. 💡 直觉理解

> 💡 **直觉**：Padding Mask 像考试时把草稿纸（pad 位置）折起来——不能看草稿。Causal Mask 像做阅读理解题目一页一页发——你只能看到当前页和之前页，不能翻到后面看答案。两个 Mask 都在 Softmax 打分前把不该看的地方打 $-\infty$ 标签，让 Softmax 自动忽略。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 Mask 在 Softmax 之后乘 0。那样破坏归一化（权重和不再为 1）。正确做法是 Softmax **之前**把 logits 设 $-\infty$。
>
> ⚠️ **误区二**：以为 Causal Mask 只在训练时需要。推理时也需要——KV cache 里已有历史，当前位置只能关注已生成 token。
>
> ⚠️ **误区三**：以为 Mask 方向无所谓。Causal Mask 写反（上三角为 1）会让模型看到未来，训练直接作弊。

## 5. 💻 代码实现

下面构造 Causal Mask 和 Padding Mask，并验证效果。工程注意：`torch.tril` 生成下三角；`masked_fill(mask==0, -1e9)` 里 mask 的 True 表示允许；bool 掩码在 GPU/CPU 间行为一致。

```python
import torch
import torch.nn.functional as F

# Causal Mask: 下三角全1，上三角全0，True 表示允许关注
def causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).bool()

# Padding Mask
def padding_mask(seq_lens, max_len):
    """seq_lens: 每个样本真实长度列表"""
    mask = torch.zeros(len(seq_lens), 1, max_len, dtype=torch.bool)
    for i, l in enumerate(seq_lens):
        mask[i, 0, :l] = True
    return mask

# 测试 Causal Mask
n = 5
cm = causal_mask(n)
print("Causal Mask (True=允许):")
print(cm.int())

# 验证：位置 i 只能看到 j<=i
scores = torch.randn(1, 1, n, n)  # (batch, heads, q_len, k_len)
scores_masked = scores.masked_fill(~cm, -1e9)
alpha = F.softmax(scores_masked, dim=-1)
print("\n位置 0 的注意力权重 (应只在位置0非零):", alpha[0,0,0].detach().round())
print("位置 2 的权重 (应在 0,1,2 非零, 3,4 为0):", alpha[0,0,2].detach().round(decimals=3))

# 验证 Padding Mask
pm = padding_mask([3, 5], max_len=5)
print("\nPadding Mask (batch=2, lens=[3,5]):")
print(pm[0,0].int(), "(第一个句子只有3个真实位置)")
print(pm[1,0].int(), "(第二个句子有5个)")
```

## 6. 小结

1. Padding Mask 屏蔽 `<pad>` 位置，避免无意义位置参与注意力。
2. Causal Mask 屏蔽未来位置，保证 Decoder 自回归性质。
3. Mask 在 Softmax 前注入 $-\infty$，利用 Softmax 数学特性自然产生零权重。
4. Cross-Attention 的 Mask 来自 Encoder 侧 Padding Mask。

## 7. 延伸阅读/进一步思考

- Causal Mask 是 Decoder-only 模型（GPT）训练并行化的关键：一次前向算出所有位置的预测，靠 mask 保证不偷看。
- KV cache（ch23）下，推理时 Causal Mask 退化成"当前 query 关注所有已缓存 key"，不再是完整 $n\times n$ 矩阵。
- FlashAttention（ch24）的一个优化就是把 mask 融合进 kernel，避免显式构造大掩码矩阵。

## 8. 思考题

**基础题**

1. 如果把 Causal Mask 从下三角改成上三角，模型会变成什么？

**进阶题**

2. 推理时 KV cache 中已有 $t$ 个 token，新位置 $t$ 的 Causal Mask 应该是什么形状？
3. 为什么不在输入嵌入层就把 pad 位置向量置零，而要在注意力里 Mask？

**开放题**

4. 滑窗注意力（Sliding Window Attention）把每个 query 的可见范围限制在附近 $w$ 个位置（类似局部 mask）。从 ch16 复杂度和长程依赖两个角度分析：这种局部 mask 牺牲了什么、换来了什么？
