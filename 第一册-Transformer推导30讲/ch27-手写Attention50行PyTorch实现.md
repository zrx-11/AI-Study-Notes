# 第27讲 手写 Attention：50 行 PyTorch 实现

> 📌 **本讲问题**：前面讲了那么多公式，现在动手把注意力从零写出来。不调任何现成的注意力 API，不依赖黑盒，一行一行把 SDPA 和 MultiHeadAttention 搭出来，并用单元测试证明它没有"偷看未来"。

## 1. 数学工具/预备知识

- **ch08 的 Scaled Dot-Product Attention**：$\mathrm{softmax}(QK^\top/\sqrt{d_k})V$。
- **ch09 的多头**：把 $d$ 维切成 $h$ 份，每份 $d_k=d/h$，各自做注意力再拼回来。
- **ch14 的 Causal Mask**：把未来位置的分数换成 $-\infty$，softmax 后权重为 0。

这一讲不推新公式，重点是把公式正确翻译成能跑的代码，并想清楚每个 reshape 的维度含义。

## 2. 完整推导

### 2.1 SDPA 的五个步骤

Scaled Dot-Product Attention 拆开就是五步：

1. $QK^\top$：算每个 query 和每个 key 的相似度；
2. 除以 $\sqrt{d_k}$：防止点积过大导致 softmax 饱和（ch08）；
3. 加 mask：把未来/填充位置的分数设成 $-\infty$；
4. softmax：把分数归一化成权重；
5. 乘 $V$：按权重加权求和。

翻译成代码就是一个 `forward` 函数，下面 2.2 给出。

### 2.2 MultiHeadAttention 的数据流

多头就是把上面这套做 $h$ 遍，只是在不同子空间里。关键在 reshape：

- 先用 $W_Q,W_K,W_V$ 把 $d$ 维投影到 $d$ 维；
- 把最后一维 $(d)$ 拆成 $(h, d_k)$，再把 head 维搬到前面，让注意力对每个头独立做；
- 注意力算完后，把头拼回去（transpose + reshape）；
- 最后用 $W_O$ 投影回 $d$ 维。

### 2.3 单元测试：因果掩码不能泄露未来

最关键的验证是：位置 0 的输出只能看到位置 0 及以前。具体做法：把位置 4 的输入改一下，如果位置 0 的输出变了，就说明它偷看到了位置 4——掩码失败。位置 4 自己的输出当然应该变。

## 3. 💡 直觉理解

> 💡 **直觉**：把 Q/K/V 投影 → 分头 → 点积 → 缩放 → Mask → Softmax → 乘 V → 合并头 → 输出投影，就是全部。没有任何玄学，全是矩阵乘法加 reshape。真正容易错的不是公式，是维度搬来搬去时把 head 维放错位置。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 MHA 需要很多特殊算子。它就是矩阵乘法 + reshape + softmax。
>
> **误区二**：以为 `transpose` 之后直接 `view` 就行。transpose 会让张量在内存里不连续，必须先 `.contiguous()` 再 `.view`，否则结果错乱。

## 5. 💻 代码实现

下面从零写 `ScaledDotProductAttention` 和 `MultiHeadAttention`，并跑两个单元测试：因果无泄露、输出形状正确。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力: softmax(QK^T/sqrt(d_k)) V"""
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        # 步骤1+2: 点积并缩放。Q:(b,h,nq,dk) K:(b,h,nk,dk) -> scores:(b,h,nq,nk)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        # 步骤3: 把 mask=0 的位置分数换成 -inf, softmax 后权重为0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 步骤4: softmax 归一化成注意力权重
        alpha = F.softmax(scores, dim=-1)
        # 步骤5: 按权重加权 V
        return torch.matmul(alpha, V), alpha

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被头数整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads          # 每个头的维度
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()

    def split_heads(self, x):
        """(b, s, d_model) -> (b, n_heads, s, d_k): 把最后一维拆成头并换到前面"""
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        b = Q.size(0)
        q = self.split_heads(self.W_Q(Q))     # (b,h,nq,dk)
        k = self.split_heads(self.W_K(K))
        v = self.split_heads(self.W_V(V))
        out, alpha = self.attn(q, k, v, mask)  # out:(b,h,nq,dk)
        # 合并头: transpose 回 (b,s,h,dk) 再展平成 (b,s,d_model)
        # transpose 后必须 contiguous() 才能 view
        out = out.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        return self.W_O(out), alpha

# ===== 单元测试 =====
def test_causal_no_leak():
    """因果掩码: 改了位置4, 位置0的输出绝不能变"""
    torch.manual_seed(42)
    mha = MultiHeadAttention(64, 4)
    x = torch.randn(1, 5, 64)
    # 下三角掩码: 位置 i 只能看 0..i
    mask = torch.tril(torch.ones(5, 5)).unsqueeze(0).unsqueeze(0).bool()
    out1, _ = mha(x, x, x, mask)
    # 把位置4的输入打乱
    x2 = x.clone(); x2[0, 4, :] += 100
    out2, _ = mha(x2, x2, x2, mask)
    diff0 = (out1[0, 0] - out2[0, 0]).abs().max().item()
    diff4 = (out1[0, 4] - out2[0, 4]).abs().max().item()
    print(f"位置0输出差异: {diff0:.2e} (应≈0)")
    print(f"位置4输出差异: {diff4:.2e} (应>0)")
    assert diff0 < 1e-4, "因果掩码泄露！位置0看到了未来！"
    print("✓ 因果无泄露测试通过")

def test_output_shape():
    mha = MultiHeadAttention(512, 8)
    x = torch.randn(2, 10, 512)
    out, alpha = mha(x, x, x)
    assert out.shape == (2, 10, 512), f"输出形状错: {out.shape}"
    assert alpha.shape == (2, 8, 10, 10), f"注意力权重形状错: {alpha.shape}"
    print("✓ 输出形状测试通过")

test_causal_no_leak()
test_output_shape()
```

## 6. 小结

1. SDPA = 点积 → 缩放 → Mask → Softmax → 乘 V，五步。
2. MHA = 投影 → 分头 → SDPA → 合并头 → 输出投影。
3. transpose 后要 `.contiguous()` 才能 `view`，否则维度错乱。
4. 因果无泄露测试是验证 Causal Mask 正确性的标准方法。

## 7. 思考题

**基础题**

1. `split_heads` 里先 `view(b,s,n_heads,d_k)` 再 `transpose(1,2)`，为什么一定要 transpose 把头维换到前面？不换会怎样？

**进阶题**

2. 注意力权重 `alpha` 的形状是 `(batch, n_heads, seq, seq)`。请解释为什么 n_heads 维要放在 batch 和 seq 之间，而不是最后。

**开放题**

3. 这个实现里 Q、K、V 都来自同一个输入（自注意力）。怎么改成 Cross-Attention——让 Q 来自 decoder、K/V 来自 encoder？只需要改 forward 的哪几个参数？

## 8. 延伸阅读 / 进一步思考

- 本讲是 ch08、ch09、ch14 的代码落地，可对照原论文图 2。
- 进阶：把这份手写 MHA 换成 `F.scaled_dot_product_attention`（FlashAttention 后端），体会"同一个数学、不同工程实现"。
- ch28 会把这个 MHA 装进完整 Transformer Block。
