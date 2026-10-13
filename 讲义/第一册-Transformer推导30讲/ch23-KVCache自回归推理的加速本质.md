# 第23讲 KV Cache：自回归推理的加速本质

> 📌 **本讲问题**：为什么大模型推理不能像训练那样一次并行算完？KV Cache 到底缓存了什么？它把推理复杂度从什么量级降到了什么量级？又为什么省了计算却吃显存？

## 1. 数学工具/预备知识

**（1）自回归解码。** Decoder 一次只生成一个 token，然后把它接回输入再生成下一个（ch10）。位置 $t$ 的输出只依赖位置 $1:t$。

**（2）K/V 的计算只依赖自己。** $K_t=W_K\boldsymbol{x}_t$，$V_t=W_V\boldsymbol{x}_t$。注意 $K_t,V_t$ 只由第 $t$ 个 token 自己的嵌入决定，和其他位置的 Query 无关——这是 KV Cache 能成立的根本原因。

**（3）复杂度求和。** 本讲会用到 $\sum_{t=1}^{n}t=n(n+1)/2\approx n^2/2$。

**（4）显存 ≈ 参数量 × 字节数。** 一个 FP16 数占 2 字节，FP32 占 4 字节。

## 2. 完整推导

### 2.1 自回归解码为什么是 $O(n^2)$

生成第 $t$ 个新 token 时，模型的输入是前面全部 $x_1,\dots,x_t$。如果每生成一个新 token 都把整个序列从头到尾重新算一遍：

- 第 1 步算长度 1 的序列，
- 第 2 步算长度 2 的序列，
- ...
- 第 $n$ 步算长度 $n$ 的序列。

关键观察：**生成第 $t$ 步时，前 $t-1$ 个 token 的 K 和 V，和上一步（生成第 $t-1$ 步时）算出来的完全一样。** 因为 $K_i=W_K\boldsymbol{x}_i$ 只依赖 $\boldsymbol{x}_i$ 自己，而 $\boldsymbol{x}_{1:t-1}$ 这一步和上一步没有任何变化。每次重算纯粹是重复劳动。

### 2.2 FLOPs 定量对比

**无 Cache**：第 $t$ 步要处理长度 $t$ 的序列，投影部分代价约 $O(td^2)$。把 $n$ 步加起来：

$$
\sum_{t=1}^{n}O(td^2)=O(d^2)\sum_{t=1}^{n}t=O(n^2 d^2)
$$

这是关于 $n$ 的二次方。

**有 Cache**：每一步只需要对**新 token** 算一次自己的 Q/K/V（代价 $O(d^2)$），然后拿新 Q 去和**已缓存的全部 K/V** 做注意力（代价 $O(td)$）。每步：

$$
\text{每步 FLOPs}=O(d^2+td)
$$

总 FLOPs：

$$
\sum_{t=1}^{n}O(d^2+td)=O(nd^2+d\sum_{t=1}^{n}t)=O(nd^2+n^2d)
$$

对比一下：无 Cache 是 $O(n^2 d^2)$，有 Cache 是 $O(nd^2+n^2 d)$。当 $d$ 很大时，$nd^2$ 项主导，复杂度从关于 $n$ 的二次（$n^2 d^2$）降到了**关于 $n$ 近似线性**（$nd^2$）。这就是为什么 KV Cache 是推理加速的第一功臣。

### 2.3 显存占用推导

KV Cache 要把所有历史 token 的 K 和 V 存下来。设 $L$ 层、$h$ 个头、头维度 $d_k=d/h$、序列长度 $n$。每层要存：

$$
\text{每层 KV}=2\times h\times d_k\times n=2\times d\times n
$$

（前面的 2 是 K、V 两份；$h\times d_k=d$ 是整层的宽度。）$L$ 层全部：

$$
\text{KV Cache 大小}=2Ld n\times\text{bytes}
$$

用 FP16 存（每个数 2 字节）：

$$
\text{显存}=2Ldn\times 2=4Ldn\text{ bytes}
$$

代入 LLaMA-7B（$L=32,d=4096$），序列长度 2048：

$$
\text{显存}=4\times32\times4096\times2048\approx1.07\times10^9\text{ bytes}\approx1\text{ GB}
$$

序列拉到 32k，这一个数就涨 16 倍——KV Cache 是长上下文推理的显存大头。

### 2.4 Memory-Bound 与 MQA/GQA

推理时瓶颈往往不是计算（Compute-Bound），而是**显存带宽**：每生成一个 token，都要把整份 KV Cache 从显存搬到计算单元读一遍。KV Cache 越大，搬得越慢。这就是 Memory-Bound。

既然 KV Cache 吃显存又吃带宽，就有了两个省它的办法：

- **MQA（Multi-Query Attention）**：所有头共享同一份 K/V。KV Cache 直接缩小 $h$ 倍，但表达力略降。
- **GQA（Grouped-Query Attention）**：把头分成若干组，组内共享 K/V，是 MQA 和普通 MHA 的折中。LLaMA-2 70B、LLaMA-3 都用 GQA。

## 3. 💡 直觉理解

> 💡 **直觉**：KV Cache 像上课做笔记。老师每讲一段，你就把这段的要点（K/V）记在本子上。下次写新段落时，不用把之前讲过的内容重新听一遍，直接翻笔记（缓存的 K/V）就行。没有 KV Cache，相当于每写一个字都把整堂课的录音重放一遍——蠢得要死。MQA/GQA 则是全班共用一份笔记：省纸，但笔记记得没那么细。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 KV Cache 缓存的是注意力输出。它缓存的是每层的 K 和 V——注意力输出是每次拿新 Q 和缓存 K/V 实时现算的。
>
> **误区二**：以为 KV Cache 只省时间不占空间。它把计算量降下来了，但代价是显存——长序列时 KV Cache 可能比模型权重本身还占地方。
>
> **误区三**：以为训练时也该开 KV Cache。训练时一次并行看到全部 token，每个位置的 Q/K/V 都要算，没有"历史"可缓存——缓存是为自回归的逐 token 生成服务的。

## 5. 💻 代码实现

下面模拟逐 token 生成：先算前 3 个 token 的 K/V 存进缓存，再生成第 4 个 token 时直接复用缓存。最后验证：用缓存算出来的第 4 个 token 输出，和"把前 4 个 token 从头全量算一遍"的结果完全一致。

```python
import torch
import torch.nn.functional as F

def full_attention(Q, K, V):
    """不带缓存: 一次性算完整序列的注意力"""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    return F.softmax(scores, dim=-1) @ V

def one_step_with_cache(q_new, k_cache, v_cache, k_new, v_new):
    """带缓存: 新 token 的 K/V 拼到缓存末尾, 再算注意力"""
    K = torch.cat([k_cache, k_new], dim=1)   # 历史 K + 新 K
    V = torch.cat([v_cache, v_new], dim=1)  # 历史 V + 新 V
    out = full_attention(q_new, K, V)
    return out, K, V

torch.manual_seed(0)
d = 64
x = torch.randn(1, 5, d)   # 模拟 5 个 token 的嵌入

# 第一步: 处理前 3 个 token, 它们的 K/V 进缓存
Q0 = x[:, :3, :]
K0, V0 = x[:, :3, :], x[:, :3, :]
_ = full_attention(Q0, K0, V0)
print("前3个token的 K/V 缓存 shape:", K0.shape)

# 第二步: 生成第 4 个 token, 复用缓存
q_new = x[:, 3:4, :]
k_new, v_new = x[:, 3:4, :], x[:, 3:4, :]
out4, K_upd, V_upd = one_step_with_cache(q_new, K0, V0, k_new, v_new)
print("第4个token输出 shape:", out4.shape)
print("更新后 K 缓存 shape:", K_upd.shape, "(应为 (1,4,d))")

# 验证: 缓存法 == 从头全量算
out_full = full_attention(x[:, :4, :], x[:, :4, :], x[:, :4, :])
match = torch.allclose(out_full[:, 3:4, :], out4, atol=1e-5)
print("缓存结果与全量计算一致:", match)
```

## 6. 小结

1. KV Cache 缓存每层历史 token 的 K/V，避免自回归生成时重复计算。
2. 无 Cache 推理 $O(n^2 d^2)$，有 Cache 降到 $O(nd^2+n^2d)$，约等于把 $n$ 从二次变线性。
3. KV Cache 显存约 $4Ldn$ 字节（FP16），长序列时很大，是推理的显存/带宽瓶颈。
4. MQA/GQA 通过共享 K/V 缩小缓存，是推理优化的关键。

## 7. 思考题

**基础题**

1. batch size=1、序列长度 4096、LLaMA-7B（$L=32,d=4096$）的 KV Cache 用 FP16 存约多大？用本讲公式 $4Ldn$ 算一遍。

**进阶题**

2. MQA 把所有头的 K/V 合并成一份。它把 KV Cache 缩小了几倍？这样做会损失什么表达能力？为什么 GQA 是折中？

**开放题**

3. KV Cache 推理是 Memory-Bound 的——瓶颈在带宽不在算力。vLLM 的 PagedAttention 把 KV Cache 像操作系统虚拟内存一样分页管理，解决了什么问题？请从"显存碎片"的角度想一想。

## 8. 延伸阅读 / 进一步思考

- KV Cache 与推理优化：PagedAttention（Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023）。
- MQA：Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019；GQA：Ainslie et al., 2023。
- 进阶：连续批处理（continuous batching）、投机采样（speculative decoding），是推理系统方向的延伸。
