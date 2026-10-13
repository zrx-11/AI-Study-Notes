# 第9讲 Multi-Head Attention：多头的本质与维度分析

> 📌 **本讲问题**：为什么 Transformer 要把 $d_{\text{model}}$ 切成好几份分别做注意力、再拼回来？一个大头和八个小头，表达能力有什么本质区别？本讲用参数量和计算量两笔账说清这件事。

## 1. 数学工具/预备知识

**ch08 的 SDPA。** $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$。

**参数量与计算量估算。** 一个 $\mathbb{R}^{m\times k}$ 矩阵有 $mk$ 个参数；矩阵乘 $\mathbb{R}^{m\times k}\mathbb{R}^{k\times n}$ 约需 $2mnk$ 次浮点运算（乘加各算一次，ch15 展开）。

**子空间投影。** 一个 $d$ 维向量被矩阵 $\boldsymbol{W}\in\mathbb{R}^{r\times d}$ 投到 $r$ 维子空间，可以理解成"换一组基看它"。不同的 $\boldsymbol{W}$ 对应不同的观察角度。

## 2. 完整推导

### 2.1 单头注意力的局限

单头注意力 $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ 中，每个位置只有一个权重分布。但实际句子里，一个词可能同时需要关注多种关系："苹果"既要关注主语（谁吃了苹果）、又要关注宾语（苹果被怎么了）、还要关注修饰语（红苹果还是青苹果）。一个权重分布无法同时建模多种关系——softmax 是一个单一的概率分布，没法同时给出"对主语的关注"和"对修饰语的关注"两套独立权重。

### 2.2 MHA 完整公式

Multi-Head Attention 把 $d_{\text{model}}$ 维特征切成 $h$ 个头，每个头维度 $d_k=d_{\text{model}}/h$。每个头独立做 SDPA：

$$\mathrm{head}_i=\mathrm{Attention}(Q\boldsymbol{W}_Q^{(i)},\,K\boldsymbol{W}_K^{(i)},\,V\boldsymbol{W}_V^{(i)})$$

其中 $\boldsymbol{W}_Q^{(i)}\in\mathbb{R}^{d_{\text{model}}\times d_k}$，$\boldsymbol{W}_K^{(i)}\in\mathbb{R}^{d_{\text{model}}\times d_k}$，$\boldsymbol{W}_V^{(i)}\in\mathbb{R}^{d_{\text{model}}\times d_v}$。

拼接所有头再过一个输出投影：

$$\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\boldsymbol{W}_O$$

其中 $\boldsymbol{W}_O\in\mathbb{R}^{h d_v\times d_{\text{model}}}$。原论文取 $h=8$，$d_k=d_v=d_{\text{model}}/h=512/8=64$。

工程上，$\boldsymbol{W}_Q^{(i)}$ 这 $h$ 个矩阵被合并成一个大矩阵 $\boldsymbol{W}_Q\in\mathbb{R}^{d_{\text{model}}\times d_{\text{model}}}$，一次投影后再 reshape 切头，避免 $h$ 次小矩阵乘。

### 2.3 参数量分析

设 $d_{\text{model}}=d$，$h$ 个头，每头 $d_k=d/h$。

**单头（直接用 $d$ 维做 Q/K/V）：**
- $\boldsymbol{W}_Q: d\times d$
- $\boldsymbol{W}_K: d\times d$
- $\boldsymbol{W}_V: d\times d$
- $\boldsymbol{W}_O$ 不需要（输出已经是 $d$ 维）
- 合计 $3d^2$

**多头（$h$ 头，每头 $d_k=d/h$）：**
- $\boldsymbol{W}_Q: h\times d\times(d/h)=d^2$（合并成一个 $d\times d$ 矩阵）
- $\boldsymbol{W}_K: d^2$
- $\boldsymbol{W}_V: d^2$
- $\boldsymbol{W}_O: d\times d$（拼接后 $h\cdot d_k=d$ 维，再投回 $d$ 维）
- 合计 $4d^2$

多头比单头只多了一个 $\boldsymbol{W}_O$（$d^2$ 参数）。也就是说，**多头几乎不增加参数量**，但得到了 $h$ 套独立的注意力权重分布。

### 2.4 计算量分析

对序列长度 $n$。注意力 FLOPs 主要在两处：$QK^\top$ 和 $\mathrm{softmax}\cdot V$。

- 单头：$QK^\top$ 是 $n\times n\times d$，约 $2n^2d$；乘 $V$ 又是 $2n^2d$；合计 $4n^2d$（投影 $nd^2$ 另算）。
- 多头：每个头维度 $d_k=d/h$，每个头的注意力是 $4n^2 d_k$；$h$ 个头合计 $h\cdot 4n^2(d/h)=4n^2d$。

**注意力的 FLOPs 和单头完全一样**——切头只是在不同子空间里做同样的事，总量被 $h$ 和 $1/h$ 约掉了。投影部分（$Q/K/V/O$ 四个 $d\times d$ 矩阵乘 $n$ 维序列）也是 $O(nd^2)$，和单头对比只是多了 $W_O$。

### 2.5 多头的本质

多头不是"做更多次注意力"，而是"在 $h$ 个不同子空间里并行做注意力"。每个头的 $\boldsymbol{W}_Q^{(i)}$ 把输入投到一个不同子空间，在那个子空间里算相似度。不同子空间对应不同类型的关系：有的头关注语法依赖，有的头关注语义指代，有的头关注局部相邻位置。$\boldsymbol{W}_O$ 负责把这些不同视角的结果融合回统一的 $d$ 维表征。

## 3. 💡 直觉理解

> 💡 **直觉**：单头注意力像一只眼睛看世界——只能聚焦一个焦点。多头像八只眼睛，每只戴不同颜色的滤镜：一只看语法关系，一只看语义关系，一只看位置邻近。最后把八只眼睛看到的画面拼起来、再由 $\boldsymbol{W}_O$ 融合，比一只眼睛看得全面。
>
> 关键是：八只眼睛的总"像素数"和一只眼睛一样——只是分通道处理了。所以多头几乎不花额外算力，买到的是视角的多样性。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为多头是"并行做更多次注意力所以更强"。多头总计算量和单头相同，优势在于**子空间多样性**。
>
> ⚠️ **误区二**：以为头数越多越好。头数太多时每头维度太小（$d=512,h=64$ 则每头 8 维），单头表达能力不足，性能反而下降。
>
> ⚠️ **误区三**：以为 $\boldsymbol{W}_O$ 可有可无。没有它，八个头的输出只是粗暴拼接，头与头之间没有信息交换；$\boldsymbol{W}_O$ 负责让各视角融合成统一表征。

## 5. 💻 代码实现

下面手写 MHA。工程关键：`view(b, s, h, d_k).transpose(1, 2)` 切头后，合并头时要先 `transpose(1,2)` 再 `contiguous().view(...)`——`contiguous()` 是因为 transpose 后的张量不连续，直接 view 会报错。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 四个 d_model x d_model 矩阵：Q/K/V 投影 + 输出投影
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # x: (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        # 线性投影后切头
        q = self.split_heads(self.W_Q(Q))   # (b, h, n_q, d_k)
        k = self.split_heads(self.W_K(K))   # (b, h, n_k, d_k)
        v = self.split_heads(self.W_V(V))   # (b, h, n_k, d_k)

        # SDPA
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(scores, dim=-1)
        out = torch.matmul(alpha, v)        # (b, h, n_q, d_k)

        # 合并头：transpose 回 (b, n_q, h, d_k) 再 view 成 (b, n_q, d_model)
        b, h, n_q, _ = out.shape
        out = out.transpose(1, 2).contiguous().view(b, n_q, self.d_model)
        return self.W_O(out)

# 测试
mha = MultiHeadAttention(d_model=512, n_heads=8)
x = torch.randn(2, 10, 512)  # batch=2, seq=10
out = mha(x, x, x)
print("输入 shape:", tuple(x.shape))
print("输出 shape:", tuple(out.shape))
print("参数量:", sum(p.numel() for p in mha.parameters()))
# 四个 512x512 矩阵 + 四个 bias(512) ≈ 4*262144 + 4*512
```

## 6. 小结

1. MHA 把 $d_{\text{model}}$ 切成 $h$ 个子空间，每个独立做 SDPA，最后拼接并线性投影。
2. 多头总参数量（$4d^2$）和计算量与单头（$3d^2$）基本同阶，优势在于子空间多样性。
3. 不同头可学到不同注意力模式（语法、语义、位置等）。
4. 头数选择受限于 $d_k=d_{\text{model}}/h$ 不能太小；$\boldsymbol{W}_O$ 负责融合各头。

## 7. 延伸阅读/进一步思考

- 原论文 Vaswani 等 2017 是多头的出处；后续研究（如 Voita 等 2019）分析发现很多头其实是冗余的，可以剪枝。
- 现代大模型为了省 KV cache（ch23）发展出 MQA（所有头共享同一组 K/V）和 GQA（分组共享），是对"多头"的进一步工程压缩。
- 想理解"子空间多样性"的实证，可看注意力可视化研究（ch29）：不同头确实倾向于捕捉不同的语言现象。

## 8. 思考题

**基础题**

1. $d_{\text{model}}=512$、$h=8$ 时每头维度是多少？$h=16$ 呢？为什么 $h$ 不能取到 512？

**进阶题**

2. 为什么 MHA 中需要额外的输出投影 $\boldsymbol{W}_O$？没有它，拼接八个头的输出会缺什么能力？
3. 多头注意力和"把输入复制 $h$ 份分别过 $h$ 个独立的单头注意力、每份用不同的 $d_k$"有什么区别？从参数共享角度分析。

**开放题**

4. MQA/GQA 把 $h$ 个头的 K/V 共享成 1 组（或少数几组），Q 仍然多头。从参数量和 KV cache 大小角度分析：这为什么能显著加速推理？（预告 ch23。）它牺牲了什么？
