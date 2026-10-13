# 第13讲 前馈网络FFN：逐位置MLP的作用

> 📌 **本讲问题**：注意力负责"信息交换"，FFN 负责什么？为什么中间层维度是 $4d_{\text{model}}$？这个 4 倍系数是怎么来的？为什么说大部分参数和知识都在 FFN 里？

## 1. 数学工具/预备知识

**逐位置（position-wise）。** 同一个 MLP 被序列每个位置独立、相同地应用一次。

**非线性激活函数。** 没有非线性，两层线性变换等价于一层。ReLU/GELU/SwiGLU 提供非线性表达能力。

**参数量估算。** $\boldsymbol{W}_1\in\mathbb{R}^{d\times d_{ff}}$ 有 $d\cdot d_{ff}$ 个参数。

## 2. 完整推导

### 2.1 逐位置本质

FFN 对序列的**每个位置独立、相同**地应用同一个两层 MLP：

$$\mathrm{FFN}(\boldsymbol{x})=\max(0,\boldsymbol{x}\boldsymbol{W}_1+\boldsymbol{b}_1)\boldsymbol{W}_2+\boldsymbol{b}_2$$

关键：同一个 $\boldsymbol{W}_1,\boldsymbol{W}_2$ 被所有位置共享。和 Self-Attention 不同，FFN **不跨位置交换信息**——它只对每个位置的向量做独立变换。

这意味着分工：注意力层负责"位置之间通信"，FFN 负责"每个位置自己的信息加工"。两者交替，形成"交换信息 → 各自思考 → 再交换 → 再思考"的循环。

工程上，因为逐位置独立，可以把 batch 和 seq 两个维合并，写成一个大矩阵乘法，不需要 for 循环遍历位置。

### 2.2 为什么中间层是 $4d_{\text{model}}$

原论文取 $d_{ff}=4d_{\text{model}}=2048$（$d_{\text{model}}=512$ 时）。这个 4 倍不是理论最优值，而是经验选择，但有合理解释：中间层要比输入维宽，才能在"升维 → 非线性 → 降维"中提供足够的容量。

FFN 参数量：
- $\boldsymbol{W}_1: 512\times 2048=1{,}048{,}576$
- $\boldsymbol{W}_2: 2048\times 512=1{,}048{,}576$
- 合计约 2.1M 参数

对比 MHA 参数量（约 $4\times 512\times 512=1{,}048{,}576$，见 ch09），FFN 占整个 Transformer 层约 2/3 参数量。也就是说**大部分参数和知识存储在 FFN 里**，而不是注意力里。

### 2.3 事实存储器假说

研究发现，Transformer 的事实知识（"法国首都是巴黎"这类）主要存储在 FFN 的权重里。注意力路由负责"把对的信息送到对的位置"，FFN 负责"在那个位置上查询和调用知识"。这也是 FFN 需要更大维度的原因——它要存大量参数化知识。

### 2.4 激活函数演进

**ReLU**：$\max(0,x)$。简单，但负值区梯度为 0，可能导致"神经元死亡"。

**GELU**：$x\Phi(x)$，$\Phi$ 是标准正态 CDF。近似为 $0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$。BERT、GPT-2 起用 GELU 替代 ReLU，更平滑。

**SwiGLU**：$(\mathrm{Swish}(x\boldsymbol{W}_1)\odot(x\boldsymbol{W}_2))\boldsymbol{W}_3$。引入门控，PaLM、LLaMA 用。SwiGLU 通常把中间层从 $4d$ 调成 $\frac{8}{3}d$，因为它用三个矩阵而不是两个，参数量大致持平。

## 3. 💡 直觉理解

> 💡 **直觉**：注意力是开会讨论——每个人从别人那里获取信息。FFN 是会后各自整理笔记——把刚听到的信息消化、重组、存进自己的笔记本。开会本身不产生新知识，只是交换；真正的知识加工发生在独自思考的时候。FFN 维度设大 4 倍，是因为"笔记本"要够大能装下足够多的知识。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 FFN 和普通 MLP 一样是"全局变换"。FFN 逐位置独立，位置之间没有信息交换。这也是它能写成一个大矩阵乘法的原因。
>
> ⚠️ **误区二**：以为 FFN 可有可无。消融实验表明去掉 FFN 性能大幅下降，它不是冗余的。
>
> ⚠️ **误区三**：以为 FFN 的 4 倍是理论最优。它是经验值，不同模型从 2.67 到 8 都有。

## 5. 💻 代码实现

下面对比 ReLU FFN 和 SwiGLU，并验证 FFN 的逐位置独立性：改一个位置的输入，只影响那个位置的输出。工程注意：SwiGLU 用三个矩阵，为保持参数量和 $4d$ 版持平，中间维取 $\frac{8}{3}d$。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class SwiGLU(nn.Module):
    def __init__(self, d_model=512, d_ff=None):
        super().__init__()
        # 三矩阵保持参数量与两矩阵 4d 版相当: 3*(d*d_ff) ≈ 2*(d*4d)
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
        self.w_gate = nn.Linear(d_model, d_ff)
        self.w_up = nn.Linear(d_model, d_ff)
        self.w_down = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

# 对比参数量
ffn = SimpleFFN(512, 2048)
swiglu = SwiGLU(512)
print(f"ReLU FFN 参数量: {sum(p.numel() for p in ffn.parameters()):,}")
print(f"SwiGLU 参数量:   {sum(p.numel() for p in swiglu.parameters()):,}")

# 验证逐位置：改位置0，只影响位置0的输出
x = torch.randn(2, 5, 512)
out = ffn(x)
x_mod = x.clone()
x_mod[0, 0, :] += 100
out_mod = ffn(x_mod)
diff = (out_mod - out).abs()
print(f"\n输入 shape: {tuple(x.shape)}, 输出 shape: {tuple(out.shape)}")
print(f"改变位置0后, 位置0输出差异: {diff[0,0].max().item():.4f}")
print(f"改变位置0后, 位置1输出差异: {diff[0,1].max().item():.6f} (应≈0)")
```

## 6. 小结

1. FFN 对每个位置独立应用同一个两层 MLP，不跨位置交换信息。
2. 中间层 $d_{ff}=4d_{\text{model}}$ 是经验选择，FFN 占整层约 2/3 参数量。
3. FFN 被认为是 Transformer 的"事实存储器"。
4. 激活从 ReLU 演进到 GELU 再到 SwiGLU，门控提升表达能力。

## 7. 延伸阅读/进一步思考

- 事实存储器假说可参考 Geva 等 2020《Transformer Feed-Forward Layers Are Key-Value Memories》。
- SwiGLU 出自 Shazeer 2020《GLU Variants Improve Transformer》；LLaMA、PaLM 均采用。
- 把 FFN 扩成多个专家就是 MoE（ch25）——逐位置地路由到不同 FFN，是"条件计算"。

## 8. 思考题

**基础题**

1. 如果把 FFN 中间层从 $4d$ 改成 $d$（瓶颈），会发生什么？

**进阶题**

2. 为什么 FFN 可以写成一个大矩阵乘法，而不用 for 循环遍历每个位置？
3. SwiGLU 用三个矩阵代替原来两个，为什么还能保持参数量相当？

**开放题**

4. MoE（ch25）把 FFN 换成多个专家网络、每个 token 只激活其中一两个。从"FFN 存事实"的角度分析：为什么把 FFN 做成稀疏多个专家能在不显著增加计算量的前提下扩大知识容量？
