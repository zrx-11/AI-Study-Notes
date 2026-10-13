# 第8讲 Scaled Dot-Product Attention：完整推导

> 📌 **本讲问题**：为什么点积注意力要除以 $\sqrt{d_k}$？不除会发生什么？这一个小小的除法，是 Transformer 训练稳定的关键工程细节。本讲从方差出发把它推到底。

## 1. 数学工具/预备知识

**ch04 的 Softmax 饱和区。** 单独 Softmax 的雅可比 $J_{ij}=p_i(\delta_{ij}-p_j)$，当某个 $p_i\to 1$、其余 $\to 0$ 时，整个雅可比趋近 0，梯度消失。

**期望与方差。** 对随机变量 $X$，$\mathrm{Var}(X)=\mathbb{E}[X^2]-(\mathbb{E}[X])^2$。独立变量之和的方差等于方差之和：$\mathrm{Var}(\sum_l X_l)=\sum_l\mathrm{Var}(X_l)$。

**ch07 的 Q/K/V 抽象。** $\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}$ 的定义和打分 $S_{ij}=\boldsymbol{q}_i^\top\boldsymbol{k}_j$。

## 2. 完整推导

### 2.1 维度标注

设 $\boldsymbol{Q}\in\mathbb{R}^{n\times d_k}$，$\boldsymbol{K}\in\mathbb{R}^{m\times d_k}$，$\boldsymbol{V}\in\mathbb{R}^{m\times d_v}$。缩放点积注意力（SDPA）定义为：

$$\mathrm{Attention}(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})=\mathrm{softmax}\!\left(\frac{\boldsymbol{Q}\boldsymbol{K}^\top}{\sqrt{d_k}}\right)\boldsymbol{V}$$

打分矩阵 $\boldsymbol{S}=\boldsymbol{Q}\boldsymbol{K}^\top\in\mathbb{R}^{n\times m}$，每个元素 $S_{ij}=\boldsymbol{q}_i^\top\boldsymbol{k}_j=\sum_{l=1}^{d_k}q_{il}k_{jl}$。

### 2.2 为什么除以 $\sqrt{d_k}$：方差推导

假设 $\boldsymbol{q}_i$ 和 $\boldsymbol{k}_j$ 的各分量是独立同分布的随机变量，均值为 0、方差为 1（这是 Xavier/He 初始化附近的典型假设，ch20 会讲为什么这样初始化）。计算 $S_{ij}=\sum_{l=1}^{d_k}q_{il}k_{jl}$ 的期望和方差。

**期望**：

$$\mathbb{E}[S_{ij}]=\sum_{l=1}^{d_k}\mathbb{E}[q_{il}k_{jl}]=\sum_{l=1}^{d_k}\mathbb{E}[q_{il}]\mathbb{E}[k_{jl}]=0$$

第二步用了独立性（$\mathbb{E}[XY]=\mathbb{E}[X]\mathbb{E}[Y]$）和零均值。

**方差**：因为 $\mathbb{E}[S_{ij}]=0$，$\mathrm{Var}(S_{ij})=\mathbb{E}[S_{ij}^2]$。展开：

$$\mathrm{Var}(S_{ij})=\sum_{l=1}^{d_k}\mathrm{Var}(q_{il}k_{jl})$$

跨 $l$ 独立，方差可加。对单个乘积 $q_{il}k_{jl}$，由独立性和零均值：

$$\mathrm{Var}(q_{il}k_{jl})=\mathbb{E}[q_{il}^2 k_{jl}^2]=\mathbb{E}[q_{il}^2]\mathbb{E}[k_{jl}^2]=1\cdot 1=1$$

（这里用了 $\mathbb{E}[q^2]=\mathrm{Var}(q)+(\mathbb{E}q)^2=1+0=1$。）所以：

$$\mathrm{Var}(S_{ij})=\sum_{l=1}^{d_k}1=d_k$$

$S_{ij}$ 的标准差是 $\sqrt{d_k}$。当 $d_k=64$ 时，打分典型量级是 $\sqrt{64}=8$；$d_k=512$ 时是 $\sqrt{512}\approx 22.6$。维度越高，打分的"音量"越大。

### 2.3 Softmax 饱和区的梯度消失

把这些很大的 logits 喂给 Softmax。假设 $S_{ij}$ 量级是 $O(\sqrt{d_k})$，那么 $n$ 个分数中最大的那个会比其他值大很多（差值是 $O(\sqrt{d_k})$）。回忆 ch04：Softmax 输入差越大，输出越接近 one-hot。比如 $d_k=512$ 时分数标准差 22，最大分数比平均高两三个标准差，softmax 后那个位置权重几乎就是 1。

一旦 Softmax 输出接近 one-hot，回忆 ch04 雅可比 $J_{ij}=p_i(\delta_{ij}-p_j)\to 0$——梯度几乎为零，模型学不动。更糟的是，注意力权重接近 one-hot 意味着加权求和退化成"只取一个 value"，模型的软检索能力退化。

除以 $\sqrt{d_k}$ 后：

$$\mathrm{Var}\!\left(\frac{S_{ij}}{\sqrt{d_k}}\right)=\frac{1}{d_k}\mathrm{Var}(S_{ij})=\frac{d_k}{d_k}=1$$

打分的标准差回到 1，分数之间的典型差值在 $O(1)$ 量级，Softmax 工作在非饱和区，权重分布合理、梯度正常流动。

### 2.4 与加性注意力的效率对比

加性注意力（ch06）用小 MLP 打分：$e_{ij}=v^\top\tanh(\boldsymbol{W}_1\boldsymbol{q}_i+\boldsymbol{W}_2\boldsymbol{k}_j)$。

- 加性：每个 $(i,j)$ 对都要过一个 $d_a$ 维的小网络，每对 $O(d_a d_k)$。
- 点积：一次矩阵乘法 $\boldsymbol{Q}\boldsymbol{K}^\top$，全部 $n\times m$ 对一共 $O(nmd_k)$，而且矩阵乘法在 GPU 上高度优化。

当 $d_k$ 不大（原论文 $d_k=64$）时，点积注意力快得多，这是原论文选择点积的主因。加性注意力在小 $d_k$ 下没有速度优势，又多参数，被淘汰。

### 2.5 直觉核对

把 2.2 的推导和 ch04 串起来：不缩放 → 点积方差 $d_k$ → logits 过大 → softmax 饱和 → 雅可比趋零 → 梯度消失。缩放 $\sqrt{d_k}$ → 方差 1 → softmax 在线性区。这就是为什么缩放因子恰好是 $\sqrt{d_k}$ 而不是 $d_k$ 或别的——它正好把方差拉回 1。

## 3. 💡 直觉理解

> 💡 **直觉**：维度越高，点积的"数值音量"越大——就像一千个人同时说话，噪音叠加起来震耳欲聋。除以 $\sqrt{d_k}$ 就是给这个嘈杂的打分做音量归一化，让 Softmax 这个"耳朵"能正常分辨谁在说什么，而不是被最大的声音震聋（饱和成 one-hot，梯度消失）。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为除以 $\sqrt{d_k}$ 是为了"数值稳定防上溢"。数值稳定是靠减最大值实现的（ch04），除以 $\sqrt{d_k}$ 的目的是**控制 Softmax 输入分布、避免饱和导致梯度消失**。
>
> ⚠️ **误区二**：以为这个缩放是经验调出来的。它有严格的方差推导支撑——前提是 Q/K 分量零均值单位方差。
>
> ⚠️ **误区三**：以为缩放因子该用 $d_k$ 而不是 $\sqrt{d_k}$。$d_k$ 是方差，$\sqrt{d_k}$ 才是标准差；要把标准差拉回 1，除的必须是标准差。

## 5. 💻 代码实现

下面从零实现 SDPA 并对照 PyTorch 官方 `F.scaled_dot_product_attention`。工程注意：官方实现会自动处理 mask 和数值稳定，手写版要自己 `masked_fill`；验证饱和现象时，让 $d_k$ 很大就能看到原始点积分数量级是 $\sqrt{d_k}$。

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch, n_heads, n_q, d_k)
    K: (batch, n_heads, n_k, d_k)
    V: (batch, n_heads, n_k, d_v)
    """
    d_k = Q.size(-1)
    # 第一步：点积打分 (b, h, n_q, n_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))
    # 第二步：缩放，把方差从 d_k 拉回 1
    scores = scores / (d_k ** 0.5)
    # 第三步：mask（如果有），把禁止位置填大负数
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 第四步：softmax 归一化
    alpha = F.softmax(scores, dim=-1)
    # 第五步：加权求和
    output = torch.matmul(alpha, V)
    return output, alpha

# 对照官方实现
torch.manual_seed(0)
b, h, n, d_k = 2, 4, 10, 64
Q = torch.randn(b, h, n, d_k)
K = torch.randn(b, h, n, d_k)
V = torch.randn(b, h, n, d_k)

out_mine, alpha_mine = scaled_dot_product_attention(Q, K, V)
out_official = F.scaled_dot_product_attention(Q, K, V)

print("手动 SDPA 输出 shape:", tuple(out_mine.shape))
print("与官方实现差异:", torch.allclose(out_mine, out_official, atol=1e-5))

# 验证：不除以 sqrt(d_k) 会导致 softmax 饱和（权重接近 one-hot）
d_k_large = 512
torch.manual_seed(0)
Q2 = torch.randn(1, 1, 1, d_k_large)
K2 = torch.randn(1, 1, d_k_large, d_k_large)  # 1 个 query，很多 key
raw = (Q2 @ K2.transpose(-2, -1)).squeeze()
scaled = raw / (d_k_large ** 0.5)
p_raw = F.softmax(raw, dim=-1)
p_scaled = F.softmax(scaled, dim=-1)
print(f"\nd_k={d_k_large} 时:")
print(f"  原始点积分数 std: {raw.std().item():.2f} (应≈sqrt({d_k_large})={d_k_large**0.5:.1f})")
print(f"  缩放后分数 std:   {scaled.std().item():.2f} (应≈1)")
print(f"  不缩放 softmax 最大权重: {p_raw.max().item():.4f} (接近1=饱和)")
print(f"  缩放后 softmax 最大权重: {p_scaled.max().item():.4f} (更分散)")
```

## 6. 小结

1. SDPA 公式：$\mathrm{softmax}(QK^\top/\sqrt{d_k})V$。
2. 除以 $\sqrt{d_k}$ 的原因：Q/K 分量独立零均值单位方差时，点积方差为 $d_k$，不缩放会导致打分过大、Softmax 饱和、梯度消失。
3. 缩放后方差为 1，Softmax 工作在线性区。
4. 点积注意力比加性注意力计算高效，因为可以用高度优化的矩阵乘法。

## 7. 延伸阅读/进一步思考

- 缩放因子的原始论证见 Vaswani 等 2017《Attention Is All You Need》第 3.2.2 节；注意论文里也只是定性猜测了"大维度下点积方差变大"，本讲 2.2 是严格化。
- 现代变体（如 GQA、MQA）改变的是 K/V 的头数，不改变 $1/\sqrt{d_k}$ 这个缩放。
- 若 Q/K 经过 LayerNorm（ch12）后再接注意力，点积方差会被 LN 重新归一，缩放的必要性会变化——这是某些架构选择的讨论点。

## 8. 思考题

**基础题**

1. 如果 Q 和 K 的分量方差不是 1 而是 $\sigma^2$，推导点积 $S_{ij}$ 的方差，并给出正确的缩放因子。

**进阶题**

2. 当 $d_k=1$ 时缩放因子是 1。此时点积注意力 $\mathrm{softmax}(\boldsymbol{q}\boldsymbol{k}^\top)\boldsymbol{v}$ 等价于什么操作？
3. 为什么不直接对 Q/K 做 LayerNorm 来控制方差，而要在打分后除以 $\sqrt{d_k}$？（提示：考虑训练速度和参数量。）

**开放题**

4. 实验发现：即使除以 $\sqrt{d_k}$，当 $d_k$ 继续增大（比如到 4096），训练仍会不稳。这说明"方差归一化到 1"可能还不够——从 softmax 的熵和梯度量级角度，你觉得还缺什么？（提示：分数 std=1 时，$n$ 个分数的最大值期望大约是 $\sqrt{2\log n}$，它随 key 数量 $n$ 增长，这又会造成新的饱和。）
