# 第24讲 FlashAttention：IO 感知的精确注意力

> 📌 **本讲问题**：注意力的 FLOPs 明明那么多，为什么 GPU 上真正的瓶颈不是算不动，而是"搬不动数据"？FlashAttention 凭什么不改一个 FLOPs、不做任何近似，却能把注意力提速好几倍？

## 1. 数学工具/预备知识

**（1）GPU 内存层级。** GPU 里有两种存储：HBM（显存，容量大但带宽低，相当于冰箱）和 SRAM（片上缓存，容量小但带宽极高，相当于灶台）。计算单元（SM）只能直接用 SRAM 里的数据。

**（2）Compute-Bound vs Memory-Bound。** 如果时间主要花在计算上，叫 Compute-Bound；如果主要花在把数据从 HBM 搬到 SRAM，叫 Memory-Bound。现代 GPU 的算力增长远快于显存带宽增长，所以大矩阵操作常常是 Memory-Bound。

**（3）Online Softmax。** 标准 softmax 需要先看到全部分数才能归一化。但其实可以边看边维护一个"当前最大值"和"修正后的指数和"，逐块累加，最后归一化。这是 FlashAttention 的数学核心。

**（4）为什么 softmax 要减最大值。** 直接算 $\exp(s)$ 容易上溢，减去当前最大值 $m=\max s$ 后，$\exp(s-m)\le 1$，数值稳定。

## 2. 完整推导

### 2.1 标准注意力为什么是 Memory-Bound

标准注意力流程在 HBM 和 SRAM 之间反复搬数据：

1. 从 HBM 读 Q,K,V 进 SRAM；
2. 算 $S=QK^\top$，把 $S$ 写回 HBM；
3. 从 HBM 读回 $S$，算 softmax 得 $A$，把 $A$ 写回 HBM；
4. 从 HBM 读回 $A$ 和 $V$，算 $AV$，写回 HBM。

中间矩阵 $S$ 和 $A$ 都是 $n\times n$ 的。当序列很长时，这两个矩阵大到装不进 SRAM，只能在 HBM 里存着，一来一回搬好几趟。算力其实没占满，时间全耗在搬数据上——这就是 Memory-Bound。

### 2.2 标准注意力的 IO 复杂度

设 SRAM 容量为 $M$（几十 KB），Q/K/V 每行维度 $d$。标准注意力把 $n\times n$ 的中间矩阵物化在 HBM 里，每处理一个 query 块都要和全部 key 块交互，总 HBM 读写量约为：

$$
\text{IO}_{\text{standard}}=O\left(n^2+\frac{n^2 d}{M}\right)
$$

关键是那个 $n^2$ 项——它来自"把 $n\times n$ 的 scores 矩阵整个写进 HBM 再读回来"。$n$ 一大，这一项就压垮一切。

### 2.3 Online Softmax：边算边归一化

FlashAttention 的核心洞察：**根本不需要把 $n\times n$ 的 scores 矩阵整个写进 HBM**。可以把 K/V 切成小块，在 SRAM 里逐块算，同时维护三个量：当前最大分数 $m$、修正后的指数和 $l$、累加输出 $O$。

假设处理到第 $j$ 个 key 块时，手里有：
- $m_j$：目前见过的最大 score；
- $l_j=\sum_{i\le j}\exp(s_i-m_j)$：以 $m_j$ 为基准修正后的指数和；
- $O_j=\sum_{i\le j}\exp(s_i-m_j)v_i$：修正后的加权输出。

现在来一个新块，新分数是 $\{s'_i\}$。分四步更新：

**第一步：更新最大值。**

$$
m_{j+1}=\max\left(m_j,\ \max_i s'_i\right)
$$

**第二步：把旧的和、旧的输出按新旧最大值的差重标定。** 因为之前的 $\exp(s_i-m_j)$ 是拿 $m_j$ 当基准算的，现在基准换成了 $m_{j+1}$，得乘上 $\exp(m_j-m_{j+1})$ 修正：

$$
l_j^{\text{old}}=l_j\cdot\exp(m_j-m_{j+1}),\qquad O_j^{\text{old}}=O_j\cdot\exp(m_j-m_{j+1})
$$

**第三步：把新块的贡献加进去。**

$$
l_{j+1}=l_j^{\text{old}}+\sum_i\exp(s'_i-m_{j+1})
$$

$$
O_{j+1}=O_j^{\text{old}}+\sum_i\exp(s'_i-m_{j+1})v'_i
$$

**第四步（最后）：归一化。** 全部块处理完后，$\text{out}=O_{j+1}/l_{j+1}$。

整个过程只在 SRAM 里维护 $m,l,O$ 三个向量，$n\times n$ 的中间矩阵**从来不需要写进 HBM**。数学上，这和先算完整 $A=\mathrm{softmax}(S)$ 再乘 $V$ 完全等价——只是计算顺序换了。

### 2.4 分块后的 IO 复杂度

FlashAttention 把 Q 切成 $B_r$ 行的块、K/V 切成 $B_c$ 列的块。对每个 Q 块：加载进 SRAM，遍历所有 K/V 块做上面的 online softmax 累积，最后把结果写回 HBM。中间产物全程留在 SRAM。

于是 IO 复杂度从 $O(n^2+n^2d/M)$ 降到：

$$
\text{IO}_{\text{flash}}=O\left(\frac{n^2d}{M}\right)
$$

那个 $n^2$ 的一次性写回项消失了。FLOPs 一个没少，但 HBM 访问量大幅下降——在 Memory-Bound 的场景下，这就是数倍提速。

### 2.5 为什么说它"精确"而不是"近似"

Online softmax 的重标定公式在数学上和一次性 softmax 完全等价（只是用了不同的基准最大值，最后都归一化掉）。它不丢精度，不分块近似，不丢任何 token。FlashAttention 提速纯粹来自"改 IO 顺序"，不是"降低计算精度"。这是它能直接替换标准注意力、且数值对得上的原因。

## 3. 💡 直觉理解

> 💡 **直觉**：标准注意力像在小厨房里做菜——每做一步就要把食材从冰箱（HBM）拿出来用一下，用完又塞回冰箱，下一步再拿出来，中间的碗碟堆在冰箱里。FlashAttention 像把这一顿需要的食材一次性摊开在灶台台面上（SRAM），在台面上一路做完，最后只把成品端出去。中间过程的临时碗碟从不离开台面——省掉了无数次开关冰箱门的时间。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 FlashAttention 是近似算法。它是**精确**注意力，数学结果和标准注意力完全一致，只是 IO 顺序不同。
>
> **误区二**：以为 FlashAttention 减少了 FLOPs。它不减少 FLOPs，减少的是 HBM 访问次数。在 Memory-Bound 场景下，少搬数据比少算更值钱。
>
> **误区三**：以为有了 FlashAttention 就不用关心 $n^2$ 了。它只是把常数和 IO 大幅优化，$n^2$ 的 FLOPs 本质还在，超长序列仍需稀疏/线性注意力配合。

## 5. 💻 代码实现

下面实现一个简化版 FlashAttention：按块切分 Q 和 K/V，用 online softmax 逐块累积，最后和 `scaled_dot_product_attention` 对拍，验证误差极小。

```python
import torch
import torch.nn.functional as F

def flash_attention_sim(Q, K, V, block=32):
    """简化版 FlashAttention: 分块 + online softmax。数学上等于标准注意力。"""
    b, h, n, d = Q.shape
    out = torch.zeros_like(Q)
    for i in range(0, n, block):
        q = Q[:, :, i:i+block, :]              # 当前 query 块 (b,h,Br,d)
        m = torch.full((b, h, block, 1), -1e9) # running max
        l = torch.zeros((b, h, block, 1))      # running 指数和
        O = torch.zeros((b, h, block, d))      # running 输出
        for j in range(0, n, block):
            k = K[:, :, j:j+block, :]          # key 块 (b,h,Bc,d)
            v = V[:, :, j:j+block, :]
            s = q @ k.transpose(-2, -1) / (d ** 0.5)   # 这块的 scores
            m_new = torch.maximum(m, s.max(dim=-1, keepdim=True).values)
            # 旧量按新旧最大值差重标定
            rescale = torch.exp(m - m_new)
            l = l * rescale
            O = O * rescale
            # 累加新块
            exp_s = torch.exp(s - m_new)
            l = l + exp_s.sum(dim=-1, keepdim=True)
            O = O + exp_s @ v
            m = m_new
        out[:, :, i:i+block, :] = O / l         # 最后归一化
    return out

torch.manual_seed(0)
Q = torch.randn(1, 2, 128, 64)
K = torch.randn(1, 2, 128, 64)
V = torch.randn(1, 2, 128, 64)

ref = F.scaled_dot_product_attention(Q, K, V)
out = flash_attention_sim(Q, K, V, block=32)
print("标准注意力 shape:", ref.shape)
print("Flash 模拟 shape:", out.shape)
print("两者最大逐元素误差:", (ref - out).abs().max().item(), "(应≈0)")
```

## 6. 小结

1. 标准注意力是 Memory-Bound：$n\times n$ 中间矩阵在 HBM 和 SRAM 间反复搬运。
2. Online Softmax 在 SRAM 里维护 running max/sum/output，逐块累积，从不物化完整 scores。
3. IO 复杂度从 $O(n^2+n^2d/M)$ 降到 $O(n^2d/M)$，$n^2$ 项消失。
4. FlashAttention 是精确算法，不降精度、不减 FLOPs，只改 IO 顺序。

## 7. 思考题

**基础题**

1. FlashAttention 减少的是 FLOPs 还是 HBM 访问量？为什么在 GPU 上减少后者反而更值钱？

**进阶题**

2. 如果 SRAM 无限大（能装下整个 $n\times n$ scores 矩阵），FlashAttention 还有意义吗？此时它退化成什么？

**开放题**

3. FlashAttention 的反向传播也得分块重算 scores（不存中间结果）。这意味着反向时要把前向的块再算一遍——你愿意为了省 HBM 多算一遍吗？这个 trade-off 怎么算才划算？

## 8. 延伸阅读 / 进一步思考

- FlashAttention 原始论文：Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022；FlashAttention-2/3。
- Online softmax：Milakov & Gimelshein, *Online normalizer calculation for softmax*, 2018。
- 进阶：FlashDecoding、与 PagedAttention 结合的推理引擎，以及稀疏注意力 + FlashAttention 的混合。
