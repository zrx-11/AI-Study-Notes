# 第22讲 RoPE 旋转位置编码：复数域的优雅推导

> 📌 **本讲问题**：为什么现代大模型（LLaMA、Qwen 等）都抛弃了正余弦位置编码，改用 RoPE？"旋转"到底比"加法"好在哪？为什么它能让注意力分数只取决于相对位置？

## 1. 数学工具/预备知识

**（1）复数乘法 = 旋转。** 一个复数 $z=re^{i\theta}$。把它乘上 $e^{i\phi}$，得到 $re^{i(\theta+\phi)}$——模长 $r$ 不变，角度加了 $\phi$。也就是说，**乘 $e^{i\phi}$ 等价于把这个复数在复平面上逆时针旋转 $\phi$**。这是 RoPE 的全部几何基础。

**（2）二维旋转矩阵。** 旋转角度 $\theta$ 的实矩阵形式是：

$$
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
$$

它作用在二维向量 $\begin{bmatrix}x\\y\end{bmatrix}$ 上，正好和复数乘 $e^{i\theta}$ 一一对应。

**（3）内积与共轭。** 两个复数的内积取实部：$\langle z, w\rangle=\mathrm{Re}(z\overline{w})$，其中 $\overline{w}$ 是共轭。共轭的作用是把指数上的符号翻号。

**（4）ch11 的正余弦编码。** 它是把一个位置向量**加**到输入嵌入上，属于绝对位置编码。本讲要讲的 RoPE 是另一条路。

## 2. 完整推导

### 2.1 绝对位置编码的痛点

ch11 的正余弦编码是把位置信息**加在输入上**：$\boldsymbol{x}_t+\boldsymbol{p}_t$。问题在于：这是绝对位置——每个位置有一个固定的 $\boldsymbol{p}_t$。当序列长度超过训练时见过的最大长度，模型就要用到从没见过的 $\boldsymbol{p}_t$，外推出去完全失真，性能断崖式下跌。

### 2.2 RoPE 的核心思想

RoPE（Rotary Position Embedding）换了个思路：**不把位置加在输入上，而是在算注意力分数之前，把 Query 和 Key 各旋转一个和位置有关的角度**。这样点积的结果会自动只依赖相对位置。

把 $d_k$ 维的 Q/K 向量两两分组，每组 2 维看成一个复数。比如 $\boldsymbol{q}=(q_1,q_2,q_3,q_4,\dots)$ 就拆成 $(q_1+iq_2,\;q_3+iq_4,\;\dots)$。对位置 $t$ 的 query，给每个复数乘上 $e^{it\theta_i}$（$\theta_i$ 是第 $i$ 组的旋转频率）：

$$
\boldsymbol{q}_t' = \boldsymbol{q}_t \cdot e^{it\theta_i}
$$

同理，位置 $s$ 的 key 变成 $\boldsymbol{k}_s'=\boldsymbol{k}_s\cdot e^{is\theta_i}$。

### 2.3 为什么点积只依赖相对位置

现在算旋转后两者的内积。用复数内积公式 $\langle z,w\rangle=\mathrm{Re}(z\overline{w})$：

$$
\langle\boldsymbol{q}_t',\boldsymbol{k}_s'\rangle
=\mathrm{Re}\left(\boldsymbol{q}_t e^{it\theta_i}\cdot \overline{\boldsymbol{k}_s e^{is\theta_i}}\right)
$$

关键一步：共轭把指数上的符号翻号，$\overline{e^{is\theta_i}}=e^{-is\theta_i}$。所以：

$$
\langle\boldsymbol{q}_t',\boldsymbol{k}_s'\rangle
=\mathrm{Re}\left(\boldsymbol{q}_t\overline{\boldsymbol{k}_s}\cdot e^{i(t-s)\theta_i}\right)
$$

看指数：$it\theta_i-is\theta_i=i(t-s)\theta_i$——**绝对位置 $t$ 和 $s$ 单独出现的地方全消掉了，只剩它们的差 $(t-s)$**。这就是 RoPE 最漂亮的性质：注意力分数只取决于相对距离 $t-s$，和它们站在哪里无关。

**几何意义**：把 Q 和 K 都看成复平面上的箭头。位置 $t$ 的 query 箭头转了 $t\theta$，位置 $s$ 的 key 箭头转了 $s\theta$。两个箭头之间的夹角差就是 $(t-s)\theta$——只跟"隔了多远"有关，跟"绝对站在哪"无关。所以模型天生就知道"这两个词隔了几个位置"。

### 2.4 用实矩阵实现：二维分块旋转

复数乘 $e^{it\theta_i}$ 对应一个二维旋转矩阵。对第 $i$ 组维度 $(2i,2i+1)$ 施加：

$$
\begin{bmatrix}\cos(t\theta_i)&-\sin(t\theta_i)\\\sin(t\theta_i)&\cos(t\theta_i)\end{bmatrix}
\begin{bmatrix}q_{2i}\\q_{2i+1}\end{bmatrix}
$$

不同组用不同频率 $\theta_i$：

$$
\theta_i = 10000^{-2i/d_k}
$$

低频组（$\theta_i$ 小）转得慢，负责"长距离粗粒度"的位置感知；高频组（$\theta_i$ 大）转得快，负责"短距离细粒度"。这套频率安排和 ch11 的正余弦编码同源。

### 2.5 为什么只转 Q 和 K，不转 V

注意力分数 $\boldsymbol{q}^\top\boldsymbol{k}$ 只涉及 Q 和 K，所以位置信息只需要进 Q、K。V 不参与点积——它只是被 softmax 出来的权重加权求和。如果也转 V，反而会给输出强行加一个和位置有关的旋转，没有任何好处，还破坏了"V 是内容本身"的纯粹性。

### 2.6 外推：好是好，但不是无限好

RoPE 的相对位置性质让它在一定长度范围内的外推能力远强于绝对编码。但当序列远超训练长度时仍会退化：高频组的旋转角度积累过大，不同相对位置的角度差变得难以区分。

现代做法：调整频率基来延长外推，如 NTK-aware scaling、YaRN。它们的思路都是"重新分配各频率组的角度预算"。

## 3. 💡 直觉理解

> 💡 **直觉**：正余弦编码像给每个 token 发一张写着绝对门牌号的标签——标签是死的，一旦出了熟悉的街区，门牌号就对不上了。RoPE 像让每个 token 站在转盘上，按自己的位置转一个角度——两个 token 之间的夹角差只取决于它们隔了多远，不取决于它们站在操场的哪个角落。不管队伍拉多长，相对关系始终成立。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 RoPE 是加在输入上的。RoPE 是**乘在 Q/K 上**的——在注意力打分之前旋转它们，而不是在嵌入层加位置向量。
>
> **误区二**：以为 RoPE 完美解决了外推。它只是比绝对编码外推好，超长序列（远超训练长度）仍然需要 NTK、YaRN 等额外缩放。
>
> **误区三**：以为相对位置性质意味着模型不在乎绝对位置。其实通过"相对距离的累积"，模型仍能间接感知大致位置，只是不显式编码。

## 5. 💻 代码实现

下面预计算 RoPE 频率表并实现旋转，然后**亲手验证**那个最关键的性质：同一对 Q/K 向量，放在 (10,12) 和 (50,52) 两处（相对距离都是 2），旋转后的点积应该相等。

```python
import torch

def precompute_rope_freqs(d_k, max_len, theta_base=10000.0):
    """预计算每组维度的 cos/sin 频率表"""
    # 第 i 组的频率 theta_i = base^(-2i/d_k)
    freqs = 1.0 / (theta_base ** (torch.arange(0, d_k, 2).float() / d_k))
    positions = torch.arange(max_len).float()
    # 外积: positions[i] * freqs[j] -> (max_len, d_k//2)
    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos_freqs, sin_freqs):
    """x: (batch, n_heads, seq_len, d_k)。把相邻两维当成一组做旋转"""
    x1 = x[..., 0::2]   # 偶数维: q_{2i}
    x2 = x[..., 1::2]   # 奇数维: q_{2i+1}
    # cos/sin: (seq_len, d_k//2) -> 广播到 (1,1,seq_len,d_k//2)
    cos = cos_freqs.unsqueeze(0).unsqueeze(0)
    sin = sin_freqs.unsqueeze(0).unsqueeze(0)
    # 旋转矩阵作用: [c -s; s c] @ [x1; x2]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    # 交错拼回原来的 (..., d_k) 布局
    return torch.stack([rot1, rot2], dim=-1).flatten(-2)

# ===== 验证：点积只依赖相对位置 =====
torch.manual_seed(0)
d_k = 8
cos_f, sin_f = precompute_rope_freqs(d_k, max_len=100)

q = torch.randn(1, 1, 1, d_k)   # 同一个 query 向量
k = torch.randn(1, 1, 1, d_k)  # 同一个 key 向量

# 情形 A: q 在位置10, k 在位置12，相对距离 = 2
qA = apply_rope(q, cos_f[10:11], sin_f[10:11])
kA = apply_rope(k, cos_f[12:13], sin_f[12:13])
dotA = (qA * kA).sum().item()

# 情形 B: 同样的 q/k 向量，放到位置50和52，相对距离仍 = 2
qB = apply_rope(q, cos_f[50:51], sin_f[50:51])
kB = apply_rope(k, cos_f[52:53], sin_f[52:53])
dotB = (qB * kB).sum().item()

print(f"相对距离=2, 位置(10,12) 点积: {dotA:.6f}")
print(f"相对距离=2, 位置(50,52) 点积: {dotB:.6f}")
print(f"两者相等（相对位置性质）: {abs(dotA - dotB) < 1e-4}")
```

## 6. 小结

1. RoPE 在注意力打分前对 Q/K 施加与位置相关的旋转，使点积只依赖相对位置 $t-s$。
2. 旋转由二维旋转矩阵实现，不同维度组对应不同频率 $\theta_i=10000^{-2i/d_k}$。
3. 只转 Q/K 不转 V，因为位置信息只需要进入点积。
4. 相比绝对位置编码，RoPE 外推能力更强；超长序列仍需 NTK、YaRN 等额外缩放。

## 7. 思考题

**基础题**

1. 为什么 RoPE 只旋转 Q 和 K，不旋转 V？（提示：V 进不进点积。）

**进阶题**

2. 当相对位置 $t-s=0$ 时，旋转角差为 0，RoPE 点积等于什么？这个结果说明位置 $t$ 的 query 对"它自己"的打分有什么特点？

**开放题**

3. 如果把频率基从 10000 改成 1000000，高频组的频率会怎么变？这对"短距离敏感度"和"长距离外推能力"分别有什么影响？NTK、YaRN 又是怎么动态调整这个频率基的？

## 8. 延伸阅读 / 进一步思考

- RoPE 原始论文：Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021。
- 外推扩展：NTK-aware scaling、YaRN（Peng et al., 2023）、Position Interpolation。
- 其他位置编码路线对比：ALiBi（直接给注意力分数加一个和距离成反比的偏置），可作为对比阅读。
