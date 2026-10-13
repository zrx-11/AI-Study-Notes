# 第18讲 学习率调度：Warmup 与衰减的设计原理

> 📌 **本讲问题**：为什么大模型训练不能从一开始就用最大学习率？那个"先慢慢升温、再慢慢降温"的曲线，背后到底在补偿什么？原论文那个带 $d_{\text{model}}^{-0.5}$ 和 $\min$ 的奇怪公式，为什么峰值恰好落在 warmup 步数上？

## 1. 数学工具/预备知识

**（1）梯度下降的更新规则。** $\theta\leftarrow\theta-\eta\,g_t$，其中 $g_t=\nabla L(\theta_t)$ 是当前梯度，$\eta$ 是学习率。$\eta$ 是每步走多大。这是 ch03 的基础。

**（2）为什么"太大的 $\eta$ 会发散"。** 把损失在当前点附近二阶泰勒展开：$L(\theta-\eta g)\approx L(\theta)-\eta g^2+\frac{1}{2}\eta^2 g^\top H g$，$H$ 是 Hessian。要让损失下降，需要 $\eta g^2>\frac{1}{2}\eta^2 g^\top H g$，即 $\eta<2/(g^\top H g/g^2)=2/\lambda_{\max}$（$\lambda_{\max}$ 是 Hessian 最大特征值）。$\eta$ 一旦超过这个上限，更新反而把损失推高——发散。

**（3）训练初期为什么特别脆弱。** 随机初始化的权重落在一个完全陌生的区域，此时梯度方向随机性大、量级也不稳。第 1 步如果用大 $\eta$，可能一步就把权重踢出"可学习盆地"，之后再也回不来。这就是 Warmup 要解决的问题。

**（4）平方根函数与余弦函数的形状。** 后面会用到 $\mathrm{step}^{-0.5}$（随步数缓慢下降的曲线）和 $0.5(1+\cos(\pi t))$（从 1 平滑降到 0 的钟形）。记住它们"后期下降越来越慢"的特点。

## 2. 完整推导

### 2.1 Warmup 在补偿什么

训练第 1 步，权重是随机的，梯度 $g_1$ 的方向和大小都没校准过。如果直接用最大学习率 $\eta_{\max}$，更新量 $\eta_{\max}g_1$ 可能大到把权重直接推飞。

Warmup 的做法：前 $W$ 步让学习率从 0 **线性**升到 $\eta_{\max}$：

$$
\eta_{\text{step}}=\eta_{\max}\cdot\frac{\text{step}}{W},\qquad \text{step}=1,\dots,W
$$

这样开头几步更新量很小，模型有时间"先在小步幅下摸清梯度的大致方向"，等动量（ch19 的 Adam 状态）积累稳定了，再加大步幅。

### 2.2 原论文公式逐块拆解

原论文没有直接写 $\eta_{\max}$，而是给出一个自归一化的公式：

$$
\eta_{\text{step}}=d_{\text{model}}^{-0.5}\cdot\min\left(\text{step}^{-0.5},\;\text{step}\cdot W^{-1.5}\right)
$$

其中 $W=4000$ 是 warmup 步数。这个 $\min$ 套了两条曲线，我们拆开看。

**第一条曲线（warmup 段）**：当 step $\le W$ 时，$\text{step}\cdot W^{-1.5}$ 更小，取它。它关于 step 是**线性增长**的（step 的一次方）。

**第二条曲线（衰减段）**：当 step $>W$ 时，$\text{step}^{-0.5}$ 更小，取它。它关于 step 是**平方根衰减**的。

**为什么峰值恰好落在 step $=W$？** 让两条曲线相等，解出交点：

$$
\text{step}^{-0.5}=\text{step}\cdot W^{-1.5}
\;\Longrightarrow\;
\text{step}^{-1.5}=W^{-1.5}
\;\Longrightarrow\;
\text{step}=W
$$

所以 $\min$ 操作自动在 step $=W$ 处"换档"：之前线性升，之后平方根降，峰值就卡在 warmup 终点。峰值处的学习率为：

$$
\eta_{\text{peak}}=d_{\text{model}}^{-0.5}\cdot W^{-0.5}=\frac{1}{\sqrt{d_{\text{model}}\,W}}
$$

代入原论文 $d_{\text{model}}=512, W=4000$：$\eta_{\text{peak}}=1/\sqrt{512\times4000}\approx 7.0\times10^{-4}$，和实际用的 ~$7.6\times10^{-4}$ 吻合。

**为什么前面要乘 $d_{\text{model}}^{-0.5}$？** 因为注意力打分的方差和 $d_k$ 成正比（ch08：点积方差为 $d_k$）。隐藏维度 $d_{\text{model}}$ 越大，梯度的整体量级会跟着涨。用 $d_{\text{model}}^{-0.5}$ 做归一化，让"峰值学习率"和维度自动匹配——换一个 $d_{\text{model}}$ 不用重新手调学习率。

### 2.3 为什么后期要衰减

训练后期，权重已经落到最优点附近。此时如果还用大学习率，每一步都在最优点两边来回跳（震荡），永远停不下来。逐步降低学习率，等于"越接近目标越小心迈步"，让模型在最优点附近精细打磨。

从 2.1 节那个二阶展开看：后期 Hessian 的条件数可能变差，需要更小的 $\eta$ 才能保证单调下降。

### 2.4 主流调度方案对比

**Warmup + 线性衰减**：升到 $\eta_{\max}$ 后线性降到 0。BERT 用这个。

**Warmup + 余弦衰减**：升到 $\eta_{\max}$ 后，按余弦曲线降到 0：

$$
\eta_{\text{step}}=\eta_{\max}\cdot\frac{1}{2}\left(1+\cos\left(\pi\cdot\frac{\text{step}-W}{T-W}\right)\right)
$$

其中 $T$ 是总步数。验证两端：$\text{step}=W$ 时括号里是 $0$，$\cos 0=1$，$\eta=\eta_{\max}$；$\text{step}=T$ 时括号里是 $1$，$\cos\pi=-1$，$\eta=0$。余弦曲线在中段下降快、后期下降慢——后期长时间维持一个很小但非零的学习率，对收敛有好处。GPT 系列用这个。

**Warmup + 常数**：升到 $\eta_{\max}$ 后保持不变。早期小模型用，后期不衰减容易震荡。

### 2.5 学习率和 batch size 的关系

实践中有一条经验法则叫**线性缩放规则**：batch size 翻 $k$ 倍，学习率也大约翻 $k$ 倍。道理是：batch 越大，梯度估计越准（平均了更多样本），梯度方向的噪声越小，于是可以放心迈更大的步。batch 小的时候梯度噪声大，步子迈大了会被噪声带偏。

这也是为什么"调 batch size"和"调学习率"必须联动看，不能孤立地改一个。

### 2.6 为什么是平方根衰减，而不是线性

原论文衰减段用的是 $\text{step}^{-0.5}$，不是线性下降。选平方根有个隐含道理：随训练推进，我们对"最优点在哪"的估计越来越准，但准得越来越慢——信息量按步数的平方根增长，相应地学习率就该按平方根下降。平方根曲线前期降得快、后期降得极慢，正好匹配"先粗调后细调"的节奏。线性衰减是匀速刹车，后期反而降得太干脆，容易在还没收敛完时就停了。

现代大模型还出现了 **WSD（Warmup-Stable-Decay）** 曲线：先 warmup，再在最大学习率上稳定很长一段，最后才短距离衰减。它服务于"先把模型能力练出来，最后再精修"的新训练范式，和传统余弦曲线的"一直降"思路不同。

## 3. 💡 直觉理解

> 💡 **直觉**：学习率像开车。起步时不能一脚油门踩到底（Warmup），否则会直接冲出路面——你还没摸清这台车的脾气呢。开上高速后可以全速前进（常数段/峰值段），可以尽情跑。快到目的地了必须踩刹车（衰减），否则会冲过终点线停不下来。不同的衰减曲线就是不同的刹车脚法：线性刹车是一脚踩死，余弦刹车是先重后轻、稳稳停下。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 Warmup 是可选的小技巧。对大模型（>1B 参数），没有 Warmup 几乎必发散——这不是调优，是保命。
>
> **误区二**：以为学习率越大收敛越快。太大直接发散，太小收敛极慢，存在一个最优区间。Warmup + 衰减就是为了让你在不同阶段待在该待的区间里。
>
> **误区三**：以为衰减到 0 才好。余弦衰减到接近 0 即可，实践中常常留一个最小学习率（比如峰值的 10%）不降到 0，完全降到 0 反而会让后期训练停滞。

## 5. 💻 代码实现

下面实现两种调度并画出它们的数值（沙箱里不依赖 GUI，直接打印关键点）。注意原论文公式里 step 必须从 1 开始，否则 step $^{-0.5}$ 在 step $=0$ 处除零。

```python
import numpy as np

def original_transformer_lr(step, d_model=512, warmup=4000):
    """原论文调度: d_model^-0.5 * min(step^-0.5, step*warmup^-1.5)"""
    step = max(step, 1)   # 防止 step=0 时除零
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))

def cosine_lr(step, max_lr=1e-3, warmup=4000, total_steps=100000):
    """Warmup + 余弦衰减"""
    if step < warmup:
        return max_lr * step / warmup          # 线性升温
    progress = (step - warmup) / (total_steps - warmup)
    return max_lr * 0.5 * (1 + np.cos(np.pi * progress))

print("=== 原论文调度 (d=512, warmup=4000) ===")
for s in [1, 1000, 4000, 10000, 50000]:
    print(f"  step={s:>6}: lr={original_transformer_lr(s):.6f}")
print("  (峰值应出现在 step=4000)")

print("\n=== 余弦调度 (max_lr=1e-3, warmup=4000, total=100000) ===")
for s in [1, 4000, 25000, 50000, 100000]:
    print(f"  step={s:>6}: lr={cosine_lr(s):.6f}")
print("  (step=4000 应等于 max_lr=0.001, step=100000 应≈0)")
```

## 6. 小结

1. Warmup 防止训练初期梯度不稳导致发散，前 $W$ 步线性升学习率。
2. 原论文调度 $\eta=d_{\text{model}}^{-0.5}\min(\text{step}^{-0.5},\text{step}\cdot W^{-1.5})$，$\min$ 自动在 step $=W$ 处换档，峰值 $\eta_{\text{peak}}=1/\sqrt{d_{\text{model}}W}$。
3. $d_{\text{model}}^{-0.5}$ 把学习率和隐藏维度自动匹配，换维度不用重调。
4. 主流调度：Warmup + 线性/余弦/平方根衰减；余弦因后期降得慢更常用。
5. 学习率与 batch size 联动：batch 大可以用更大学习率。

## 7. 思考题

**基础题**

1. 如果把 warmup 步数 $W$ 设成 0（第 1 步就用峰值学习率），训练初期最可能出现什么现象？

**进阶题**

2. 验证原论文公式的换档点：手算证明当 step $=W$ 时，$\text{step}^{-0.5}$ 和 $\text{step}\cdot W^{-1.5}$ 恰好相等。再算一遍 $d=512, W=4000$ 时的峰值学习率。

**开放题**

3. 余弦衰减后期"下降慢"，线性衰减后期"匀速下降"。请从"训练后期需要长时间精细打磨"的角度，论证为什么现代大模型更偏爱余弦衰减。如果总训练步数比预期短很多，余弦曲线会停在一个还很高的学习率上，这会带来什么问题？

## 8. 延伸阅读 / 进一步思考

- 原论文第 5.3 节关于优化器与学习率调度的讨论。
- 线性缩放规则：Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, 2017。
- 余弦衰减：Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, 2016（余弦退火与热重启的源头）。
- 进阶：WSD（Warmup-Stable-Decay）调度，是 LLaMA-3 之后训练长上下文时兴起的新曲线，可作为进一步阅读方向。
