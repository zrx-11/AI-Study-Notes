# 第4讲 KL散度与f-散度家族

> 📌 **本讲问题**：两个概率分布 $p$ 和 $q$ 之间的"距离"该怎么定义？为什么KL散度不是真正的距离（它不对称）？变分推断中那个著名的ELBO下界，为什么本质上是一个KL散度的变形？Gibbs不等式为什么说"KL散度永远非负"？

## 1. 数学工具/预备知识

回顾前3讲中反复出现的KL散度：

$$
D_{\mathrm{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

本讲严格推导它的非负性，引入f-散度家族，并把KL散度与变分推断的核心工具——ELBO——联系起来。

### 1.1 Jensen不等式

Jensen不等式是证明KL散度非负的核心工具。回顾它的内容：

- 如果 $f$ 是凸函数（二阶导数 $\geq 0$），则 $\mathbb{E}[f(X)] \geq f(\mathbb{E}[X])$。
- 如果 $f$ 是凹函数（二阶导数 $\leq 0$），则 $\mathbb{E}[f(X)] \leq f(\mathbb{E}[X])$。

**对数函数 $\log t$ 是凹函数**：二阶导数 $d^2/dt^2 \log t = -1/t^2 < 0$。因此对 $\log$ 应用Jensen不等式，得到：

$$
\mathbb{E}[\log T] \leq \log \mathbb{E}[T]
$$

这个不等式在下面的Gibbs不等式证明中会直接用到。

### 1.2 什么是"散度"？

散度（divergence）是两个分布之间的"差异度量"。它比"距离"（metric）的要求低：

- 距离要求：非负、对称、三角不等式。
- 散度只要求：非负（$D(p\|q) \geq 0$，等号当且仅当 $p = q$）。

KL散度不对称（$D(p\|q) \neq D(q\|p)$），也不满足三角不等式，所以它不是严格意义上的"距离"。但它在信息论中极其有用。

## 2. 完整推导

### 2.1 KL散度的定义与非负性

**定义**：对两个离散分布 $p(x)$ 和 $q(x)$，KL散度为

$$
D_{\mathrm{KL}}(p \| q) = \mathbb{E}_{p(x)}\left[\log \frac{p(x)}{q(x)}\right] = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

连续情形将求和换成积分：$D_{\mathrm{KL}}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} dx$。

**几何/物理意义**：KL散度度量的是"用 $q$ 来编码来自 $p$ 的数据时，比最优编码（用 $p$ 自己编码）多出来的平均比特数"。如果 $p = q$，多余的比特数为0。如果 $p$ 和 $q$ 差距大，多余的比特数就大。

**定理**（Gibbs不等式）：$D_{\mathrm{KL}}(p \| q) \geq 0$，等号当且仅当 $p(x) = q(x)$ 对所有 $x$ 成立。

**证明**：利用 $\log$ 的凹性（Jensen不等式）。

第一步：把KL散度改写成适合Jensen不等式的形式。

$$
D_{\mathrm{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = -\sum_x p(x) \log \frac{q(x)}{p(x)}
$$

第二步：令 $T = q(x)/p(x)$，在分布 $p$ 下取期望。

$$
D_{\mathrm{KL}}(p \| q) = -\mathbb{E}_{p}\!\left[\log \frac{q(x)}{p(x)}\right]
$$

第三步：对 $\mathbb{E}_p[\log T]$ 应用Jensen不等式（因为 $\log$ 是凹函数）。

$$
\mathbb{E}_{p}\!\left[\log T\right] \leq \log \mathbb{E}_{p}\!\left[T\right]
$$

计算右边：

$$
\mathbb{E}_{p}\!\left[T\right] = \sum_x p(x) \cdot \frac{q(x)}{p(x)} = \sum_x q(x) = 1
$$

（因为 $q$ 是归一化分布，所有概率加起来等于1。）

所以：

$$
\mathbb{E}_{p}\!\left[\log \frac{q(x)}{p(x)}\right] \leq \log 1 = 0
$$

第四步：代入KL散度的表达式。

$$
D_{\mathrm{KL}}(p \| q) = -\mathbb{E}_{p}\!\left[\log \frac{q}{p}\right] \geq -0 = 0
$$

∎

**等号成立条件**：Jensen不等式等号当 $T$ 为常数（不随 $x$ 变化），即 $q(x)/p(x) = c$ 对所有 $x$。由于 $p$ 和 $q$ 都是归一化分布，$c = 1$（因为 $\sum q(x) = \sum c\,p(x) = c = 1$），即 $p = q$。∎

### 2.2 KL散度的非对称性

KL散度不满足对称性：$D_{\mathrm{KL}}(p\|q) \neq D_{\mathrm{KL}}(q\|p)$。

**数值例子**：设 $p$ 是二项分布 $(0.9, 0.1)$，$q$ 是 $(0.5, 0.5)$。

$$
D_{\mathrm{KL}}(p\|q) = 0.9\log\frac{0.9}{0.5} + 0.1\log\frac{0.1}{0.5}
$$

$$
= 0.9 \times 0.588 + 0.1 \times (-1.609) = 0.529 - 0.161 = 0.368
$$

$$
D_{\mathrm{KL}}(q\|p) = 0.5\log\frac{0.5}{0.9} + 0.5\log\frac{0.5}{0.1}
$$

$$
= 0.5 \times (-0.588) + 0.5 \times 1.609 = -0.294 + 0.805 = 0.511
$$

两者不等。$D_{\mathrm{KL}}(p\|q) = 0.368$，$D_{\mathrm{KL}}(q\|p) = 0.511$。

**直觉差异**：

- **前向KL** $D_{\mathrm{KL}}(p\|q)$（"mean-seeking"）：当 $p(x) > 0$ 而 $q(x) \to 0$ 时，项 $p(x)\log(p(x)/q(x)) \to \infty$。这迫使 $q$ 必须在 $p$ 有概率的地方都有概率覆盖——不能漏掉任何一个 $p$ 的模式。
- **反向KL** $D_{\mathrm{KL}}(q\|p)$（"mode-seeking"）：当 $q(x) > 0$ 而 $p(x) \to 0$ 时，项 $q(x)\log(q(x)/p(x)) \to \infty$。这迫使 $q$ 只在 $p$ 的高概率区域分布——不去覆盖 $p$ 的低概率区域。

**为什么这个区别重要**：VAE用的是反向KL（$D(q\|p)$），所以VAE生成的图像往往模糊——它倾向于覆盖所有模式的平均，而不是锐化某个模式。GAN用的是某种f-散度的变体，更接近前向KL，所以GAN生成的图像更尖锐。

### 2.3 f-散度家族

KL散度不是唯一的散度。更一般的**f-散度**定义为：

$$
D_f(p \| q) = \sum_x q(x)\, f\!\left(\frac{p(x)}{q(x)}\right)
$$

其中 $f$ 是一个凸函数，满足 $f(1) = 0$。

**为什么 $f(1) = 0$**：当 $p = q$ 时，$p(x)/q(x) = 1$，所以 $D_f(p\|q) = f(1) = 0$。这是散度"等号当且仅当两分布相同"的保证。

不同的 $f$ 给出不同的散度：

| 散度名称 | $f(t)$ | 表达式 |
|---|---|---|
| KL散度 | $t\log t$ | $\sum p \log(p/q)$ |
| 反向KL | $-\log t$ | $\sum q \log(q/p)$ |
| 皮尔逊$\chi^2$ | $(t-1)^2$ | $\sum (p-q)^2 / q$ |
| 总变差距离 | $\|t-1\|/2$ | $\frac{1}{2}\sum\|p-q\|$ |
| Hellinger | $(\sqrt{t}-1)^2$ | $\sum(\sqrt{p}-\sqrt{q})^2$ |

f-散度的共同性质：
1. 非负（由 $f$ 凸 + Jensen不等式）。
2. $D_f(p\|q) = 0$ 当且仅当 $p = q$。
3. 大多数不对称（除了总变差和Hellinger）。

**为什么需要f-散度家族**：不同的散度在不同场景下有不同的计算性质。比如GAN使用Jensen-Shannon散度（KL的对称化版本），而VAE使用KL。理解f-散度家族能帮助你判断一个模型在用什么方式"度量分布差异"。

### 2.4 变分推断与ELBO：从KL散度出发

考虑一个隐变量模型：观测数据 $x$，隐变量 $z$，联合分布 $p(x, z) = p(z)p(x|z)$。我们想推断后验 $p(z|x) = p(x,z)/p(x)$。

**问题**：$p(x) = \int p(x,z) dz$（证据）通常难以计算，因为需要在连续的 $z$ 空间积分。如果 $z$ 是高维的，这个积分是指数级困难的。

**变分推断**的思路：用一个简单的分布 $q_\phi(z|x)$（变分分布，由参数 $\phi$ 控制）去逼近真实后验 $p(z|x)$。

我们要最小化 $D_{\mathrm{KL}}(q_\phi(z|x) \| p(z|x))$。

**但这个KL里包含 $p(z|x)$，而 $p(z|x)$ 正是我们想算的东西！** 这是一个鸡生蛋蛋生鸡的问题。变分推断的聪明之处在于：把这个不可计算的KL改写成一个可计算的目标。

**推导过程**：

第一步：展开KL散度。

$$
D_{\mathrm{KL}}(q_\phi(z|x) \| p(z|x)) = \mathbb{E}_{q_\phi}\left[\log q_\phi(z|x) - \log p(z|x)\right]
$$

第二步：利用贝叶斯公式 $p(z|x) = p(x,z)/p(x)$，所以 $\log p(z|x) = \log p(x,z) - \log p(x)$。

$$
= \mathbb{E}_{q_\phi}\left[\log q_\phi(z|x) - \log p(x,z) + \log p(x)\right]
$$

第三步：注意 $\log p(x)$ 与 $z$ 无关，可以提出期望。

$$
= \mathbb{E}_{q_\phi}\left[\log q_\phi(z|x) - \log p(x,z)\right] + \log p(x)
$$

第四步：把 $\log p(x)$ 移到左边。

$$
\log p(x) - D_{\mathrm{KL}}(q_\phi(z|x) \| p(z|x)) = \mathbb{E}_{q_\phi}\left[\log p(x,z) - \log q_\phi(z|x)\right]
$$

右边就是**证据下界（ELBO）**：

$$
\mathrm{ELBO}(\phi) = \mathbb{E}_{q_\phi(z|x)}\left[\log p(x,z) - \log q_\phi(z|x)\right]
$$

**关键结论**：

$$
\log p(x) = \mathrm{ELBO}(\phi) + D_{\mathrm{KL}}(q_\phi(z|x) \| p(z|x))
$$

因为 $D_{\mathrm{KL}} \geq 0$（Gibbs不等式），所以 $\mathrm{ELBO}(\phi) \leq \log p(x)$。ELBO是对数证据的下界。最大化ELBO等价于最小化那个难以计算的KL散度——因为 $\log p(x)$ 与 $\phi$ 无关。

**为什么这是一个绝妙的转化**：

- 原来的问题：计算 $p(x) = \int p(x,z)dz$（高维积分，不可行）。
- 转化后的问题：最大化ELBO（可以采样估计，可行）。

这就是变分推断的全部核心：**把一个不可计算的后验推断问题，转化为一个可计算的优化问题**。第10讲将在VAE中完整展开这个推导。

## 3. 💡 直觉理解

> 💡 **直觉**：KL散度就像"用 $q$ 来编码 $p$ 的代价"。想象你知道真实分布 $p$，但你必须用一个简化分布 $q$ 来做编码。每看到一个来自 $p$ 的样本 $x$，你实际用了 $q(x)$ 的概率去编码它，而不是最优的 $p(x)$。多余的代价就是 $\log(p(x)/q(x))$。
>
> 前向KL $D(p\|q)$ 说的是"我从 $p$ 中抽样，用 $q$ 编码的多余代价"——这迫使 $q$ 不能漏掉 $p$ 有概率的任何地方（mean-seeking）。反向KL $D(q\|p)$ 说的是"我从 $q$ 中抽样，用 $p$ 编码的多余代价"——这允许 $q$ 只挑 $p$ 的高峰处待着（mode-seeking）。
>
> 变分推断的ELBO之所以是"下界"，是因为你永远不知道真实的 $p(x)$ 是多少，但你知道 ELBO + KL = 真实值，而 KL ≥ 0，所以 ELBO 是你能摸到的最接近真实值的下界。这就像你不知道一座山的精确高度，但你知道"山高 = 我测量的高度 + 测量误差"，而误差非负，所以你的测量值是山高的下界。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：把KL散度当距离。距离要求对称性 $d(p,q) = d(q,p)$ 和三角不等式。KL两者都不满足。它是"散度"，不是"度量"。说"KL距离"是口语化的说法，严格来说应该叫"KL散度"。
>
> **误区二**：认为 $D_{\mathrm{KL}}(p\|q)$ 可以取负值。由Gibbs不等式，KL恒非负。如果计算出负值，一定是归一化或数值精度出了问题（比如某些概率为0导致的数值不稳定）。
>
> **误区三**：混淆前向KL和反向KL在GAN/VAE中的角色。VAE用的是 $D(q\|p)$（反向KL，mode-seeking），所以VAE生成的图像往往模糊——它倾向于覆盖所有模式的平均。GAN用的是某种f-散度（更接近前向KL），所以GAN生成的图像更尖锐但可能模式崩溃。
>
> **误区四**：认为KL散度的单位不重要。用自然对数单位是nat，以2为底是bit。1 nat ≈ 1.443 bit。在比较不同文献的KL数值时，先确认单位。

## 5. 💻 代码实现

数值对比前向KL与反向KL的行为差异，并可视化ELBO与真实证据的关系。

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ========== 实验1：前向KL vs 反向KL ==========
print("=" * 60)
print("实验1：前向KL vs 反向KL的行为差异")
print("=" * 60)

# 真实分布 p 是一个双峰高斯混合
# 为什么用双峰分布？因为双峰分布最能体现前向/反向KL的差异：
#   前向KL（mean-seeking）会把q放在两个峰中间
#   反向KL（mode-seeking）会把q放在其中一个峰上
x = np.linspace(-5, 5, 1000)
p = 0.5 * norm.pdf(x, -2, 0.5) + 0.5 * norm.pdf(x, 2, 0.5)
p = p / p.sum()  # 归一化到离散近似（因为我们用求和近似积分）

# 用单峰高斯 q 去逼近它，扫描不同均值
# 为什么扫描均值？因为我们要看q的最优位置在哪里。
means = np.linspace(-3, 3, 50)
kl_forward = []  # D(p || q)
kl_backward = [] # D(q || p)

for mu in means:
    q = norm.pdf(x, mu, 0.8)
    q = q / q.sum()
    # 前向 KL(p || q) = sum p * log(p/q)
    # 只在 p > 1e-10 的位置求和（避免 log(0)）
    mask = p > 1e-10
    kl_f = np.sum(p[mask] * np.log(p[mask] / q[mask]))
    kl_forward.append(kl_f)
    # 反向 KL(q || p) = sum q * log(q/p)
    mask2 = q > 1e-10
    kl_b = np.sum(q[mask2] * np.log(q[mask2] / p[mask2]))
    kl_backward.append(kl_b)

plt.figure(figsize=(10, 5))
plt.plot(means, kl_forward, 'r-', linewidth=2, label='前向 KL(p || q)  (mean-seeking)')
plt.plot(means, kl_backward, 'b-', linewidth=2, label='反向 KL(q || p)  (mode-seeking)')
plt.xlabel('q 的均值 μ')
plt.ylabel('KL 散度')
plt.title('前向 vs 反向KL：逼近双峰分布时的不同偏好')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=-2, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=2, color='gray', linestyle='--', alpha=0.5)
plt.text(-2, plt.ylim()[1]*0.9, 'p左峰', ha='center', fontsize=9)
plt.text(2, plt.ylim()[1]*0.9, 'p右峰', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('ch04_kl_forward_backward.png', dpi=100)
print("图已保存为 ch04_kl_forward_backward.png")
print("观察：反向KL在靠近左峰或右峰处都有局部极小值(mode-seeking)")
print("      前向KL在中间处才最小(mean-seeking，试图覆盖两个峰)")

# ========== 实验2：ELBO 演示 ==========
print("\n" + "=" * 60)
print("实验2：ELBO = log p(x) - KL(q||p)")
print("=" * 60)

# 简单的隐变量模型：z ~ N(0,1), x|z ~ N(z, 1)
# 真实后验 p(z|x) 是高斯，可以解析计算
# 为什么用这个模型？因为后验可以解析求解，我们能直接验证ELBO。
x_obs = 1.5  # 观测值
# 真实后验：p(z|x) = N(μ_post, σ_post²)
# 由贝叶斯公式推导：μ_post = x/(1+1) = x/2, σ_post² = 1/(1+1) = 0.5
mu_post = x_obs / 2
var_post = 0.5

# 变分分布 q(z) = N(mu_q, sigma_q²)，扫描不同 mu_q
mu_qs = np.linspace(-1, 2.5, 50)
elbo_list = []
kl_list = []

for mu_q in mu_qs:
    var_q = 0.5  # 固定方差，只扫均值
    # 两个高斯之间KL的解析公式：
    # KL(N(μ0,σ0²) || N(μ1,σ1²)) = log(σ1/σ0) + (σ0² + (μ0-μ1)²)/(2σ1²) - 1/2
    kl = 0.5 * (np.log(var_post/var_q) + (var_q + (mu_q - mu_post)**2)/var_post - 1)
    kl_list.append(kl)
    # log p(x) 的解析值（边际似然）
    # p(x) = N(0, 1+1) = N(0, 2)
    log_px = norm.logpdf(x_obs, 0, np.sqrt(2))
    elbo = log_px - kl
    elbo_list.append(elbo)

print(f"真实 log p(x) = {norm.logpdf(x_obs, 0, np.sqrt(2)):.4f}")
print(f"最优 ELBO（q=p时） = {max(elbo_list):.4f}")
print(f"最小 KL = {min(kl_list):.6f}")

plt.figure(figsize=(10, 4))
plt.plot(mu_qs, elbo_list, 'g-', linewidth=2, label='ELBO(μ_q)')
plt.axhline(y=norm.logpdf(x_obs, 0, np.sqrt(2)), color='r', linestyle='--',
            label=f'真实 log p(x) = {norm.logpdf(x_obs, 0, np.sqrt(2)):.3f}')
plt.plot(mu_qs, kl_list, 'b-', linewidth=2, label='KL(q||p)')
plt.xlabel('变分均值 μ_q')
plt.ylabel('nats')
plt.title('ELBO 是 log p(x) 的下界，max ELBO ⟺ min KL')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ch04_elbo_demo.png', dpi=100)
print("图已保存为 ch04_elbo_demo.png")
```

**工程注意事项**：上面的ELBO演示用的是解析可解的简单模型。实际VAE中，$p(z|x)$ 不可解析计算，ELBO需要用蒙特卡洛采样估计。采样的方差是VAE训练不稳定性的主要来源之一，重参数化技巧（第10讲）就是为了解决这个问题。

## 6. 小结

1. KL散度 $D_{\mathrm{KL}}(p\|q) = \mathbb{E}_p[\log(p/q)]$ 度量用 $q$ 逼近 $p$ 的信息代价，由Gibbs不等式保证非负（证明用Jensen不等式）。
2. KL不对称：前向KL $D(p\|q)$ 是mean-seeking（不遗漏任何模式），反向KL $D(q\|p)$ 是mode-seeking（只覆盖高概率区域）。
3. f-散度家族以凸函数 $f$ 统一了KL、反向KL、$\chi^2$、总变差、Hellinger等散度，不同散度适合不同的模型设计。
4. 变分推断将不可计算的后验 $p(z|x)$ 转化为可计算的ELBO最大化：$\log p(x) = \mathrm{ELBO} + D_{\mathrm{KL}}(q_\phi\|p)$。
5. 最大化ELBO等价于最小化变分分布与真实后验之间的KL散度，这是VAE（第10讲）的理论基石。

## 7. 思考题

### 基础档（巩固概念）

1. 写出KL散度的定义式，并证明 $D_{\mathrm{KL}}(p\|p) = 0$。
2. 解释为什么 $D_{\mathrm{KL}}(p\|q) \geq 0$（用Jensen不等式的思路）。
3. 前向KL和反向KL的行为有什么区别？哪个是mean-seeking，哪个是mode-seeking？

### 进阶档（推导与分析）

4. 证明：当 $p$ 和 $q$ 都是 $n$ 维高斯分布时，KL散度有闭式表达：

$$
D_{\mathrm{KL}}(\mathcal{N}_0 \| \mathcal{N}_1) = \frac{1}{2}\left[\mathrm{tr}(\Sigma_1^{-1}\Sigma_0) + (\mu_1-\mu_0)^\top\Sigma_1^{-1}(\mu_1-\mu_0) - d + \log\frac{|\Sigma_1|}{|\Sigma_0|}\right]
$$

（提示：代入高斯分布的对数密度，求期望，整理。）

5. 为什么GAN用反向KL的某种变体训练会产生"模式崩溃"（只生成单一模式）？用mode-seeking的直觉解释。
6. 如果 $q_\phi$ 的容量无限大（可以拟合任意分布），ELBO的上界等于多少？这对实际的VAE意味着什么？

### 开放档（联系研究）

7. JS散度（Jensen-Shannon Divergence）是KL散度的对称化版本：$JS(p\|q) = \frac{1}{2}KL(p\|M) + \frac{1}{2}KL(q\|M)$，其中 $M = (p+q)/2$。为什么GAN最初用JS散度训练会遇到梯度消失问题？后来的Wasserstein GAN用什么f-散度的极限来解决这个问题？
8. f-散度的变分表示（Donsker-Varadhan）把KL散度写成一个优化问题：$D_{\mathrm{KL}}(p\|q) = \sup_f \mathbb{E}_p[f] - \mathbb{E}_q[e^f]$。这个表示为什么重要？它和MINE互信息估计器有什么关系？

## 8. 延伸阅读/进一步思考

- **Kullback & Leibler (1951)**，*On Information and Sufficiency*。KL散度的原始论文。
- **Nowozin et al. (2016)**，*f-GAN: Training Generative Neural Samplers using Variational Variational Minimization of f-divergences*。f-散度在GAN中的系统应用。
- **Kingma & Welling (2013)**，*Auto-Encoding Variational Bayes*。VAE的原始论文，ELBO推导的经典参考。
- **Goodfellow et al. (2014)**，*Generative Adversarial Networks*。GAN的原始论文，讨论了JS散度在生成模型中的应用。
