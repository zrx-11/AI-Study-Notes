# 第19讲 量子生成模型 QCBM：用 Born 机采样

> 📌 **本讲问题**：量子线路在计算基下测量，天然给出一个概率分布 $p_\theta(z)=|\langle z|U(\theta)|0\rangle|^2$。这本身就是一个**生成模型**！如何训练这个"Born 机"去拟合目标分布？它与 GAN、扩散模型是什么关系？

## 1. 数学工具/预备知识

复习 Born 规则：测量概率 $p_\theta(z)=|\langle z|U(\theta)|0\rangle|^2$。QCBM（Quantum Circuit Born Machine）就是一个以 $U(\theta)$ 为参数的生成模型。

**MMD（最大均值差异）**：两个分布 $p,q$ 在 RKHS 中的距离

$$
\mathrm{MMD}^2(p,q)=\mathbb{E}_{z,z'\sim p}[k(z,z')]+\mathbb{E}_{z,z'\sim q}[k(z,z')]-2\mathbb{E}_{z\sim p,z'\sim q}[k(z,z')].
$$

**参数位移法则**（第 17 讲）：对 $e^{-i\theta P/2}$ 形式的门，$\partial f/\partial\theta=\frac12[f(\theta+\pi/2)-f(\theta-\pi/2)]$。

## 2. 完整推导

### 2.1 Born 规则即概率分布

$n$ 比特 QCBM 输出一个 $2^n$ 维概率分布

$$
p_\theta(z)=|\langle z|U(\theta)|0\rangle|^2,\qquad z\in\{0,1\}^n.
$$

它天然归一、非负。训练目标是让 $p_\theta$ 逼近目标分布 $p_{\mathrm{data}}$。

**为什么这是生成模型？** 因为给定参数 $\theta$，你可以采样得到离散比特串 $z$。这与 GAN 从高斯噪声采样图像、扩散模型从去噪过程采样图像是同一类操作。

### 2.2 MMD 损失推导

最大均值差异（MMD）度量两个分布在再生核 Hilbert 空间（RKHS）中的距离：

$$
\mathrm{MMD}^2(p,q)=\mathbb{E}_{z,z'\sim p}[k(z,z')]+\mathbb{E}_{z,z'\sim q}[k(z,z')]-2\mathbb{E}_{z\sim p,z'\sim q}[k(z,z')].
$$

对 QCBM，$p=p_\theta$，$q=p_{\mathrm{data}}$。选一个简单核 $k(z,z')=1$（当 $z=z'$）或 RBF 核。

**为什么用 MMD 而不是 KL 散度？** 因为 $p_\theta(z)$ 可以在量子硬件上直接采样得到样本，不需要枚举所有 $z$；MMD 只需样本均值估计。KL 散度需要知道 $p_\theta(z)$ 的精确值，在大 $n$ 时不可行。

**MMD 的样本估计**：

$$
\widehat{\mathrm{MMD}}^2=\frac{1}{m^2}\sum_{i,j}k(z_i,z_j)+\frac{1}{n^2}\sum_{i,j}k(z'_i,z'_j)-\frac{2}{mn}\sum_{i,j}k(z_i,z'_j).
$$

### 2.3 参数位移穿过 Born 规则

梯度 $\partial \mathrm{MMD}/\partial\theta_k$ 需要穿过 $p_\theta(z)=|\langle z|U|0\rangle|^2$。用第 17 讲的参数位移法则：

$$
\frac{\partial p_\theta(z)}{\partial\theta_k}=\frac12\bigl[p_{\theta+\pi/2 e_k}(z)-p_{\theta-\pi/2 e_k}(z)\bigr].
$$

于是

$$
\frac{\partial \mathrm{MMD}}{\partial\theta_k}
=\frac12\Bigl[\mathrm{MMD}(p_{\theta+\pi/2 e_k},p_{\mathrm{data}})-\mathrm{MMD}(p_{\theta-\pi/2 e_k},p_{\mathrm{data}})\Bigr].
$$

两次前向采样即可估计梯度。

### 2.4 与 GAN、扩散的对照

| 模型 | 生成机制 | 训练信号 |
|---|---|---|
| GAN | 神经网络 $\epsilon=\mathcal{G}_\theta(z)$ | 对抗损失 |
| 扩散 | 逐步去噪 $\epsilon_\theta(x_t,t)$ | 去噪分数匹配 |
| QCBM | $p_\theta(z)=\|\langle z\|U\|0\rangle\|^2$ | MMD / KL |

**共同点**：都是从一个简单分布（高斯/均匀）变换到目标数据分布。
**差别**：QCBM 的"变换"是酉演化，输出是离散比特串分布；GAN/扩散输出连续向量。

**QCBM 的独特性**：它的输出分布天然住在 $2^n$ 维空间里，且振幅是复数。这意味着它可以表达经典生成模型难以表达的分布（如需要高阶干涉的分布）。

### 2.5 NumPy 训练 3 比特 QCBM

枚举 $2^3=8$ 个基态，手动扫参数。

## 3. 💡 直觉理解

> 💡 **直觉**：Born 机像一个"概率调音器"。你有 $2^n$ 个音孔（计算基），每个音孔出音的概率由量子线路的振幅决定。调参就是在拧每个音孔的旋钮。MMD 损失像一个耳朵：它听"机器发出的声音"和"目标声音"之间的差别，告诉旋钮往哪拧。参数位移法则说："想知道某个旋钮往哪边拧更好？把它往左拧 $\pi/2$、往右拧 $\pi/2$，各听一次，差就是梯度。"
>
> 生活类比：想象你在调一台管风琴。每根管子对应一个计算基，管子响度由振幅决定。MMD 像一个音乐老师：他听你弹的曲子和目标曲子的差别，告诉你哪根管子该调响、哪根该调轻。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**："QCBM 能生成任意分布。"
>
> 澄清：浅线路表达能力有限。要表达复杂分布需要深线路，又会遇贫瘠高原。
>
> **误区二**："MMD 比 KL 好。"
>
> 澄清：MMD 对分布尾部不敏感；KL 对尾部敏感。选哪个取决于任务。
>
> **误区三**："QCBM 训练就是反向传播。"
>
> 澄清：QCBM 用参数位移法则，不是反向传播。它需要两次采样，不是一次前向 + 一次反向。
>
> **误区四**："QCBM 输出图像。"
>
> 澄清：QCBM 输出离散比特串。要生成图像需要额外解码（如把比特串映射到像素）。

## 5. 💻 代码实现

```python
import numpy as np

rng = np.random.default_rng(0)

# ---- 1. 3 比特 QCBM ----
n = 3
dim = 2**n
I = np.eye(2, dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

def Ry(theta):
    return np.cos(theta/2)*I - 1j*np.sin(theta/2)*Y

def build_state(params):
    """3 比特：Ry 层 + CNOT 纠缠 + Ry 层，共 6 参数
    （仅用 3 个独立 Ry 无法产生 |111⟩ 的相干叠加，必须加纠缠层）"""
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    # 第一层 Ry
    for k in range(n):
        ops = [I]*n
        ops[k] = Ry(params[k])
        Uk = ops[0]
        for j in range(1, n):
            Uk = np.kron(Uk, ops[j])
        psi = Uk @ psi
    # CNOT 纠缠层（控制 0→1，控制 1→2）
    for c, t in [(0,1),(1,2)]:
        M = np.eye(dim, dtype=np.complex128)
        for s in range(dim):
            bits = [(s >> k) & 1 for k in range(n)]
            if bits[c] == 1:
                bits[t] = 1 - bits[t]
                s2 = sum(b << k for k, b in enumerate(bits))
                M[s, s] = 0; M[s, s2] = 1
        psi = M @ psi
    # 第二层 Ry
    for k in range(n):
        ops = [I]*n
        ops[k] = Ry(params[3+k])
        Uk = ops[0]
        for j in range(1, n):
            Uk = np.kron(Uk, ops[j])
        psi = Uk @ psi
    return psi

def probs(params):
    psi = build_state(params)
    return np.abs(psi)**2

# ---- 2. 目标分布：双峰，集中在 |000⟩ 和 |111⟩ ----
p_data = np.zeros(dim)
p_data[0] = 0.5
p_data[7] = 0.5

# ---- 3. MMD 损失（用 RBF 核） ----
def mmd(p, q, sigma=1.0):
    """简化：把索引当作整数，用 |z-z'| 距离"""
    z = np.arange(dim)
    def kernel(a, b):
        return np.exp(-(a-b)**2/(2*sigma**2))
    loss = 0
    for i in range(dim):
        for j in range(dim):
            loss += p[i]*p[j]*kernel(i,j) + q[i]*q[j]*kernel(i,j) - 2*p[i]*q[j]*kernel(i,j)
    return loss

# ---- 4. 参数位移训练 ----
params = rng.uniform(0, 0.1, size=6)
lr = 0.5
for step in range(100):
    p = probs(params)
    loss = mmd(p, p_data)
    grad = np.zeros(6)
    for k in range(6):
        pp = params.copy(); pp[k] += np.pi/2
        pm = params.copy(); pm[k] -= np.pi/2
        grad[k] = (mmd(probs(pp), p_data) - mmd(probs(pm), p_data)) / 2
    params -= lr * grad
    if step % 20 == 0:
        print(f"step {step}: loss={loss:.4f}, p={np.round(p,3)}")

print(f"最终 p_θ = {np.round(probs(params), 3)}")
print(f"目标 p_data = {p_data}")
# 注：这个玩具例子用 RBF 核在整数索引上，MMD landscape 容易坍缩到 |000⟩ 局部最优。
# 这恰恰说明 QCBM 训练需要仔细设计损失和 ansatz——不是"接上线就能收敛"。
# 真实任务中会用更大的 ansatz、多个随机初始化、或 KL 散度损失来缓解。
```

## 6. 小结

1. QCBM 把量子线路的 Born 概率分布 $p_\theta(z)$ 当作生成模型，目标是拟合 $p_{\mathrm{data}}$。
2. MMD 损失只需样本估计，避免枚举 $2^n$ 个基态；参数位移法则穿过 Born 规则给出梯度。
3. 与 GAN、扩散同属"简单分布→数据分布"范式，但 QCBM 的变换是酉演化、输出离散比特。
4. QCBM 的表达能力受限于 ansatz 深度，深线路又遇贫瘠高原。

## 7. 思考题

**基础档**

1. 为什么 QCBM 的输出是**离散**比特串分布，而 GAN 输出连续向量？这对图像生成意味着什么？
2. 写出 MMD 损失的简化形式（当核 $k(z,z')=\delta_{z,z'}$ 时）。

**进阶档**

3. 若目标分布 $p_{\mathrm{data}}$ 本身就是从某个已知量子线路 $U_{\mathrm{data}}$ 生成的，QCBM 理论上能否精确恢复？需要满足什么条件？
4. MMD 与 KL 散度在"尾部概率"上的行为有何不同？对训练稳定度有何影响？
5. 证明：参数位移法则穿过 Born 规则后，$\partial p_\theta(z)/\partial\theta_k$ 仍是一个合法概率分布的导数（即 $\sum_z \partial p/\partial\theta_k=0$）。

**开放档**

6. QCBM 能表达哪些经典生成模型难以表达的分布？这些分布的共同特征是什么（如需要高阶干涉）？
7. 若 QCBM 输出 $2^n$ 维分布，但实际数据维度远小于 $n$，QCBM 会不会"过参数化"？这与经典生成模型的过参数化有何异同？

**延伸阅读**

- Liu & Wang, *Quantum-born machine for quantum state tomography* (2018)：QCBM 原始论文。
- Goodfellow et al., *Generative Adversarial Networks* (2014)：GAN 原始论文，用于对比。
- Song et al., *Quantum Circuit Born Machine* (2019)：训练方法与 MMD 损失。
