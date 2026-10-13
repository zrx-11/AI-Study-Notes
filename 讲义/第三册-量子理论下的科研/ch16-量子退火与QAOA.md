# 第16讲 量子退火与 QAOA：组合优化新引擎

> 📌 **本讲问题**：旅行商、Max-Cut、图着色……这些 NP-hard 组合优化问题，经典模拟退火靠"热涨落跳出局部极小"。量子退火说：我们不靠热，靠**量子隧穿**——让概率幅从一个局部极小"穿"过势垒到另一个。本讲从伊辛模型出发，严格推导 QAOA 的 $p=1$ 情形。

## 1. 数学工具/预备知识

### 1.1 伊辛哈密顿量

$$
H_C=-\sum_{\langle i,j\rangle} J_{ij} z_i z_j-\sum_i h_i z_i,
$$

$z_i=\pm 1$（在量子化版本里是泡利 $Z$ 算符）。这是经典 Ising 模型的量子化版本。

**为什么这能编码组合优化？** 因为任意组合优化问题都可以写成"找一个比特串 $z\in\{\pm1\}^n$ 最小化某个二次函数"的形式。Max-Cut、旅行商、图着色都能规约到伊辛模型。

### 1.2 绝热量子计算

从一个简单哈密顿量 $H_B$ 的基态出发，缓慢开启 $H_C$，系统始终跟随瞬时基态，结束时即为 $H_C$ 的基态——也就是优化问题的最优解。

**绝热定理**：若演化时间 $T$ 远大于能隙 $\Delta$ 的倒数平方（$T\gg 1/\Delta^2$），系统以高概率停留在瞬时基态。

## 2. 完整推导

### 2.1 组合优化 → 伊辛模型

Max-Cut 问题：给一张图 $G=(V,E)$，把顶点分成两组 $A$、$B$，最大化跨越 $A-B$ 边界的边数。

编码：每个顶点 $i$ 配一个量子比特 $z_i\in\{+1,-1\}$，+1 代表在 $A$，-1 代表在 $B$。一条边 $(i,j)$ 被 cut 当且仅当 $z_i z_j=-1$。总 cut 大小

$$
C=\sum_{(i,j)\in E}\frac{1-z_i z_j}{2}.
$$

最大化 $C$ 等价于最小化

$$
H_C=-\sum_{(i,j)\in E} z_i z_j.
$$

这就是伊辛哈密顿量（无外场 $h_i=0$）。

### 2.2 绝热定理与量子退火

初始哈密顿量 $H_B=-\sum_i X_i$（$X$ 是泡利 $X$），其基态是 $\bigotimes_i |+\rangle_i$，所有比特叠加。时间依赖哈密顿量

$$
H(t)=(1-t/T)H_B+(t/T)H_C.
$$

绝热定理说：若 $T$ 足够大，系统从 $H_B$ 基态平滑演化到 $H_C$ 基态。最终在 $Z$ 基下测量，得到的就是 Max-Cut 最优解。

**为什么 $H_B=-\sum X_i$ 的基态是 $|+\rangle^{\otimes n}$？** 因为 $X$ 的本征值 $\pm 1$，$X$ 的本征态 $|+\rangle$ 对应本征值 $+1$。$-\sum X_i$ 的基态是所有比特都取 $X=+1$，即 $|+\rangle^{\otimes n}$。

### 2.3 隧穿 vs 热涨落

经典模拟退火：靠温度 $T$ 给粒子"能量预算"，让它翻过势垒。势垒越宽越高，热隧穿越慢。

量子退火：靠 $H_B$ 的 $X$ 项让粒子"量子隧穿"过势垒。隧穿概率指数依赖于势垒**宽度**而非高度。对"宽而矮"的势垒，量子隧穿远优于热涨落——这是量子退火的潜在优势。

**物理直觉**：热涨落像"翻墙"（需要克服势垒高度），量子隧穿像"穿墙"（只需要势垒窄）。

### 2.4 QAOA：把绝热演化离散化

QAOA（Quantum Approximate Optimization Algorithm）把连续绝热演化切成 $p$ 段：

$$
|\psi(\boldsymbol{\gamma},\boldsymbol{\beta})\rangle
=e^{-i\beta_p H_B}e^{-i\gamma_p H_C}\cdots e^{-i\beta_1 H_B}e^{-i\gamma_1 H_C}|+\rangle^{\otimes n}.
$$

每段是一个固定的"cost 演化" $e^{-i\gamma H_C}$ 和一个"mixer 演化" $e^{-i\beta H_B}$。参数 $\gamma,\beta$ 经典优化，目标是

$$
F_p(\boldsymbol{\gamma},\boldsymbol{\beta})=\langle\psi|H_C|\psi\rangle
$$

最小化。

**为什么这样离散化？** 因为 $e^{-iH(t)t}$ 在小段时间内可以近似为 $e^{-iH_C\Delta t}e^{-iH_B\Delta t}$（Trotter 分解）。QAOA 就是把这个分解截断为 $p$ 段。

### 2.5 $p=1$ 情形的直观

$p=1$ 时：

$$
|\psi(\gamma,\beta)\rangle=e^{-i\beta H_B}e^{-i\gamma H_C}|+\rangle^{\otimes n}.
$$

**第一步 $e^{-i\gamma H_C}$**：在初始叠加态 $|+\rangle^{\otimes n}$ 上施加 cost 哈密顿量相位。每个计算基态 $|z\rangle$ 获得一个相位 $e^{-i\gamma E(z)}$，其中 $E(z)=\langle z|H_C|z\rangle=-\sum_{(i,j)\in E}z_i z_j$。

**第二步 $e^{-i\beta H_B}$**：施加 mixer，让不同计算基态之间产生跃迁。$H_B=-\sum X_i$，每个 $X_i$ 翻转第 $i$ 个比特。

最终在 $Z$ 基下测量，得到每个比特串的概率。优化 $\gamma,\beta$ 让期望 cut 最大。

### 2.6 NumPy 模拟 $p=1$ QAOA 求 Max-Cut

直接在 $2^n$ 维复向量上模拟 $e^{-i\gamma H_C}$、$e^{-i\beta H_B}$，扫参数找最小 $\langle H_C\rangle$。

## 3. 💡 直觉理解

> 💡 **直觉**：经典模拟退火像一只跳蚤在起伏的地形上跳——温度高时跳得高，能翻过小山；温度低时只能在一个坑里。量子退火像这只跳蚤突然有了"穿墙术"：它不必翻过山头，可以直接从坑底**穿过山体**到相邻坑。QAOA 就是把这个"穿墙过程"切成 $p$ 小步，每小步调整一下穿墙强度（$\gamma$）和跳跃强度（$\beta$），用经典优化器调参。$p$ 越大，模拟越精细，解越好——但 $p$ 大到一定程度就遇到"贫瘠高原"（第 20 讲）。
>
> 生活类比：想象你在迷宫里找出口。经典退火像"每次随机走一步，偶尔退回"；量子退火像"你有概率同时存在于多条路径上，最后坍缩到最优路径"。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**："量子退火一定比经典退火快。"
>
> 澄清：隧穿优势只在"宽而矮"势垒上明显。对很多实际优化 landscape，势垒窄而高，经典热涨落已经足够。量子退火是否有实际优势仍是开放问题。
>
> **误区二**："QAOA 是通用量子计算。"
>
> 澄清：QAOA 是专用变分算法，只针对组合优化。它不比经典退火"通用"，只是另一类算法。
>
> **误区三**："$p$ 越大越好。"
>
> 澄清：$p$ 大了参数多了，但梯度会指数消失（贫瘠高原），实际训练不动。$p=1,2$ 反而最常用。
>
> **误区四**："D-Wave 量子退火机是通用量子计算机。"
>
> 澄清：D-Wave 是专用退火机，不能跑 Grover、Shor 等通用量子算法。

## 5. 💻 代码实现：NumPy 模拟 $p=1$ QAOA 求 Max-Cut

```python
import numpy as np

# ---- 1. 一个 3 顶点 3 边的小图（三角形） ----
n = 3
edges = [(0, 1), (1, 2), (0, 2)]

# ---- 2. 构造 H_C = -Σ z_i z_j ----
def z_op(i, n):
    """第 i 个比特上的 Z 算符，n 比特系统"""
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    ops = [I] * n
    ops[i] = Z
    result = ops[0]
    for k in range(1, n):
        result = np.kron(result, ops[k])
    return result

HC = np.zeros((2**n, 2**n), dtype=np.complex128)
for i, j in edges:
    HC -= z_op(i, n) @ z_op(j, n)

# ---- 3. 构造 H_B = -Σ X_i ----
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)
HB = np.zeros((2**n, 2**n), dtype=np.complex128)
for i in range(n):
    ops = [I2] * n
    ops[i] = X
    Xi = ops[0]
    for k in range(1, n):
        Xi = np.kron(Xi, ops[k])
    HB -= Xi

# ---- 4. 初态 |+>^n ----
plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
psi0 = plus
for _ in range(n - 1):
    psi0 = np.kron(psi0, plus)

# ---- 5. p=1 QAOA：扫 γ, β，找最大 ⟨HC⟩ ----
# 注意：HC = -Σ z_i z_j，最大化 C 等价于最大化 ⟨HC⟩
# （因为 C = (|E| + ⟨HC⟩)/2，每条边贡献 (1 - z_i z_j)/2）
def expect_HC(psi):
    return np.real(np.vdot(psi, HC @ psi))

best = (0, 0, -1e9)
for gamma in np.linspace(0, 2*np.pi, 41):
    for beta in np.linspace(0, np.pi, 21):
        # 应用 e^{-i γ HC}（HC 是对角的，直接乘相位）
        Uc = np.diag(np.exp(-1j * gamma * np.diag(HC)))
        psi_t = Uc @ psi0
        # 应用 e^{-i β HB}
        Ub = np.eye(2**n, dtype=np.complex128)
        for i in range(n):
            # e^{-i β X_i} = cos(β) I - i sin(β) X_i
            Ub_i = np.cos(beta) * I2 - 1j * np.sin(beta) * X
            ops = [I2] * n
            ops[i] = Ub_i
            Ui = ops[0]
            for k in range(1, n):
                Ui = np.kron(Ui, ops[k])
            Ub = Ui @ Ub
        psi_t = Ub @ psi_t
        f = expect_HC(psi_t)
        if f > best[2]:
            best = (gamma, beta, f)

print(f"最优 γ={best[0]:.3f}, β={best[1]:.3f}, ⟨HC⟩={best[2]:.4f}")
# 三角形 Max-Cut 最优 cut = 2，对应 ⟨HC⟩ = 1（两条边 z_i z_j=-1 贡献 +1，一条 +1 贡献 -1）
print(f"对应 cut 大小 = {(len(edges) + best[2])/2:.4f}")
# 验证：三角形最大 cut 确实是 2（两条边跨组分）
```

## 6. 小结

1. Max-Cut 等组合优化可编码为伊辛哈密顿量 $H_C$ 的基态问题；量子退火从 $H_B=-\sum X_i$ 基态缓慢演化到 $H_C$ 基态。
2. 量子隧穿相对于热涨落的优势在"宽而矮"势垒上明显；并非普适优势。
3. QAOA 把绝热演化离散为 $p$ 层变分线路，参数 $\gamma,\beta$ 经典优化；$p=1$ 可数值求解。
4. $p$ 增大时表达力增强，但梯度会指数消失（贫瘠高原），实际多用小 $p$。

## 7. 思考题

**基础档**

1. 把 Max-Cut 的目标函数 $C=\sum_{(i,j)}(1-z_i z_j)/2$ 与 $H_C=-\sum z_i z_j$ 对照，说明最小化 $H_C$ 为什么等价于最大化 $C$。
2. 写出 $H_B=-\sum X_i$ 在 $n=2$ 时的矩阵形式。

**进阶档**

3. 绝热定理要求演化时间 $T$ 反比于能隙的平方。若能隙随 $n$ 指数缩小，$T$ 需要多大？这对量子退火的可扩展性意味着什么？
4. 为什么 $p=1$ QAOA 不能达到任意精度？要想逼近最优解，$p$ 需要怎么增长？
5. 证明：$e^{-i\gamma H_C}$ 作用在计算基 $|z\rangle$ 上只改变相位 $e^{-i\gamma E(z)}$，不改变概率幅的绝对值。

**开放档**

6. D-Wave 量子退火机已商用多年，但至今没有明确展示超越经典退火的实用优势。为什么？是硬件问题还是问题结构问题？
7. QAOA 的参数 $\gamma,\beta$ 是经典优化的。若把它们换成可学习的参数（如在神经网络中），QAOA 与经典启发式算法（如模拟退火）的边界在哪？

**延伸阅读**

- Farhi, Goldstone, Gutmann, *A Quantum Approximate Optimization Algorithm* (2014)：QAOA 原始论文。
- Johnson et al., *Quantum annealing with manufactured spins* (Nature, 2011)：D-Wave 退火机原理。
- Headley et al., *How much entanglement is needed in QAOA?*：QAOA 纠缠结构分析。
