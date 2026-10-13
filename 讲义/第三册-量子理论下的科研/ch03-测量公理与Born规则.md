# 第3讲 测量公理与 Born 规则：概率从哪里来

> 📌 **本讲问题**：态矢量 $|\psi\rangle$ 是一个确定性的复向量，为什么测量结果却是概率性的？这个概率到底是"我们知识的缺欠"，还是"世界本身的内禀随机性"？Born 规则 $p(i)=|\langle e_i|\psi\rangle|^2$ 究竟是一条需要外加的公理，还是可以从别的公理推出来的定理？

## 1. 数学工具/预备知识

复习第 2 讲：厄米算符 $A$ 有谱分解 $A=\sum_i\lambda_i P_i$，其中 $P_i$ 是到本征值 $\lambda_i$ 对应本征子空间的投影算符，满足

$$
P_iP_j=\delta_{ij}P_i,\qquad \sum_iP_i=\mathbb{I},\qquad P_i^\dagger=P_i.
$$

在非退化情形 $P_i=|e_i\rangle\langle e_i|$。本讲把这套投影和"测量动作"直接挂钩。

**记号约定**：

- $\langle A\rangle_\psi:=\langle\psi|A|\psi\rangle$ 称为可观测量 $A$ 在态 $|\psi\rangle$ 下的**期望值**；
- $p(\lambda_i):=\langle\psi|P_i|\psi\rangle$ 是测得本征值 $\lambda_i$ 的概率；
- 测量后态 $|\psi'\rangle$ 必须重新归一。

## 2. 完整推导

### 2.1 投影测量公理

**公理（投影测量）**：设可观测量 $A$ 厄米，谱分解 $A=\sum_i\lambda_i P_i$（把简并子空间合并，$P_i$ 是到本征值 $\lambda_i$ 对应本征子空间的投影）。若系统处于归一化态 $|\psi\rangle$，则：

1. 测得结果为 $\lambda_i$ 的概率
   $$p(\lambda_i)=\langle\psi|P_i|\psi\rangle=\|P_i|\psi\rangle\|^2;$$
2. 若测得 $\lambda_i$，则测量后系统的态坍缩为
   $$|\psi'\rangle=\frac{P_i|\psi\rangle}{\sqrt{\langle\psi|P_i|\psi\rangle}}=\frac{P_i|\psi\rangle}{\sqrt{p(\lambda_i)}}.$$

在非退化情形 $P_i=|e_i\rangle\langle e_i|$ 下，$p(\lambda_i)=|\langle e_i|\psi\rangle|^2$，即 **Born 规则**。

**为什么分母要 $\sqrt{p_i}$？** 因为 $P_i|\psi\rangle$ 不再归一：$\langle\psi|P_i^\dagger P_i|\psi\rangle=\langle\psi|P_i|\psi\rangle=p_i$。除以 $\sqrt{p_i}$ 把它重新归一到单位长度，否则后续概率不归一。

### 2.2 概率归一性的自洽检查

验证 $\sum_i p(\lambda_i)=1$：

$$
\sum_i p(\lambda_i)=\sum_i\langle\psi|P_i|\psi\rangle
=\langle\psi|\Bigl(\sum_i P_i\Bigr)|\psi\rangle
=\langle\psi|\mathbb{I}|\psi\rangle
=\langle\psi|\psi\rangle=1.
$$

这一步把"概率必须归一"这件事直接归约到了"态必须归一"——再次说明内积几何与概率几何是同构的。

### 2.3 期望值作为谱加权平均

可观测量 $A$ 在态 $|\psi\rangle$ 下的期望值定义为

$$
\langle A\rangle_\psi:=\sum_i \lambda_i\,p(\lambda_i)
=\sum_i\lambda_i\langle\psi|P_i|\psi\rangle
=\langle\psi|\Bigl(\sum_i\lambda_i P_i\Bigr)|\psi\rangle
=\langle\psi|A|\psi\rangle.
$$

最后一个等号用了谱分解。这就解释了为什么物理学家写 $\langle A\rangle=\langle\psi|A|\psi\rangle$——它不是一个随意的记号，而是"按概率加权平均本征值"的简写。

**方差**：

$$
\mathrm{Var}(A)=\langle A^2\rangle-\langle A\rangle^2
=\langle\psi|A^2|\psi\rangle-\langle\psi|A|\psi\rangle^2.
$$

方差刻画测量结果的散布程度。若 $|\psi\rangle$ 已是 $A$ 的本征态，则 $\mathrm{Var}(A)=0$，测量结果完全确定。

### 2.4 Born 规则是公理还是推导？

这是量子基础里争论了一百年的问题，这里给出三种立场的客观陈述：

1. **哥本哈根立场（本讲义默认）**：Born 规则是**独立公理**，与薛定谔方程（幺正演化）并列。它不能从幺正演化导出，必须外加。优点是简单、工程上够用；代价是它引入了一个非幺正、非线性、非决定论的"坍缩"过程，与薛定谔方程不兼容。
2. **多世界立场（Everett）**：幺正演化是唯一的演化定律，Born 规则不是独立公理，而是"世界分支上的自测量概率"——即理性观察者应当按权重 $|\langle e_i|\psi\rangle|^2$ 对自己落在哪个分支下注。这一派声称可以从决策论对称性（Deutsch–Wallace）导出 Born 规则，但严格性仍有争议。
3. **玻姆力学（de Broglie–Bohm）**：粒子有确定轨迹，Born 规则是"量子平衡态"下的统计分布，是一条类似统计力学中"宏观平衡"的涌现定律，不是基本公理。偏离量子平衡的分布原则上存在，但弛豫时间极短。

本讲不站队，但要记住：**在做量子信息/量子机器学习时，我们默认采用立场 1**——把 Born 规则当作工程事实来用。

### 2.5 坍缩：非幺正的一步

测量后的态 $|\psi'\rangle=P_i|\psi\rangle/\sqrt{p_i}$ 一般不再满足幺正演化（因为 $P_i$ 不是酉算符）。这就是所谓的"波函数坍缩"。它的特征是：

- **非线性**：$P_i(\alpha|\psi\rangle+\beta|\phi\rangle)\ne\alpha P_i|\psi\rangle+\beta P_i|\phi\rangle$ 一般不成立（归一化分母是状态相关的）；
- **非决定论**：即便 $|\psi\rangle$ 完全已知，下一次测量结果仍随机；
- **瞬时性**：与测量装置的空间尺度无关，不需要时间参数。

**关键观察**：若测得 $\lambda_i$，再测一次同一个 $A$，结果必为 $\lambda_i$。证明：坍缩后态是 $P_i|\psi\rangle/\sqrt{p_i}$，它落在 $P_i$ 的本征子空间里，所以 $\langle\psi'|P_j|\psi'\rangle=\delta_{ij}$。这就是"测量可重复"公理。

### 2.6 Born 规则 vs Softmax 采样：类比的边界

神经网络里常用 Softmax：

$$
q(i)=\frac{e^{s_i}}{\sum_j e^{s_j}}.
$$

量子 Born 规则：

$$
p(i)=|\langle e_i|\psi\rangle|^2=\frac{|\psi_i|^2}{\sum_j|\psi_j|^2}.
$$

**相似处**：都是把一组实数/复数压成概率分布。
**本质差别**：

1. Softmax 的 $s_i$ 是任意打分，没有几何意义；Born 概率来自**希尔伯特空间内积**，$|\psi\rangle$ 是单位向量。
2. Softmax 没有"坍缩"概念；量子测量一旦测得 $i$，后续态就是 $|e_i\rangle$，再测一次必仍得 $i$。
3. Softmax 里的 $e^{s_i}$ 全为正；Born 里允许 $\psi_i$ 是复数，**相位在模平方时被抹掉**——这正是为什么相位信息在单比特测量中不可见，但在干涉（第 5 讲）和纠缠（第 6 讲）中至关重要。

**类比的有效边界**：当你只做"读出概率"这一步时，Softmax ≈ Born；一旦涉及"测量后态如何更新"，量子就必须用坍缩，Softmax 没有对应物。第 14 讲会回到这一点。

### 2.7 广义测量（POVM）预告

投影测量是"理想测量"。现实中测量装置不完美，可以用**正算符值测度（POVM）**推广：一组半正定算符 $E_i\succeq 0$，满足 $\sum_i E_i=\mathbb{I}$，测得结果 $i$ 的概率为 $p(i)=\langle\psi|E_i|\psi\rangle$。POVM 不要求 $E_i$ 是投影（不要求 $E_i^2=E_i$），也不要求测量后态有简单形式。本讲义主线用投影测量，POVM 在需要时再引入。

## 3. 💡 直觉理解

> 💡 **直觉**：把 $|\psi\rangle$ 想成一支箭，斜插在由本征轴 $\{|e_i\rangle\}$ 张成的空间里。$|\langle e_i|\psi\rangle|$ 就是这支箭在第 $i$ 个轴上投下的影子长度。Born 规则说：你闭眼抽一支箭射出，它落在第 $i$ 个轴上的概率，正比于那个影子长度的**平方**——不是长度本身，是平方。为什么是平方？因为概率必须是正实数，而 $|\cdot|^2$ 是复数唯一自然给出的非负实函数。一旦射中第 $i$ 个轴，箭就"啪"地一下贴在那个轴上，变成 $|e_i\rangle$——这就是坍缩。再射一次，必仍在 $|e_i\rangle$。
>
> 生活类比：想象你在黑暗中摸一把钥匙串上的钥匙。每把钥匙对应一个本征方向。你摸到哪把钥匙的"概率"正比于你手心在那把钥匙形状上的"投影面积"。一旦你摸到某把钥匙，它就滑到你手心正中央（坍缩），再摸一次还是同一把。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**："测量只是'发现'了系统原本就有的值。"
>
> 澄清：除非 $|\psi\rangle$ 本身就是 $|e_i\rangle$ 之一，否则系统在测量前**没有**一个确定的 $\lambda_i$。测量不是"读取"，而是"参与制造"结果。这是量子力学与经典概率论最深刻的差别。
>
> **误区二**："Born 规则里的 $|\langle e_i|\psi\rangle|^2$ 就是 $|\psi_i|^2$，所以相位不重要。"
>
> 澄清：在**单计算基测量**下相位确实被抹掉，但在**不同基**下测量时，相位会通过干涉重新冒出来（第 5 讲双缝）。把相位当垃圾是初学者最常犯的错。
>
> **误区三**："坍缩是一个真实的物理过程，需要时间。"
>
> 澄清：在哥本哈根诠释里，坍缩是一个**公设层面**的更新规则，不是一个动力学过程。它没有时间参数，也没有微观机制——这正是爱因斯坦觉得"不完备"的地方。若把它当真，会导致薛定谔方程（幺正、线性、决定论）与坍缩（非幺正、非线性、非决定论）之间的矛盾，即"测量问题"。

## 5. 💻 代码实现：模拟投影测量实验

```python
import numpy as np

rng = np.random.default_rng(42)

# ---- 1. 准备一个非平凡单比特态 |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}sin(θ/2)|1⟩ ----
theta, phi = np.pi / 3.0, np.pi / 4.0
psi = np.array([np.cos(theta / 2),
                np.exp(1j * phi) * np.sin(theta / 2)], dtype=np.complex128)
psi /= np.linalg.norm(psi)
print("|ψ⟩ =", psi)

# ---- 2. 计算 Born 概率 p(0), p(1) ----
probs_theory = np.abs(psi) ** 2
print("理论概率 p(0), p(1) =", probs_theory)

# ---- 3. 模拟 N 次投影测量 ----
N = 20000
results = rng.choice([0, 1], size=N, p=probs_theory)
print("频率统计 p̂(0), p̂(1) =", np.bincount(results) / N)

# ---- 4. 测量后的坍缩：若测得 i，态变为 |i⟩ ----
# 验证：连续测两次，第二次结果必与第一次相同
def measure_once(state, basis=None):
    """在给定标准基下投影测量一次，返回 (结果, 坍缩后态)"""
    probs = np.abs(state) ** 2
    outcome = rng.choice(len(state), p=probs)
    post = np.zeros_like(state)
    post[outcome] = 1.0   # P_i|ψ⟩/√p_i 在标准基下就是 |i⟩
    return outcome, post

o1, s1 = measure_once(psi)
o2, s2 = measure_once(s1)
print(f"第一次测得 {o1}，第二次测得 {o2}（应相同）")

# ---- 5. 期望值 ⟨Z⟩ = ⟨ψ|Z|ψ⟩，用泡利 Z 矩阵 ----
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
expect_Z = np.vdot(psi, Z @ psi)
expect_Z_via_freq = probs_theory @ np.array([1.0, -1.0])
print("直接 ⟨Z⟩ =", expect_Z, " 频率加权 =", expect_Z_via_freq)

# ---- 6. 换基测量：在 X 基 {|+⟩,|-⟩} 下测，看相位如何复活 ----
plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
minus = np.array([1, -1], dtype=np.complex128) / np.sqrt(2)
p_plus = np.abs(np.vdot(plus, psi)) ** 2
p_minus = np.abs(np.vdot(minus, psi)) ** 2
print(f"X 基测量：p(+)={p_plus:.4f}, p(-)={p_minus:.4f}")
# 注意：这两个概率依赖于相位 φ！
psi2 = psi * np.exp(-1j * phi)   # 把相位去掉（全局相位不影响，但这里是构造对比）
print(f"去掉相位后 X 基：p(+)={np.abs(np.vdot(plus, psi2))**2:.4f}, "
      f"p(-)={np.abs(np.vdot(minus, psi2))**2:.4f}")

# ---- 7. 方差演示：在 Z 本征态 |0⟩ 上方差为 0 ----
psi0 = np.array([1, 0], dtype=np.complex128)
var_Z_0 = np.vdot(psi0, Z@Z@psi0).real - np.vdot(psi0, Z@psi0).real**2
print("|0⟩ 下 Var(Z) =", var_Z_0, "(应为 0)")
```

## 6. 小结

1. 投影测量公理：测得 $\lambda_i$ 的概率为 $\langle\psi|P_i|\psi\rangle$，测量后态坍缩为 $P_i|\psi\rangle/\sqrt{p_i}$。
2. 归一性 $\sum_i p_i=1$ 与期望值 $\langle A\rangle=\langle\psi|A|\psi\rangle$ 都直接由谱分解导出，不需要额外假设。
3. Born 规则在哥本哈根诠释下是独立公理；它与 Softmax 的类比仅在"读出概率"这一步成立，一旦涉及坍缩与相位就失效。
4. 单计算基测量会抹掉相位，但换基测量（如 $X$ 基）会让相位通过干涉重新显现——这是第 5 讲双缝的伏笔。

## 7. 思考题

**基础档**

1. 设 $|\psi\rangle=\frac{1}{\sqrt2}(|0\rangle+|1\rangle)$。在 $Z$ 基下测量，概率是多少？在 $X$ 基下测量呢？比较二者，说明为什么"先测 $Z$ 再测 $X$"与"先测 $X$ 再测 $Z$"结果分布不同。
2. 证明：若 $|\psi\rangle$ 已经是 $A$ 的某个本征态 $|e_k\rangle$，则测得 $\lambda_k$ 的概率为 1，且坍缩后态不变。

**进阶档**

3. 设 $A,B$ 是两个不对易的可观测量。从投影测量公理出发，说明为什么"先测 $A$ 再测 $B$"与"先测 $B$ 再测 $A$"一般得不到相同的概率分布。
4. 设 $A=\sum_i\lambda_iP_i$。证明 $\langle A^2\rangle=\sum_i\lambda_i^2p_i$，从而 $\mathrm{Var}(A)=\sum_i(\lambda_i-\langle A\rangle)^2p_i$（经典方差形式）。
5. POVM：设 $E_i=\frac12P_i$，$\sum_i E_i=\frac12\mathbb{I}$。这是合法 POVM 吗？若不是，缺了什么？给出一个合法的两结果 POVM 例子（非投影）。

**开放档**

6. 多世界诠释声称 Born 规则可从决策论对称性导出。这一导出的关键假设是什么？为什么有人认为它是循环论证？
7. 若测量坍缩是真实物理过程，它与狭义相对论的"类空间隔不能有因果联系"是否冲突？纠缠态的非局域关联（第 6 讲）如何与这一点自洽？

**延伸阅读**

- Born, *Zur Quantenmechanik der Stoßvorgänge* (1926)：Born 规则的原始论文（获诺贝尔奖）。
- Nielsen & Chuang, 第 2.2 节：投影测量与 POVM 的标准推导。
- Wallace, *The Emergent Multiverse* (2012)：多世界视角下 Born 规则的决策论导出。
- Mermin, *Is the moon there when nobody looks?* (1985)：测量问题的通俗但严格的讨论。
