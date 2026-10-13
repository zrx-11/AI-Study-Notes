# 第17讲 变分量子线路 VQC：参数化量子门与梯度

> 📌 **本讲问题**：在 NISQ（带噪声中等规模量子）时代，我们没法跑很深的线路。于是大家退而求其次：搭一个**浅层**线路，里面塞满可训练的旋转门参数，用经典优化器调参——这就是变分量子线路 VQC。它的梯度怎么算？为什么有一个特别干净的"参数位移法则"？

## 1. 数学工具/预备知识

### 1.1 单比特旋转门

$$
R_x(\theta)=e^{-i\theta X/2}=\cos(\theta/2)\mathbb{I}-i\sin(\theta/2)X,
$$

$$
R_y(\theta)=\cos(\theta/2)\mathbb{I}-i\sin(\theta/2)Y,\qquad
R_z(\theta)=\cos(\theta/2)\mathbb{I}-i\sin(\theta/2)Z.
$$

**为什么是 $\theta/2$？** 因为泡利矩阵本征值是 $\pm 1$，$e^{-i\theta \sigma/2}$ 在 $\sigma$ 本征基下是 $\mathrm{diag}(e^{-i\theta/2},e^{i\theta/2})$，相位差为 $\theta$。这是"旋转 $\theta$ 角度"的标准约定。

### 1.2 两比特纠缠门 CNOT

$|a,b\rangle\mapsto|a,a\oplus b\rangle$。它把控制比特 $a$ 原样保留，目标比特 $b$ 翻转为 $a\oplus b$。

### 1.3 参数化量子门的一般形式

$$
G(\theta)=e^{-i\theta P/2},
$$

其中 $P$ 是厄米算符（通常是泡利算符或其张量积）。这类门的期望值对 $\theta$ 的依赖是正弦/余弦函数，这是参数位移法则的根源。

## 2. 完整推导

### 2.1 VQC 可微编程框架

一个 VQC 由三部分组成：

1. **数据编码**：把经典数据 $x$ 编码为初态 $|\psi(x)\rangle$（常用角度编码 $R_x(x_1)R_y(x_2)\cdots$）；
2. **变分 ansatz**：一串参数化门 $U(\boldsymbol{\theta})=U_L(\theta_L)\cdots U_1(\theta_1)$；
3. **读出**：测某个厄米可观测量 $O$，得到期望值

$$
f(\boldsymbol{\theta};x)=\langle\psi(x)|U(\boldsymbol{\theta})^\dagger O U(\boldsymbol{\theta})|\psi(x)\rangle.
$$

训练目标是最小化损失 $\mathcal{L}(f(\boldsymbol{\theta};x), y)$。

**为什么用期望值而不是采样？** 因为期望值可以通过多次测量平均近似，而梯度需要精确。在模拟中我们直接算期望值。

### 2.2 参数位移法则（Parameter Shift Rule）

对一个含参数 $\theta$ 的门 $G(\theta)=e^{-i\theta P/2}$（$P$ 是泡利算符，本征值 $\pm 1$），期望值 $f(\theta)$ 满足

$$
\frac{\mathrm{d}f}{\mathrm{d}\theta}=\frac12\Bigl[f(\theta+\pi/2)-f(\theta-\pi/2)\Bigr].
$$

**推导**：设 $f(\theta)=\langle\phi|e^{i\theta P/2}O e^{-i\theta P/2}|\phi\rangle$。把 $O$ 分解为 $P$ 的本征空间上的矩阵元：$O=O_+ + O_-$，其中 $O_\pm$ 是 $P=\pm 1$ 子空间上的投影。

关键观察：$e^{i\theta P/2}O_\pm e^{-i\theta P/2}=e^{\pm i\theta}O_\pm$。于是

$$
f(\theta)=c_0 + c_+ e^{i\theta} + c_- e^{-i\theta},
$$

其中 $c_0=\langle\phi|O_0|\phi\rangle$（$O$ 在 $P=0$ 部分，若 $P$ 泡利则 $c_0=0$），$c_\pm=\langle\phi|O_\pm|\phi\rangle$。

求导：

$$
f'(\theta)=i c_+ e^{i\theta} - i c_- e^{-i\theta}.
$$

另一方面：

$$
f(\theta+\pi/2)-f(\theta-\pi/2)
=c_+(e^{i(\theta+\pi/2)}-e^{i(\theta-\pi/2)}) + c_-(e^{-i(\theta+\pi/2)}-e^{-i(\theta-\pi/2)})
=c_+(2i e^{i\theta}) + c_-(2(-i)e^{-i\theta})
=2(i c_+ e^{i\theta} - i c_- e^{-i\theta})=2f'(\theta).
$$

$\square$

**重要**：这个法则对任何泡利旋转门都成立，与线路其余部分无关。它是量子版的"自动微分"，但只需要两次前向模拟。

**为什么位移量是 $\pi/2$？** 因为 $P$ 本征值是 $\pm 1$，$e^{i\pi/2}=i$ 正好让 $e^{i\theta}$ 和 $e^{-i\theta}$ 交换符号。若 $P$ 本征值是 $\pm \lambda$，位移量改为 $\pi/(2\lambda)$。

### 2.3 Ansatz 与期望值读出

常用 hardware-efficient ansatz：每层是"所有比特单层 $R_y$ 旋转 + CNOT 纠缠层"，重复 $L$ 层。读出通常是 $\langle Z_0\rangle$（第一个比特上测 $Z$）。

**为什么用 $R_y$ 而不是 $R_x$？** 因为 $R_y$ 在 $|0\rangle$ 基下的旋转是实矩阵，数值稳定。任何单比特旋转都可以分解为 $R_z R_y R_z$。

### 2.4 NumPy 模拟 2 比特 VQC 二分类

构造一个 2 比特 VQC，输入是 2D 数据点 $(x_1,x_2)$，用角度编码，变分 ansatz 含 4 个参数，读出 $\langle Z_0\rangle$ 作为分类 logit。

## 3. 💡 直觉理解

> 💡 **直觉**：VQC 像一个"量子版的神经网络"。每个旋转门 $R_y(\theta)$ 是一个带参数的"神经元"——它把态矢量在 Bloch 球上转一个角度。CNOT 是"突触"，把两个比特的态纠缠起来。读出 $\langle Z_0\rangle$ 是"输出层"，给出一个标量。参数位移法则说："想知道某个 $\theta$ 的梯度？把它往左拨 $\pi/2$、往右拨 $\pi/2$，各测一次输出，差除以 2 就是梯度。"这比反向传播还干净——不需要链式法则。
>
> 生活类比：想象你在调一盏灯的旋钮（$\theta$）。参数位移法则说："想知道旋钮转一点，亮度变化多少？把旋钮往右转 90 度看亮度，再往左转 90 度看亮度，两个亮度差除以 2 就是灵敏度。"

## 4. ⚠️ 常见误区

> ⚠️ **误区一**："VQC 就是量子神经网络。"
>
> 澄清：VQC 是一个变分量子线路，它的"神经元"是旋转门，不是经典神经元。二者结构相似但物理载体不同。
>
> **误区二**："参数位移法则需要高阶导数时也能用。"
>
> 澄清：参数位移法则是一阶梯度公式。二阶导数要做两次位移，成本高。
>
> **误区三**："更深的 ansatz 一定更好。"
>
> 澄清：深线路会遇到贫瘠高原（第 20 讲），梯度指数消失。浅线路反而更易训练。
>
> **误区四**："参数位移法则对任意参数化门都成立。"
>
> 澄清：它只对 $e^{-i\theta P/2}$ 形式的门成立，其中 $P$ 是泡利型（本征值 $\pm 1$）。对 $P$ 本征值不同的门，位移量要调整。

## 5. 💻 代码实现

```python
import numpy as np

# ---- 1. 基本门 ----
I = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

def Ry(theta):
    return np.cos(theta/2)*I - 1j*np.sin(theta/2)*Y

def CNOT(n, control, target):
    """n 比特 CNOT 门"""
    dim = 2**n
    M = np.eye(dim, dtype=np.complex128)
    for s in range(dim):
        bits = [(s >> k) & 1 for k in range(n)]
        if bits[control] == 1:
            bits[target] = 1 - bits[target]
            s2 = sum(b << k for k, b in enumerate(bits))
            M[s, s] = 0
            M[s, s2] = 1
    return M

# ---- 2. 2 比特 VQC ----
def vqc_forward(x, params):
    """x: (x1, x2) 输入；params: (θ1, θ2, θ3, θ4)"""
    # 角度编码：先 |00⟩，然后对第 0 比特 Ry(x1)，对第 1 比特 Ry(x2)
    psi = np.array([1, 0], dtype=np.complex128)
    psi = Ry(x[0]) @ psi
    psi = np.kron(psi, np.array([1, 0], dtype=np.complex128))
    psi = np.kron(I, Ry(x[1])) @ psi

    # 变分层 1：两个 Ry
    psi = np.kron(Ry(params[0]), I) @ psi
    psi = np.kron(I, Ry(params[1])) @ psi
    # CNOT
    psi = CNOT(2, 0, 1) @ psi
    # 变分层 2：两个 Ry
    psi = np.kron(Ry(params[2]), I) @ psi
    psi = np.kron(I, Ry(params[3])) @ psi

    # 测 ⟨Z0⟩
    Z0 = np.kron(Z, I)
    return np.real(np.vdot(psi, Z0 @ psi))

# ---- 3. 玩具二分类数据：同心圆 ----
rng = np.random.default_rng(0)
N = 40
x_in = rng.uniform(0, 1, (N//2, 2)) * 0.5
x_out = rng.uniform(1.5, 2, (N//2, 2)) * 0.5
X = np.vstack([x_in, x_out])
y = np.array([-1]*(N//2) + [1]*(N//2))

# ---- 4. 用参数位移法则训练 ----
params = rng.uniform(0, 0.1, size=4)
lr = 0.1
for step in range(100):
    grads = np.zeros(4)
    loss = 0
    for xi, yi in zip(X, y):
        f = vqc_forward(xi, params)
        loss += (f - yi)**2
        for k in range(4):
            p_plus = params.copy(); p_plus[k] += np.pi/2
            p_minus = params.copy(); p_minus[k] -= np.pi/2
            # d(f)/dθk = (f(θ+π/2) - f(θ-π/2))/2
            df_dtheta = (vqc_forward(xi, p_plus) - vqc_forward(xi, p_minus)) / 2
            # d loss / dθk = 2(f-y) * df/dθk
            grads[k] += 2 * (f - yi) * df_dtheta
    params -= lr * grads / N
    if step % 20 == 0:
        print(f"step {step}: loss={loss/N:.4f}")

# ---- 5. 评估准确率 ----
preds = np.array([np.sign(vqc_forward(xi, params)) for xi in X])
acc = np.mean(preds == y)
print(f"训练准确率 = {acc:.3f}")
```

## 6. 小结

1. VQC = 数据编码 + 参数化 ansatz + 期望值读出；训练靠经典优化器。
2. 参数位移法则 $\partial f/\partial\theta=\frac12[f(\theta+\pi/2)-f(\theta-\pi/2)]$ 对泡利旋转门严格成立，只需两次前向模拟。
3. 常用 hardware-efficient ansatz：$R_y$ 层 + CNOT 层堆叠。
4. VQC 是 NISQ 时代的主力范式，但深线路会遇贫瘠高原。

## 7. 思考题

**基础档**

1. 推导参数位移法则时，为什么要求 $P$ 是泡利算符（本征值 $\pm 1$）？若 $P$ 本征值为 $\pm 2$，位移点应改为多少？
2. VQC 读出 $\langle Z_0\rangle$ 输出的是 $[-1,1]$ 间实数。要做多分类，需要几个读出比特？

**进阶档**

3. 参数位移法则每次梯度需要 $2\times$ 前向模拟。相比反向传播，这是贵还是便宜？在量子硬件上这意味着什么？
4. 证明：若 ansatz 只有一层 $R_y(\theta)$，$f(\theta)=\langle Z\rangle$ 是 $\cos\theta$ 形式，从而参数位移法则给出精确梯度。
5. 为什么 hardware-efficient ansatz 在深时会贫瘠高原？它与结构化 ansatz（如 QAOA）的差别在哪？

**开放档**

6. VQC 与经典神经网络在表达能力上有何本质差别？VQC 能逼近任意函数吗（量子通用逼近定理）？
7. 噪声对参数位移法则有什么影响？在有噪声硬件上，两次位移测量的方差会放大多少？

**延伸阅读**

- Mitarai et al., *Quantum circuit learning* (2018)：VQC 原始框架。
- Schuld et al., *Evaluating analytic gradients on quantum hardware* (2019)：参数位移法则的严格推导。
- McClean et al., *Barren plateaus in quantum neural network training landscapes* (2018)：贫瘠高原原始论文。
