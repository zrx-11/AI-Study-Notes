# 第18讲 量子核机 QSVM：量子优势第一候选

> 📌 **本讲问题**：第 10 讲我们定义了量子核 $k_Q(x,x')=|\langle\psi(x')|\psi(x)\rangle|^2$。为什么大家觉得"量子核 SVM"是量子机器学习最有可能率先展示量子优势的方向？本讲从 SVM 对偶问题出发，把量子核接到经典 SVM 上，并解释 Havlicek 的量子优势论证。

## 1. 数学工具/预备知识

复习第 10 讲量子核。SVM 对偶问题：

$$
\max_\alpha \sum_i\alpha_i-\frac12\sum_{ij}\alpha_i\alpha_j y_i y_j k(x_i,x_j),\qquad 0\le\alpha_i\le C.
$$

一旦核矩阵 $K_{ij}=k(x_i,x_j)$ 算出来，剩下的就是标准二次规划。

**SVM 预测**：$f(x)=\sum_i \alpha_i y_i k(x_i,x)$。预测时只需要算新样本与支持向量的核。

## 2. 完整推导

### 2.1 SVM 对偶：核是唯一入口

SVM 的训练和预测都只依赖核函数 $k(x_i,x_j)$，不显式访问特征 $\phi(x)$。这意味着：只要能给出核矩阵，SVM 本身**完全是经典的**。量子部分只负责"估计核矩阵"。

**为什么这很重要？** 因为经典 QP 求解器成熟、稳定。我们不需要在量子硬件上跑整个 SVM，只需要用量子硬件算一个 $N\times N$ 的核矩阵，剩下的全交给经典求解器。

### 2.2 量子核线路估计

对每对 $(x_i,x_j)$，运行两次量子线路：

1. 制备 $|\psi(x_i)\rangle=U(x_i)|0\rangle$，在计算基下测 $N$ 次，得频率 $\hat p_i(z)$；
2. 同样测 $|\psi(x_j)\rangle$ 得 $\hat q_j(z)$；
3. 用**交换测试**（swap test）估计 $k_Q(x_i,x_j)=|\langle\psi(x_j)|\psi(x_i)\rangle|^2$。

**交换测试**：制备辅助比特 $|0\rangle$，控制 Hadamard，然后控制交换 $|\psi(x_i)\rangle$ 和 $|\psi(x_j)\rangle$，再测辅助比特。测得 $|0\rangle$ 的概率是 $\frac12+\frac12|\langle\psi(x_j)|\psi(x_i)\rangle|^2$。

所有 $N^2$ 对估计完，得到核矩阵 $K\in\mathbb{R}^{N\times N}$，送入经典 SVM 求解。

### 2.3 与经典核的有效维度差异

**有效维度**定义为核矩阵本征值分布的集中度：

$$
d_{\mathrm{eff}}=\frac{(\mathrm{Tr}(K))^2}{\mathrm{Tr}(K^2)}.
$$

若量子核映射到 $2^n$ 维空间，核矩阵的有效维度可随 $n$ 多项式甚至指数增长；经典核（RBF、多项式）的有效维度最多随数据维数多项式增长。

若数据集的真实"分类复杂度"需要高有效维度，经典核需要指数多的支持向量才能拟合，而量子核只用多项式多的样本。

### 2.4 Havlicek 量子优势论证

Havlicek 等人（2019）构造了一类特殊的量子特征映射 $U(x)$，对应的量子核函数不能用任何多项式大小的经典线路近似估计。具体地：若输入 $x\in\{0,1\}^n$ 编码为 $n$ 比特角态，特征线路包含一层"纠缠网络"，则 $k_Q(x,x')$ 是一个**经典计算上困难**的函数——经典计算机要估计它需要指数时间。

这一论证的关键：量子核不是"换了个核"，而是"经典计算机算不动的核"。当核矩阵本身的估计就是指数困难时，SVM 训练自然受益。

**边界**：这个优势只在**合成数据集**上展示。在真实工业数据集上，至今没有超越经典方法。

### 2.5 NumPy 模拟 4 比特量子核 SVM

直接在 $2^4=16$ 维复空间里枚举所有 $|0\rangle^4$ 经参数化线路后的态，计算核矩阵，喂给 sklearn SVM。

## 3. 💡 直觉理解

> 💡 **直觉**：经典 SVM 像一个阅卷老师，他只关心"两份答卷有多像"（核函数），不关心答卷背后的思考过程。量子核说："这份答卷的思考过程藏在一个 $2^n$ 维的量子晶体里，经典老师要重建这个思考过程得花指数时间，但量子老师直接测一下两份答卷的重叠就能给出相似度。"所以量子核 SVM 的优势不在 SVM 本身，而在"测重叠"这件事量子比经典快。
>
> 生活类比：想象两个画家的作品。经典核像"比较两幅画的像素距离"；量子核像"把两幅画放进一个全息装置，直接读出它们的干涉图样"。后者能捕捉到前者看不到的结构。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**："QSVM 是在量子硬件上跑 SVM。"
>
> 澄清：SVM 求解器是经典的；量子硬件只用来估计核矩阵。
>
> **误区二**："量子核一定难经典模拟。"
>
> 澄清：很多浅层量子核其实可以被经典高效模拟（如直接解薛定谔方程）。Havlicek 的优势论证针对的是**特定深层纠缠线路**。
>
> **误区三**："QSVM 已经在真实任务上展示了量子优势。"
>
> 澄清：截至目前，量子优势仍停留在**合成数据集**上，尚未在任何真实工业任务上超越经典方法。
>
> **误区四**："核矩阵估计只需测一次。"
>
> 澄清：每对 $(x_i,x_j)$ 都需要一次交换测试，总共 $N^2$ 次。这在大 $N$ 时是瓶颈。

## 5. 💻 代码实现

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_circles

rng = np.random.default_rng(0)

# ---- 1. 玩具数据集 ----
X, y = make_circles(n_samples=40, factor=0.5, noise=0.1, random_state=0)

# ---- 2. 4 比特量子特征映射：把 2D 输入编码为 16 维态 ----
def quantum_feature(x):
    """x: 2D 实数；映射到 4 比特态
    简化：把每个计算基 z 赋予振幅 exp(i * (x1*z0 + x2*z1))
    """
    psi = np.zeros(16, dtype=np.complex128)
    for z in range(16):
        z0 = (z >> 0) & 1
        z1 = (z >> 1) & 1
        psi[z] = np.exp(1j * (x[0] * z0 + x[1] * z1))
    psi /= np.linalg.norm(psi)
    return psi

# ---- 3. 计算核矩阵 ----
def quantum_kernel(X1, X2):
    K = np.zeros((len(X1), len(X2)))
    for i, x1 in enumerate(X1):
        psi1 = quantum_feature(x1)
        for j, x2 in enumerate(X2):
            psi2 = quantum_feature(x2)
            K[i, j] = np.abs(np.vdot(psi2, psi1)) ** 2
    return K

K = quantum_kernel(X, X)
print("核矩阵形状 =", K.shape, " 对角元 =", np.round(np.diag(K)[:5], 3))

# ---- 4. 训练 SVM ----
clf = SVC(kernel="precomputed")
clf.fit(K, y)
acc = clf.score(K, y)
print(f"量子核 SVM 训练准确率 = {acc:.3f}")

# ---- 5. 对照：线性核、RBF 核 ----
clf_lin = SVC(kernel="linear").fit(X, y)
clf_rbf = SVC(kernel="rbf").fit(X, y)
print(f"线性核 SVM = {clf_lin.score(X, y):.3f}")
print(f"RBF 核 SVM = {clf_rbf.score(X, y):.3f}")

# ---- 6. 有效维度 ----
trK = np.trace(K)
trK2 = np.trace(K @ K)
d_eff = trK**2 / trK2
print(f"量子核有效维度 = {d_eff:.2f}")
```

## 6. 小结

1. QSVM = 量子核矩阵估计 + 经典 SVM 求解；量子部分只负责"测重叠"。
2. 量子特征空间维数 $2^n$，有效维度可随 $n$ 指数增长；经典核最多多项式增长。
3. Havlicek 论证：特定深层量子核的经典估计是指数困难的，这是量子优势的理论基础。
4. 实际优势仍局限于合成数据集；真实工业任务上的量子优势尚未达成。

## 7. 思考题

**基础档**

1. 为什么 SVM 对偶问题只依赖核矩阵？把原始 SVM 问题写成关于 $\phi(x)$ 的形式，再用核替换 $\phi(x_i)^\top\phi(x_j)$，说明这一步为什么合法。
2. 写出交换测试测得 $|0\rangle$ 概率的推导。

**进阶档**

3. 若量子核矩阵 $K$ 的秩为 $r$，则 SVM 最多需要 $r$ 个支持向量。为什么量子核的高有效维度意味着更少的支持向量？
4. Havlicek 论证的"经典估计困难"依赖于量子线路有足够深度的纠缠。若线路很浅，经典计算机能否直接模拟？这对 QSVM 的实际部署意味着什么？
5. 证明：核矩阵 $K_{ij}=|\langle\psi(x_j)|\psi(x_i)\rangle|^2$ 的对角元全为 1。

**开放档**

6. 为什么真实工业数据集上量子核没有展示优势？是因为数据本身的"量子结构"不够，还是因为硬件噪声？
7. 若量子核的经典估计是困难的，那么在 NISQ 硬件上估计核矩阵的噪声误差会如何影响 SVM 训练？噪声是平滑掉核的量子优势，还是反而让它更鲁棒？

**延伸阅读**

- Havlicek et al., *Quantum enhanced feature maps in kernel methods* (Nature, 2019)：量子核优势原始论文。
- Schölkopf & Smola, *Learning with Kernels* (2002)：SVM 对偶与核方法。
- Schuld et al., *Quantum Machine Learning* (2021)：量子核方法综述。
