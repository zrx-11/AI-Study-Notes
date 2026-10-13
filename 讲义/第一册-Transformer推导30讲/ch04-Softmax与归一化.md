# 第4讲 Softmax与归一化：从logits到概率分布

> 📌 **本讲问题**：为什么注意力权重必须经过 Softmax 才能乘到 V 上？Softmax 有什么性质让它和交叉熵组成"梯度最干净"的组合？这一讲把 Softmax 的雅可比、复合梯度、温度系数全部推到底，并解释 ch08 缩放为什么必要。

## 1. 数学工具/预备知识

**指数函数 $e^x$。** 单调递增、处处可导，$\frac{d}{dx}e^x=e^x$；$e^x>0$ 恒成立，$x\to-\infty$ 时 $e^x\to 0$。这两条保证 Softmax 输出恒正。

**概率单纯形（probability simplex）。** $k$ 维概率分布集合 $\{p\in\mathbb{R}^{k}:p_i\geq 0,\ \sum_i p_i=1\}$。它是一个 $(k-1)$ 维的三角形（$k=3$ 时是平面上的三角形区域）。Softmax 的作用就是把任意实数向量 $\boldsymbol{z}\in\mathbb{R}^{k}$ 映射到这个单纯形内部。

**ch02 的交叉熵。** $L=-\sum_k t_k\ln p_k$，$\boldsymbol{t}$ 是真实分布。

**偏导数链式法则。** 对复合函数 $L(\boldsymbol{p}(\boldsymbol{z}))$，$\frac{\partial L}{\partial z_j}=\sum_i\frac{\partial L}{\partial p_i}\frac{\partial p_i}{\partial z_j}$。

## 2. 完整推导

### 2.1 Softmax 的定义与平移不变性

对 $k$ 维 logits 向量 $\boldsymbol{z}\in\mathbb{R}^{k}$，Softmax 定义为：

$$p_i=\mathrm{softmax}(z_i)=\frac{e^{z_i}}{\sum_{j=1}^{k}e^{z_j}}$$

分母是所有指数项之和，保证 $\sum_i p_i=1$；分子是单项指数，保证 $p_i>0$。输出落在概率单纯形内部。

**平移不变性**：给所有 $z_i$ 同时加上常数 $C$：

$$\frac{e^{z_i+C}}{\sum_j e^{z_j+C}}=\frac{e^C e^{z_i}}{e^C\sum_j e^{z_j}}=\frac{e^{z_i}}{\sum_j e^{z_j}}$$

$e^C$ 分子分母约掉，输出不变。也就是说 Softmax 只关心 logits 之间的**相对差**，不关心整体平移。这个性质是数值稳定的关键（2.5 节）。

### 2.2 Softmax 的雅可比矩阵

求 $\frac{\partial p_i}{\partial z_j}$。记 $S=\sum_m e^{z_m}$，则 $p_i=e^{z_i}/S$。分两种情况。

**当 $i=j$：** 用商法则，分子 $e^{z_i}$ 对 $z_i$ 求导是 $e^{z_i}$：

$$\frac{\partial p_i}{\partial z_i}=\frac{e^{z_i}\cdot S - e^{z_i}\cdot e^{z_i}}{S^2}=\frac{e^{z_i}}{S}-\frac{e^{2z_i}}{S^2}=p_i-p_i^2=p_i(1-p_i)$$

**当 $i\neq j$：** 分子 $e^{z_i}$ 对 $z_j$ 求导是 0（$i\neq j$），但分母 $S$ 对 $z_j$ 求导是 $e^{z_j}$：

$$\frac{\partial p_i}{\partial z_j}=\frac{0\cdot S - e^{z_i}\cdot e^{z_j}}{S^2}=-\frac{e^{z_i}e^{z_j}}{S^2}=-p_i p_j$$

写成矩阵形式 $\boldsymbol{J}\in\mathbb{R}^{k\times k}$：

$$J_{ij}=\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)$$

这里 $\delta_{ij}$ 是 Kronecker 符号（$i=j$ 时为 1，否则为 0）。

**这个雅可比告诉我们什么？** 对角元 $p_i(1-p_i)$ 在 $p_i\to 0$ 或 $p_i\to 1$ 时都趋近 0。也就是说，一旦 Softmax 输出接近 one-hot（某个 $p_i\approx 1$，其余 $\approx 0$），雅可比的所有元素都趋近 0——梯度消失。这就是 ch08 要对打分做缩放的根本原因。

### 2.3 Softmax + 交叉熵的复合梯度

设损失 $L=-\sum_k t_k\ln p_k$，$\boldsymbol{t}$ 是 one-hot 标签。求 $\frac{\partial L}{\partial z_j}$。

先用链式法则展开：

$$\frac{\partial L}{\partial z_j}=\sum_i\frac{\partial L}{\partial p_i}\frac{\partial p_i}{\partial z_j}$$

第一步算 $\frac{\partial L}{\partial p_i}=-\frac{t_i}{p_i}$（对 $L=-\sum_k t_k\ln p_k$ 求导，只有 $k=i$ 项贡献）。

第二步代入雅可比 $J_{ij}=p_i(\delta_{ij}-p_j)$：

$$\frac{\partial L}{\partial z_j}=\sum_i\left(-\frac{t_i}{p_i}\right)\cdot p_i(\delta_{ij}-p_j)=-\sum_i t_i(\delta_{ij}-p_j)$$

$p_i$ 约掉了——这是关键的一步。展开求和：

$$=-\sum_i t_i\delta_{ij}+\sum_i t_i p_j=-t_j+p_j\sum_i t_i$$

由于 $\sum_i t_i=1$（one-hot 标签归一）：

$$\frac{\partial L}{\partial z_j}=p_j-t_j$$

**结论**：Softmax+CE 的复合梯度就是"预测概率减真实标签"。对比 2.2 节单独 Softmax 的雅可比 $p_i(\delta_{ij}-p_j)$——那些 $p_i$ 因子在和 $-\frac{t_i}{p_i}$ 相乘时恰好约掉了。所以复合梯度里**没有饱和项**：无论 $p_j$ 接近 0 还是 1，梯度都是 $p_j-t_j$，量级正常。这就是分类网络敢把 logits 放得很大、却不会梯度消失的原因。

### 2.4 温度系数

若在 Softmax 前除以温度 $T$：

$$p_i=\frac{e^{z_i/T}}{\sum_j e^{z_j/T}}$$

- $T\to 0$：$z_i/T$ 的差异被无限放大，分布趋向 one-hot（argmax）。
- $T=1$：标准 Softmax。
- $T\to\infty$：$z_i/T\to 0$，所有 $e^{z_i/T}\to 1$，分布趋向均匀。

从方差角度看：logits 经过温度缩放后，有效 logit 差缩小 $T$ 倍，输出分布的熵升高（更均匀）。推理时调高 $T$ 让生成更多样，调低 $T$ 让生成更确定。训练时 $T=1$。

### 2.5 数值稳定 Softmax

直接算 $e^{z_i}$ 有两个风险：$z_i$ 很大时 $e^{z_i}$ 上溢到 $\infty$；$z_i$ 很小时 $e^{z_i}$ 下溢到 0（分母变成 0）。利用 2.1 节的平移不变性：令 $m=\max_i z_i$，令 $\tilde{z}_i=z_i-m$，则 $\tilde{z}_i\leq 0$，$e^{\tilde{z}_i}\in(0,1]$，最大的指数项是 $e^0=1$，不会上溢：

$$p_i=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}$$

分母至少有一项是 $e^0=1$，也不会下溢成 0。工程上进一步用 `log_softmax` 直接计算 $\log p_i=\tilde{z}_i-\log\sum_j e^{\tilde{z}_j}$，避免先 exp 再 log 的两次精度损失。

## 3. 💡 直觉理解

> 💡 **直觉**：Softmax 是一个"投票器"——每个 logit 是一个候选人的"得票指数"，然后归一化成百分比。它的平移不变性意味着"大家同时加 100 分"不改变排名，只改变自信程度。
>
> 单独 Softmax 的雅可比 $p_i(\delta_{ij}-p_j)$ 之所以在饱和区消失，是因为当一个候选人几乎全拿票时，你再微调任何人的分数，结果都几乎不变——函数平坦了。而 Softmax+CE 的复合梯度 $p_j-t_j$ 之所以干净，是因为交叉熵损失对"错得多"惩罚极重（ch02），两个函数的导数里那些饱和因子恰好互相约掉——这是两个函数天作之合的结果。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 Softmax 只是"把数归一化成概率"。关键陷阱是：**单独对 Softmax 求梯度会饱和**（$p_i\to 0$ 或 1 时 $p_i(1-p_i)\to 0$），但和 CE 组合后就不饱和了。如果你在中间层单独用 Softmax（比如中间特征归一化），梯度仍然会消失。
>
> ⚠️ **误区二**：以为温度 $T$ 是训练时调的超参。训练时 $T=1$；温度只在推理（采样）时调，用来控制生成的锐度。
>
> ⚠️ **误区三**：以为除以最大值是为了"提高精度"。它的首要目的是防止上溢/下溢；顺便它也不改变输出（平移不变性）。

## 5. 💻 代码实现

下面手写数值稳定 Softmax 并验证复合梯度。注意 `log_softmax` 是稳定实现，不要手写 `log(softmax)`；温度缩放要在 softmax 之前做。

```python
import torch
import torch.nn.functional as F

# 数值稳定 Softmax 手动实现
def stable_softmax(z, dim=-1):
    z_max, _ = z.max(dim=dim, keepdim=True)
    z_shifted = z - z_max  # 平移，防止上溢；平移不改变输出
    exp_z = torch.exp(z_shifted)
    return exp_z / exp_z.sum(dim=dim, keepdim=True)

z = torch.tensor([[1000.0, 1001.0, 999.0]])  # 大数值，直接 exp 会 inf
print("稳定 softmax:", stable_softmax(z).detach().round(decimals=4))
print("官方 softmax:", F.softmax(z, dim=-1).detach().round(decimals=4))
assert torch.allclose(stable_softmax(z), F.softmax(z, dim=-1), atol=1e-5)

# 验证平移不变性
z2 = z + 1000
print("\n加常数后 softmax 不变:",
      torch.allclose(stable_softmax(z), stable_softmax(z2), atol=1e-5))

# 验证复合梯度: dL/dz = p - t
z3 = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)
target = torch.tensor([0, 1, 0], dtype=torch.float32)  # one-hot

p = F.softmax(z3, dim=0)
loss = F.cross_entropy(z3.unsqueeze(0), target.argmax().unsqueeze(0))
loss.backward()
print("\n复合梯度 (autograd):", z3.grad.detach().round(decimals=4))
print("p - t (手动):       ", (p - target).detach().round(decimals=4))
assert torch.allclose(z3.grad, p - target, atol=1e-5)

# 温度系数效果：T 越大越均匀
logits = torch.tensor([3.0, 1.0, -1.0])
print("\n温度对分布的影响:")
for T in [0.5, 1.0, 2.0, 10.0]:
    p_T = F.softmax(logits / T, dim=0)
    ent = -(p_T * torch.log(p_T + 1e-12)).sum()
    print(f"  T={T:5.1f}: 分布={p_T.tolist()}  熵={ent:.3f}")

# 验证饱和区：单独 softmax 的雅可比趋近 0
z_sat = torch.tensor([100.0, 0.0, 0.0], requires_grad=True)
p_sat = F.softmax(z_sat, dim=0)
p_sat.sum().backward()  # 对 p1 求梯度（单独 softmax，不加 CE）
print("\n单独 softmax 在饱和区的梯度 (应≈0):", z_sat.grad.detach().round(decimals=6))
```

## 6. 小结

1. Softmax 把任意实数向量映射到概率单纯形，具有平移不变性——只看相对差。
2. 单独 Softmax 的雅可比 $J_{ij}=p_i(\delta_{ij}-p_j)$，在饱和区梯度趋近 0。
3. Softmax+CE 的复合梯度 $\partial L/\partial z_j=p_j-t_j$，饱和因子被约掉，完全消除饱和问题。
4. 数值稳定做法：先减最大值再指数；温度系数在推理时控制生成锐度。

## 7. 延伸阅读/进一步思考

- Softmax 的饱和问题是 Ch08 引入 $\sqrt{d_k}$ 缩放的直接动机；想理解完整因果链，把本讲 2.2 和 Ch08 2.3 对照着读。
- Label Smoothing（ch17）把 one-hot 标签 $t$ 软化成 $(1-\epsilon)\cdot\text{onehot}+\epsilon/k$，动机之一就是防止 Softmax 输出过度自信导致饱和。
- 若把单标签 Softmax 换成多标签 Sigmoid（$\sigma(z_i)=1/(1+e^{-z_i})$），复合梯度不再是 $p-t$ 的干净形式，每个标签独立用 BCE 损失。

## 8. 思考题

**基础题**

1. 若 Softmax 输入 $z=(100,0,0)$，输出约等于什么？此时对 $z_1$ 的单独 Softmax 雅可比 $\partial p_1/\partial z_1=p_1(1-p_1)$ 约等于多少？

**进阶题**

2. 为什么推理时调高温度会让生成更随机？从输出分布的熵角度解释：$T\to\infty$ 时熵趋向多少？
3. 如果把 Softmax 换成 Sigmoid（多标签分类，每个 logit 独立过 $\sigma$，用 BCE 损失 $L=-\sum_i[t_i\ln\sigma(z_i)+(1-t_i)\ln(1-\sigma(z_i))]$），求 $\partial L/\partial z_i$，它还是 $p_i-t_i$ 这个形式吗？

**开放题**

4. 实验发现：当 logits 方差很大时，即使 Softmax+CE 复合梯度本身不饱和，训练仍然不稳。结合 ch08 的方差推导解释：大 logits 除了让 Softmax 饱和，还会通过什么途径影响训练？（提示：想想注意力权重一旦接近 one-hot，value 加权求和就退化成了只取一个 value，模型的"软检索"能力会发生什么变化？）
