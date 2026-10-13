# 第20讲 初始化与梯度稳定性：Xavier 与 He 初始化推导

> 📌 **本讲问题**：为什么权重不能全初始化为零？为什么 Xavier 要让前向和反向的方差都不变？He 初始化里那个"2"到底是从哪冒出来的？初始化选错了，网络会在第一次前向就崩。

## 1. 数学工具/预备知识

**（1）独立随机变量乘积的方差。** 若 $X,Y$ 独立、均值都是 0，则 $\mathrm{Var}(XY)=\mathrm{Var}(X)\mathrm{Var}(Y)$。推导：$E[XY]=E[X]E[Y]=0$，所以 $\mathrm{Var}(XY)=E[X^2Y^2]=E[X^2]E[Y^2]=\mathrm{Var}(X)\mathrm{Var}(Y)$。本讲的核心就是反复用这条。

**（2）独立随机变量和的方差。** 若各项独立，则 $\mathrm{Var}(\sum_j X_j)=\sum_j\mathrm{Var}(X_j)$——方差相加，不交叉。

**（3）前向与反向传播。** 前向是信号从输入传到输出；反向是梯度从输出传回来（ch03）。两者都要经过同样的权重矩阵，所以方差在两个方向上都要照顾到。

**（4）ReLU 的二阶矩。** 对零均值高斯 $X\sim\mathcal N(0,\sigma^2)$，$\mathrm{ReLU}(X)=\max(0,X)$。负半轴被置零，剩下正半轴。它的二阶期望 $E[\mathrm{ReLU}(X)^2]=\sigma^2/2$——正好是原方差的一半。这就是 He 初始化里那个因子的来源。

## 2. 完整推导

### 2.1 为什么不能全零初始化

如果所有权重 $W=0$，那么同一层的所有神经元算出一模一样的输出，梯度也一模一样——无论怎么更新，这一堆神经元永远做相同的事，学不出不同特征。这叫**对称性问题**。所以初始化必须给权重加随机扰动，打破对称。

### 2.2 前向方差怎么一步步演化

设第 $l-1$ 层输入 $\boldsymbol{h}^{l-1}\in\mathbb{R}^{n_{l-1}}$，权重矩阵 $\boldsymbol{W}^l\in\mathbb{R}^{n_l\times n_{l-1}}$，每个 $W_{ij}^l$ 独立同分布、均值 0、方差 $\sigma_w^2$。第 $i$ 个输出神经元：

$$
h_i^l=\sum_{j=1}^{n_{l-1}} W_{ij}^l h_j^{l-1}
$$

假设 $h_j^{l-1}$ 均值 0、方差 $\sigma_h^2$，且和 $W$ 独立。先用乘积方差公式看每一项 $W_{ij}^l h_j^{l-1}$：

$$
\mathrm{Var}(W_{ij}^l h_j^{l-1})=\mathrm{Var}(W_{ij}^l)\mathrm{Var}(h_j^{l-1})=\sigma_w^2\sigma_h^2
$$

再用和的方差公式把 $n_{l-1}$ 项加起来：

$$
\mathrm{Var}(h_i^l)=\sum_{j=1}^{n_{l-1}}\sigma_w^2\sigma_h^2=n_{l-1}\sigma_w^2\sigma_h^2
$$

我们希望信号每穿过一层，方差既不放大也不缩小（否则叠 100 层后信号要么爆炸成无穷大、要么萎缩成 0）。令 $\mathrm{Var}(h^l)=\mathrm{Var}(h^{l-1})=\sigma_h^2$：

$$
n_{l-1}\sigma_w^2=1\;\Longrightarrow\;\sigma_w^2=\frac{1}{n_{l-1}}
$$

这就是**前向传播保持方差稳定**的条件。

### 2.3 反向传播的对称条件

梯度 $\delta^l=\partial L/\partial h^l$ 从输出往回传。反向时，梯度要乘 $\boldsymbol{W}^l$ 的转置，走的方向和前向相反。用同样的方差推导（把 $n_{l-1}$ 换成 $n_l$），得到：

$$
\mathrm{Var}(\delta^{l-1})=n_l\sigma_w^2\,\mathrm{Var}(\delta^l)
$$

要让反向方差也不变（梯度不随层数消失或爆炸）：

$$
n_l\sigma_w^2=1\;\Longrightarrow\;\sigma_w^2=\frac{1}{n_l}
$$

### 2.4 Xavier：前向要 $1/n_{in}$，反向要 $1/n_{out}$，取个折中

前向要求 $\sigma_w^2=1/n_{l-1}=1/n_{in}$，反向要求 $\sigma_w^2=1/n_l=1/n_{out}$。两个要求打架，Glorot 的办法是取平均：

$$
\sigma_w^2=\frac{2}{n_{in}+n_{out}}
$$

这就是 **Xavier 初始化**（也叫 Glorot 初始化）。它假设激活函数是线性的（或近似线性的 tanh/sigmoid），让前向和反向的方差都大致稳住。

**几何意义**：把每层看成一个"信号放大器"。Xavier 让这个放大器的增益约等于 1——信号穿过去不增不减，梯度传回来也不增不减。

### 2.5 He 初始化：ReLU 悄悄吃掉了一半方差

Xavier 假设激活函数是线性的。但 ReLU 会把负值全部置零——大约一半的神经元输出变成 0。这等于悄悄把方差砍了一半。

具体说，前向传播经过 ReLU 后，方差关系变成（用 2.1 预备知识里的 $E[\mathrm{ReLU}(X)^2]=\sigma^2/2$）：

$$
\mathrm{Var}(h^l)=\frac{1}{2}\,n_{l-1}\sigma_w^2\,\mathrm{Var}(h^{l-1})
$$

那个 $1/2$ 就是 ReLU 置零负半轴造成的。要让方差仍然不变：

$$
\frac{1}{2}n_{l-1}\sigma_w^2=1\;\Longrightarrow\;\sigma_w^2=\frac{2}{n_{in}}
$$

这就是 **He 初始化**（何恺明初始化）。相比 Xavier，它把方差从 $2/(n_{in}+n_{out})$ 放大成 $2/n_{in}$，多出来的那一点正好补偿 ReLU 吃掉的一半。Transformer 用 ReLU/GELU，所以对应 He 初始化。

### 2.6 Transformer 里的实际做法

原论文：权重矩阵 $W\sim\mathcal N(0, d_{\text{model}}^{-0.5})$，偏置全 0，嵌入层 $\sim\mathcal N(0,1)$。

现代实践更简单：直接用 PyTorch 默认初始化，再加上 LayerNorm（ch12）。LayerNorm 会在每层主动把激活归一化回单位方差，等于给"方差漂移"上了个保险，所以对初始化的敏感度比早期网络低很多。

## 3. 💡 直觉理解

> 💡 **直觉**：初始化像给一长串接力赛安排起跑。每个运动员（神经元）传棒给下一个时，不能把棒扔得太快太猛（信号爆炸），也不能软乎乎传不到（信号消失）。前向是"信号从起点传到终点"，反向是"教练的反馈从终点喊回起点"，两个方向都得保证声音不衰减。Xavier 让两个方向音量差不多；后来发现 ReLU 这个运动员有个怪习惯——一半情况下直接把棒扔了（置零），所以得让他起跑时音量加倍补偿，这就是 He。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为初始化只是个"初始值"，训久了就无所谓。糟糕的初始化可能让第一步前向就把激活打出 inf，之后再也救不回来。
>
> **误区二**：以为 Xavier 和 He 可以随便用。Xavier 适合 tanh/sigmoid，He 适合 ReLU/GELU。用反了——比如对 ReLU 用 Xavier——前向方差会逐层缩小，几十层后激活全变成 0，梯度也跟着消失。
>
> **误区三**：以为 LayerNorm 能完全替代好初始化。LayerNorm 缓解敏感度，但糟糕的初始化仍可能在第一层就破坏训练。

## 5. 💻 代码实现

下面模拟一个 50 层的全连接网络，分别用 Xavier 和 He 初始化，打印前向传播中各层激活的方差变化。用 ReLU 激活，应该看到：He 初始化的方差基本稳住，Xavier 的方差逐层缩小。

```python
import torch

def check_forward_variance(weight_std, n_in=512, n_layers=50, batch=32):
    """模拟前向传播，观察各层激活方差如何变化（带 ReLU）"""
    x = torch.randn(batch, n_in)
    vars_list = [x.var().item()]
    for _ in range(n_layers):
        W = torch.randn(n_in, n_in) * weight_std   # 固定标准差的权重
        x = torch.relu(x @ W)                       # 线性层 + ReLU
        vars_list.append(x.var().item())
    return vars_list

# Xavier: 假设线性激活，对 ReLU 来说标准差偏小
std_xavier = (2 / (512 + 512)) ** 0.5
vars_xavier = check_forward_variance(std_xavier)

# He: 补偿了 ReLU 置零一半，标准差应保持方差稳定
std_he = (2 / 512) ** 0.5
vars_he = check_forward_variance(std_he)

print(f"Xavier std={std_xavier:.4f} (对 ReLU 偏小):")
print(f"  第1层={vars_xavier[0]:.3f}  第10层={vars_xavier[10]:.4f}  第50层={vars_xavier[50]:.6f}")

print(f"\nHe std={std_he:.4f} (补偿 ReLU):")
print(f"  第1层={vars_he[0]:.3f}  第10层={vars_he[10]:.4f}  第50层={vars_he[50]:.4f}")
print("\n预期：He 的方差基本稳住；Xavier 的方差逐层萎缩到接近 0。")
```

## 6. 小结

1. 全零初始化导致对称性问题，必须用随机权重打破对称。
2. Xavier：$\sigma_w^2=2/(n_{in}+n_{out})$，同时照顾前向和反向方差，适合 tanh/sigmoid。
3. He：$\sigma_w^2=2/n_{in}$，多出来的因子补偿 ReLU/GELU 置零负半轴损失的一半方差。
4. 选错初始化会让前向信号爆炸或消失；LayerNorm 能缓解但不能替代好初始化。

## 7. 思考题

**基础题**

1. 如果对一个用 sigmoid 激活的网络错误地用了 He 初始化（方差偏大），前向传播的激活会发生什么？

**进阶题**

2. 用 $E[\mathrm{ReLU}(X)^2]=\sigma^2/2$ 这一条，重新推导为什么 He 初始化的方差因子是 $2/n_{in}$ 而不是 $1/n_{in}$。GELU 也大约置零一半，它能用 He 初始化吗？

**开放题**

3. LayerNorm 在每层主动把激活拉回单位方差。请论证：为什么有了 LayerNorm 之后，现代 Transformer 对初始化的敏感度比早期 VGG/ResNet 低得多？这是否意味着初始化可以随便选？

## 8. 延伸阅读 / 进一步思考

- Xavier 原始推导：Glorot & Bengio, *Understanding the Difficulty of Training Deep Feedforward Neural Networks*, 2010。
- He 初始化：He et al., *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*, 2015。
- 进阶：针对深层 Transformer 的初始化技巧（如缩小残差分支初始化、GPT-2 的缩放初始化），以及 LayerScale 等，可作为进一步阅读。
