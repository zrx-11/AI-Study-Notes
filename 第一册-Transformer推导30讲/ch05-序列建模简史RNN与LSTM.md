# 第5讲 序列建模简史：RNN与LSTM的能力边界

> 📌 **本讲问题**：在 Transformer 出现之前，RNN 是序列建模的标准答案。为什么它训不深？LSTM 到底解决了什么问题？理解这些历史包袱，才能理解 Transformer 为什么要抛弃循环、改用全局注意力。

## 1. 数学工具/预备知识

**序列数据的形式化。** 一条序列 $\boldsymbol{x}_{1:T}=(\boldsymbol{x}_1,\boldsymbol{x}_2,\dots,\boldsymbol{x}_T)$，每个 $\boldsymbol{x}_t\in\mathbb{R}^{d}$ 是一个向量（比如 token embedding）。语言模型的目标是建模条件概率 $p(x_t\mid \boldsymbol{x}_{<t})$。

**ch03 的反向传播与梯度消失。** 本讲把"层间连乘导致指数衰减"这个分析搬到"时间维"上——同一套数学，只是连乘方向从"层"变成了"时间步"。

**逐元素激活函数。** $\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}}\in(-1,1)$，导数 $\tanh'(z)=1-\tanh^2(z)\in(0,1]$，在 $z=0$ 处取最大值 1，在饱和区 $|z|\gg 1$ 时趋近 0。Sigmoid $\sigma(z)\in(0,1)$，导数 $\sigma'(z)=\sigma(z)(1-\sigma(z))\leq 0.25$。

**门控直觉。** Sigmoid 输出 $(0,1)$，可以当成一个"软开关"：0 是完全关闭，1 是完全打开。门控机制就是用一个 Sigmoid 输出逐元素地控制"让多少信息通过"。

## 2. 完整推导

### 2.1 序列建模的形式化

序列模型要建模条件概率 $p(x_t\mid \boldsymbol{x}_{<t})$。RNN 用一个隐藏状态 $\boldsymbol{h}_t$ 递归地压缩全部历史：

$$\boldsymbol{h}_t=f(\boldsymbol{h}_{t-1},\boldsymbol{x}_t),\quad \boldsymbol{y}_t=g(\boldsymbol{h}_t)$$

其中 $f$ 是转移函数，$g$ 是输出函数。对最简单的 Elman RNN：

$$\boldsymbol{h}_t=\tanh(\boldsymbol{W}_{ih}\boldsymbol{x}_t+\boldsymbol{W}_{hh}\boldsymbol{h}_{t-1}+\boldsymbol{b}_h)$$

$$\boldsymbol{y}_t=\boldsymbol{W}_{ho}\boldsymbol{h}_t+\boldsymbol{b}_o$$

关键特点：**所有时间步共享同一套参数 $\boldsymbol{W}_{hh},\boldsymbol{W}_{ih}$**；当前隐藏状态只依赖上一步隐藏状态和当前输入——信息沿时间步单向流动。

### 2.2 BPTT：沿时间反向传播

总损失 $L=\sum_{t=1}^{T}L_t$，其中 $L_t$ 是第 $t$ 步预测的损失。求 $\frac{\partial L}{\partial \boldsymbol{W}_{hh}}$，要把梯度沿时间传回所有历史步：

$$\frac{\partial L}{\partial \boldsymbol{W}_{hh}}=\sum_{t=1}^{T}\frac{\partial L_t}{\partial \boldsymbol{W}_{hh}}$$

而单个 $L_t$ 对 $\boldsymbol{W}_{hh}$ 的梯度，要对从时刻 1 到时刻 $t$ 的每一步求贡献：

$$\frac{\partial L_t}{\partial \boldsymbol{W}_{hh}}=\sum_{k=1}^{t}\frac{\partial L_t}{\partial \boldsymbol{h}_t}\frac{\partial \boldsymbol{h}_t}{\partial \boldsymbol{h}_k}\frac{\partial \boldsymbol{h}_k}{\partial \boldsymbol{W}_{hh}}$$

关键项是雅可比连乘 $\frac{\partial \boldsymbol{h}_t}{\partial \boldsymbol{h}_k}$，它衡量"时刻 $k$ 的隐藏状态变动，经过 $t-k$ 步后，对时刻 $t$ 的影响"：

$$\frac{\partial \boldsymbol{h}_t}{\partial \boldsymbol{h}_k}=\prod_{j=k+1}^{t}\frac{\partial \boldsymbol{h}_j}{\partial \boldsymbol{h}_{j-1}}$$

这就是沿时间展开的链式法则，叫 BPTT（Backpropagation Through Time）。

### 2.3 梯度消失的数学证明

对 Elman RNN，$\boldsymbol{h}_j=\tanh(\boldsymbol{W}_{hh}\boldsymbol{h}_{j-1}+\cdots)$。逐元素求导：

$$\frac{\partial \boldsymbol{h}_j}{\partial \boldsymbol{h}_{j-1}}=\mathrm{diag}(\tanh'(\boldsymbol{z}_j))\cdot \boldsymbol{W}_{hh}$$

其中 $\boldsymbol{z}_j=\boldsymbol{W}_{ih}\boldsymbol{x}_j+\boldsymbol{W}_{hh}\boldsymbol{h}_{j-1}+\boldsymbol{b}_h$。取谱范数上界：

$$\left\|\frac{\partial \boldsymbol{h}_j}{\partial \boldsymbol{h}_{j-1}}\right\|\leq \max_i |\tanh'(z_{j,i})|\cdot \|\boldsymbol{W}_{hh}\|$$

由于 $\tanh'(z)=1-\tanh^2(z)\leq 1$，且在饱和区 $|\tanh(z)|\approx 1$ 时 $\tanh'(z)\approx 0$。记 $c=\max_i\tanh'(z_{j,i})\cdot\|\boldsymbol{W}_{hh}\|$。若 $\|\boldsymbol{W}_{hh}\|<1$ 且激活经常饱和，则 $c<1$。经过 $t-k$ 步：

$$\left\|\frac{\partial \boldsymbol{h}_t}{\partial \boldsymbol{h}_k}\right\|\leq c^{t-k}\to 0\quad (t-k\to\infty)$$

遥远过去的梯度信号传到当前位置时已经指数级消失。RNN 实际只能记住"最近几步"的信息。这个分析和 ch03 层间梯度消失完全同构，只是把"层数 $L$"换成了"时间距离 $t-k$"。

### 2.4 LSTM 的梯度高速公路

LSTM 的核心思想：引入一个**细胞状态** $\boldsymbol{c}_t$，让梯度可以在不经过 $\tanh$ 饱和变换的情况下直接流过。六组公式：

$$\boldsymbol{f}_t=\sigma(\boldsymbol{W}_f[\boldsymbol{h}_{t-1},\boldsymbol{x}_t]+\boldsymbol{b}_f)\quad\text{（遗忘门）}$$

$$\boldsymbol{i}_t=\sigma(\boldsymbol{W}_i[\boldsymbol{h}_{t-1},\boldsymbol{x}_t]+\boldsymbol{b}_i)\quad\text{（输入门）}$$

$$\tilde{\boldsymbol{c}}_t=\tanh(\boldsymbol{W}_c[\boldsymbol{h}_{t-1},\boldsymbol{x}_t]+\boldsymbol{b}_c)\quad\text{（候选细胞）}$$

$$\boldsymbol{c}_t=\boldsymbol{f}_t\odot \boldsymbol{c}_{t-1}+\boldsymbol{i}_t\odot \tilde{\boldsymbol{c}}_t\quad\text{（细胞状态更新）}$$

$$\boldsymbol{o}_t=\sigma(\boldsymbol{W}_o[\boldsymbol{h}_{t-1},\boldsymbol{x}_t]+\boldsymbol{b}_o)\quad\text{（输出门）}$$

$$\boldsymbol{h}_t=\boldsymbol{o}_t\odot\tanh(\boldsymbol{c}_t)$$

关键区别在细胞状态的梯度。对 $\boldsymbol{c}_t=\boldsymbol{f}_t\odot\boldsymbol{c}_{t-1}+\dots$ 两边关于 $\boldsymbol{c}_{t-1}$ 求导（固定其他项）：

$$\frac{\partial \boldsymbol{c}_t}{\partial \boldsymbol{c}_{t-1}}=\boldsymbol{f}_t$$

遗忘门 $\boldsymbol{f}_t$ 是 Sigmoid 输出，取值 $(0,1)$。如果训练学得让 $\boldsymbol{f}_t\approx \mathbf{1}$（遗忘门打开），那么梯度在细胞状态上的传递近似恒等映射，没有 $\tanh'$ 的饱和缩放，不会指数衰减。这就是"梯度高速公路"。

但 LSTM 并没有根治：$\boldsymbol{f}_t$ 仍然逐元素小于 1，长期连乘 $\prod_j \boldsymbol{f}_j$ 仍可能缓慢衰减；而且某个维度的遗忘门若长期接近 0，那个维度的梯度仍然消失。更根本的是——LSTM 仍然是**串行**的：$\boldsymbol{h}_t$ 必须等 $\boldsymbol{h}_{t-1}$ 算完才能算，序列长度 $T$ 决定了计算必须串行 $T$ 步，并行度为 1。这在算力时代是不可接受的。

### 2.5 串行瓶颈：RNN vs 注意力

RNN 任意两个位置 $i,j$ 之间的信息路径长度是 $|i-j|$（要沿时间一步步传）。Transformer 的自注意力里，任意两个位置直接连边，路径长度是 $O(1)$（ch16 会展开复杂度对比）。路径短意味着梯度不需要穿过长连乘，也意味着长距离依赖可以直接建模——这是 Transformer 取代 RNN 的两个根本原因。

## 3. 💡 直觉理解

> 💡 **直觉**：RNN 像传话游戏里站成一排的人——第一个人说的话传到第十个人时已经面目全非（梯度消失）。LSTM 给每个人发了一个笔记本，可以直接把上一个人的笔记原样传下去，不用经过"口头翻译"。但问题是——这一排人还是得一个接一个地传，没法同时开工。
>
> Transformer 直接把所有人叫到一个房间里开会（注意力），谁需要谁直接对话，不用排队传话。训练时所有人同时发言，GPU 把整个矩阵一次算完。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 LSTM 彻底解决了梯度消失。它只是缓解——通过细胞状态的加法通路代替乘法通路。遗忘门接近 1 时梯度近似恒等，但连乘 $\prod f_t$ 仍可能缓慢衰减。
>
> ⚠️ **误区二**：把 RNN 的失败只归因于梯度消失。即使梯度不消失，固定维度 $\boldsymbol{h}_t\in\mathbb{R}^{d}$ 也是一个信息瓶颈——任意长的历史都要压进同一个 $d$ 维向量，装不下。Transformer 的全局注意力让每个输出直接访问全部历史，绕开了这个瓶颈。
>
> ⚠️ **误区三**：以为 LSTM 慢是因为层数多。慢的根源是**时间维串行**：第 $t$ 步必须等第 $t-1$ 步，GPU 再大也只能一步一步算。

## 5. 💻 代码实现

下面观察 RNN 和 LSTM 在不同序列长度下的梯度范数。工程注意：梯度范数要对所有参数求和，且每个模型实例要独立建、独立反向，否则梯度会累积；`sum().backward()` 对输出求和构造一个标量损失，方便观测。

```python
import torch
import torch.nn as nn

# 直接观测"长程梯度"：末端损失对第 0 步输入的梯度范数，随序列长度 T 的变化
torch.manual_seed(0)

class VanillaRNN(nn.Module):
    def __init__(self, d=16, scale=0.9):
        super().__init__()
        self.d = d
        self.W_ih = nn.Linear(d, d)
        self.W_hh = nn.Linear(d, d)
        with torch.no_grad():
            self.W_hh.weight.mul_(scale)  # 循环权重谱半径 < 1，制造梯度消失条件
    def forward(self, x):  # x: (T, d)，要求 x 保留梯度
        h = torch.zeros(self.d)
        for t in range(x.size(0)):
            h = torch.tanh(self.W_ih(x[t]) + self.W_hh(h))
        return h

class LSTMWrap(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.lstm = nn.LSTM(d, d, batch_first=False)
    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))
        return out[-1, 0]

# 关键观测对象：末端 h[T] 对 x[0] 的梯度范数。
# RNN 应随 T 增大明显缩小（每步乘 ~0.9，长程信号指数衰减）；
# LSTM 靠细胞加法通路衰减慢得多。
print("T  | RNN 末端对 x[0] 梯度范数 | LSTM 末端对 x[0] 梯度范数")
for T in [10, 20, 40, 80]:
    x = torch.randn(T, 16, requires_grad=True)
    rnn = VanillaRNN(scale=0.9)
    rnn(x).sum().backward()
    rnn_grad0 = x.grad[0].norm().item()

    x2 = torch.randn(T, 16, requires_grad=True)
    lstm = LSTMWrap()
    lstm(x2).sum().backward()
    lstm_grad0 = x2.grad[0].norm().item()

    print(f"{T:3d} | {rnn_grad0:22.6f} | {lstm_grad0:22.6f}")
```

## 6. 小结

1. RNN 通过隐藏状态递归压缩历史，BPTT 导致梯度沿时间维雅可比连乘。
2. 梯度消失根源是 $\tanh'$ 饱和性和 $\|\boldsymbol{W}_{hh}\|<1$ 的连乘效应 $c^{t-k}$。
3. LSTM 通过细胞状态加法通路 $\boldsymbol{c}_t=\boldsymbol{f}_t\odot\boldsymbol{c}_{t-1}+\dots$ 提供梯度高速公路，$\partial\boldsymbol{c}_t/\partial\boldsymbol{c}_{t-1}=\boldsymbol{f}_t$，缓解但未根治。
4. RNN/LSTM 的致命伤是时间维串行无法并行，加上固定维度隐藏状态的信息瓶颈。

## 7. 延伸阅读/进一步思考

- 梯度消失的早期系统分析见 Hochreiter 1991 的论文和 Pascanu 等 2013 年《On the difficulty of training Recurrent Neural Networks》；LSTM 出自 Hochreiter & Schmidhuber 1997。
- 想理解"路径长度"和"长距离依赖"的关系，对照 He 等 2015 残差网络（ch12）：残差让前馈网络的有效路径也变成近似 $O(1)$。
- Transformer 的自注意力把串行 $T$ 步变成一次 $O(n^2d)$ 的矩阵乘法，这是它能并行训练的根本原因；ch16 会系统对比复杂度。

## 8. 思考题

**基础题**

1. 在 LSTM 中，如果遗忘门 $\boldsymbol{f}_t=\mathbf{0}$（完全关闭），细胞状态 $\boldsymbol{c}_t$ 会变成什么？梯度还能沿细胞状态传过这一步吗？

**进阶题**

2. RNN 中位置 $i$ 到位置 $j$ 的最大路径长度是 $|i-j|$。对比 Transformer 自注意力中任意两位置路径长度为 $O(1)$。从梯度连乘 $c^{|i-j|}$ 的角度解释：为什么路径短对长距离依赖训练更友好？
3. 为什么双向 RNN 在训练时可以并行（双向同时算），但推理时仍然是串行的？

**开放题**

4. LSTM 的细胞状态是加法更新 $\boldsymbol{c}_t=\boldsymbol{f}_t\odot\boldsymbol{c}_{t-1}+\boldsymbol{i}_t\odot\tilde{\boldsymbol{c}}_t$，这和残差连接 $x+F(x)$（ch12）在数学结构上有什么共通之处？为什么"加法通路"比"乘法通路"更利于梯度流动？
