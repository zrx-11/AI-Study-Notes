# 第12讲 Add & Norm：残差连接与LayerNorm的原理

> 📌 **本讲问题**：为什么 Transformer 能堆到几十上百层而不崩？残差连接到底修好了什么病？LayerNorm 又在归一化什么？这两个零件合起来，是深层 Transformer 训练稳定的基石。

## 1. 数学工具/预备知识

**ch03 的梯度消失/爆炸。** 梯度沿层间连乘 $c^L$ 指数衰减。

**向量的均值与方差。** $\mu=\frac{1}{d}\sum_i x_i$，$\sigma^2=\frac{1}{d}\sum_i(x_i-\mu)^2$。

**批归一化（BatchNorm）。** 对 batch 维做归一化，需要足够大的 batch 才稳定。

## 2. 完整推导

### 2.1 残差连接：梯度恒等通路

设子层函数为 $F(x)$，残差连接输出为：

$$\mathrm{Layer}(x)=x+F(x)$$

反向传播时，损失 $L$ 对输入 $x$ 的梯度：

$$\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\cdot\frac{\partial (x+F(x))}{\partial x}=\frac{\partial L}{\partial y}\cdot\left(1+\frac{\partial F}{\partial x}\right)$$

关键是那个 **1**。即使 $\frac{\partial F}{\partial x}\to 0$（子层梯度消失），梯度仍有一个恒等通路 1 直接传回。不管子层多深多复杂，梯度至少能以系数 1 无损回传。

对比没有残差：$\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y}\cdot\frac{\partial F}{\partial x}$，一旦 $\frac{\partial F}{\partial x}\to 0$，梯度就消失了。这从结构上解决了 ch03 的连乘衰减问题——每一层的梯度都拆成 $1+$ 一个小量，$L$ 层连乘后那个 1 始终保留。

### 2.2 LayerNorm 公式

LayerNorm 对每个样本的特征维做归一化。设输入 $\boldsymbol{x}\in\mathbb{R}^{d}$（单个样本的 $d$ 维特征）：

$$\mu=\frac{1}{d}\sum_{i=1}^{d}x_i,\qquad \sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2$$

$$\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}$$

$$y_i=\gamma_i\hat{x}_i+\beta_i$$

其中 $\gamma,\beta\in\mathbb{R}^{d}$ 是可学习缩放和偏置，$\epsilon$ 防止除零。

**为什么不用 BatchNorm？** BatchNorm 对 batch 维归一化，统计量依赖 batch 大小。序列长度变化时（不同句子长度不同），BatchNorm 的 batch 统计量不稳定，且 train/inference 行为不一致。LayerNorm 对特征维归一化，每个样本独立，与 batch 大小、序列长度都无关。

### 2.3 Pre-LN vs Post-LN

原论文用 **Post-LN**：先子层，再残差，再 Norm：

$$\mathrm{output}=\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))$$

现代大模型普遍改用 **Pre-LN**：先 Norm，再子层，再残差：

$$\mathrm{output}=x+\mathrm{Sublayer}(\mathrm{LayerNorm}(x))$$

**梯度流分析**：Post-LN 中 Norm 在残差通路外面，梯度要穿过 Norm 才能走恒等通路；Pre-LN 中 Norm 在子层内部，残差通路 $x$ 完全不受 Norm 影响，梯度恒等通路更干净。这就是 Pre-LN 训练更稳定、能堆更深的原因（代价是最后一层输出没有被归一化，通常额外加一个最终 LN）。

### 2.4 为什么需要 LayerNorm

没有归一化时，网络加深后每层输出的均值方差不断漂移（covariate shift）。LayerNorm 强制每层输入均值 0、方差 1（再由 $\gamma,\beta$ 恢复表达能力），让每层输入分布保持稳定，训练更容易。配合残差，它把 ch03 里"每层谱半径乘积"这个不受控的量重新拉回稳定区间。

## 3. 💡 直觉理解

> 💡 **直觉**：残差连接像高速公路——不管中间出口（子层）堵不堵，主路永远畅通（系数 1）。LayerNorm 像车速限制——不管谁上高速都先校准到统一速度，防止某辆车速度太快导致连环相撞。Pre-LN 是先校准再上高速，Post-LN 是先上高速再校准——显然前者更安全。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为残差连接是"把输入复制一份"。它是 $x+F(x)$，输入加子层输出，不是简单复制。
>
> ⚠️ **误区二**：以为 LayerNorm 和 BatchNorm 只是归一化维度不同。关键区别：BatchNorm 训练用 batch 统计量、推理用滑动平均，有 train/inference 不一致；LayerNorm 每样本独立，行为一致。
>
> ⚠️ **误区三**：以为 $\gamma,\beta$ 是多余的。初始 $\gamma=1,\beta=0$ 时 LayerNorm 只是标准化，$\gamma,\beta$ 让网络能自己恢复需要的尺度和偏移。

## 5. 💻 代码实现

下面手写 LayerNorm 对照官方实现，并演示残差的梯度恒等通路。注意 `torch.var(unbiased=False)` 用有偏方差（除以 $d$），和公式一致；PyTorch 官方 LayerNorm 也用有偏估计。

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x: (..., d_model)，对最后一维归一化
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 有偏方差
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

# 对照官方实现
torch.manual_seed(0)
x = torch.randn(2, 10, 512)

my_ln = LayerNorm(512)
official_ln = nn.LayerNorm(512)  # 默认 gamma=1, beta=0
out_mine = my_ln(x)
out_official = official_ln(x)
print("手动 LayerNorm 与官方差异:", torch.allclose(out_mine, out_official, atol=1e-5))

# 验证残差的梯度恒等通路
x = torch.randn(4, requires_grad=True)
f = lambda z: z * 0.1  # 梯度很小的子层
y = x + f(x)           # 残差
y.sum().backward()
print("\n残差连接下 x 的梯度:", x.grad.tolist(), "(应为 1 + 0.1 = 1.1)")

x2 = torch.randn(4, requires_grad=True)
y2 = f(x2)             # 无残差
y2.sum().backward()
print("无残差时 x 的梯度:", x2.grad.tolist(), "(应为 0.1)")
```

## 6. 小结

1. 残差连接 $x+F(x)$ 提供梯度恒等通路，$\partial L/\partial x=\partial L/\partial y\cdot(1+\partial F/\partial x)$，从结构上缓解梯度消失。
2. LayerNorm 对特征维归一化，每样本独立，与 batch 无关。
3. Pre-LN（先 Norm 再子层）比 Post-LN 梯度流更干净，现代大模型普遍采用。
4. 归一化防止激活分布漂移，是深层网络训练稳定的必要条件。

## 7. 延伸阅读/进一步思考

- 残差连接出自 He 等 2015《Deep Residual Learning for Image Recognition》；LayerNorm 出自 Ba 等 2016《Layer Normalization》。
- Pre-LN vs Post-LN 的系统对比可参考 Xiong 等 2020《On Layer Normalization in the Transformer Architecture》。
- RMSNorm（仅缩放不中心化）是 LayerNorm 的简化变体，LLaMA 在用，计算更省。

## 8. 思考题

**基础题**

1. 如果残差改成 $x+0\cdot F(x)$（完全绕过子层），网络还能训练吗？这对应什么极端情况？

**进阶题**

2. LayerNorm 的 $\gamma,\beta$ 初始化为 1 和 0，意味着训练开始时 LayerNorm 在做什么操作？
3. 为什么 Pre-LN 的原始 Transformer 性能略差于 Post-LN？（提示：考虑最后一层输出有没有被归一化。）

**开放题**

4. RMSNorm 只做 $\hat{x}_i=x_i/\sqrt{\mathrm{mean}(x^2)+\epsilon}\cdot\gamma$，不减均值。从"归一化到底在稳定什么"的角度分析：去掉减均值这一步，会损失什么、省了什么？为什么大模型愿意接受这个 trade-off？
