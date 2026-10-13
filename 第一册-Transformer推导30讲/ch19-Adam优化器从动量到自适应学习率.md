# 第19讲 Adam 优化器：从动量到自适应学习率

> 📌 **本讲问题**：为什么几乎所有 Transformer 都用 Adam 而不是 SGD？Adam 到底"自适"了什么？偏差修正又在修什么"偏"？AdamW 为什么非要把权重衰减从梯度里拆出来？

## 1. 数学工具/预备知识

**（1）梯度下降。** $\theta_t=\theta_{t-1}-\eta g_t$，$g_t=\nabla L(\theta_{t-1})$。每步只看当前梯度。

**（2）指数移动平均（EMA）。** $m_t=\beta m_{t-1}+(1-\beta)g_t$，$\beta\in[0,1)$。它把最近一串梯度加权求和，越近的权重越大，越远的按几何级数衰减。可以把 $m_t$ 理解成"最近梯度的滑动平均"。

**（3）几何级数求和。** $\sum_{i=0}^{n-1}r^i=(1-r^n)/(1-r)$。本讲推偏差修正时会反复用到它。

**（4）二阶矩。** $v_t$ 跟踪的是 $g_t^2$ 的滑动平均，它衡量"这个参数的梯度最近有多大"。$v_t$ 大说明梯度长期偏大，$v_t$ 小说明长期偏小。

## 2. 完整推导

### 2.1 SGD + 动量：把梯度平均掉

朴素 SGD 每步只看当前 $g_t$。问题是：当损失曲面狭长（一个方向陡、另一个方向缓）时，SGD 会在陡的方向上剧烈来回震荡，在缓的方向上却几乎不前进。

动量（Momentum）解决这个问题：累积历史梯度的 EMA，让更新方向沿"平均方向"走：

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t
$$

$$
\theta_t=\theta_{t-1}-\eta m_t
$$

把 $m_t$ 展开，$m_t=(1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i$。注意震荡方向上正负梯度会互相抵消（EMA 里一加一减），而持续方向上的梯度会叠加——所以动量相当于一个"低通滤波器"，滤掉高频震荡、保留持续趋势。$\beta_1$ 常取 0.9。

### 2.2 RMSProp：给每个参数配自己的步幅

动量只管了"方向"。RMSProp 管"每个参数走多快"。它对梯度平方做 EMA：

$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\theta_t=\theta_{t-1}-\eta\frac{g_t}{\sqrt{v_t}+\epsilon}
$$

$v_t$ 是梯度平方的滑动平均。看更新量里的 $g_t/\sqrt{v_t}$：

- 如果某参数梯度一直很大，$v_t$ 大，$g_t/\sqrt{v_t}\approx$ 个位数——步幅被压小。
- 如果某参数梯度一直很小，$v_t$ 小，$g_t/\sqrt{v_t}$ 相对变大——步幅被放大。

这就是"自适应"：**梯度大的参数走小步，梯度小的参数走大步**，把各方向的有效步幅拉到差不多。$\epsilon$（$10^{-8}$）防止除零。

### 2.3 Adam：动量 + 自适应合体

Adam 同时维护一阶矩（动量）和二阶矩（自适应）：

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t\qquad\text{（一阶矩，方向）}
$$

$$
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2\qquad\text{（二阶矩，步幅）}
$$

$$
\theta_t=\theta_{t-1}-\eta\frac{m_t}{\sqrt{v_t}+\epsilon}
$$

默认超参：$\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$。

### 2.4 偏差修正：为什么 $m_0=v_0=0$ 会出问题

问题出在初始化：$m_0=0$，$v_0=0$。开头几步 $m_t$、$v_t$ 都被零起点拖着，系统性地**偏小**。

以 $m_t$ 为例，把递推展开成求和：

$$
m_t=(1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}g_i
$$

现在假设梯度是个常数 $g_i\equiv g$（理想情况，没有噪声），那应该有 $m_t\approx g$ 才对。代进去用几何级数求和：

$$
m_t=g(1-\beta_1)\sum_{i=1}^{t}\beta_1^{t-i}
=g(1-\beta_1)\frac{1-\beta_1^t}{1-\beta_1}
=g(1-\beta_1^t)
$$

看出来了吗？当 $t$ 很小时，$\beta_1^t\approx 1$，所以 $m_t=g(1-\text{接近}1)\approx 0$——明明真实梯度是 $g$，动量却被算成接近 0，被严重低估了。$v_t$ 同理。

修正方法就是把这个 $(1-\beta^t)$ 的折扣除掉：

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t},\qquad \hat{v}_t=\frac{v_t}{1-\beta_2^t}
$$

更新规则随之变成：

$$
\theta_t=\theta_{t-1}-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

**为什么 $t$ 大了以后修正就失效了？** 因为 $t\to\infty$ 时 $\beta^t\to 0$，$1-\beta^t\to 1$，$\hat{m}_t\to m_t$——初始的零起点影响被时间冲淡了，修正项趋近于 1。所以偏差修正是个"只在前几千步起作用"的补丁。

### 2.5 AdamW：把权重衰减从梯度里解耦出来

原始 Adam 里，权重衰减（weight decay，即把权重往 0 拉的正则化）是直接加进梯度的：$g_t\leftarrow g_t+\lambda\theta_{t-1}$。但这样有个问题：加进去的 $\lambda\theta$ 随后又被 $\sqrt{v_t}$ 除了一遍——于是不同参数实际受到的衰减强度被自适应学习率搅乱了，正则化变得不干净。

AdamW 干脆把权重衰减从梯度更新里**解耦**出来，单独乘一个学习率：

$$
\theta_t=\theta_{t-1}-\eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}+\lambda\theta_{t-1}\right)
$$

其中 $\lambda\theta_{t-1}$ 是独立的衰减项，每个参数都以同样的比例 $\eta\lambda$ 向 0 收缩，不受自适应学习率干扰。这让正则化效果更干净，是现代大模型的标配。

## 3. 💡 直觉理解

> 💡 **直觉**：SGD 像蒙眼下山——每步只看脚下那一点坡度，路窄就左右横跳。动量像一颗带惯性的滚石——顺着之前滚的方向继续走，被小坑洼绊一下也不会立刻改向，震荡被惯性磨平了。RMSProp 像给不同陡度配不同步幅——陡坡走小步、缓坡走大步，把横移压下去。Adam 就是"滚石 + 自动步幅"的合体。偏差修正则是因为这颗石头起步时是静止的，得把"启动延迟"补偿掉，否则头几步石头根本不滚。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 Adam 不需要调学习率。Adam 只是把每个参数之间的**相对**步幅自适应了，全局学习率 $\eta$ 仍然是必须手调的头号超参。
>
> **误区二**：以为 Adam 总比 SGD 好。在很多 CV 任务上，SGD+动量泛化反而更好——因为 Adam 的自适应步幅会让模型过早收敛到"尖锐"的极小值，而 SGD 倾向于停在"平坦"的极小值，平坦点泛化更好。Transformer 用 Adam 主要是因为它收敛快、对超参不敏感。
>
> **误区三**：以为 AdamW 的权重衰减就是 L2 正则化。两者在 SGD 里等价，但在 Adam 里不等价——正是这个不等价催生了 AdamW。

## 5. 💻 代码实现

下面手写一个最小 AdamW，并和 PyTorch 官方实现对拍，验证前向几步的参数更新完全一致。手写版严格按上面公式实现偏差修正与解耦衰减。

```python
import torch

class ManualAdamW:
    """严格按公式实现: 一阶矩 + 二阶矩 + 偏差修正 + 解耦权重衰减"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = [p.clone().detach() for p in params]   # 维护一份参数副本
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]  # 一阶矩
        self.v = [torch.zeros_like(p) for p in self.params]  # 二阶矩

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            # 更新矩估计
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            # 偏差修正: 除以 (1 - beta^t)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # 更新 + 解耦权重衰减: 衰减项独立于自适应步幅
            p.data -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p.data)

# ===== 对拍测试 =====
torch.manual_seed(0)
y_target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# 官方 AdamW
x = torch.randn(5, requires_grad=True)
opt_official = torch.optim.AdamW([x], lr=0.01, betas=(0.9, 0.999), weight_decay=0.01)

# 手动 AdamW（从相同初值出发）
x2 = x.detach().clone().requires_grad_(True)
opt_mine = ManualAdamW([x2], lr=0.01, betas=(0.9, 0.999), weight_decay=0.01)

for step in range(5):
    # 官方：反向传播自动求梯度
    loss = ((x - y_target) ** 2).sum()
    opt_official.zero_grad(); loss.backward(); opt_official.step()
    # 手动：梯度就是 2*(当前参数 - y_target)，从【当前】参数算，不能用初值
    g = 2 * (opt_mine.params[0] - y_target)
    opt_mine.step([g])

    same = torch.allclose(x.detach(), opt_mine.params[0], atol=1e-6)
    print(f"step {step+1}: 官方={x.data.tolist()[:2]}  手动={opt_mine.params[0].tolist()[:2]}  一致={same}")
```

## 6. 小结

1. Adam = 动量（梯度一阶矩 EMA）+ RMSProp（梯度二阶矩 EMA）：一个管方向，一个管步幅。
2. 偏差修正 $\hat m_t=m_t/(1-\beta_1^t)$ 补偿零初始化在前几千步造成的低估，$t$ 大后自动失效。
3. AdamW 把权重衰减从梯度更新里解耦，正则化不受自适应步幅干扰，是现代标配。
4. 默认超参：$\beta_1=0.9$，$\beta_2=0.999$，$\epsilon=10^{-8}$；全局学习率仍需手调。

## 7. 思考题

**基础题**

1. 如果把 $\beta_1$ 设成 0（不要动量），Adam 的更新退化成什么？如果把 $\beta_2$ 设成 0 呢？

**进阶题**

2. 用几何级数求和，推导出偏差修正项 $(1-\beta_1^t)$ 的来历：假设 $g_i\equiv g$，证明 $m_t=g(1-\beta_1^t)$，并说明为什么除以 $(1-\beta_1^t)$ 就还原成了 $g$。

**开放题**

3. 为什么 SGD+动量在 CV 上泛化常常优于 Adam，而 Transformer 却几乎只用 Adam？请从"平坦 vs 尖锐极小值"和"收敛速度 vs 泛化"两个角度讨论。最新的某些优化器（如 Lion、Adafactor）又是在 Adam 的哪一点上做的改进？

## 8. 延伸阅读 / 进一步思考

- Adam 原始论文：Kingma & Ba, *Adam: A Method for Stochastic Optimization*, 2015。
- AdamW 与解耦权重衰减：Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, 2017。
- Adam 泛化问题：Wilson et al., *The Marginal Value of Adaptive Gradient Methods in Machine Learning*, 2017。
- 进阶优化器：Lion（Google，用符号动量）、Adafactor（节省二阶矩显存）、Sophia（二阶信息），可作为第二册内容。
