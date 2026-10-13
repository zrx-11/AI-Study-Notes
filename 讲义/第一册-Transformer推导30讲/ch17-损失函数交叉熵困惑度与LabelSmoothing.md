# 第17讲 损失函数：交叉熵、困惑度与 Label Smoothing

> 📌 **本讲问题**：语言模型的损失为什么是交叉熵而不是 MSE？"困惑度"到底在量什么？Label Smoothing 凭什么能泛化变好，它又是怎么悄悄给 logits 戴了个紧箍咒的？

## 1. 数学工具/预备知识

**（1）熵与交叉熵的定义。** 对一个真实分布 $p$ 和一个模型给出的分布 $q$，离散熵 $H(p)=-\sum_k p_k\log p_k$，交叉熵 $H(p,q)=-\sum_k p_k\log q_k$。它们之间差一个 KL 散度：$H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q)$。ch02 已经建立过这层关系，本讲把它用到语言模型上。

**（2）对数似然。** 模型给第 $t$ 个真实 token 分配的概率是 $q(x_t)$。看到一串文本 $x_{1:T}$，"这串文本在模型眼里有多合理"就等于它的联合概率 $\prod_t q(x_t)$。我们希望这个乘积越大越好——这就是最大似然估计（MLE）的出发点。

**（3）softmax 的梯度（ch04）。** 对 logits $\boldsymbol{z}=(z_1,\dots,z_V)$，softmax 输出 $q_k=e^{z_k}/\sum_j e^{z_j}$。本讲最关键的一条结论会反复用到：**交叉熵对 logits 的梯度恰好是 $q-\boldsymbol{y}$**，即预测分布减去真实标签。我们会在 2.2 把它推出来。

**（4）one-hot 向量。** 真实 token 是词汇表里的某一个，我们用一个长度 $V$ 的向量 $\boldsymbol{y}$ 表示：正确位置是 1，其余全是 0。这叫 one-hot 标签。

## 2. 完整推导

### 2.1 从最大似然到交叉熵

位置 $t$ 处，模型根据前文 $x_{<t}$ 输出一组 logits $z_{1:V}$，经 softmax 得到概率分布 $q_t=\mathrm{softmax}(z_{1:V})$。真实 token 记为 $x_t$，它在词汇表里的序号对应的 one-hot 标签是 $\boldsymbol{y}_t$。

整段文本的似然是各位置预测概率的连乘：

$$
P(x_{1:T})=\prod_{t=1}^{T} q_t(x_t\mid x_{<t})
$$

直接乘很多个小于 1 的数会下溢，所以取对数（对数单调递增，最大化对数似然等价）：

$$
\log P(x_{1:T})=\sum_{t=1}^{T}\log q_t(x_t)
$$

我们要最大化它，等价于最小化它的负数。再除以 $T$ 做平均，得到标准的训练损失：

$$
\mathcal{L}=-\frac{1}{T}\sum_{t=1}^{T}\log q_t(x_t)
$$

这正是 one-hot 标签下的交叉熵。写成向量形式更清楚：因为 $q_t(x_t)=q_{t,k^\ast}$（$k^\ast$ 是正确类），而 one-hot 标签 $\boldsymbol{y}_t$ 只在 $k^\ast$ 处为 1，所以 $\sum_k y_{t,k}\log q_{t,k}=\log q_{t,k^\ast}$。于是：

$$
\mathcal{L}=-\frac{1}{T}\sum_{t=1}^{T}\sum_{k=1}^{V} y_{t,k}\log q_{t,k}
$$

**为什么不用 MSE？** MSE 假设输出是连续值、误差服从高斯分布；而这里我们要的是一个概率分布，真实标签是离散类别。交叉熵直接度量"模型把概率分配给正确类的效率"，梯度干净，且在分类问题上不会像 MSE 那样被 softmax 饱和拖慢学习。

### 2.2 交叉熵对 logits 的梯度：$q-\boldsymbol{y}$

这是本讲最重要的一步，推出来以后很多现象都会变得一目了然。

单个样本的损失 $\mathcal{L}=-\sum_k y_k\log q_k$，而 $q_k=e^{z_k}/\sum_j e^{z_j}$。对任意一个 logit $z_i$ 求偏导：

$$
\frac{\partial \mathcal{L}}{\partial z_i}
=-\sum_k y_k\frac{\partial \log q_k}{\partial z_i}
$$

用 softmax 的求导公式 $\partial q_k/\partial z_i=q_k(\delta_{ki}-q_i)$（ch04 推过），即 $\partial \log q_k/\partial z_i=\delta_{ki}-q_i$。代进去：

$$
\frac{\partial \mathcal{L}}{\partial z_i}
=-\sum_k y_k(\delta_{ki}-q_i)
=-y_i+\sum_k y_k q_i=-y_i+q_i\underbrace{\sum_k y_k}_{=1}=q_i-y_i
$$

写成向量就是 $\nabla_{\boldsymbol{z}}\mathcal{L}=\boldsymbol{q}-\boldsymbol{y}$。

**这个结果为什么重要？** 它说明：训练时 logits 收到的"推力"正好等于"预测概率减去标签概率"。如果模型对正确类预测了 $q_{k^\ast}=0.8$，而标签是 1，那这一项的梯度就是 $0.8-1=-0.2$——把正确类 logit 往上推；预测错得越离谱，梯度越大，推得越狠。这个形式干净、尺度适中，是交叉熵适合梯度下降的根本原因。

### 2.3 困惑度：把损失翻译成"犹豫程度"

困惑度（Perplexity, PPL）定义为损失的指数：

$$
\mathrm{PPL}=\exp(\mathcal{L})=\exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log q_t(x_t)\right)
$$

把指数和对数交换一下（$e^{-\frac{1}{T}\sum\log q}=\prod q^{-1/T}$）：

$$
\mathrm{PPL}=\left(\prod_{t=1}^{T}\frac{1}{q_t(x_t)}\right)^{1/T}
$$

这是所有 token 预测概率倒数的**几何平均**。它的直观含义是：**模型在每个位置上，平均相当于从多少个等概率选项里做选择。**

- PPL $=1$：每个 token 都被预测概率 1，模型"胸有成竹"。
- PPL $=V$：模型在 $V$ 个词上均匀分布（$q_t(x_t)=1/V$，$\mathcal{L}=\log V$），等于瞎猜。
- PPL 越小越好。

**为什么用 PPL 而不是直接报 CE？** 因为 CE 取了对数，读者对"损失 5"没有感觉；指数还原成"在多少个候选词之间犹豫"就直观了。而且 PPL 可以跨数据集、跨人比较——CE 依赖词表大小，PPL 不依赖。

**几何意义**：CE 是在"对数概率空间"里量平均距离，PPL 是把它映射回原始概率尺度。CE 每降 1，PPL 就除以 $e$——这是一个乘法关系，不是加法关系。

### 2.4 Label Smoothing：把 one-hot 变软

原始标签是纯 one-hot：正确类 $y_{k^\ast}=1$，其余 0。Label Smoothing 把它改成一个软目标：

$$
y'_k=(1-\epsilon)\,y_k+\frac{\epsilon}{V}
$$

即正确类保留 $(1-\epsilon)$ 的概率质量，剩下 $\epsilon$ 的质量均匀撒给全部 $V$ 个类。原论文取 $\epsilon=0.1$。新标签仍然是个合法分布（所有分量加起来等于 1，验证一下：$(1-\epsilon)\cdot 1+\epsilon/V\cdot V=1$）。

**带 Label Smoothing 的损失**：

$$
\mathcal{L}_{\mathrm{LS}}=-\sum_k y'_k\log q_k
$$

### 2.5 为什么 Label Smoothing 能正则化 logits 尺度

先回答一个问题：交叉熵损失 $\mathcal{L}=-\sum_k y'_k\log q_k$ 关于 $q$ 的最小值在哪？由于 $-\log$ 是严格凸函数，最小值在 $q_k=y'_k$ 处取得（用 2.2 的梯度结论：$\nabla_{z}\mathcal{L}=q-y'$，令它为 0 即 $q=y'$）。

也就是说，模型被推向的最优输出是"正确类给 $1-\epsilon$、其他类各给 $\epsilon/V$"，而不是"正确类给 1、其他给 0"。

这直接掐住了 logits 的脖子。假设没有 Label Smoothing，最优解是 $q_{k^\ast}\to 1$，这要求正确类 logit $z_{k^\ast}\to+\infty$、其余 $\to-\infty$——logits 的尺度可以无限涨大。softmax 在极端 logits 下会饱和（ch04），梯度趋近 0，而且大 logit 对输入扰动极度敏感，泛化差。

加了 Label Smoothing 后，目标本身就要求 $q_{k^\ast}=1-\epsilon<1$。于是当 $z_{k^\ast}$ 大到使 $q_{k^\ast}=1-\epsilon$ 时，梯度 $q_{k^\ast}-y'_{k^\ast}=0$，损失停止下降——**logits 不再被推往无穷大**。这就是它对 logits 尺度的隐式正则化：不是显式加一个权重衰减项，而是通过改变目标分布，从根上杜绝了过度自信。

**几何/物理意义**：one-hot 目标把模型拽向 softmax 曲面的"角落"（某个分量为 1，其余为 0），角落处梯度为零且曲面尖锐；Label Smoothing 把目标移到角落往内一点的位置，模型收敛到一个更"圆润"的点，对输入扰动更鲁棒。

### 2.6 边界讨论：为什么大模型后来不用了

Label Smoothing 早期（Transformer 翻译模型）用得很多。但 GPT-3 之后的很多大模型训练**关掉了它**。原因是：Label Smoothing 的本质是一种"人为注入不确定性"的正则化，它在数据少、模型容易过拟合时有用；当数据量极大、模型本身就欠拟合时，再把标签软化反而会干扰模型去拟合真实的类别结构，白白损失一点拟合能力。所以它是一个"看数据量下菜碟"的正则化开关。

## 3. 💡 直觉理解

> 💡 **直觉**：交叉熵是"猜错的代价"，猜错越离谱罚得越狠；困惑度是这个代价的指数版——代价翻一倍，犹豫的候选词数翻 $e$ 倍。Label Smoothing 像老师改卷时说："即使你确定答案是 A，也给 B、C、D 各留 0.1 分情面。"这么做不是纵容错误，而是防止学生把所有宝全押在一个选项上、押到最后一被反问就崩盘。它教会学生"其他选项也各有几分道理"，考试时更稳。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 PPL 能压到 1。PPL 的下限不是 1，而是数据本身的熵——人类写的文本本身就有随机性，同一个上文可以接很多合理的下一个词，完美模型也预测不出"唯一正确答案"。
>
> **误区二**：以为 Label Smoothing 是给数据加噪声。它不是随机扰动，而是**改变了目标分布本身**——从 one-hot 变成软分布，数学上等价于一个正则项，是确定性的。
>
> **误区三**：以为 CE 越小越好到无上限。CE 低到一定程度就说明模型开始"记住"训练集了，泛化未必跟着变好——这正是 Label Smoothing 要防的。

## 5. 💻 代码实现

下面把 2.5 的结论亲手验证一遍：(1) 手动实现 Label Smoothing 并和官方 `cross_entropy(..., label_smoothing=...)` 对拍；(2) 验证梯度 $q-y$；(3) 看 PPL 随 CE 怎么变。

```python
import torch
import torch.nn.functional as F

# ---------- (1) 手动 Label Smoothing 与官方对拍 ----------
def label_smoothed_ce(logits, target, eps=0.1, vocab_size=1000):
    # logits: (batch, V), target: (batch,)
    log_probs = F.log_softmax(logits, dim=-1)
    # 正确类那一项: -log q_{k*}
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
    # 所有类别平均 log prob（即把 eps 的质量均匀分给所有类）
    smooth_loss = -log_probs.mean(dim=-1)
    # 混合: (1-eps)*正确项 + eps*所有项平均
    loss = (1 - eps) * nll_loss + eps * smooth_loss
    return loss.mean()

torch.manual_seed(0)
logits = torch.randn(32, 1000)
target = torch.randint(0, 1000, (32,))

my_loss = label_smoothed_ce(logits, target, eps=0.1)
official_loss = F.cross_entropy(logits, target, label_smoothing=0.1)
print(f"手动 LS loss:   {my_loss.item():.4f}")
print(f"官方 LS loss:   {official_loss.item():.4f}")
print(f"两者一致: {torch.allclose(my_loss, official_loss, atol=1e-5)}")

# ---------- (2) 验证梯度 = q - y ----------
z = torch.randn(5, requires_grad=True)
y_idx = torch.tensor([2])
loss = F.cross_entropy(z.unsqueeze(0), y_idx)
loss.backward()
q = F.softmax(z, dim=0).detach()
y_onehot = torch.zeros(5); y_onehot[2] = 1.0
print(f"\n梯度 q - y: {z.grad.tolist()}")
print(f"q - y 手算: {(q - y_onehot).tolist()}")
print(f"梯度一致: {torch.allclose(z.grad, q - y_onehot, atol=1e-6)}")

# ---------- (3) PPL 与 CE 的指数关系 ----------
for ce in [6.9, 5.0, 4.0, 2.3]:
    print(f"CE={ce:.1f} -> PPL={torch.exp(torch.tensor(ce)).item():.1f}")
print(f"均匀分布(V=1000)的 PPL = {torch.exp(torch.log(torch.tensor(1000.0))).item():.0f}")
```

## 6. 小结

1. 语言模型损失 = 对 one-hot 标签的交叉熵 = 负对数似然，它来自最大似然估计。
2. 交叉熵对 logits 的梯度恰好是 $q-\boldsymbol{y}$——预测减标签，这个干净形式是它好用的根源。
3. 困惑度 $=\exp(\mathrm{CE})$，是"平均在多少个等概率候选间犹豫"，越小越好，下限是数据自身熵。
4. Label Smoothing 把 one-hot 软化为 $(1-\epsilon)$ 正确类 $+\epsilon/V$ 均匀分布，从目标分布上掐住 logits 尺度，防止过度自信。
5. Label Smoothing 是正则化开关：小数据开，超大数据可关。

## 7. 思考题

**基础题**

1. 如果把 $\epsilon$ 设成 1.0，Label Smoothing 的目标分布变成什么？模型还学得到正确类吗？

**进阶题**

2. CE 从 6.9 降到 4.0，PPL 从多少降到多少？请用 $\mathrm{PPL}=\exp(\mathrm{CE})$ 的关系手算，并解释为什么说"CE 是对数尺度上的损失"。

**开放题**

3. 大模型（如 GPT-3、LLaMA）训练时普遍关掉了 Label Smoothing。请结合"正则化是用来对抗过拟合的"这一原理，论证：当数据量远大于模型容量时，为什么关掉它反而更好？

## 8. 延伸阅读 / 进一步思考

- Label Smoothing 原始动机：Szegedy et al., *Rethinking the Inception Architecture for Computer Vision*, 2016（首次系统提出）。
- 为什么大模型不用 LS：Chinchilla、LLaMA 系列技术报告里关于正则化选择的讨论。
- 温度 softmax 与蒸馏里的温度是同一个量，留到 ch26 知识蒸馏再深入。
- 想继续挖：熵正则化（entropy regularization）、 focal loss 等"改损失改标签"的一族方法。
