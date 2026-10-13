# 第17讲 RLHF与DPO：KL正则化完整推导

> 📌 **本讲问题**：RLHF为什么要在奖励最大化的同时加一个KL惩罚项？DPO是怎么把强化学习问题变成一个监督学习问题的？KL正则化的理论依据是什么？为什么"对齐"本质上是一个约束优化问题？

## 1. 数学工具/预备知识

- KL散度：$D_{\mathrm{KL}}(p\|q)$ 度量两个分布的差异。
- 奖励模型：$r_\phi(x, y)$ 给文本对打分。
- PPO：近端策略优化，强化学习算法。
- DPO：直接偏好优化，把RLHF转化为监督学习。

### 1.1 RLHF的三阶段

RLHF（Reinforcement Learning from Human Feedback）分三个阶段：
1. **监督微调（SFT）**：用人工标注的高质量回复微调基础模型。
2. **奖励模型训练**：训练一个模型 $r_\phi(x, y)$ 来预测人类偏好。
3. **强化学习微调**：用PPO优化语言模型，最大化奖励同时加KL惩罚。

本讲重点在第三阶段的KL正则化。

## 2. 完整推导

### 2.1 为什么需要KL惩罚？

如果只最大化奖励 $r_\phi(x, y)$，语言模型会"钻空子"——生成奖励模型喜欢但质量差的文本（reward hacking）。

KL惩罚的作用：约束优化后的语言模型 $\pi_\theta(y|x)$ 不要偏离初始模型 $\pi_{\text{ref}}(y|x)$ 太远。

**目标函数**：

$$
\max_\theta \mathbb{E}_{(x,y)\sim \pi_\theta}\left[r_\phi(x, y)\right] - \beta\, \mathbb{E}_x\left[D_{\mathrm{KL}}(\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x))\right]
$$

**两项的含义**：
- 第一项：最大化奖励（生成人类喜欢的回复）。
- 第二项：最小化与参考模型的KL散度（不要偏离基础模型太远）。
- $\beta$ 控制权衡：$\beta$ 大 = 保守，$\beta$ 小 = 激进。

### 2.2 KL正则化的理论依据

**为什么KL惩罚能防止reward hacking**？因为reward hacking本质上是"模型找到了奖励模型的盲区"——在这些地方奖励模型给高分，但人类其实不喜欢。

KL惩罚强制模型在参考模型的"邻域"内优化——不允许跳到参考模型之外的"奇怪区域"。这就像在基础模型的能力范围内做微调，而不是重新造一个模型。

**信息论视角**：KL惩罚是一个"软约束"——它不是硬性禁止偏离，而是对偏离的程度施加代价。这和信息瓶颈的 $\beta$ 系数是同一个思想：在两个目标之间取折中。

### 2.3 DPO：从RL到监督学习

DPO（Direct Preference Optimization）的核心洞察：上面那个带KL惩罚的优化问题，有一个解析解。

**推导**：

上面的目标函数可以改写为：

$$
\max_\theta \mathbb{E}_{x}\left[\mathbb{E}_{y\sim\pi_\theta}\left[r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right]\right]
$$

对固定的 $x$，最优策略 $\pi_\theta^*(y|x)$ 是什么？

这是一个在所有分布上最大化 $\mathbb{E}[r - \beta\log(\pi/\pi_{\text{ref}})]$ 的问题。利用变分法或拉格朗日乘子，可以得到解析解：

$$
\pi_\theta^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r_\phi(x, y)\right)
$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r_\phi(x, y)/\beta)$ 是归一化常数。

**直觉**：最优策略 = 参考模型分布 × 指数化的奖励。奖励高的回复概率变大，奖励低的概率变小。

### 2.4 DPO损失函数

从上面的解析解，可以推出DPO的损失函数：

$$
\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

其中 $y_w$ 是人类偏好的好回复（winner），$y_l$ 是差回复（loser），$\sigma$ 是sigmoid函数。

**直觉**：DPO损失说的是——好回复相对于参考模型的对数概率比，应该比差回复的更大。换句话说，好回复的概率要相对提高，差回复的概率要相对降低。

**为什么DPO比RLHF简单**：因为DPO不需要训练奖励模型，也不需要跑PPO强化学习。它直接用偏好数据训练语言模型，把RL问题变成了监督学习问题。

### 2.5 DPO的推导细节

上面的解析解 $\pi_\theta^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp(r_\phi(x,y)/\beta)$ 是怎么来的？

**推导过程**：

目标函数：

$$
\max_\theta \mathbb{E}_{x}\left[\mathbb{E}_{y\sim\pi_\theta}\left[r_\phi(x, y) - \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right]\right]
$$

对固定的 $x$，这是一个在所有分布 $\pi_\theta(y|x)$ 上最大化的问题。用变分法或拉格朗日乘子：

构造拉格朗日函数：

$$
\mathcal{L}(\pi) = \sum_y \pi(y|x) r(x,y) - \beta \sum_y \pi(y|x) \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}
$$

对 $\pi(y|x)$ 求导并令其为0：

$$
\frac{\partial \mathcal{L}}{\partial \pi(y|x)} = r(x,y) - \beta \left(\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} + 1\right) = 0
$$

解得：

$$
\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} = \frac{r(x,y)}{\beta} - 1
$$

$$
\pi(y|x) = C \cdot \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)
$$

其中 $C$ 是归一化常数（保证 $\sum_y \pi(y|x) = 1$）。

**为什么这个推导重要**：它告诉你DPO的解析解是怎么来的——不是凭空猜的，而是从带KL惩罚的优化问题严格推导出来的。

### 2.5 DPO的直观理解

DPO的直觉：如果 $y_w$ 比 $y_l$ 好，那么 $p(y_w|x)/p_{\text{ref}}(y_w|x)$ 应该比 $p(y_l|x)/p_{\text{ref}}(y_l|x)$ 大。
- $\beta$ 控制"偏离参考模型"的程度。
- $\beta$ 大：偏离小，更保守。
- $\beta$ 小：偏离大，更激进。

**为什么DPO有效**：因为它直接在偏好数据上优化模型，不需要训练奖励模型，也不需要强化学习——简单直接。

**DPO和RLHF的对比**：
- RLHF：训练奖励模型 → PPO优化 → 复杂但灵活。
- DPO：直接在偏好数据上优化 → 简单但可能不如RLHF灵活。
- DPO是RLHF的"简化版"——牺牲了一些灵活性，换来了简单和稳定。

### 2.6 DPO的损失函数推导

从解析解可以推出DPO损失。

**第一步**：从解析解中解出奖励函数。

$$
\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} = \frac{1}{Z(x)} \exp\left(\frac{r(x,y)}{\beta}\right)
$$

取对数：

$$
\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} = \frac{r(x,y)}{\beta} - \log Z(x)
$$

**第二步**：好回复和差回复的奖励差。

$$
r(x, y_w) - r(x, y_l) = \beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right]
$$

$Z(x)$ 项抵消了。

**第三步**：偏好概率的logistic形式。

人类偏好"好回复优于差回复"的概率为：

$$
P(y_w \succ y_l | x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)
$$

代入奖励差的表达式：

$$
P(y_w \succ y_l | x) = \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right]\right)
$$

**第四步**：最大化偏好概率 = 最小化负对数似然。

$$
\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}\left[\log P(y_w \succ y_l | x)\right]
$$

这就是DPO损失的完整推导。

**为什么这个推导重要**：它告诉你DPO损失不是凭空设计的，而是从偏好概率的logistic模型严格推导出来的。

## 3. 💡 直觉理解

> 💡 **直觉**：RLHF就像"在基础模型的基础上微调，但不要走太远"。参考模型是你的"舒适区"，奖励模型告诉你"往哪个方向走更好"。KL惩罚就像一根橡皮筋——你走得越远，橡皮筋拉得越紧，把你往回拽。
>
> DPO的聪明之处在于：既然这个优化问题有解析解，为什么还要用PPO慢慢迭代？直接用偏好数据训练模型就行了。就像爬山——既然你知道山顶在哪里，为什么还要一步一步爬？直接跳过去。
>
> KL惩罚的本质是"防止模型跑偏"。没有KL惩罚，模型可能生成一堆看起来很像人类但其实很奇怪的文本——因为奖励模型只见过有限的数据，不知道这些奇怪文本是不好的。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：认为DPO不需要参考模型。错。DPO损失里有 $\pi_{\text{ref}}$，而且它很重要。参考模型是"锚点"，没有锚点模型会跑偏。
>
> **误区二**：认为DPO比RLHF好。DPO更简单，但在某些场景下（如在线RLHF），PPO仍然有优势。两者各有适用场景。
>
> **误区三**：认为 $\beta$ 越小越好。$\beta$ 太小意味着KL惩罚太弱，模型会跑偏。$\beta$ 太大意味着模型不动。最优的 $\beta$ 需要调。
>
> **误区四**：认为RLHF的KL惩罚是正则化。KL惩罚确实是一种正则化，但它的特殊之处在于它约束的是"与参考模型的距离"，不是权重大小。

### 4.1 DPO的实践建议

**$\beta$ 系数的选择**：
- $\beta$ 太大：偏离参考模型太少，改进有限。
- $\beta$ 太小：偏离参考模型太多，可能跑偏。
- 推荐范围：0.1 - 0.5。

**参考模型的选择**：
- 参考模型应该是一个"好的起点"——已经有不错的基础能力。
- 如果参考模型太差，DPO的改进空间有限。

**偏好数据的质量**：
- 偏好数据的质量直接决定DPO的效果。
- 噪声多的偏好数据会导致模型跑偏。

## 5. 💻 代码实现

实现一个简化版的DPO损失，对比不同 $\beta$ 的效果。

```python
import numpy as np
import matplotlib.pyplot as plt

def dpo_loss(logp_theta_w, logp_ref_w, logp_theta_l, logp_ref_l, beta=0.1):
    """
    简化版DPO损失。
    
    logp_theta_w: 当前模型对好回复的对数概率
    logp_ref_w: 参考模型对好回复的对数概率
    logp_theta_l: 当前模型对差回复的对数概率
    logp_ref_l: 参考模型对差回复的对数概率
    beta: KL系数
    """
    # log-ratio
    log_ratio_w = logp_theta_w - logp_ref_w
    log_ratio_l = logp_theta_l - logp_ref_l
    # DPO loss: -log sigmoid(beta * (log_ratio_w - log_ratio_l))
    diff = beta * (log_ratio_w - log_ratio_l)
    # 数值稳定的sigmoid
    loss = -np.log(1 / (1 + np.exp(-diff)) + 1e-10)
    return loss

# ========== 实验：不同beta的效果 ==========
print("=" * 60)
print("实验：DPO损失对模型偏移的影响")
print("=" * 60)

# 模拟：好回复和差回复的对数概率差
diffs = np.linspace(-2, 2, 100)

betas = [0.01, 0.1, 0.5, 1.0]
colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

plt.figure(figsize=(10, 5))
for beta, c in zip(betas, colors):
    losses = [-np.log(1 / (1 + np.exp(-beta * d)) + 1e-10) for d in diffs]
    plt.plot(diffs, losses, color=c, linewidth=2, label=f'β = {beta}')

plt.xlabel('log-ratio差 (好回复 - 差回复)')
plt.ylabel('DPO损失')
plt.title('不同β值下的DPO损失曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('ch17_dpo_loss.png', dpi=100)
print("图已保存为 ch17_dpo_loss.png")
print("\n观察：β越大，损失曲线越陡峭（模型更容易偏移）")
print("β越小，损失曲线越平缓（模型更保守）")
```

## 6. 小结

1. RLHF的第三阶段用PPO优化语言模型，目标是最大化奖励同时加KL惩罚。
2. KL惩罚约束模型不要偏离参考模型太远，防止reward hacking。
3. DPO把带KL惩罚的RL问题转化为解析解，直接用偏好数据训练。
4. DPO损失：好回复相对于参考模型的对数概率比，应该比差回复的更大。
5. DPO比RLHF简单，但参考模型仍然是必要的"锚点"。

## 7. 思考题

### 基础档（巩固概念）

1. RLHF的三个阶段是什么？
2. 为什么需要KL惩罚？
3. DPO损失函数的形式是什么？

### 进阶档（推导与分析）

4. 为什么DPO不需要训练奖励模型？它从哪里得到"好回复"和"差回复"的信号？
5. 如果 $\beta \to 0$，DPO会退化成什么？$\beta \to \infty$ 呢？
6. DPO和RLHF的KL惩罚在数学上是等价的——这个等价性的前提是什么？

### 开放档（联系研究）

7. 大模型的"对齐"问题可以用信息瓶颈分析吗？对齐是在压缩什么、保留什么？
8. 为什么RLHF会导致"对齐税"（模型在某些任务上性能下降）？用KL惩罚的直觉解释。

## 8. 延伸阅读/进一步思考

- **Christiano et al. (2017)**，*Deep Reinforcement Learning from Human Preferences*。RLHF的早期工作。
- **Ouyang et al. (2022)**，*Training Language Models to Follow Instructions with Human Feedback*（InstructGPT）。RLHF的经典论文。
- **Rafailov et al. (2023)**，*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*。DPO的原始论文。
- **Ziegler et al. (2023)**，*DPO vs RLHF: A Comparison*。DPO和RLHF的对比研究。
