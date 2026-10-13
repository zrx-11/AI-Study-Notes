# 第7讲 Query-Key-Value：注意力的一般化抽象

> 📌 **本讲问题**：Bahdanau 注意力里只有一个"查询"（Decoder 状态）和一堆"被查内容"（Encoder 隐藏状态）。怎么把它抽象成数据库检索的三要素 Q/K/V？为什么这个抽象是 Transformer 的灵魂？

## 1. 数学工具/预备知识

**ch06 的 Bahdanau/Luong 注意力。** 已知打分 $e_{st}$、softmax 归一化、加权求和三步。

**数据库检索直觉。** 检索系统三要素：query（你输入的搜索词）、key（每条文档的索引/标签）、value（文档正文）。匹配度由 query 和 key 决定，返回的是 value。

**概率分布与期望。** softmax 输出 $\alpha_i$ 是概率分布，输出 $\sum_i\alpha_i\boldsymbol{v}_i$ 是 value 的期望。

## 2. 完整推导

### 2.1 从数据库检索到数学抽象

数据库检索流程：输入 query，系统把每个文档的 key 和 query 匹配，匹配度高的排前面，返回这些文档的 value。翻译成数学：

- **Query** $\boldsymbol{q}$：当前需要查询的信息。
- **Key** $\boldsymbol{k}_1,\dots,\boldsymbol{k}_n$：每个被检索对象的索引特征。
- **Value** $\boldsymbol{v}_1,\dots,\boldsymbol{v}_n$：每个被检索对象的实际内容。

匹配函数 $s(\boldsymbol{q},\boldsymbol{k}_i)$ 衡量 query 和第 $i$ 个 key 的相似度，经过 softmax 变成权重 $\alpha_i$，输出是 value 的加权期望：

$$\mathrm{Attn}(\boldsymbol{q},\mathcal{K},\mathcal{V})=\sum_{i=1}^{n}\alpha_i\boldsymbol{v}_i,\quad \alpha_i=\frac{\exp(s(\boldsymbol{q},\boldsymbol{k}_i))}{\sum_j\exp(s(\boldsymbol{q},\boldsymbol{k}_j))}$$

这就是注意力的最一般形式。ch06 的 Bahdanau 是它的一个特例：$\boldsymbol{q}=\boldsymbol{s}_s$，$\boldsymbol{k}_i=\boldsymbol{v}_i=\boldsymbol{h}_i$，$s$ 是加性 MLP。

### 2.2 为什么需要三套不同的投影

在 Bahdanau 中，Decoder 状态直接当 query，Encoder 隐藏状态直接当 key 和 value。但这有个问题：query、key、value 若都来自同一个未变换的表征空间，表达能力受限——因为"该怎么匹配"和"匹配后输出什么"这两件事被绑死在同一个向量里。

Transformer 引入三个可学习的投影矩阵，把同一输入分别投到三个子空间：

$$\boldsymbol{Q}=\boldsymbol{X}\boldsymbol{W}_Q,\quad \boldsymbol{K}=\boldsymbol{X}\boldsymbol{W}_K,\quad \boldsymbol{V}=\boldsymbol{X}\boldsymbol{W}_V$$

- Self-Attention：$\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}$ 都来自同一个输入序列 $\boldsymbol{X}$。
- Cross-Attention：$\boldsymbol{Q}$ 来自 Decoder，$\boldsymbol{K},\boldsymbol{V}$ 来自 Encoder。

这个设计的好处：

- Q/K 投影学"该怎么匹配"，V 投影学"匹配后提取什么内容"，三者解耦。
- 同一个 token 向量，在不同投影下扮演不同角色（它既是 query 又是 key 又是 value）。
- 维度可不同：Q/K 维度 $d_k$，V 维度 $d_v$（实际 Transformer 中通常 $d_k=d_v=d_{\text{model}}/h$，见 ch09）。

### 2.3 Self-Attention vs Cross-Attention

**Self-Attention**：Q、K、V 来自同一序列。输入 $\boldsymbol{X}\in\mathbb{R}^{n\times d}$，则 $\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V}\in\mathbb{R}^{n\times d_k}$（投影后）。每个位置可以直接关注序列中任何其他位置——不管距离多远，路径长度都是 1。这直接解决了 ch05 RNN 的长距离路径问题。

**Cross-Attention**：Q 来自 Decoder 上一层输出，K/V 来自 Encoder 最终输出。这对应 Bahdanau 的原始设定：Decoder 查询 Encoder 的记忆。

### 2.4 注意力权重的概率解释

$\alpha_i$ 是概率分布：非负、和为 1。注意力输出 $\sum_i\alpha_i\boldsymbol{v}_i$ 就是 value 的期望。这个视角让我们能用概率工具分析注意力：

- 权重熵高 → 分布均匀 → "广泛关注"，对所有 key 一视同仁。
- 权重熵低 → 分布尖锐 → "聚焦于少数几个位置"。

训练好的语言模型里，不同位置、不同层的注意力熵差异很大，对应不同的注意力模式。

### 2.5 通用 attention 函数的形式

把打分函数 $s$ 一般化：

$$\mathrm{Attention}(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})=\mathrm{softmax}\!\left(\frac{s(\boldsymbol{Q},\boldsymbol{K})}{\sqrt{d_k}}\right)\boldsymbol{V}$$

不同的 $s$ 对应不同变体：

- $s(\boldsymbol{q},\boldsymbol{k})=\boldsymbol{q}^\top\boldsymbol{k}$ → 缩放点积注意力（ch08）。
- $s(\boldsymbol{q},\boldsymbol{k})=v^\top\tanh(\boldsymbol{W}_1\boldsymbol{q}+\boldsymbol{W}_2\boldsymbol{k})$ → 加性注意力（ch06）。

除以 $\sqrt{d_k}$ 的原因 ch08 专门推导，这里先记住它是为了防止打分过大导致 softmax 饱和。

## 3. 💡 直觉理解

> 💡 **直觉**：Q/K/V 就像图书馆检索——Q 是你要找的书的主题，K 是每本书的书名标签，V 是书的内容。你先把每个书名和你的主题做匹配（打分），匹配度高的书你多看几页（权重高），最后把所有书的内容按阅读比例混合成一份笔记输出。
>
> Self-Attention 就是图书馆里每本书都在同时检索其他所有书——"我是一本讲猫的书，我想和讲动物学、讲宠物、讲生物学的书都聊聊"。Cross-Attention 是你拿着自己的笔记（Q）去图书馆书架（K/V）里找参考资料。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 Q/K/V 是三个不同的输入。它们来自同一个输入 $\boldsymbol{X}$，只是通过三个不同投影矩阵变换到了三个不同子空间。
>
> ⚠️ **误区二**：以为注意力"选择"了最相关的 value。它不是 hard selection（选一个），而是 soft mixture（所有 value 都参与，按权重混合）。权重可以接近 one-hot，但数学上始终是加权平均。
>
> ⚠️ **误区三**：以为投影矩阵 $\boldsymbol{W}_Q,\boldsymbol{W}_K,\boldsymbol{W}_V$ 是多余的。没有它们，query 和 key 被锁死在原始 token 空间，模型学不到"该用什么标准衡量匹配"。

## 5. 💻 代码实现

下面实现一个通用注意力函数，支持 dot 和 additive 两种打分。工程注意：additive 分支里的投影向量应当是传入的参数而不是函数内随机生成（否则每次调用都不一样，无法训练）；`masked_fill` 把被屏蔽位置填成大负数，配合 ch04 的 softmax 产生近零权重。

```python
import torch
import torch.nn.functional as F

def general_attention(Q, K, V, score='dot', scale=None, mask=None):
    """
    Q: (..., n_q, d_k)
    K: (..., n_k, d_k)
    V: (..., n_k, d_v)
    score: 'dot' 或 'additive'
    """
    if score == 'dot':
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (..., n_q, n_k)
        if scale is not None:
            scores = scores / scale
    elif score == 'additive':
        # 简化版：Q.unsqueeze(-2) + K.unsqueeze(-3) 后过 tanh，再和可学向量 w 做点积
        d_k = Q.size(-1)
        w = torch.randn(d_k, device=Q.device) * 0.1  # 演示用，实际应作为参数传入
        combined = torch.tanh(Q.unsqueeze(-2) + K.unsqueeze(-3))  # (..., n_q, n_k, d_k)
        scores = combined @ w  # (..., n_q, n_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    alpha = F.softmax(scores, dim=-1)  # (..., n_q, n_k)
    output = torch.matmul(alpha, V)    # (..., n_q, d_v)
    return output, alpha

# 测试 Self-Attention
torch.manual_seed(0)
X = torch.randn(1, 4, 8)  # batch=1, seq=4, dim=8
W_Q = torch.randn(8, 8) * 0.1
W_K = torch.randn(8, 8) * 0.1
W_V = torch.randn(8, 8) * 0.1

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

out, alpha = general_attention(Q, K, V, score='dot')
print("Self-Attention 输出 shape:", tuple(out.shape))
print("注意力权重矩阵（每行和为1）:")
print(alpha[0].detach().round(decimals=3))
print("每行权重和:", alpha[0].sum(dim=-1).detach().round(decimals=4).tolist())
```

## 6. 小结

1. Q/K/V 抽象来自数据库检索：Q 是查询，K 是索引，V 是内容；通用公式 $\mathrm{softmax}(s(Q,K))V$。
2. 三个投影矩阵把"怎么匹配"（Q/K）和"提取什么"（V）解耦，三者来自同一输入但处于不同子空间。
3. Self-Attention 中 Q/K/V 来自同一序列，任意两位置路径长度为 1；Cross-Attention 中 Q 来自 Decoder、K/V 来自 Encoder。
4. 注意力输出是 value 的加权期望，权重是概率分布——软检索而非硬选择。

## 7. 延伸阅读/进一步思考

- Q/K/V 三要素的明确区分见 Vaswani 等 2017《Attention Is All You Need》（ch08/ch09 的主线）；在那之前 Bahdanau 版本里 key 和 value 是同一个向量。
- 想理解"为什么把 key 和 value 拆开是重要一步"，可对比：Dense Retrieval / 双塔模型里 query 塔和 doc 塔也是分开投影的，思想同源。
- 注意力权重的可解释性研究（"attention is not explanation"之争）提醒我们：高注意力权重不等于因果重要性，ch29 可视化时会再讨论。

## 8. 思考题

**基础题**

1. 如果把 Q 和 K 投影到同一个空间（即 $\boldsymbol{W}_Q=\boldsymbol{W}_K$），从"匹配标准"角度看会损失什么？

**进阶题**

2. 注意力权重的熵越高，说明模型在做什么？熵越低呢？分别对应什么语义场景？
3. 为什么 Cross-Attention 中 Q 来自 Decoder 而 K/V 来自 Encoder？反过来（Q 来自 Encoder、K/V 来自 Decoder）对应什么任务？

**开放题**

4. 通用注意力公式里打分函数 $s$ 可以替换成任意可学习函数。除了点积和加性，你能想到第三种打分函数吗？（提示：考虑余弦相似度、核方法、或把 key 拆成多块。）这种替换会如何影响 ch08 的缩放分析？
