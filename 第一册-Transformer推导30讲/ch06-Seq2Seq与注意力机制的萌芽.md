# 第6讲 Seq2Seq与注意力机制的萌芽

> 📌 **本讲问题**：把一句话压缩成一个固定维度向量再解码，信息瓶颈有多严重？Bahdanau 注意力是怎么打破这个瓶颈的？从"固定向量"到"动态回看"，这一步为什么是 Transformer 的思想源头？

## 1. 数学工具/预备知识

**ch05 的 RNN/LSTM。** Encoder 和 Decoder 都是循环网络，状态沿时间步递推。

**对齐（alignment）。** 在机器翻译里，生成目标语言第 $s$ 个词时，应该重点参照源语言的哪个（哪些）词？这个对应关系就叫对齐。Bahdanau 之前对齐是隐式的（压在固定向量里），之后变成显式的概率分布。

**前馈网络作为打分函数。** 一个小的 MLP 输入两个向量、输出一个标量分数，就是一个可学习的"相似度函数"。

**softmax（ch04）。** 把一组分数变成和为 1 的权重。

## 2. 完整推导

### 2.1 Encoder-Decoder 的信息瓶颈

Seq2Seq 架构分两段。Encoder 把源序列 $\boldsymbol{x}_{1:T}$ 编码成一个固定向量 $\boldsymbol{c}$；Decoder 从 $\boldsymbol{c}$ 出发自回归生成目标序列 $\boldsymbol{y}_{1:S}$：

$$\boldsymbol{h}_t^{enc}=f_{enc}(\boldsymbol{h}_{t-1}^{enc},\boldsymbol{x}_t),\quad \boldsymbol{c}=\boldsymbol{h}_T^{enc}$$

$$p(y_s\mid y_{<s},\boldsymbol{c})=g_{dec}(\boldsymbol{h}_s^{dec},\boldsymbol{c}),\quad \boldsymbol{h}_s^{dec}=f_{dec}(\boldsymbol{h}_{s-1}^{dec},y_{s-1})$$

问题出在 $\boldsymbol{c}$：无论源序列多长，全部信息都要塞进一个 $\boldsymbol{c}\in\mathbb{R}^{d}$。当 $T$ 很大时，$\boldsymbol{c}$ 容量不足，翻译质量随句子变长急剧下降。这就是**信息瓶颈**。打个比方：让人把一整篇文章读完后合上书背诵全文，文章越长背错越多。

### 2.2 Bahdanau 加性注意力

Bahdanau 的突破：不再用固定向量 $\boldsymbol{c}$，而是让 Decoder 在每一步**动态地回看** Encoder 的全部隐藏状态，为每个目标位置生成一个专属上下文向量 $\boldsymbol{c}_s$。

设 Encoder 用双向 RNN，隐藏状态为 $\boldsymbol{h}_1,\dots,\boldsymbol{h}_T$（每个 $\boldsymbol{h}_t$ 已包含左右文信息）。Decoder 在第 $s$ 步的隐藏状态是 $\boldsymbol{s}_s$。三步：

**第一步：打分函数（加性）**

$$e_{st}=v^\top\tanh(\boldsymbol{W}_1\boldsymbol{s}_s+\boldsymbol{W}_2\boldsymbol{h}_t)$$

其中 $\boldsymbol{W}_1\in\mathbb{R}^{d_a\times d_s}$，$\boldsymbol{W}_2\in\mathbb{R}^{d_a\times d_h}$，$v\in\mathbb{R}^{d_a}$。这是一个小型前馈网络：输入"当前 Decoder 状态 $\boldsymbol{s}_s$"和"某个 Encoder 位置 $\boldsymbol{h}_t$"，输出标量分数 $e_{st}$。为什么用 $\tanh$ 而不是直接点积？因为它能建模 $\boldsymbol{s}_s$ 和 $\boldsymbol{h}_t$ 之间更复杂的非线性匹配，代价是参数更多。

**第二步：归一化**

$$\alpha_{st}=\frac{\exp(e_{st})}{\sum_{t'=1}^{T}\exp(e_{st'})}$$

对固定的 $s$，$\alpha_{s\cdot}$ 是一个关于源位置 $t$ 的概率分布（ch04 softmax）。

**第三步：加权求和得到上下文向量**

$$\boldsymbol{c}_s=\sum_{t=1}^{T}\alpha_{st}\boldsymbol{h}_t$$

这个 $\boldsymbol{c}_s$ 替代原来的固定 $\boldsymbol{c}$，和 $\boldsymbol{s}_s$ 拼接后输入 Decoder 下一步。关键区别：$\boldsymbol{c}_s$ 随 $s$ 变化——生成每个目标词时，注意力都重新分配一次。

### 2.3 Luong 乘性注意力

Luong 提出三种打分函数，比加性更省参数、更快：

1. **Dot product**：$e_{st}=\boldsymbol{s}_s^\top\boldsymbol{h}_t$（要求 $d_s=d_h$）。
2. **General**：$e_{st}=\boldsymbol{s}_s^\top\boldsymbol{W}\boldsymbol{h}_t$，$\boldsymbol{W}\in\mathbb{R}^{d_s\times d_h}$。
3. **Concat**：等价于 Bahdanau 加性注意力。

乘性注意力比加性快，因为点积可以写成一次矩阵乘法 $\boldsymbol{S}\boldsymbol{H}^\top$ 批量算出所有 $(s,t)$ 对的分数，而加性要逐对过 MLP。这直接导向了 ch08 的 Scaled Dot-Product Attention。

### 2.4 注意力权重的概率解释

$\alpha_{st}$ 是一个关于 $t$ 的概率分布（对固定 $s$）。它回答："Decoder 在生成第 $s$ 个词时，应该把多少注意力放在源序列的位置 $t$？"这个软分配（soft alignment）和硬选择（hard selection，只挑一个位置）不同——所有位置都按权重参与，权重可以接近 one-hot 但永远是混合。

这个概率视角后来被抽象为 Query-Key-Value 框架（ch07）：$\boldsymbol{s}_s$ 是 query，$\boldsymbol{h}_t$ 既是 key 也是 value。

## 3. 💡 直觉理解

> 💡 **直觉**：传统 Seq2Seq 像让人把整本书背下来再默写——背不下来就丢内容。Bahdanau 注意力像让人看着原书一页一页地翻译：每翻到目标位置，眼睛扫一遍原文，把目光聚焦在最相关的那几行上。$\alpha_{st}$ 就是眼球的注视分布。
>
> 加性注意力像"两个人各说一句话，第三个人综合判断相关度"；乘性注意力像"直接把两句话的向量对齐算分"——后者省掉了中间人，更快，但要求两边在同一个向量空间里可比。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为注意力是"新的网络层"。它本质是一个**带软检索的加权求和**，不是层、不是激活函数，而是一种计算范式。
>
> ⚠️ **误区二**：以为 Bahdanau 注意力补的是"上下文"。它用双向 RNN，每个 $\boldsymbol{h}_t$ 本来就含左右文；注意力补的是"Decoder 每一步需要聚焦不同源位置"的能力。
>
> ⚠️ **误区三**：以为 $\alpha_{st}$ 会均匀分布。实际训练好的模型，$\alpha_{st}$ 往往很尖锐——一个目标词只盯着一两个源词，这正是对齐关系。

## 5. 💻 代码实现

下面实现一个迷你 Bahdanau 加性注意力。注意维度广播：`dec_proj` 是 `(1, d_a)`，`enc_proj` 是 `(T, d_a)`，要各自 unsqueeze 成 `(1,1,d_a)` 和 `(1,T,d_a)` 才能相加。`torch.tanh` 是函数式接口，不要用已废弃的 `F.tanh`。

```python
import torch
import torch.nn.functional as F

# 迷你 Bahdanau 加性注意力
torch.manual_seed(0)

T, d_h, d_s, d_a = 5, 8, 6, 10  # 源长度=5, encoder维=8, decoder维=6, 中间维=10

# Encoder 隐藏状态 (batch=1, T, d_h)
encoder_out = torch.randn(1, T, d_h)
# Decoder 当前状态 (1, d_s)
decoder_state = torch.randn(1, d_s)

# 加性注意力参数
W1 = torch.randn(d_a, d_s)  # 作用于 decoder state
W2 = torch.randn(d_a, d_h)  # 作用于 encoder hidden
v = torch.randn(d_a)        # 最终打分向量

# 第一步：计算每个位置的分数 e_st
# decoder_state: (1, d_s) @ W1.T: (d_s, d_a) -> (1, d_a)
dec_proj = torch.tanh(decoder_state @ W1.T)          # (1, d_a)
# encoder_out[0]: (T, d_h) @ W2.T: (d_h, d_a) -> (T, d_a)
enc_proj = torch.tanh(encoder_out[0] @ W2.T)         # (T, d_a)

# 广播相加: dec_proj(1,1,d_a) + enc_proj(1,T,d_a) -> (1,T,d_a)
combined = torch.tanh(dec_proj.unsqueeze(1) + enc_proj.unsqueeze(0))  # (1, T, d_a)
scores = combined @ v                                 # (1, T)

# 第二步：softmax 归一化（沿 key 维 T）
alpha = F.softmax(scores, dim=-1)                     # (1, T)
print("注意力权重 alpha:", alpha.detach().round(decimals=4))
print("权重和:", alpha.sum().item())                 # 应为 1

# 第三步：加权求和得到上下文向量
# alpha.unsqueeze(1): (1,1,T) 乘 encoder_out: (1,T,d_h) -> (1,1,d_h)
context = torch.bmm(alpha.unsqueeze(1), encoder_out).squeeze(1)  # (1, d_h)
print("上下文向量 shape:", tuple(context.shape))
```

## 6. 小结

1. Encoder-Decoder 架构的固定向量 $\boldsymbol{c}$ 是信息瓶颈，长序列下性能崩塌。
2. Bahdanau 加性注意力：$e_{st}=v^\top\tanh(\boldsymbol{W}_1\boldsymbol{s}_s+\boldsymbol{W}_2\boldsymbol{h}_t)$，softmax 后加权求和得到动态上下文向量 $\boldsymbol{c}_s$。
3. Luong 乘性注意力（Dot/General/Concat）是加性注意力的高效变体，可批量矩阵乘法，直接导向 SDPA。
4. 注意力权重 $\alpha_{st}$ 是概率分布，回答"当前步该关注源序列哪个位置"，是软检索而非硬选择。

## 7. 延伸阅读/进一步思考

- Bahdanau, Cho, Bengio 2014《Neural Machine Translation by Jointly Learning to Align and Translate》是注意力的开山之作；Luong 等 2015《Effective Approaches to Attention-based Neural Machine Translation》系统对比了三种打分函数。
- 注意力最初是为了解决"长句子的翻译瓶颈"，后来被证明可以脱离 RNN 独立使用（ch07 的自注意力），这是思想上的关键一跳。
- 软对齐 vs 硬对齐：Xu 等 2015 把注意力用于图像描述生成，提出了 hard/soft attention 的对比，是视觉 Transformer 的先声。

## 8. 思考题

**基础题**

1. 加性注意力对每个 $(s,t)$ 对都要过一个小 MLP；乘性注意力直接矩阵乘。为什么乘性可以批量加速而加性不能？

**进阶题**

2. 如果 Encoder 用单向 RNN（而非双向），$\boldsymbol{h}_t$ 只含左侧信息。这对依赖"后文"的翻译会造成什么影响？
3. Bahdanau 注意力中 $\sum_t\alpha_{st}=1$。若源序列有 1000 个词，均匀分布时每位置权重 0.001——softmax 后如果分数差异不大，权重会摊得很平，这对上下文向量 $\boldsymbol{c}_s$ 有什么坏处？

**开放题**

4. Bahdanau 注意力里 Decoder 状态是 query、Encoder 隐藏状态既是 key 又是 value。ch07 会把 key 和 value 拆成两个不同的投影。你觉得"key 和 value 共用同一个向量"在什么情况下是个限制？（提示：一个向量既要负责"好不好匹配"又要负责"匹配后输出什么"，这两个角色能解耦吗？）
