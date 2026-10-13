# 第10讲 Transformer总体架构：Encoder-Decoder数据流全览

> 📌 **本讲问题**：把前面九讲的零件组装起来，一个完整的 Transformer 长什么样？数据从 token 到 logits 的每一步维度变化是什么？训练和推理的数据流有什么本质区别？

## 1. 数学工具/预备知识

**ch09 的 Multi-Head Attention。** Self-Attention 和 Cross-Attention 的区别只在 Q/K/V 从哪来。

**残差连接与 LayerNorm（ch12 详解）。** 每个子层后都跟一个 `x + Sublayer(x)` 再加 LayerNorm，保证深层网络梯度能流通。

**Teacher Forcing。** 训练 Decoder 时，输入不是模型自己上一步的预测，而是真实的上一步 token——这样所有位置可以并行算损失。

## 2. 完整推导

### 2.1 原论文架构

Transformer 由 $N=6$ 个 Encoder 层和 $N=6$ 个 Decoder 层堆叠而成。

**Encoder 每层两个子层：**
1. Multi-Head Self-Attention（残差 + LayerNorm）
2. Position-wise Feed-Forward Network（残差 + LayerNorm）

**Decoder 每层三个子层：**
1. Masked Multi-Head Self-Attention（残差 + LayerNorm，禁止看到未来位置，ch14）
2. Encoder-Decoder Cross-Attention（Q 来自 Decoder，K/V 来自 Encoder）
3. Position-wise Feed-Forward Network（残差 + LayerNorm）

### 2.2 逐子层维度账本

设 $d_{\text{model}}=512$，$d_{ff}=2048$，$n_{\text{heads}}=8$，$d_k=64$。

**输入嵌入**：token ID 查表。输入 $\boldsymbol{X}_{\text{id}}\in\mathbb{R}^{n\times 1}$（整数 token IDs），乘嵌入矩阵 $\boldsymbol{E}\in\mathbb{R}^{V\times d_{\text{model}}$ 变成 $\boldsymbol{X}\in\mathbb{R}^{n\times 512}$。

**加位置编码**：$\boldsymbol{X}=\boldsymbol{X}_{\text{embed}}+\boldsymbol{P}_{\text{pos}}$（ch11），形状不变。

**Encoder 第 $l$ 层：**
- Self-Attention：$n\times 512 \to n\times 512$
- Add & Norm：$n\times 512$
- FFN：$n\times 512 \to n\times 2048 \to n\times 512$
- Add & Norm：$n\times 512$

6 层后 Encoder 输出 $\boldsymbol{Z}\in\mathbb{R}^{n\times 512}$。

**Decoder 第 $l$ 层：**
- Masked Self-Attention：$m\times 512 \to m\times 512$
- Add & Norm：$m\times 512$
- Cross-Attention：Q 是 $m\times 512$，K/V 来自 Encoder 输出 $n\times 512$，输出 $m\times 512$
- Add & Norm：$m\times 512$
- FFN：$m\times 512 \to m\times 2048 \to m\times 512$
- Add & Norm：$m\times 512$

6 层后 Decoder 输出 $\boldsymbol{D}\in\mathbb{R}^{m\times 512}$。

**输出投影**：$\boldsymbol{D}\in\mathbb{R}^{m\times 512}$ 乘 $\boldsymbol{W}_{\text{out}}\in\mathbb{R}^{512\times V}$ 得到 logits $\in\mathbb{R}^{m\times V}$，逐位置 softmax 得到下一个 token 概率。

### 2.3 训练：Teacher Forcing

训练时 Decoder 输入是**完整目标序列右移一位**：`<bos> y1 y2 ... y_{m-1}`。Self-Attention 上加 Causal Mask（ch14），保证位置 $t$ 只能看到 $\leq t$。

损失：对每个位置 $t$，预测 $y_t$，用交叉熵（ch02）。

Teacher Forcing 的含义：即使模型在位置 $t$ 预测错了，位置 $t+1$ 的输入仍然是**真实的 $y_t$**，而不是模型自己预测的 $\hat{y}_t$。这让训练可以并行（所有位置同时算损失），代价是 train-test 不一致（推理时模型看到的是自己的错误预测，ch29 讨论 exposure bias）。

### 2.4 推理：自回归

推理时没有完整目标序列，必须一步一步生成：
1. 输入 `<bos>`，得到第一个 token 概率分布，采样/贪心得 $\hat{y}_1$。
2. 把 $\hat{y}_1$ 接在输入后面，得到第二个 token。
3. 重复直到 `<eos>` 或达到最大长度。

每一步都要重新过所有层的注意力——这就是 KV Cache（ch23）要解决的问题。

## 3. 💡 直觉理解

> 💡 **直觉**：Encoder 像一个读完原文后做笔记的人——把整句话压成一组向量表示。Decoder 像边看笔记边翻译的人——每翻一个词，先回顾自己已经翻了什么（Self-Attention），再看原文笔记（Cross-Attention），然后输出下一个词。训练时是看着答案翻译（Teacher Forcing 并行），推理时是自己边想边翻（自回归串行）。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 Decoder 的 Cross-Attention 也是 Self-Attention。Cross-Attention 中 Q 来自 Decoder，K/V 来自 Encoder——两个不同来源。
>
> ⚠️ **误区二**：以为 Encoder 和 Decoder 共享权重。原论文中它们独立；现代 GPT 系列只有 Decoder 没有 Encoder。
>
> ⚠️ **误区三**：以为位置编码加在注意力打分上。它加在输入嵌入上，只在第一层进入网络。

## 5. 💻 代码实现

下面搭一个最小 Transformer。工程注意：`nn.MultiheadAttention` 的 cross-attention 里 `query` 是 Decoder 状态、`key/value` 是 Encoder 输出；这里为清晰起见没接 causal mask（ch14 专门讲），只验证维度。

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """单层 Encoder 或 Decoder（简化版，Post-LN）"""
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, is_decoder=False):
        super().__init__()
        self.is_decoder = is_decoder
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = (nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                           if is_decoder else None)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model) if is_decoder else None

    def forward(self, x, enc_out=None):
        # Self-Attention + 残差 + Norm
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        if self.is_decoder and enc_out is not None:
            # Cross-Attention: Q=decoder, K/V=encoder
            cross_out, _ = self.cross_attn(x, enc_out, enc_out)
            x = self.norm2(x + cross_out)
            x = self.norm3(x + self.ffn(x))
        else:
            x = self.norm2(x + self.ffn(x))
        return x

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=512, n_heads=8, n_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, is_decoder=False) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, is_decoder=True) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        enc = self.embed(src)
        for layer in self.encoder_layers:
            enc = layer(enc)
        dec = self.embed(tgt)
        for layer in self.decoder_layers:
            dec = layer(dec, enc_out=enc)
        return self.head(dec)

# 测试维度
model = MiniTransformer()
src = torch.randint(0, 1000, (2, 15))  # batch=2, src_len=15
tgt = torch.randint(0, 1000, (2, 10))  # tgt_len=10
logits = model(src, tgt)
print("src shape:", tuple(src.shape))
print("tgt shape:", tuple(tgt.shape))
print("输出 logits shape:", tuple(logits.shape))  # (2, 10, 1000)
```

## 6. 小结

1. Transformer = Encoder 栈 + Decoder 栈，每层由 Attention + Add&Norm + FFN + Add&Norm 组成；Decoder 多一层 Cross-Attention。
2. 维度账本：输入嵌入 $n\times d_{\text{model}}$，各层维度不变，最后投影到 $n\times V$。
3. 训练用 Teacher Forcing（并行），推理用自回归（串行）。
4. Cross-Attention 中 Q 来自 Decoder，K/V 来自 Encoder。

## 7. 延伸阅读/进一步思考

- 原论文 Vaswani 等 2017 是 Encoder-Decoder 架构的出处；之后 BERT 只留 Encoder（理解类任务），GPT 只留 Decoder（生成类任务），这是架构上的两条分支。
- Teacher Forcing 带来的 exposure bias 是经典话题，scheduled sampling 等方法试图缓解；ch29 训练小语言模型时会再遇到。
- Encoder-Decoder（如 T5、BART）至今仍用于翻译、摘要等"输入输出都是序列"的任务。

## 8. 思考题

**基础题**

1. GPT 系列只有 Decoder 没有 Encoder，它的 Cross-Attention 去哪里了？

**进阶题**

2. 为什么训练时可以并行计算所有位置的损失，而推理时必须一个一个生成？
3. 如果把 Encoder 层数从 6 加到 24，参数量大约翻几倍？（提示：每层参数量近似常数，看 ch15。）

**开放题**

4. Encoder-Decoder 结构有"编码一次、多次解码"的优势（翻译时 Encoder 跑一遍，Decoder 自回归）。GPT 类 Decoder-only 模型没有独立 Encoder，它怎么处理"长输入"？从 KV cache 和计算分配角度分析。
