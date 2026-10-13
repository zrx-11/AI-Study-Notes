# 第28讲 手写 mini-Transformer：完整模型搭建

> 📌 **本讲问题**：把所有零件组装起来——从 token ID 到 logits，一个完整的 Decoder-only Transformer 长什么样？权重共享为什么能省这么多参数？搭好后真的能训出 loss 下降吗？

## 1. 数学工具/预备知识

- **ch10 的整体架构**：嵌入 → 位置编码 → N 层 Block → 输出投影。
- **ch12 的残差 + LayerNorm**：Pre-LN 结构。
- **ch13 的 FFN**：两层 MLP。
- **ch14 的 Causal Mask**：下三角掩码。

这一讲把这些零件拼成一个能前向、能反向、能训练的完整模型。

## 2. 完整推导

### 2.1 Token 到 logits 的数据流

一个 Decoder-only 语言模型的前向数据流：

1. Token ID → Embedding 查表，得到 $d$ 维向量；
2. 加位置编码；
3. 过 $N$ 层 Block（Self-Attention + 残差，再接 FFN + 残差）；
4. 最后一次 LayerNorm；
5. 线性投影到词表大小 $V$，得到 logits。

### 2.2 Pre-LN 残差结构

每层 Block 用 Pre-LN（先归一化再进子层）：

$$
x = x + \mathrm{Attn}(\mathrm{LN}(x)),\qquad x = x + \mathrm{FFN}(\mathrm{LN}(x))
$$

残差通路保证梯度能直达输入（ch12），Pre-LN 让训练更稳。

### 2.3 权重共享

输入嵌入矩阵 $E\in\mathbb{R}^{V\times d}$ 和输出投影矩阵 $W_{\text{out}}\in\mathbb{R}^{d\times V}$ 共享同一份权重（互为转置）。这省掉了 $V\times d$ 个参数——对大词表来说这是一大笔。直觉是："这个词长什么样"和"什么样的向量对应这个词"本质是同一份知识的正反两面。

### 2.4 参数量手算与代码对拍

搭完模型后，用代码数出真实参数量，再和手算公式对照，验证没有漏算或多算。

## 3. 💡 直觉理解

> 💡 **直觉**：嵌入层和输出层共享权重像"进出门用同一把钥匙"——进去时查这个 token 对应什么向量，出来时把向量映射回 token 空间，同一份知识用两次。整个模型就是一串"加残差"的积木块叠起来。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为权重共享必须做。早期模型共享，后来大模型发现不共享效果更好——因为嵌入空间和输出空间的需求不完全一致。
>
> **误区二**：以为 Pre-LN 可以不要最后的 LayerNorm。去掉它，最后一层的输出可能幅度失控，logits 爆掉。

## 5. 💻 代码实现

下面搭一个完整的 mini-Transformer，并做一个**迷你训练**：用随机数据跑几步，确认 loss 真的在下降（证明模型能训）。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """Pre-LN Transformer Block: 自注意力 + FFN, 各带残差"""
    def __init__(self, d_model=128, n_heads=4, d_ff=512):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )
    def forward(self, x, mask):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_heads=4, n_layers=4, max_seq=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_model * 4) for _ in range(n_layers)]
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight      # 权重共享

    def forward(self, idx):
        b, s = idx.shape
        pos = torch.arange(s, device=idx.device).unsqueeze(0)
        x = self.embed(idx) + self.pos(pos)
        # 因果掩码: 上三角为 True 表示"这里被屏蔽"
        mask = torch.triu(torch.ones(s, s, device=idx.device), diagonal=1).bool()
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.norm_f(x))

# ===== 搭模型 + 前向 =====
torch.manual_seed(0)
model = MiniTransformer(vocab_size=1000, d_model=128, n_heads=4, n_layers=4)
x = torch.randint(0, 1000, (2, 20))
logits = model(x)
print(f"输入: {tuple(x.shape)}, 输出 logits: {tuple(logits.shape)}")

# ===== 参数量对拍 =====
total = sum(p.numel() for p in model.parameters())
# 逐项手算(含 bias):
#   embed: 1000*128 ; pos: 64*128 ; head 共享 embed 不重复算
#   每 block: attn in_proj(3*128*128+3*128) + out_proj(128*128+128)
#             + ffn(128*512+512)+(512*128+128) + 两个LN(2*2*128)
#   最后 norm_f: 2*128
manual = (1000*128 + 64*128
          + 4*((3*128*128+3*128) + (128*128+128) + (128*512+512) + (512*128+128) + 2*2*128)
          + 2*128)
print(f"代码数参数量: {total/1e6:.3f}M, 手算: {manual/1e6:.3f}M, 一致: {total==manual}")

# ===== 迷你训练: 确认 loss 在下降 =====
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
y = torch.randint(0, 1000, (2, 20))   # 随机目标, 只验证训练回路通不通
losses = []
for step in range(30):
    out = model(x)
    loss = F.cross_entropy(out.reshape(-1, 1000), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
print(f"\n训练 loss: 第1步={losses[0]:.3f}, 第30步={losses[-1]:.3f}")
print(f"loss 下降了: {losses[-1] < losses[0]}")
```

## 6. 小结

1. 完整 mini-Transformer = 嵌入 + 位置编码 + N 层 Pre-LN Block + 最终 LN + 输出投影。
2. Pre-LN：先 Norm 再子层再残差，训练更稳。
3. 权重共享：嵌入和输出投影共用一份矩阵，省 $V\times d$ 参数。
4. 搭好后必须跑一个迷你训练，确认 loss 下降——证明前向反向回路真的通。

## 7. 思考题

**基础题**

1. 如果不共享权重，这个模型要多花多少参数？（提示：$V\times d$。）

**进阶题**

2. Pre-LN 为什么最后还要单独来一次 LayerNorm？如果把 `self.norm_f` 去掉，logits 的幅度可能出什么问题？

**开放题**

3. 这个模型是 Decoder-only。要改成 Encoder-Decoder 结构（加 Cross-Attention），需要在 Block 里加什么模块？参考 ch06 的 Seq2Seq 结构想一想。

## 8. 延伸阅读 / 进一步思考

- 本讲是 ch10 架构图的代码落地。
- 进阶：把这里的 `nn.MultiheadAttention` 换成 ch27 手写的 MHA，观察参数量是否一致。
- ch29 会用这个模型在真实小语料上训练并可视化注意力。
