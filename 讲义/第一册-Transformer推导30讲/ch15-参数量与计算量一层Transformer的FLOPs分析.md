# 第15讲 参数量与计算量：一层Transformer的FLOPs分析

> 📌 **本讲问题**：一个"7B 模型"的 7B 到底是什么？训练一次要花多少钱？推理一秒能跑多少 token？这些数字怎么从架构算出来？本讲把一层 Transformer 的参数和算力账算到底。

## 1. 数学工具/预备知识

**矩阵乘法 FLOPs。** $\boldsymbol{A}\in\mathbb{R}^{m\times k}$ 乘 $\boldsymbol{B}\in\mathbb{R}^{k\times n}$，要算 $mn$ 个点积，每个点积 $k$ 次乘加，约 $2mnk$ 次浮点运算（乘一次加一次各算一次）。

**参数量。** 等于权重矩阵元素总数，一个 $\mathbb{R}^{m\times k}$ 矩阵有 $mk$ 个参数。

**前向 vs 反向。** 反向传播梯度计算量约为前向的 2 倍，所以训练（前向+反向）约为 $3\times$ 前向。

## 2. 完整推导

### 2.1 一层 Transformer 的参数量

设 $d_{\text{model}}=d$，FFN 中间层 $d_{ff}=4d$。

**MHA 参数量**（ch09）：四个 $d\times d$ 矩阵 $W_Q,W_K,W_V,W_O$，合计 $4d^2$。偏置忽略（现代大模型通常不带偏置）。

**FFN 参数量**（ch13）：$\boldsymbol{W}_1: d\times 4d$，$\boldsymbol{W}_2: 4d\times d$，合计 $8d^2$。

**LayerNorm 参数量**：每层 2 个 LN（MHA 后一个，FFN 后一个），每个有 $\gamma,\beta$ 各 $d$ 个参数，合计 $4d$。

**一层总参数量**：$4d^2+8d^2+4d=12d^2+4d\approx 12d^2$。

### 2.2 注意力 FLOPs

对序列长度 $n$。

- $Q,K,V$ 投影：三个 $n\times d\times d$ 矩阵乘，$3\times 2nd^2=6nd^2$。
- $QK^\top$：$n\times n\times d$，约 $2n^2d$。
- Softmax：$O(n^2)$，可忽略。
- $\mathrm{Attention}\cdot V$：$n\times n\times d$，约 $2n^2d$。
- $W_O$ 投影：$2nd^2$。

MHA 总 FLOPs：$8nd^2+4n^2d$。

### 2.3 FFN FLOPs

- $\boldsymbol{W}_1$：$n\times d\times 4d$，约 $2n\cdot d\cdot 4d=8nd^2$。
- $\boldsymbol{W}_2$：$n\times 4d\times d$，约 $8nd^2$。

FFN 总 FLOPs：$16nd^2$。

### 2.4 一层总 FLOPs

$$\text{FLOPs}_{\text{layer}}=(8nd^2+16nd^2)+4n^2d=24nd^2+4n^2d$$

短序列 $n\ll d$ 时 $24nd^2$（线性项）主导；长序列 $n\gg d$ 时 $4n^2d$（二次项）主导。

### 2.5 原论文配置

$d=512, n=1024, L=6$（Encoder 6 层 + Decoder 6 层）：
- 每层参数量 $12\times 512^2\approx 3.7M$。
- Encoder+Decoder 共 12 层：$12\times 3.7M\approx 44M$。
- 加嵌入层和输出投影（共享权重时约 $V\times d=37000\times 512\approx 19M$）。
- 总参数量约 65M。

### 2.6 推理成本

自回归推理每生成一个 token：前向一遍所有层，约 $2N_{\text{params}}$ FLOPs（乘加各算一次）。对 $L$ 层模型，每 token 推理 FLOPs $\approx 2N_{\text{params}}$。

## 3. 💡 直觉理解

> 💡 **直觉**：参数量像书的字数，FLOPs 像读书花的时间。FFN 占大部分字数（$8d^2$ vs 注意力 $4d^2$），但读长句子时注意力部分（$n^2d$）越来越费时——每个词要和所有词比较。这就是长序列推理特别贵的原因。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 FLOPs 和参数量成正比。短序列是的，但长序列注意力 $n^2$ 项主导——序列翻倍，注意力成本翻四倍。
>
> ⚠️ **误区二**：把训练和推理 FLOPs 混为一谈。训练要反向传播约 $3\times$ 前向；推理只做前向。
>
> ⚠️ **误区三**：以为"7B"是 FLOPs。7B 指 7B 参数，不是计算量。

## 5. 💻 代码实现

下面用代码核算参数量和 FLOPs。工程注意：FLOPs 是估算，实际含 softmax、LayerNorm、激活等小项，通常只数矩阵乘；训练 FLOPs 乘 3 是经验系数（前向 1 + 反向约 2）。

```python
def count_transformer_params(d_model=512, n_layers=6, vocab_size=37000):
    """统计 Transformer 参数量"""
    mha_params = 4 * d_model * d_model           # W_Q/K/V/O
    ffn_params = d_model * 4*d_model + 4*d_model * d_model  # W1, W2
    ln_params = 2 * 2 * d_model                   # 每层2个LN, 每个 gamma+beta
    per_layer = mha_params + ffn_params + ln_params
    total = per_layer * n_layers + vocab_size * d_model  # 加嵌入
    return total, per_layer

def count_transformer_flops(d_model=512, seq_len=1024, n_layers=6):
    """估算一层训练 FLOPs (前向+反向≈3x)"""
    mha_flops = 8 * seq_len * d_model**2 + 4 * seq_len**2 * d_model
    ffn_flops = 16 * seq_len * d_model**2
    per_layer = mha_flops + ffn_flops
    total_train = per_layer * n_layers * 3
    return total_train, per_layer

# 原论文配置: Encoder 6 层 + Decoder 6 层 = 12 层
total_params, per_layer = count_transformer_params(n_layers=12)
print(f"原论文 Transformer 总参数量(12层+嵌入): {total_params/1e6:.1f}M")
print(f"每层参数量: {per_layer/1e6:.2f}M")

# FLOPs（按 12 层计）
total_flops, layer_flops = count_transformer_flops(n_layers=12)
print(f"\n每层训练 FLOPs: {layer_flops/1e9:.2f}G")
print(f"12层训练 FLOPs (每token): {total_flops/1e9:.2f}G")
print(f"训练 10亿 token 总 FLOPs: {total_flops*1e9/1e12:.0f}T")

# 验证 GPT-3 量级: 175B 参数, 96 层, d=12288
d_gpt3 = 12288
per_layer_gpt3 = 12*d_gpt3**2 + 4*d_gpt3
print(f"\nGPT-3 每层参数量约: {per_layer_gpt3/1e6:.1f}M, 96层约: {per_layer_gpt3*96/1e9:.1f}B")
```

## 6. 小结

1. 一层参数量约 $12d^2$，FFN 占 $8d^2$（2/3），MHA 占 $4d^2$（1/3）。
2. 一层 FLOPs $24nd^2+4n^2d$，短序列线性项主导，长序列 $n^2$ 项主导。
3. 训练 FLOPs 约 $3\times$ 前向。
4. 推理每生成一个 token 成本约 $2N_{\text{params}}$。

## 7. 延伸阅读/进一步思考

- FLOPs 估算的经典参考是 Kaplan 等 2020《Scaling Laws for Neural Language Models》和 Hoffmann 等 2022《Chinchilla》，它们用类似估算拟合模型性能和算力的关系。
- Chinchilla 结论：模型参数量和训练 token 数应近似等比例增长；这指导了后来大模型的配比。
- 推理成本除了 FLOPs，还有内存带宽瓶颈（ch23 KV cache、ch24 FlashAttention），实际推理常常不是算不动而是搬不动数据。

## 8. 思考题

**基础题**

1. 把 FFN 中间层从 $4d$ 改成 $8d$，一层参数量增加多少？FLOPs 增加多少？

**进阶题**

2. 序列长度从 1024 翻倍到 2048，注意力 FLOPs 翻几倍？FFN FLOPs 翻几倍？
3. GPT-3 有 175B 参数、96 层、$d=12288$，验证每层参数量是否约 $12d^2$。

**开放题**

4. 推理时每 token FLOPs $\approx 2N_{\text{params}}$，但实测速度常常远低于这个理论值。从"内存带宽 vs 计算"的角度解释：为什么大模型推理常常是 memory-bound 而不是 compute-bound？（预告 ch23/KV cache、ch24/FlashAttention。）
