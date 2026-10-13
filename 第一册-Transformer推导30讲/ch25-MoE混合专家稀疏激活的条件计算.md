# 第25讲 MoE 混合专家：稀疏激活的条件计算

> 📌 **本讲问题**：为什么传闻 GPT-4 有 1.8T 参数但推理不算太慢？MoE 怎么做到"参数量巨大、每次却只算一小部分"？它又是怎么防止门控网络偷懒、把所有 token 都塞给同一个专家的？

## 1. 数学工具/预备知识

**（1）普通 FFN 的结构（ch13）。** 一个 FFN 是两层全连接：$x\to\text{ReLU}(W_1x)\to W_2$。所有 token 都走同一个 FFN。

**（2）门控网络（gating/router）。** 一个小的线性层，输入 token，输出"该把这个 token 送给哪个专家"的打分。

**（3）稀疏激活。** 一次只激活 $N$ 个候选里的 $k$ 个，其余完全不参与计算。

**（4）负载均衡。** 如果大家都挤向同一个专家，其余专家闲着，就浪费了。需要一个机制把 token 均匀摊开。

## 2. 完整推导

### 2.1 容量与计算解耦

普通 Transformer：每次前向，所有参数都参与计算。参数量 $=$ 计算量，想加容量就得加计算。

MoE（Mixture of Experts）把 FFN 替换成 $N$ 个并行的专家 FFN。每个 token 只送给其中 $k$ 个专家（通常 $k=2$）。于是：

- 总参数量 $\approx N\times$ 单专家参数量（容量随 $N$ 涨）；
- 每次计算量 $\approx k\times$ 单专家参数量（成本随 $k$ 涨）。

容量和计算解耦了——可以堆很大的记忆容量，而推理成本只跟那 $k$ 个被激活的专家走。

### 2.2 门控网络：Top-k 路由

对每个 token，门控网络先输出 $N$ 个 logits，再选分数最高的 $k$ 个：

$$
g(x)=\mathrm{softmax}\big(\text{TopK}(\text{Gate}(x), k)\big)
$$

具体说：$\text{Gate}(x)\in\mathbb{R}^{N}$ 是线性层输出的 $N$ 个分数；TopK 选出最大的 $k$ 个，其余置成 $-\infty$，再过 softmax，于是只有 $k$ 个位置非零，且这 $k$ 个权重加起来等于 1。

该 token 的输出是被选中专家的加权和：

$$
\text{MoE}(x)=\sum_{i\in\text{TopK}}g_i(x)\cdot\text{Expert}_i(x)
$$

### 2.3 负载均衡损失：防止赢家通吃

问题：门控网络可能很快学会"偷懒"——总把 token 送给少数几个它最有把握的专家，其余专家永远收不到 token，等于白建。这叫"专家坍缩"。

解决办法：加一个辅助负载均衡损失，鼓励每个专家被选中的频率大致相等：

$$
\mathcal{L}_{\text{aux}}=\alpha\cdot N\sum_{i=1}^{N}f_i\cdot P_i
$$

其中 $f_i$ 是专家 $i$ 实际被路由到的 token 比例，$P_i$ 是门控网络给专家 $i$ 的平均打分。理想情况下 $f_i=P_i=1/N$，此时 $N\sum_i(1/N)(1/N)=1/N$，损失最小。只要某个专家被过度偏爱或冷落，$f_i$ 和 $P_i$ 偏离 $1/N$，损失就上升——逼着门控网络均匀分配。

### 2.4 容量因子：工程上的取舍

实际工程里，每个专家一次最多处理多少 token 由"容量"决定：

$$
\text{capacity}=\text{batch}\times\text{seq}\times\frac{k}{N}\times\text{capacity\_factor}
$$

如果某专家分到的 token 超过它的容量，多出来的 token 就被直接丢弃（走残差连接绕过这一层）。容量因子太大浪费计算，太小会丢 token 掉点。这是个工程权衡。

### 2.5 为什么 MoE 推理快、训练难

推理时只激活 $k$ 个专家，所以虽然总参数巨大，单次前向的计算量只相当于 $k$ 个专家——这就是"大参数、快推理"。

但训练很难：(1) 要跨 GPU 把 token 发到不同专家（All-to-All 通信）；(2) 要加负载均衡损失防止坍缩；(3) 不同专家更新步调不一致，训练不稳定。所以 MoE 只在超大规模数据下才划算。

## 3. 💡 直觉理解

> 💡 **直觉**：普通 FFN 像一个全科医生，什么病都自己看。MoE 像一家有 8 个专科医生的医院——每个病人（token）进门由分诊台（门控网络）分给最对口的 2 个医生看。医院总人力（参数量）是 8 个医生，但每个病人只看 2 个，医院运转成本（计算量）只相当于 2 个医生。负载均衡损失就是规定"分诊台不能把所有病人都塞给最火的那两个医生"，否则其他医生闲死、热门医生累死。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：以为 MoE 推理时所有专家都算一遍。只激活 Top-k 个，其余完全不参与前向。
>
> **误区二**：以为 MoE 一定比 dense 模型好。它训练难（负载均衡、All-to-All 通信）、显存占用大（所有专家权重都得驻留），只在超大规模下才体现优势。
>
> **误区三**：以为 MoE 省显存。恰恰相反——所有专家的权重都得装在显存里，省的是计算量，不是显存。

## 5. 💻 代码实现

下面实现一个干净的 MoE：门控选 Top-k，按专家分组喂 token，加权求和。最后对比"总参数量"和"每次激活参数量"。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """一个普通的 FFN 专家"""
    def __init__(self, d_model=128, d_ff=256):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class MoE(nn.Module):
    def __init__(self, d_model=128, n_experts=8, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(n_experts)])
        self.gate = nn.Linear(d_model, n_experts)
        self.top_k = top_k
        self.n_experts = n_experts

    def forward(self, x):
        b, s, d = x.shape
        x_flat = x.reshape(-1, d)                       # (N_token, d)
        # 门控: 选 top_k 个专家
        logits = self.gate(x_flat)                      # (N_token, E)
        top_logits, top_idx = logits.topk(self.top_k, dim=-1)  # (N_token, k)
        top_probs = F.softmax(top_logits, dim=-1)       # 被选专家的权重
        output = torch.zeros_like(x_flat)
        for e in range(self.n_experts):
            # 哪些 (token, 槽位) 路由到了专家 e
            mask = (top_idx == e)                        # (N_token, k)
            if not mask.any():
                continue
            token_ids, slot_ids = mask.nonzero(as_tuple=True)
            expert_out = self.experts[e](x_flat[token_ids])
            w = top_probs[token_ids, slot_ids].unsqueeze(-1)
            output[token_ids] += w * expert_out
        return output.view(b, s, d)

torch.manual_seed(0)
moe = MoE(d_model=128, n_experts=8, top_k=2)
x = torch.randn(2, 10, 128)
out = moe(x)

total = sum(p.numel() for p in moe.parameters())
print(f"输入 shape: {tuple(x.shape)}, 输出 shape: {tuple(out.shape)}")
print(f"总参数量: {total/1e6:.2f}M")
print(f"每次实际激活参数量 (top_k/8 = 1/4): {total*2/8/1e6:.2f}M")
print(f"输出与输入同形状: {out.shape == x.shape}")
```

## 6. 小结

1. MoE 用多个并行专家 FFN 替代单个 FFN，每次只激活 Top-k 个。
2. 参数量随专家数 $N$ 增长，计算量只随激活数 $k$ 增长——容量与计算解耦。
3. 负载均衡损失 $N\sum_i f_iP_i$ 防止门控坍缩到少数专家。
4. 容量因子控制每专家 token 上限，超出则丢弃；MoE 省计算但不省显存。

## 7. 思考题

**基础题**

1. 如果把 $k$ 设成 $N$（所有专家都激活并平均加权），这个 MoE 退化成了什么？

**进阶题**

2. 负载均衡损失 $N\sum_i f_iP_i$ 在理想均匀分配时取到什么值？为什么这个损失能逼门控网络别"赢家通吃"？

**开放题**

3. MoE 训练时需要 All-to-All 通信（把 token 发到不同 GPU 上的专家）。请从"模型并行"的角度想：为什么 MoE 天生适合专家并行（把不同专家放不同 GPU）？这种并行和张量并行有什么区别？

## 8. 延伸阅读 / 进一步思考

- MoE 经典：Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, 2017；Switch Transformer, 2021。
- 负载均衡与容量因子：Switch Transformer 论文的辅助损失与 token drop 机制。
- 进阶：DeepSeekMoE、专家并行的通信优化、MoE 的训练稳定性（aux-loss-free 路由），可作为第二册。
