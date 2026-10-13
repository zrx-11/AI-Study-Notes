# AI 机理深度研习系列 · 讲义索引

> 5册 / 126讲 / 约1.3MB Markdown
> 每讲独立 .md 文件，含数学推导（LaTeX）、直觉框、误区框、可运行代码、小结、思考题

---

## 学习路径

### 工程向（做Agent、大模型开发）
第一册 → 第四册 → 第五册

### 研究向（读论文、做研究）
第一册 → 第二册 → 第三册

### 全景通读
一 → 二 → 三 → 四 → 五

---

## 第一册《Transformer 推导 30 讲》（30讲）

**定位**：全系列基石。从零推导Transformer，最终手写mini-Transformer。

| 讲 | 文件 | 主题 |
|----|------|------|
| 1 | ch01-张量与矩阵乘法.md | 张量、矩阵乘法四种视角、einsum |
| 2 | ch02-概率与信息论.md | 信息量、熵、交叉熵、KL散度 |
| 3 | ch03-梯度下降与反向传播.md | 链式法则矩阵形式、两层MLP反向、梯度消失 |
| 4 | ch04-Softmax与归一化.md | Softmax雅可比、复合梯度、温度系数 |
| 5 | ch05-序列建模简史RNN与LSTM.md | RNN/BPTT、梯度消失证明、LSTM |
| 6 | ch06-Seq2Seq与注意力机制的萌芽.md | Bahdanau/Luong注意力、加性vs乘性 |
| 7 | ch07-Query-Key-Value注意力的一般化抽象.md | Q/K/V抽象、Self/Cross-Attention |
| 8 | ch08-ScaledDotProductAttention完整推导.md | **√dₖ缩放推导**、饱和区梯度消失 |
| 9 | ch09-MultiHeadAttention多头的本质与维度分析.md | MHA公式、参数量分析 |
| 10 | ch10-Transformer总体架构EncoderDecoder数据流全览.md | Encoder-Decoder数据流、teacher forcing |
| 11 | ch11-位置编码正余弦编码的完整推导.md | 正余弦PE推导、相似度可视化 |
| 12 | ch12-AddNorm残差连接与LayerNorm的原理.md | 残差连接、LayerNorm、Pre-LN vs Post-LN |
| 13 | ch13-前馈网络FFN逐位置MLP的作用.md | 4d_model、激活函数演进、SwiGLU |
| 14 | ch14-Mask机制PaddingMask与CausalMask.md | Padding/Causal Mask、泄露检测 |
| 15 | ch15-参数量与计算量一层Transformer的FLOPs分析.md | 逐模块参数量、FLOPs推导 |
| 16 | ch16-复杂度对比TransformervsRNNvsCNN.md | O(1)路径长度、归纳偏置对比 |
| 17 | ch17-损失函数交叉熵困惑度与LabelSmoothing.md | Next-Token预测、困惑度、Label Smoothing |
| 18 | ch18-学习率调度Warmup与衰减的设计原理.md | Warmup推导、调度曲线对比 |
| 19 | ch19-Adam优化器从动量到自适应学习率.md | SGD动量、RMSProp、Adam/AdamW |
| 20 | ch20-初始化与梯度稳定性Xavier与He初始化推导.md | 方差演化、He初始化 |
| 21 | ch21-混合精度训练FP16与BF16的数学与工程.md | FP16/BF16、Loss Scaling、主权重 |
| 22 | ch22-RoPE旋转位置编码复数域的优雅推导.md | RoPE复数域推导、位置外推 |
| 23 | ch23-KVCache自回归推理的加速本质.md | KV Cache、显存占用、MQA/GQA |
| 24 | ch24-FlashAttentionIO感知的精确注意力.md | GPU内存层级、Online Softmax、分块算法 |
| 25 | ch25-MoE混合专家稀疏激活的条件计算.md | Top-k路由、负载均衡损失 |
| 26 | ch26-量化与蒸馏模型压缩的核心机理.md | 线性量化、PTQ/QAT、知识蒸馏 |
| 27 | ch27-手写Attention50行PyTorch实现.md | SDPA函数、MHA类、单元测试 |
| 28 | ch28-手写miniTransformer完整模型搭建.md | 完整模型、权重共享、参数量交叉验证 |
| 29 | ch29-从零训练小语料上的语言模型与注意力可视化.md | 字符级语料、训练循环、注意力热图 |
| 30 | ch30-面试与进阶.md | 高频面试题、论文阅读路线图 |

---

## 第二册《从信息论做科研》（24讲）

**定位**：信息论统一坐标系，从看懂模型升级为做出研究。

六模块：地基(1-4) → 极限(5-8) → 桥梁(9-12) → 前沿(13-16) → 大模型专题(17-20) → 科研实战(21-24)

---

## 第三册《量子理论下的科研》（24讲）

**定位**：量子数学眼镜——表示是希尔伯特空间中的态，计算是算符作用。

六模块：地基(1-4) → 现象(5-8) → 桥梁(9-12) → 前沿(13-16) → 量子ML专题(17-20) → 科研实战(21-24)

---

## 第四册《Agent，你需要懂什么》（24讲）

**定位**：Agent = 大模型 + 工具 + 循环 + 记忆。

六模块：地基(1-4) → 核心循环(5-8) → 系统架构(9-12) → 前沿(13-16) → 实战与安全(17-20) → 科研实战(21-24)

---

## 第五册《AI 时代下的运筹学》（24讲）

**定位**：补全"智能"到"决策"的最后一环。

六模块：地基(1-4) → 经典模型(5-8) → 复杂决策(9-12) → 学习增广(13-16) → AI落地场景(17-20) → 科研实战(21-24)

---

## 阅读建议

- 每讲自包含，不按顺序读也能看懂
- 第一册是地基，强烈建议从头读
- 代码用 numpy/PyTorch/scipy，中文注释，可直接运行
- 数学公式用标准LaTeX，在 Typora / Obsidian / VS Code + Markdown插件中阅读效果最佳
