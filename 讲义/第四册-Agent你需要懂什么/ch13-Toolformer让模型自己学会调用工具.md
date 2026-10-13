# 第13讲 Toolformer：让模型自己学会调用工具

> 📌 **本讲问题**：前面几讲都是在"用提示工程"让已经训练好的模型调工具。但能不能让模型在预训练阶段就**自己学会**什么时候该调工具、怎么调？Toolformer 就是这个思路——不靠人工标注，让模型自己生成"工具调用点"，再用一个可验证的判据判断这个调用是不是真有用。本讲讲清它的自监督标注方法和损失变化判据。

## 1. 工具/预备知识

- **自监督学习（Self-supervised Learning）**：不用人工标注，从数据本身自动生成监督信号。BERT 的掩码预测、GPT 的下一 token 预测都是自监督。
- **因果语言建模损失（Causal LM Loss）**：标准 GPT 训练目标——给定前面的 token，预测下一个 token 的交叉熵损失。
- **API 调用（API Call）**：模型在文本中插入特殊格式的调用标记，如 `[search(北京天气)]`，表示"这里我想调搜索"。

> 🧩 **这是 Schick 等人 2023 年的论文**。它和第 3 讲"提示式函数调用"的关系是：第 3 讲是训练好后靠 prompt 让模型现学；Toolformer 是把"什么时候该调工具"直接训进参数。一个是"外挂提示"，一个是"内化能力"。

## 2. 核心内容

### 2.1 参数化知识的边界

模型参数里存的是训练数据见过的知识。但有些知识是**动态**的——今天天气、最新股价、你私人日历。这些不可能写进参数，因为一直在变。

传统思路是后训练：人工标注一批"什么时候用什么工具"的样例，微调学会。但人工标注贵、覆盖面有限。Toolformer 的思路：**让模型自己从海量文本里发现"哪些地方插个工具调用，预测会更准"**。

### 2.2 自监督工具调用标注

标注分四步：

**第一步：候选点生成。** 给模型一段文本，让它自己挑位置插入"建议的工具调用"。原文"北京今天的气温是 X"，模型可能在 X 处建议 `[get_weather(北京)]`。

**第二步：执行工具。** 真去调，拿真实返回。`get_weather(北京)` 返回"15°C"。

**第三步：损失变化判据（最关键）。** 比较两个损失：不插调用时预测 X 的损失、插上调用和真实返回后预测 X 的损失。插了之后损失显著下降，说明调用有用，保留；没用就丢。

**第四步：用筛选后的数据微调。** 把所有"有用的调用"收集成自监督训练集，微调模型。

### 2.3 损失变化判据的完整推导

待预测位置为 $i$，上下文 $x_{<i}$。不调工具时：

$$
L_{\text{no\_call}} = -\log p_\theta(x_i \mid x_{<i})
$$

若在位置 $j < i$ 插入调用 $c$ 和返回 $r$：

$$
L_{\text{with\_call}} = -\log p_\theta(x_i \mid x_{<i},\ \text{insert}(c, r, \text{at } j))
$$

损失变化量：

$$
\Delta L = L_{\text{no\_call}} - L_{\text{with\_call}}
$$

若 $\Delta L > \tau$（$\tau$ 为正阈值），说明插入工具后预测位置 $i$ 的困惑度显著下降——工具确实提供了有用信息，保留这个候选。

**为什么要阈值而不是看正负？** 因为模型有噪声——有时插了工具损失只是碰巧降一点点。设正阈值 $\tau$，只有降得足够多才认，过滤随机波动。

这个判据的妙处：**完全不需要人工标注**。"工具有没有用"纯由数据驱动——看预测损失降没降。模型自己发现"哪里插个外部数据会让我预测更准"，这就是自监督。

### 2.4 一个细节：调用 token 本身要被 mask 掉

Toolformer 还有个常被忽略的技术细节：插入的调用文本 $c$ 和返回 $r$ 这些 token，在算损失时要**特殊处理**——模型不应该因为"必须学会把天气查询语句背下来"而被训练，它只需要学会"在该调的时候调"。具体做法是把这些插入 token 从"被预测目标"里 mask 掉，只在它们**作为上下文出现**时影响后面的预测。这样模型学的是"何时请求外部数据"，而不是"背诵 API 长什么样"。

### 2.5 工具涌现的数据流

整个训练数据流：

1. 从海量预训练文本采样文档。
2. 每个文档让模型生成一批"建议调用点"。
3. 每个候选执行工具、算 $\Delta L$。
4. 筛出 $\Delta L > \tau$ 的高质量样本。
5. 用这批样本微调基础模型。

训练完，推理时模型会自己在合适位置输出调用标记；后端检测到就执行、把结果拼回，模型继续生成。它证明了**"工具使用能力"可以从文本里自监督涌现，而不必人工教**。

### 2.6 和提示式调用的本质区别

| | 提示式函数调用（第 3 讲） | Toolformer |
|---|---|---|
| 能力在哪 | 每次靠 prompt 现场引导 | 固化进模型参数 $\theta$ |
| 额外成本 | 每次都要发工具说明到 prompt | 推理时几乎零额外 prompt |
| 换工具 | 改 prompt 即可 | 要重新微调 |
| 适用 | 工具少、常变 | 工具多、稳定、规模大 |

## 3. 💡 直觉理解

> 💡 **直觉**：想象学生做填空题："北京今天气温是 ___ 度。"他蒙了个"20"，心里没底（损失高）。如果允许他翻手机查到 15 度，填上"15"就很有把握（损失低）。Toolformer 就是让学生自己决定："哪些空我需要翻书，哪些直接填。"判断标准就是——翻书后我填得是不是更有把握？是，这空值得翻；翻不翻都一样，就别翻。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：认为 Toolformer 是"让模型会调工具的唯一方法"。它是数据高效的方法，但有前提：你得有一批可自动执行验证的工具、足够大的预训练文本。工具少、任务垂直时，人工标几百条样例可能更划算。Toolformer 的价值在**规模化**。
>
> ⚠️ **误区二**：以为它学的是"API 语法"。其实它学的是"何时该请求外部信息"——语法部分被 mask 掉了，不该成为学习目标。
>
> ⚠️ **误区三**：以为它能调出训练时没见过的工具。不能——它只在你提供的那批工具里学会调用。

## 5. 💻 代码实现

下面模拟 Toolformer 核心：候选生成→执行→损失变化判断→筛选保留。规则模拟预测损失。

```python
from dataclasses import dataclass
from typing import List


# ============ 工具集 ============
def tool_get_weather(city: str) -> str:
    db = {"北京": "15度多云", "上海": "22度晴", "广州": "28度雨"}
    return db.get(city, "未知")


def tool_calculate(expr: str) -> str:
    allowed = set("0123456789+-*/(). ")
    cleaned = ''.join(c for c in expr if c in allowed)
    return str(eval(cleaned)) if cleaned else "错误"


TOOLS = {"get_weather": tool_get_weather, "calculate": tool_calculate}


def fake_loss(next_token: str, has_tool_result: bool = False, tool_result: str = "") -> float:
    """模拟交叉熵损失：有工具帮助且对得上→损失低，否则高。"""
    if has_tool_result and tool_result and tool_result in next_token:
        return 1.0     # 工具提供了有用信息，预测很准
    if has_tool_result:
        return 3.0     # 有结果但和目标对不上
    return 5.0         # 没工具，瞎猜


@dataclass
class CandidateCall:
    position: int
    tool_name: str
    tool_args: str
    target_token: str


class ToolformerFilter:
    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold
        self.accepted: List[CandidateCall] = []

    def evaluate(self, c: CandidateCall, tool_result: str) -> bool:
        l_no = fake_loss(c.target_token)
        l_with = fake_loss(c.target_token, has_tool_result=True, tool_result=tool_result)
        dl = l_no - l_with
        keep = dl > self.threshold
        print(f"  {c.tool_name}({c.tool_args}): 无={l_no:.0f} 有={l_with:.0f} ΔL={dl:.0f}"
              f" → {'保留' if keep else '丢弃'}")
        if keep:
            self.accepted.append(c)
        return keep


def demo():
    f = ToolformerFilter(threshold=2.0)
    candidates = [
        CandidateCall(5, "get_weather", "北京", "15度多云"),
        CandidateCall(12, "calculate", "12*34", "408"),
        CandidateCall(20, "get_weather", "月球", "春天很美好"),  # 工具结果和目标无关
    ]
    print("=== Toolformer 候选筛选 ===\n")
    for c in candidates:
        result = TOOLS[c.tool_name](c.tool_args)
        f.evaluate(c, result)
    print(f"\n保留 {len(f.accepted)} 条样本: "
          + ", ".join(f"{c.tool_name}({c.tool_args})" for c in f.accepted))


if __name__ == "__main__":
    demo()
```

> 🔬 **跑起来你会看到**：前两个候选（查北京天气、算 12×34）插了工具后损失大降，ΔL 超过阈值被保留；第三个"查月球天气来预测'春天很美好'"，工具结果和目标对不上，损失没怎么降，被丢弃。这就是 $\Delta L$ 判据在自动区分"有用的调用"和"没用的调用"。

## 6. 小结

1. Toolformer 自监督标注：模型自己生成候选调用点，用损失变化判据判断哪些调用真有用。
2. 判据 $\Delta L = L_{\text{no\_call}} - L_{\text{with\_call}} > \tau$——插入工具后困惑度显著下降才算有价值。
3. 调用 token 本身要 mask 掉，让模型学"何时调用"而非"背诵 API"。
4. 它证明工具使用能力可从大规模文本自监督涌现；前提是工具可自动验证、文本数据足够。

> 📚 **延伸阅读**
> - Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*（2023）——本讲原始论文。

## 7. 思考题

**基础档**

1. 写出 $\Delta L$ 的定义式，并解释为什么 $\Delta L > 0$ 就意味着"这个调用有用"。
2. 为什么判据要设正阈值 $\tau$，而不是 $\Delta L > 0$ 就保留？

**进阶档**

3. 候选调用的返回和目标 token 完全无关时，$\Delta L$ 大概是正还是负？结合 `fake_loss` 的逻辑说明。
4. 为什么要把插入的调用 token 从损失目标里 mask 掉？不 mask 会学成什么样？

**开放档**

5. Toolformer 的能力在参数里，提示式调用的能力在 prompt 里。换一批全新工具时，二者各自要付出什么代价？
6. 既然有 Toolformer 这种自监督方法，为什么工业界主流还是用提示式/原生函数调用？提示：想想 Toolformer 的两个前提。
