# 第2讲 Token、上下文窗口与推理预算

> 📌 **本讲问题**：为什么一个 Agent 跑着跑着就"忘了"最开始的任务？为什么长任务里中间步骤越多，后面越容易出错？为什么账单比你想象的贵？答案都指向同一个硬约束——上下文窗口。理解它，是理解 Agent 为什么不能"想多久就想多久"的第一步。

## 1. 工具/预备知识

- **Token**：大模型处理文本的最小单位。一段中文大约每 1.5 个字对应 1 个 token，英文大约每 0.75 个单词对应 1 个 token。你不需要精确知道每个词怎么切，只需要知道：prompt 越长，token 数越多，成本越高、延迟越大。
- **上下文窗口（Context Window）**：模型一次调用最多能"看到"的 token 总数，记为 $N$。超过这个数的内容会被截断，或者直接报错。这是一个硬件级+训练级的硬约束——它由模型位置编码的训练长度决定，不是软件层面加几行代码就能绕过去的。
- **推理预算**：在一次 Agent 任务中，你愿意为这个任务花多少 token（≈多少钱+多少时间）。预算上限决定了 Agent 最多能思考几步、最多能检索多少资料。

> 🧩 **两个常被忽略的计费事实**（后面推导会用到）：
> 1. **输入 token 和输出 token 分开计价**，而且输出 token 通常更贵（常见定价里输出是输入的 3–5 倍）。因为输出是逐个 token 自回归生成的，每个都要跑一次完整前向；输入则可以批量并行处理。
> 2. **KV 缓存**：连续对话里，前面的 prompt 不必每步都从头算。但"省了重复计算"不等于"免费"——缓存本身占显存，且你依然要为每步重新发送的输入 token 付费。对 Agent 来说，真正的开销大头是**每步都要把越来越长的历史重新发一遍**。

## 2. 核心内容

### 2.1 上下文窗口是硬约束

假设模型的上下文窗口是 $N$ token。每次 Agent 循环，你都要把以下内容塞进 prompt：

- 系统提示（system prompt）：任务说明、工具定义、行为约束。通常 $S$ token。
- 历史对话/轨迹：之前每一步的思考、动作、观察。假设走了 $t$ 步，平均每步产生 $H_i$ token，累计 $\sum_{i=1}^{t} H_i$ token。
- 当前新观察：这一步刚拿到的工具返回结果。$O_t$ token。
- 输出预留空间：模型要生成的回复，最多 $M$ token。

加起来必须满足：

$$
S + \sum_{i=1}^{t} H_i + O_t + M \leq N
$$

这个不等式就是 Agent 的"生存空间"。当左边逼近右边时，系统必须做取舍。常见的取舍策略有三种：

1. **截断最早的历史**：把前几步的观察丢掉，只保留最近的。问题是——如果任务是"查 A 然后根据结果查 B 然后汇总"，你把查 A 的细节丢了，汇总时就没有材料了。
2. **压缩历史**：让模型自己写一个"进展摘要"，用摘要代替原始轨迹。这增加了一步调用，但省空间。
3. **外部存储**：把完整历史存到外部数据库，prompt 里只放指针（"详情见记忆条目 #42"）。需要时再检索回来（第 9 讲展开）。

这三种策略各有代价。这里只需要记住：**上下文窗口不是"尽量塞"的问题，而是"必须做选择"的问题**。

### 2.2 长任务的预算为什么是累计增长的

一个典型的 Agent 任务可能跑 5–20 步。关键在于：每一步都要重新发一次完整 prompt。所以第 $t$ 步发出去的输入，不是只有这一步的新东西，而是**从第 1 步到第 $t$ 步的全部历史**。

第 $t$ 步的输入 token 数（不含输出）是：

$$
I_t = S + \sum_{i=1}^{t} H_i + O_t
$$

假设每步新增观察 $H_i$ 是常数 $h$，系统提示为常数 $S$，那么 $I_t \approx S + t\cdot h$，随步数 $t$ **线性增长**。

如果我们关心的是整个任务的**累计输入 token 总消耗**（每步都计费一次），那就是把每步的 $I_t$ 再加起来：

$$
\text{总输入消耗} = \sum_{t=1}^{T} I_t = \sum_{t=1}^{T} \left( S + \sum_{i=1}^{t} H_i + O_t \right)
$$

在 $H_i = h$ 的简化下，$\sum_{t=1}^{T} t\cdot h = h \cdot \frac{T(T+1)}{2}$。这就是**二次增长**：步数翻倍，总输入 token 大约翻四倍。

> 📌 **直觉换算**：第 1 步发 2000 token，每步新增 500，那么第 10 步发 $2000 + 9\times 500 = 6500$ token。十步累计输入不是 $10\times 2000$，而是大约 $20000$——因为后面几步每次都背着前面所有的旧账。这就是为什么"长任务烧钱"不是错觉。

这就引出了推理预算的分配问题。假设你给一个任务的总预算是 $B$ token，你要决定：

- 前多少步用"完整上下文"，从哪一步开始切换到"压缩模式"？
- 工具返回的结果要不要先做摘要再塞回去？搜索一次可能返回 5000 token 的网页内容，全塞进去两轮就爆了。
- 要不要在中途主动做一次"进度总结"，把之前的长轨迹压缩成一段短文？

一个粗略的经验法则是：**给上下文窗口留 20% 的余量**。不要等到刚好填满才开始压缩，因为模型输出长度是不确定的——如果它本来打算写 200 token，结果你只留了 100 token 的空位，输出会被硬截断，格式可能直接坏掉（JSON 没闭合括号）。

### 2.3 BPE 分词：为什么有的文本特别"贵"

模型怎么把文本切成 token？主流方案是字节对编码（BPE, Byte Pair Encoding）。它的构造过程是：

1. 初始词表只有单个字符（字节）。
2. 统计训练语料里所有相邻符号对的频率。
3. 把频率最高的一对合并成一个新符号，加入词表。
4. 重复步骤 2–3，直到词表大小达到预设值（比如 5 万、10 万）。

举个极简例子。假设语料里 "low"、"lower"、"newest" 反复出现：

- 初始：每个字母是一个 token，"lower" = l-o-w-e-r（5 个）。
- "er" 出现频率最高，合并成 `er`："lower" = l-o-w-`er`（4 个）。
- 接着 "low" 整体频率高，合并成 `low`："lower" = `low`-`er`（2 个）。

合并得越多，常见词越短。结果是：

- 常见词（"的""是""你好"）通常是 1 个 token。
- 罕见词、随机字符串、长 URL 会被拆成很多个 token——因为它们在训练语料里没被合并过。
- 代码里的生造变量名（比如 `xqf7_energy_delta`）可能一个变量名就拆成 8–10 个 token。

这对 Agent 设计有三条直接推论：

1. **不要在 prompt 里塞无关的长 URL、长 ID、原始 JSON dump**——这些是 token 黑洞。能用简短标识就用简短标识。
2. **工具返回值要在工具层就清洗好**。数据库查询返回一行 1000 个字段，你只需要其中 3 个，就在工具层把它剪到只剩 3 个字段再返回给模型。这一步省的是后面每一步的累计成本。
3. **估算预算要留余量**。中文、代码、结构化数据的 token 密度差异很大，拍脑袋"这行字很短嘛"很容易低估。

### 2.4 窗口大 ≠ 记得牢：迷失在中间

就算窗口够大、没截断，长上下文里模型也未必真的"看全了"。Liu 等人（2023）的论文 *Lost in the Middle* 系统评测发现：模型对位于上下文**开头和结尾**的信息利用较好，对位于**中间**的信息利用率显著下降，呈 U 形曲线。

这意味着两件事：

- 把最重要的约束（"用户预算上限是 500 元"）放在系统提示开头，而不是埋在十几步历史的中间。
- 压缩历史时，保留"开头的原始任务"和"最近的观察"两头，中间的可以大胆压——这正好和注意力的 U 形分布吻合。

### 2.5 预算分配器的思路

一个实用的预算分配器可以按以下逻辑工作：

1. 初始化：计算系统提示占用了多少 token，工具定义占了多少。
2. 每轮开始前：估算当前 prompt 总 token 数。
3. 如果总 token < 窗口的 70%：正常运行，保留完整轨迹。
4. 如果在 70%–90% 之间：触发"轻度压缩"——把最早的 1–2 步观察替换成一句摘要。
5. 如果 > 90%：触发"重度压缩"——让模型生成一段 200 token 的"任务进展报告"，替换掉全部历史轨迹。

这个 70%/90% 的阈值不是什么科学常数，而是工程经验——给自己留缓冲，避免在最忙的时候突然爆窗口。

## 3. 💡 直觉理解

> 💡 **直觉**：把上下文窗口想象成一张办公桌。桌面上能摊开的纸是有限的。刚开始任务时，桌上只有任务书和笔（系统提示），很宽敞。每做一步，你就多摊开一张纸（新观察）。做了十步之后，桌上堆了十几张纸，再想摊新的就放不下了。这时候你要么把最早的纸收进抽屉（外部记忆），要么把几张纸的内容抄成一张摘要纸条（压缩历史）。预算分配器就是那个帮你管理桌面空间的助手——它知道桌子快满了，提前告诉你该收拾了。
>
> 而"迷失在中间"就像：桌上摊了 20 张纸，你一眼扫过去，最先注意到的是最上面和最下面那张，中间那几张你其实没认真看。所以聪明的做法是把最重要的两张（原始任务、最新进展）分别压在桌顶和桌底。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：认为"上下文窗口越大越好，128K 窗口就不用管记忆管理了"。窗口大只是延缓了问题，没有解决问题。第一，窗口越大，每次推理的成本越高、延迟越大（二次累计的账单摆在那）。第二，长上下文里模型注意力分布不均，中间位置的内容容易被忽略（迷失在中间）。第三，真正的长程任务（几百步）哪怕 1M 窗口也装不下。窗口大小是资源约束，不是记忆系统的替代品。
>
> ⚠️ **误区二**：把"字符数"当"token 数"。一行满是 URL 和生造变量名的代码，看着不长，token 数可能是同长度中文的两三倍。做预算前务必用真实 tokenizer 量一次。
>
> ⚠️ **误区三**：等到窗口爆了才压缩。爆窗往往发生在某一步工具突然返回一大坨内容时，这时再压缩已经晚了——系统会硬截断，把没闭合的 JSON 吐给模型，直接搞坏格式。

## 5. 💻 代码实现

下面实现一个带 token 预算管理的模拟 Agent。`fake_tokenize` 用字符数近似 token 数，`BudgetAllocator` 负责在上下文快满时触发压缩。全程不依赖大模型 API。

```python
from dataclasses import dataclass
from typing import List, Dict


# ============ 简易 token 估算 ============
def fake_tokenize(text: str) -> int:
    """模拟 token 计数：中文按 1.5 字/token，其余按 4 字符/token 粗估。

    真实场景应该用模型自带的 tokenizer（如 tiktoken），这里保持确定性，
    让预算逻辑可以脱离任何 API 单独跑。
    """
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    return int(chinese_chars / 1.5 + other_chars / 4)


# ============ 预算分配器 ============
@dataclass
class BudgetAllocator:
    max_tokens: int = 4096       # 上下文窗口大小
    warn_threshold: float = 0.7  # 70% 触发轻度压缩
    hard_threshold: float = 0.9  # 90% 触发重度压缩
    output_reserve: int = 512    # 给模型输出预留的空间

    def check(self, used: int) -> str:
        """根据已用 token 数返回压缩策略。

        注意分母用的是 max - output_reserve：预留输出空间是不可动用的，
        否则模型生成到一半被硬截断，JSON 会坏掉。
        """
        ratio = used / (self.max_tokens - self.output_reserve)
        if ratio < self.warn_threshold:
            return "normal"
        elif ratio < self.hard_threshold:
            return "light_compress"
        else:
            return "hard_compress"


# ============ 轨迹管理 ============
class Trajectory:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.steps: List[Dict] = []  # 每步：{thought, action, observation}

    def add_step(self, thought: str, action: str, observation: str):
        self.steps.append({"thought": thought, "action": action, "observation": observation})

    def total_tokens(self) -> int:
        total = fake_tokenize(self.system_prompt)
        for s in self.steps:
            # 思考/动作/观察都要计费，因为它们都被重新塞进下一轮 prompt
            total += fake_tokenize(s["thought"])
            total += fake_tokenize(s["action"])
            total += fake_tokenize(s["observation"])
        return total

    def light_compress(self):
        """轻度压缩：把最早一步的观察替换成一句摘要（保留两头、压中间）。"""
        if len(self.steps) >= 2:
            old_obs = self.steps[0]["observation"]
            self.steps[0]["observation"] = f"[摘要] 早期观察已省略，约 {len(old_obs)} 字。"
            print("  [预算] 轻度压缩：移除最早一步的详细观察。")

    def hard_compress(self):
        """重度压缩：把全部历史步骤替换成一段进展摘要。"""
        n = len(self.steps)
        last_obs = self.steps[-1]["observation"] if self.steps else ""
        # 模拟一个确定性摘要：只保留步数和最后观察的开头 50 字
        summary = f"[进展摘要] 已执行 {n} 步。最近观察：{last_obs[:50]}..."
        self.steps = [{"thought": "(历史已压缩)", "action": "(见摘要)", "observation": summary}]
        print(f"  [预算] 重度压缩：{n} 步历史压缩为 1 条摘要。")

    def render_prompt(self) -> str:
        parts = [self.system_prompt]
        for i, s in enumerate(self.steps, 1):
            parts.append(f"--- 第 {i} 步 ---")
            parts.append(f"思考：{s['thought']}")
            parts.append(f"动作：{s['action']}")
            parts.append(f"观察：{s['observation']}")
        return "\n".join(parts)


# ============ 模拟 LLM ============
def fake_llm(step: int) -> str:
    """规则模拟：按当前步数返回一个动作（不再解析 prompt，避免误触发）。"""
    table = {1: "搜索「项目预算」", 2: "搜索「人力成本」",
             3: "搜索「时间线」", 4: "搜索「竞品价格」",
             5: "汇总所有搜索结果，生成报告"}
    if step in table:
        return f"动作：{table[step]}"
    return "动作：完成，输出最终答案。"


# ============ 主循环 ============
def run_with_budget(max_steps: int = 7):
    system_prompt = "你是一个调研助手，负责收集信息并汇总报告。"
    traj = Trajectory(system_prompt)
    budget = BudgetAllocator(max_tokens=2048)  # 故意设小，方便触发压缩

    for step in range(1, max_steps + 1):
        prompt = traj.render_prompt()
        used = traj.total_tokens()
        status = budget.check(used)
        print(f"[步骤 {step}] 已用 {used} tokens，预算状态：{status}")

        # 先按预算状态压缩，再调模型——否则这一步就爆窗了
        if status == "light_compress":
            traj.light_compress()
        elif status == "hard_compress":
            traj.hard_compress()

        thought = f"第 {step} 步的思考：我需要继续收集信息，判断当前材料是否足够汇总。"
        action = fake_llm(step)
        # 模拟工具返回一大段文本（重复多次放大 token，模拟真实网页正文）
        observation = f"[搜索结果 {step}] 项目预算约 500 万，人力 20 人，周期 6 个月，含详细分项与历史同比。" * 12

        traj.add_step(thought, action, observation)
        if "完成" in action:
            print(f"\n✅ 任务完成，共执行 {step} 步。")
            break

    print(f"\n最终 prompt 长度：{traj.total_tokens()} tokens")


if __name__ == "__main__":
    run_with_budget()
```

这个实现里，`BudgetAllocator` 是预算管理的核心，`Trajectory` 负责维护历史并在需要时压缩。把 `fake_llm` 换成真实模型、把硬编码的 observation 换成真实工具返回，就是一个带记忆管理的 Agent 骨架。

> 🔬 **跑起来你会看到**：每步 `used` 先涨，触发 `light_compress` 后涨势被摁住，再涨触发 `hard_compress` 后总量骤降。这就是"主动预算管理"和"等爆窗再处理"的区别。

## 6. 小结

1. 上下文窗口是硬约束，不是"尽量塞"而是"必须做取舍"——截断、压缩、外部存储是三种基本策略。
2. 长任务的 token 消耗是**二次累计增长**：第 $t$ 步发 $S + \sum_{i=1}^{t}H_i$，十步累计约为单步的 $T(T+1)/2$ 倍，步数越多每步边际成本越高。
3. BPE 分词决定了不同文本的 token 密度差异很大——代码、长 URL、生造变量名都是 token 黑洞；工具层要先清洗返回值。
4. 窗口大≠记得牢：存在"迷失在中间"效应，重要信息放两头。预算分配器要在填满前主动压缩，并给输出留预留空间。

> 📚 **延伸阅读**
> - Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units*（2016）——BPE 的原始论文。
> - Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*（2023）——U 形注意力分布的实验证据。
> - 各家模型的 tokenizer 文档（如 OpenAI tiktoken）——亲手量一次中文、代码、URL 的 token 密度，比记任何经验值都管用。

## 7. 思考题

**基础档**

1. 如果一个任务平均每步产生 800 token 的观察，上下文窗口是 8000 token，系统提示占 1000 token，输出预留 500 token。在不压缩的情况下，最多能跑几步？（列不等式求解）
2. BPE 为什么会让"生造变量名"比"常见词"贵？这对你写工具返回字段有什么指导？

**进阶档**

3. 设 $S=1000$，每步新增观察 $h=500$，步数 $T=10$。请算出累计输入 token 总量，并与"单步 token × 步数"的朴素估算对比，差了多少倍？这个差距说明什么？
4. "轻度压缩"和"重度压缩"各自的风险是什么？什么情况下你会选择不压缩、直接终止任务？

**开放档**

5. 假设你发现模型在第 8 步之后就开始"忘记"第 2 步的关键信息。这更可能是窗口溢出导致的截断，还是长上下文注意力衰减？设计一个实验来区分这两种原因。
6. 如果让你设计一个"成本感知"的 Agent——它每走一步前先估算"这一步值不值得花"，你会用什么信号来决定该继续、该压缩、还是该停手？
