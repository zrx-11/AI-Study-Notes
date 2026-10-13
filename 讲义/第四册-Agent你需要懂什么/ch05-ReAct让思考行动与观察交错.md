# 第5讲 ReAct：让思考、行动与观察交错

> 📌 **本讲问题**：为什么"先想完所有步骤再一次性执行"的方式在复杂任务上总是翻车？因为真实世界的信息是逐步暴露的——你查了第一次才知道第二次该查什么。ReAct 把"思考"和"行动"交错起来，让模型边做边想，这是 Agent 从"问答机器"变成"问题解决者"的关键一跃。

## 1. 工具/预备知识

- **推理（Reasoning）**：模型对当前状态做分析、决定下一步该干什么的内部过程。在 ReAct 里，这个过程被**显式地写出来**（"思考：我需要先查 A"），而不是藏在模型内部。写出来有两个作用：一是给人看（可调试），二是成为下一步的上下文（引导后续动作）。
- **行动（Action）**：模型调用某个工具或输出某个命令，记作 $A_t$。
- **观察（Observation）**：执行行动后环境返回的结果，记作 $O_t$。
- **交错（Interleaving）**：思考→行动→观察→思考→行动→观察……的循环，而不是思考完所有步骤再一次性行动。

> 🧩 **名词先对齐**：ReAct = **Rea**soning + **Act**ing，由 Yao 等人 2022 年的论文 *ReAct: Synergizing Reasoning and Acting in Language Models* 提出。本讲的 Thought/Action/Observation 三元组记号就来自那篇论文。它和第 4 讲的 CoT 是两回事：CoT 只推理不行动，ReAct 推理和行动交替。

## 2. 核心内容

### 2.1 从一次性推理到交错推理

传统的思维链（CoT，第 4 讲）是这样的：给模型一个问题，让它把所有推理步骤一次性写出来，然后给出答案。这种方式的前提是：**推理所需的所有信息都已经在问题里了**。

但很多任务不满足这个前提。比如"查一下今天北京的天气，然后告诉我该不该带伞"——问题里没有天气数据，你必须先去查。CoT 模式下模型只能"编造"一个天气数据然后基于编造的数据推理，结果当然是错的（这就是**幻觉**的一个来源）。

ReAct 的思路是：不要一次性把所有推理写完，而是每推进一步，就去行动一次，拿真实的观察回来，再基于观察继续推理。形式化地，ReAct 的轨迹是：

$$
\text{Thought}_1 \rightarrow \text{Action}_1 \rightarrow \text{Observation}_1 \rightarrow \text{Thought}_2 \rightarrow \text{Action}_2 \rightarrow \text{Observation}_2 \rightarrow \dots \rightarrow \text{Thought}_T \rightarrow \text{Final Answer}
$$

### 2.2 状态转移：ReAct 到底在转什么

把第 1 讲的形式化套到 ReAct 上，状态 $s_t$ 就是"截至目前拼好的整段轨迹"：

$$
s_t = \big( \text{问题}, \underbrace{T_1, A_1, O_1,\ T_2, A_2, O_2,\ \dots,\ T_{t-1}, A_{t-1}, O_{t-1}}_{\text{历史（逐轮追加）}} \big)
$$

策略 $\pi_\theta$ 在状态 $s_t$ 上吐出两件事：一句**思考** $T_t$ 和一个**动作** $A_t$：

$$
(T_t, A_t) = \pi_\theta(s_t)
$$

然后环境执行 $A_t$，返回观察 $O_t$，新状态 $s_{t+1} = s_t \cup (T_t, A_t, O_t)$（在后面追加）。直到模型吐出一个特殊动作"给出最终答案"，循环终止。

把这个式子拆开看，有两个关键点：

1. **Thought 本身会被写进状态**。它不是过程的副产品，而是下一步推理的"脚手架"。模型下一步会读到自己上一步写的"我需要查降水概率"，于是顺着往下做。
2. **Observation 来自环境，不来自模型**。这正是 ReAct 区别于 CoT 的本质：CoT 的中间"事实"是模型猜的，ReAct 的中间事实是环境给的。接地（grounding）由此而来。

### 2.3 思考的四种模式

ReAct 里的"思考"不是随便想，而是有几种典型模式：

1. **分解型思考**：把大目标拆成小步骤。"我需要先查 A，再查 B，然后把 A 和 B 的结果结合起来回答 C。"
2. **校验型思考**：检查上一步的观察是否够用来回答当前子问题。"刚才查到的温度只有 15 度，但我还不知道会不会下雨，需要再查降水概率。"
3. **重试型思考**：上一步行动失败了，分析原因换个方式。"搜索'北京天气'返回的结果太笼统，我换成'今天下午三点北京朝阳区降水概率'再搜一次。"
4. **收敛型思考**：信息收集得差不多了，开始整合。"现在我有温度、降水概率、风力三个数据，可以综合判断该不该带伞了。"

这四种模式不是互斥的，一个完整轨迹里通常四种都出现。好的 ReAct prompt 会在系统提示里把这几种思考类型列出来，让模型知道"你现在应该做哪种思考"。

### 2.4 三种范式对比

| | CoT | ReAct | 先规划再执行 |
|---|---|---|---|
| 信息来源 | 全靠模型已有知识 | 每步查真实环境 | 先想全，再执行 |
| 对外界意外 | 无法应对 | 边走边调整 | 计划一错全错 |
| 幻觉风险 | 高（编造中间事实） | 低（观察接地） | 中 |
| 步数/成本 | 低 | 高（多次工具调用） | 中 |
| 适合任务 | 信息已在题面 | 需逐步查资料 | 目标明确、路径可预知 |

一句话：**CoT 适合"想清楚"，ReAct 适合"边查边想"，规划式适合"想清楚再动手"**。

### 2.5 停止条件与"假完成"

ReAct 循环什么时候停？三种停止条件同时生效：

1. **模型自己说停**：Thought 里输出"信息够了"，然后给出 Final Answer。最理想，但模型可能"假完成"——觉得够了，其实漏了关键信息。
2. **达到最大步数**：工程上必须设上限（如 10 步），防止"搜索→不满意→再搜索"死循环（第 1 讲的保险丝）。
3. **异常退出**：连续 N 次工具调用失败，或工具返回明确错误，就终止并报告失败。

针对"假完成"，后面第 7 讲的"反思"机制会兜底——让另一个（或同一个）模型回头检查"答案真的覆盖了用户要求吗"。

### 2.6 优势、代价与局限

优势：**接地**（每步有真实观察支撑）、**自适应**（遇意外可调方向）、**可调试**（轨迹可回溯）。

代价：**token 消耗大**（每步重发全历史，第 2 讲的二次增长）、**延迟高**（串行）、**错误传播**（一步观察偏，步步跟着偏）。

局限：原论文也发现，ReAct 并非总是优于 CoT——在信息本就充足的任务上，多绕几步工具调用反而引入噪声。所以它只适合**需要外部信息才能解决的问题**。

### 2.7 一个完整轨迹样例

拿"今天北京需要带伞吗"走一遍：

- **Thought 1**：用户问带伞，需要天气数据，现在没有。
- **Action 1**：`search[北京今天天气]`
- **Observation 1**：北京多云，15°C，降水概率 60%。
- **Thought 2**：降水概率 60%，且多云，信息足够，可以判断了。
- **Final Answer**：今天北京多云转小雨，降水概率 60%，建议带伞。

每一步思考都短，且明确指向下一个动作。

### 2.8 再看一个需要"重试"的轨迹

不是每个任务都一两步收敛。下面这个例子展示校验型和重试型思考如何接力：

- **Thought 1**：用户问"某公司去年营收同比增速"。我需要先查它的营收数字。
- **Action 1**：`search[某公司 营收]`
- **Observation 1**：搜到一堆新闻，但只有今年营收，没有去年的。
- **Thought 2**（校验型）：观察里缺去年数据，算不出同比。我得专门搜去年的财报。
- **Action 2**：`search[某公司 去年 财报 营收]`
- **Observation 2**：去年营收 80 亿，今年 100 亿。
- **Thought 3**（收敛型）：两年数据都有了，$(100-80)/80 = 25\%$。
- **Action 3**：`calculate[(100-80)/80*100]`
- **Observation 3**：结果: 25.0
- **Final Answer**：该公司营收同比增长 25%。

注意这里：如果第一步搜完模型就"假完成"，它会拿一个不完整的数据硬算。是 Thought 2 那句"缺去年数据"的校验，把轨迹拉回了正轨。**写 Thought 的价值，在这种地方最明显。**

## 3. 💡 直觉理解

> 💡 **直觉**：ReAct 就像你去一个陌生城市找一家餐厅。一次性推理（CoT）的方式是：出门前对着地图把路线全想好——左转、右转、坐几站地铁、出哪个口。真到现场发现那条路在施工，整个计划全废。ReAct 的方式是：出门→走到第一个路口→看看路标→决定往哪走→走到下一个路口→再看看路标……每一步都基于眼前真实情况做决定。慢一点，但不会走错。
>
> 而把每一步"我刚才怎么想的"都写在纸上，还有个额外好处：你回头看纸，能想起自己为什么这么走——这就是轨迹可调试。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：认为 ReAct 的"思考"越长越好。实际上思考太长占大量 token，且常重复。更好的做法是：思考要短、要 actionable——一句话说清"我现在知道了什么、下一步干什么"。
>
> ⚠️ **误区二**：把 ReAct 和 CoT 混为一谈。CoT 是"在文本内部推理"，没有工具、没有环境反馈；ReAct 是"推理和真实动作交错"。把只有思考没有观察的轨迹也叫 ReAct，是概念错误。
>
> ⚠️ **误区三**：以为有了 ReAct 就不幻觉了。ReAct 只是把"事实类幻觉"压下去，模型依然可能在 Thought 里错误解读观察、或把两个不相关的观察错误关联。

## 5. 💻 代码实现

下面实现一个完整的 ReAct 循环，带模拟工具集。`fake_react_llm` 根据当前轨迹规则化地决定下一步。

```python
from dataclasses import dataclass, field
from typing import List, Dict
import json


# ============ 工具集（确定性模拟） ============
def tool_search(query: str) -> str:
    if "天气" in query:
        return json.dumps({"城市": "北京", "温度": "15°C",
                           "天气": "多云转小雨", "降水概率": "60%"}, ensure_ascii=False)
    elif "人口" in query:
        return json.dumps({"城市": "北京", "人口": "2189万"}, ensure_ascii=False)
    return json.dumps({"结果": f"关于「{query}」的通用结果。"}, ensure_ascii=False)


def tool_calculate(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    cleaned = ''.join(c for c in expression if c in allowed).strip()
    if not cleaned:
        return "错误：无可计算内容。"
    try:
        return f"结果: {eval(cleaned)}"
    except Exception:
        return "计算错误。"


TOOLS = {"search": tool_search, "calculate": tool_calculate}


# ============ ReAct 轨迹 ============
@dataclass
class ReActTrajectory:
    question: str
    steps: List[Dict] = field(default_factory=list)   # thought, action, observation

    def render(self) -> str:
        parts = [f"问题: {self.question}", ""]
        for i, s in enumerate(self.steps, 1):
            parts += [f"Thought {i}: {s['thought']}",
                      f"Action {i}: {s['action']}",
                      f"Observation {i}: {s['observation']}", ""]
        parts.append("Thought (下一步):")
        return "\n".join(parts)


# ============ 模拟 LLM ============
def fake_react_llm(traj: ReActTrajectory) -> Dict:
    """规则模拟：根据已走步数和已有观察，决定下一个 Thought+Action。"""
    q, n = traj.question, len(traj.steps)

    if n == 0:
        if "带伞" in q or "穿" in q:
            return {"thought": "需要天气数据，先查天气。",
                    "action_type": "search", "action_input": "北京今天天气"}
        elif "人口" in q:
            return {"thought": "需要查北京人口。",
                    "action_type": "search", "action_input": "北京人口"}
        return {"thought": "无需外部信息，直接回答。",
                "action_type": "final", "action_input": "答案就在题面里。"}

    if n == 1 and ("带伞" in q or "穿" in q):
        return {"thought": "已有温度和降水概率，信息足够，收敛作答。",
                "action_type": "final",
                "action_input": "今天北京15°C，多云转小雨，降水概率60%，建议带伞。"}

    if n == 1 and "人口" in q:
        return {"thought": "已有人口数据，收敛作答。",
                "action_type": "final", "action_input": "北京常住人口约2189万。"}

    return {"thought": "信息足够，给出最终答案。",
            "action_type": "final", "action_input": "综合已有信息回答。"}


# ============ ReAct 主循环 ============
def run_react(question: str, max_steps: int = 5):
    traj = ReActTrajectory(question=question)
    print(f"问题: {question}\n")
    for step in range(1, max_steps + 1):
        decision = fake_react_llm(traj)
        print(f"Thought {step}: {decision['thought']}")

        if decision["action_type"] == "final":
            print(f"Final Answer: {decision['action_input']}\n✅ 完成，共 {step} 步。\n")
            return decision["action_input"]

        action_str = f"{decision['action_type']}[{decision['action_input']}]"
        print(f"Action {step}: {action_str}")
        tool_func = TOOLS.get(decision["action_type"])
        observation = tool_func(decision["action_input"]) if tool_func \
            else f"[错误] 未知工具 {decision['action_type']}"
        print(f"Observation {step}: {observation}\n")

        traj.steps.append({"thought": decision["thought"],
                           "action": action_str, "observation": observation})

    print(f"❌ 达到最大步数 {max_steps}。\n")
    return None


if __name__ == "__main__":
    run_react("今天北京天气怎么样？需要带伞吗？")
    print("=" * 50)
    run_react("北京有多少人口？")
    print("=" * 50)
    run_react("1+1等于几？")   # 不需要工具的情况：直接 final
```

> 🔬 **跑起来你会看到**：前两个问题各走"查一次工具→收敛作答"两步；第三个问题模型第一步就判断"无需外部信息"直接 final——这演示了 ReAct 的一个重要设计：**不是每个问题都值得走工具循环**，模型要先判断"信息够不够"。
>
> 这个 `fake_react_llm` 用规则把"什么情况该搜、什么情况该收尾"写死了。真实场景里，这个判断由大模型根据系统提示里列的四种思考模式自己做——而 `ReActTrajectory.render()` 渲染出来的那段文本，就是喂给真实模型的 prompt。把规则换成一次 API 调用，骨架不变。

## 6. 小结

1. ReAct 的核心是"思考—行动—观察"交错循环。状态 $s_t$ 是逐轮追加的轨迹，策略在 $s_t$ 上吐出 $(T_t, A_t)$，环境返回 $O_t$。
2. 它与 CoT 的本质区别在于观察来自真实环境而非模型想象——这既降低幻觉，也带来错误传播。
3. 思考有四种模式：分解、校验、重试、收敛；停止三重保障：自主完成、最大步数、异常退出，并警惕"假完成"。
4. 代价是 token 大、延迟高；只适合需要外部信息的任务。

> 📚 **延伸阅读**
> - Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*（2022）——本讲的原始论文，看它如何用少量 few-shot 样例让模型学会 Thought/Action/Observation 格式。
> - 对比阅读：Wei et al. CoT（2022），看清"只推理"和"推理+行动"的差别。

## 7. 思考题

**基础档**

1. 写出 ReAct 状态 $s_t$ 的构成，并说明为什么 Thought 本身也要被写进状态。
2. ReAct 和 CoT 的本质区别是什么？各举一个"用错方法会失败"的例子。

**进阶档**

3. 为什么说 ReAct 的错误会"传播"？举一个两步轨迹的例子，说明第一步观察有偏差如何带偏第二步思考。
4. "帮我订一张明天去上海的机票"这个任务，ReAct 大概要走几步？写出每步的 Thought/Action/Observation。在哪一步最可能"假完成"？

**开放档**

5. ReAct 和"先写完整计划再执行"相比，各适合什么任务？什么情况下你宁愿牺牲灵活性也要先规划？
6. ReAct 靠 few-shot 样例教会模型格式，但不同任务的"思考风格"差异很大。你会如何为一个新领域（如医疗问诊）设计 ReAct 的系统提示？
