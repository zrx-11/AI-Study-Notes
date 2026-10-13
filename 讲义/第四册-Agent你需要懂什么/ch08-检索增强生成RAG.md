# 第8讲 检索增强生成 RAG

> 📌 **本讲问题**：模型的参数化知识是训练时冻结的——它不知道昨天的新闻，也不知道你公司内部文档写了什么。怎么让模型用上"训练时没见过"的新知识？RAG 的答案是：不要改模型参数，把知识放在外部数据库里，用的时候现查现用。

## 1. 工具/预备知识

- **向量嵌入（Embedding）**：把一段文本映射成一个高维向量（如 768 维），语义相近的文本向量距离也近。
- **向量数据库**：专门存大量向量、支持快速相似度搜索的数据库。给一个查询向量，它能在毫秒级返回最相近的 top-k 条。
- **余弦相似度**：衡量两个向量方向有多一致的指标，下面 2.2 推导。
- **重排序（Reranking）**：初筛回来的 top-k 条可能不够准，再用更精细但更慢的模型重新排序，把最相关的排到前面。

> 🧩 **先建立直觉**：embedding 模型把"意思相近"的句子映射到向量空间里相邻的位置。所以"电车销量"和"新能源汽车销量"虽然字不一样，向量却很近——这就是它能跨表述匹配的原因。下面会看到，这个能力很强，但也有盲区。

## 2. 核心内容

### 2.1 RAG 流水线

一个最朴素的 RAG 系统由五步组成：

1. **文档切分（Chunking）**：把长文档切成小块（如每块 300–500 字）。为什么要切？因为 embedding 模型对超长文本编码不好，而且检索时你只想把相关那一段塞回 prompt，不想塞整篇。
2. **向量化（Embedding）**：用 embedding 模型把每个文本块转成向量。
3. **入库（Indexing）**：把向量 + 原始文本块 + 元数据（来源、时间、作者）存进向量数据库。
4. **检索（Retrieval）**：用户提问时把问题也转成向量，在库里找最相似的 top-k 块。
5. **生成（Generation）**：把检索到的块作为上下文拼进 prompt，让模型基于这些上下文回答。

整个流水线写成：

$$
\text{Answer} = p_\theta\big(\text{Question} \mid \text{TopK}(\text{Embed}(\text{Question}), \text{DB})\big)
$$

其中 $\text{Embed}(\cdot)$ 是嵌入函数，$\text{TopK}(\cdot, \text{DB})$ 是从数据库取最相似的 k 条。注意这个式子和微调的区别：模型参数 $\theta$ 没变，变的只是条件里多了检索来的证据。

### 2.2 余弦相似度：为什么看方向不看长度

常用的距离度量是余弦相似度：

$$
\text{sim}(q, d) = \frac{q \cdot d}{\|q\| \,\|d\|}
$$

分子 $q \cdot d = \sum_i q_i d_i$ 是点积，分母 $\|q\|=\sqrt{\sum_i q_i^2}$ 是向量的模长。整个式子其实就是两个向量夹角的余弦。取值 $[-1,1]$，越接近 1 越同向（越相似）。

**为什么除以模长？** 因为我们关心的是"方向（语义）"，不是"长度（信息量大小）"。一段写得很长的文档和一段很短的摘要，只要语义同向，余弦相似度就高。如果不除模长（只用点积），那越长的文档天然得分越高——这是我们不想要的。这个归一化让相似度只反映"讲的是不是一回事"。

### 2.3 召回率 vs 精确率：k 不是越大越好

这是 RAG 里最核心的一组权衡，必须讲透。假设答案真的藏在知识库的某一块里，我们定义：

- **召回率（Recall@k）**：top-k 条结果里，**有没有**把那块正确文档捞回来。k 越大，越不容易漏掉正确文档，召回越高。
- **精确率（Precision@k）**：top-k 条结果里，真正相关的占多大比例。k 越大，混进来的噪声文档越多，精确率越低。

这两者是反着走的：

- k 取小（如 3）：精确率高，塞进 prompt 的都是精华，但可能把正确文档漏掉（召回低）。
- k 取大（如 20）：召回高，基本不会漏，但 prompt 里混了一堆无关文档，既浪费 token 又会误导模型。

工程上怎么解这个矛盾？就是 2.4 的"两段式"：先用大 k 做粗筛保召回，再用重排序精挑保精确。**把"要不要捞全"和"要不要挑准"拆成两步，分别优化。**

### 2.4 粗筛与精排：两段式检索

初筛阶段为了快，通常用近似最近邻搜索（ANN），返回 top-20。这 20 条里噪声多。重排序阶段用一个更精细的模型（交叉编码模型），逐条计算"问题—文档"的真实相关度，重排后取 top-3 塞进 prompt。

为什么要分两段？因为精排慢——它要把问题和文档拼在一起过一次模型，20 条过 20 次。如果直接对十万条文档都精排，延迟不可接受。所以：**粗筛保召回、精排保精确**——先用快而糙的 ANN 从十万条捞 20 条候选，再用慢而准的精排从 20 条里挑最好的 3 条。

### 2.5 向量检索的盲区：语义相似 ≠ 信息相关

向量检索的核心假设是"语义相似 = 信息相关"，但这个假设会破。比如问题是"苹果公司的营收是多少"，向量检索可能返回讲"苹果种植技术"的文档——语义上都和"苹果"有关，信息却完全不相关。这叫**语义漂移**。

补救手段：

- **混合检索（Hybrid Search）**：把向量检索（懂语义）和关键词检索（BM25，懂字面）结合，两路结果合并。"苹果公司"这种专名，关键词检索反而更稳。
- **元数据过滤**：先按"公司=苹果、年份=2024"这类硬条件过滤，再做向量相似度。
- **重排序模型**：交叉编码模型见过大量"问题—文档"对，对"看起来相关其实没用"的辨别能力比向量强。

### 2.6 RAG 与 Agent 的关系

RAG 本身不是 Agent——它只是个检索增强问答系统。但它是 Agent 工具箱里的一个重要工具：Agent 把"检索知识库"当工具，需要查资料时就调它；Agent 还能决定"查什么关键词、查几条、要不要重排"。

换句话说，RAG 是"腿和脚"，Agent 是"大脑"。RAG 负责走到书架找到那本书，Agent 负责决定"现在该去书架找书了"（第 5 讲的 ReAct 里，`search` 工具背后往往就是一整套 RAG）。

### 2.7 切分策略：块太大太小都不行

切分（chunking）直接影响检索质量，是个常被低估的工程点：

- **切太大**（如整篇文章一块）：embedding 会把整篇意思揉成一团，问题只涉及一小段时向量被稀释，相似度反而不高；塞进 prompt 也全是无关内容。
- **切太碎**（如每句一块）：每块上下文不全，单看一句"它增长了 35%"，不知道"它"指什么；检索到了也答不了。
- **常用折中**：按语义边界切（按段落/小标题），块之间留重叠（overlap），避免把一个完整论断切成两半。

企业里常见做法是"合同按条款切、手册按章节切、网页按 DOM 块切"——让切分边界贴合文档本身的结构。

## 3. 💡 直觉理解

> 💡 **直觉**：RAG 就像开卷考试。模型记住的知识点有限（参数化知识），就像你考前背的那些；考试可以带课本——遇到不会的题，翻课本找到相关章节，照着组织答案。向量检索就是"翻课本找相关段落"，embedding 就是"根据题目猜哪一页可能有答案"，重排序就是翻到那几页后再仔细看哪页最切题。
>
> 而召回/精确的权衡就像：翻太多页（k 大），答案肯定翻到了，但你把整本笔记都抄进卷子，阅卷的（模型）反而看花眼；翻太少页（k 小），干净利落但可能正好漏了那一页。

## 4. ⚠️ 常见误区

> ⚠️ **误区一**：认为"检索到了相关文档，模型就一定能用上"。模型可能忽略检索到的上下文，尤其当参数化知识和检索内容冲突时，它倾向信自己训练时见过的"更可靠"知识。解法：prompt 里写"请只基于以下资料回答"，并把资料放在开头或结尾（首因近因，第 2 讲）。
>
> ⚠️ **误区二**：k 越大越好。k 大只提升召回，却拉低精确率、撑爆上下文。正确做法是大 k 粗筛 + 重排后取小 k 进 prompt。
>
> ⚠️ **误区三**：以为 RAG 能解决一切知识问题。检索没捞对文档时，再强的生成模型也只能基于错误或缺失的证据瞎编——RAG 的天花板由检索质量决定。

## 5. 💻 代码实现

下面实现一个迷你 RAG：文档切分、词袋向量模拟 embedding、余弦相似度检索、重排序、生成。全程确定性、不调 API。

```python
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
import re


# ============ 简易向量（词袋模拟 embedding） ============
def simple_embed(text: str) -> Dict[str, int]:
    """模拟 embedding：词频作稀疏向量。真实场景是神经网络输出稠密向量。"""
    words = re.findall(r'[a-z]+|[一-鿿]', text.lower())
    vec: Dict[str, int] = {}
    for w in words:
        vec[w] = vec.get(w, 0) + 1
    return vec


def cosine_sim(v1: Dict[str, int], v2: Dict[str, int]) -> float:
    """两个稀疏向量的余弦相似度 = 点积 / (模长 * 模长)。"""
    common = set(v1) & set(v2)
    dot = sum(v1[k] * v2[k] for k in common)
    norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
    norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0.0


# ============ 文档库 ============
@dataclass
class Document:
    doc_id: str
    title: str
    content: str


class SimpleVectorDB:
    def __init__(self):
        self.chunks: List[Dict] = []

    def add_document(self, doc: Document, chunk_size: int = 60):
        content = doc.content
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
        for idx, chunk_text in enumerate(chunks):
            self.chunks.append({
                "text": chunk_text,
                "vector": simple_embed(chunk_text),
                "metadata": {"doc_id": doc.doc_id, "title": doc.title, "chunk": idx},
            })

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
        q_vec = simple_embed(query)
        scored = [(cosine_sim(q_vec, c["vector"]), c) for c in self.chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]


# ============ 重排序器 ============
def rerank(query: str, candidates: List[Tuple[float, Dict]], top_k: int = 3) -> List[Dict]:
    """在初筛相似度上加一个"关键词命中"加分，模拟交叉编码精排。"""
    q_words = set(re.findall(r'[a-z]+|[一-鿿]', query.lower()))
    scored = []
    for base_sim, chunk in candidates:
        text_words = set(re.findall(r'[一-鿿]+|[a-z]+', chunk["text"].lower()))
        overlap = len(q_words & text_words)
        scored.append((base_sim + overlap * 0.1, chunk))   # 相似度 + 关键词重叠加分
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# ============ 模拟生成器 ============
def fake_generate(query: str, contexts: List[Dict]) -> str:
    if not contexts:
        return "根据现有资料无法回答这个问题。"
    body = "\n".join(f"[{c['metadata']['title']}] {c['text']}" for c in contexts)
    return f"根据参考资料回答「{query}」：\n{body}"


# ============ 完整流水线 ============
def run_rag(query: str):
    db = SimpleVectorDB()
    db.add_document(Document("doc1", "新能源汽车市场",
        "2024年中国新能源汽车销量达1200万辆，同比增长35%。主要厂商比亚迪、特斯拉、吉利。"))
    db.add_document(Document("doc2", "电池技术",
        "主流动力电池是磷酸铁锂和三元锂。磷酸铁锂成本低但能量密度略低。"))
    db.add_document(Document("doc3", "苹果公司",
        "苹果公司2024财年营收3830亿美元，主要产品iPhone、Mac、iPad。"))

    print(f"查询: {query}\n")
    candidates = db.search(query, top_k=5)
    print("=== 初筛（粗筛，保召回）===")
    for sim, chunk in candidates:
        print(f"  相似度={sim:.3f} | [{chunk['metadata']['title']}] {chunk['text'][:25]}...")
    print()

    top = rerank(query, candidates, top_k=3)
    print("=== 重排序后（精排，保精确）===")
    for c in top:
        print(f"  [{c['metadata']['title']}] {c['text'][:35]}...")
    print()
    print("=== 生成回答 ===")
    print(fake_generate(query, top))


if __name__ == "__main__":
    run_rag("中国新能源汽车市场规模有多大？")
```

> 🔬 **跑起来你会看到**：初筛可能把"苹果营收"那条也带进来（因为都含"营收/市场"字样），重排序后相关的"新能源汽车销量"排到前面。这个"先捞全、再挑准"的两步，正是召回/精确权衡在代码里的体现。

这个 `simple_embed` 用词袋（按字）模拟 embedding，是为了让你不依赖任何模型就能跑。真实系统换成神经网络 embedding 后，"电车销量"和"新能源汽车销量"会因为语义接近而向量相近——这正是词袋做不到、而稠密 embedding 擅长的跨表述匹配。理解了余弦公式，换什么 embedding 模型，这套流水线都不用改。

## 6. 小结

1. RAG 五步：切分→向量化→入库→检索→生成；不改参数就能注入外部知识。
2. 余弦相似度 = 点积除以双模长，只看方向不看长度；向量检索匹配语义相似度，不等于信息相关性。
3. **召回率 vs 精确率的权衡**：k 大召回高但精确低，工程上用"大 k 粗筛 + 重排精排"把两者拆开优化。
4. RAG 是 Agent 的工具不是 Agent 本身；它的天花板由检索质量决定，混合检索和重排缓解语义漂移。

> 📚 **延伸阅读**
> - Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*（2020）——RAG 原始论文。
> - 关于 ANN（如 HNSW）和混合检索（BM25 + dense），可搜相关工程博客，理解"粗筛为何快"。

## 7. 思考题

**基础档**

1. 为什么余弦相似度要除以两个模长？如果只算点积会出现什么问题？
2. 文档切分时切得太碎（每句一块）和切得太大（整篇一块），分别有什么坏处？

**进阶档**

3. 定义 Recall@k 和 Precision@k。为什么增大 k 会让一个升、一个降？两段式检索是怎么同时保住两者的？
4. 用户说"电车卖得怎么样"，文档写的是"新能源汽车销量"。纯向量检索能匹配上吗？纯关键词检索呢？混合检索为什么更好？

**开放档**

5. 检索到了正确文档，模型却仍然用自己的旧知识答错。除了在 prompt 里强调"只依据资料"，你还能从工程上做什么？
6. RAG 的"知识库"和第 9 讲的"长期记忆系统"有什么区别？知识库算长期记忆的一种吗？二者该如何分工？
