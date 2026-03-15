# Language Models are Few-Shot Learners (GPT-3)

## 层 1：电梯演讲

**一句话概括**：OpenAI 团队在 2020 年提出 1750 亿参数的 GPT-3 模型，通过大规模预训练和上下文学习（in-context learning），在无需微调的情况下实现少样本（few-shot）、零样本（zero-shot）学习，在多个 NLP 任务上达到 SOTA，展示了大语言模型的涌现能力。

---

## 层 2：故事摘要

### 背景：2020 年的语言模型困境

时间来到 2020 年，语言模型领域面临一个根本性问题：

**问题：任务特定微调的局限性**

```
传统范式（2018-2019）：
┌────────────────────────────────────────┐
│ 1. 通用预训练（语言建模）               │
│ 2. 任务特定微调（分类、问答等）          │
│ 3. 每个任务需要一个微调模型             │
└────────────────────────────────────────┘

问题：
- 需要大量标注数据（数千到数万样本）
- 每个任务单独微调，无法共享知识
- 小模型容易过拟合，泛化能力有限
```

**人类的学习方式**：

想象一个受过良好教育的人面对新任务：

```
任务："把这句话翻译成法语：The cat sat on the mat"

人类反应：
1. 理解任务类型（翻译）
2. 调用已有的法语知识
3. 直接翻译，无需额外训练

但如果是一个没学过的任务呢？

任务："把这个句子改写成古英语风格"

给 1-2 个例子后：
输入："The cat sat on the mat" → 输出："The cat did sit upon the mat"
输入："The dog ran in the park" → 输出："The dog did run within the park"
输入："The bird flew over the tree" → ???

人类可以：立即学会并应用模式
```

**核心问题**：
> "能否让语言模型像人类一样，通过几个例子就学会新任务，而无需梯度更新？"

### GPT-3 的核心洞察

OpenAI 团队提出了一个大胆假设：

**假设：规模即能力**

```
┌─────────────────────────────────────────────────────────┐
│                 规模假说（Scaling Hypothesis）           │
│                                                         │
│  当模型参数量和训练数据量达到某个阈值时：                 │
│  1. 模型获得"元学习"能力（学习如何学习）                  │
│  2. 可以从上下文中的例子推断任务模式                     │
│  3. 无需梯度更新就能完成新任务                           │
└─────────────────────────────────────────────────────────┘
```

**三种学习模式**：

```
┌────────────────────────────────────────────────────────┐
│  1. Zero-Shot（零样本）                                 │
│  ─────────────────────────────────────────────────────  │
│  输入："翻译这句话为法语：The cat sat on the mat"       │
│  输出："Le chat s'est assis sur le tapis"              │
│                                                         │
│  无需任何示例，直接完成任务                             │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  2. One-Shot（单样本）                                  │
│  ─────────────────────────────────────────────────────  │
│  输入：                                                 │
│  "英语→法语：Hello → Bonjour                           │
│   英语→法语：The cat sat on the mat → ?"              │
│  输出："Le chat s'est assis sur le tapis"              │
│                                                         │
│  仅需 1 个示例，理解任务模式                              │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│  3. Few-Shot（少样本）                                  │
│  ─────────────────────────────────────────────────────  │
│  输入：                                                 │
│  "英语→法语：                                         │
│   Hello → Bonjour                                      │
│   Good morning → Bonjour la matinée                    │
│   The cat sat on the mat → Le chat s'est assis...      │
│   The dog ran in the park → ?"                        │
│  输出："Le chien a couru dans le parc"                 │
│                                                         │
│  几个示例后，任务理解更准确                             │
└────────────────────────────────────────────────────────┘
```

### 实验结果震撼学界

**GPT-3 模型系列**：

| 模型 | 参数量 | 上下文学习？ |
|------|--------|-------------|
| GPT-3 Small | 125M | 弱 |
| GPT-3 Medium | 350M | 弱 |
| GPT-3 Large | 760M | 中等 |
| GPT-3 XL | 1.3B | 明显 |
| GPT-3 Large | 2.7B | 强 |
| GPT-3 XLarge | 6.7B | 很强 |
| **GPT-3 13B** | **13B** | **极强** |
| **GPT-3 175B** | **175B** | **SOTA** |

**关键发现**：

```
任务性能
    │
SOTA│                              ● GPT-3 (175B)
    │                        ●
    │                   ●
    │              ●
    │         ●
    │    ●
    └────────────────────────────────→ 参数量
       125M  1B    10B   100B  175B

小模型：需要微调才能完成任务
大模型：few-shot 就能达到相同效果
超大模型：zero-shot 就接近 SOTA
```

**代表性任务结果**：

| 任务 | GPT-3 Few-Shot | 之前 SOTA (微调) | 差距 |
|------|---------------|-----------------|------|
| TriviaQA (知识问答) | 64.3 (12-shot) | T5+SSM 55.0 | +9.3 ✓ |
| Natural Questions | 41.5 (64-shot) | T5+SSM 36.6 | +4.9 ✓ |
| SQuAD 2.0 (阅读理解) | 50.6 (0-shot) | 微调 63 | -12.4 |
| LAMBADA (上下文理解) | 86.4 (0-shot) | 之前 63 | +23.4 ✓ |
| Winogrande (常识推理) | 77.5 (5-shot) | 微调 79 | -1.5 |
| QuAC (问答) | 52 (1-shot) | 微调 59 | -7 |

**关键成就**：
- 在知识密集型任务上超越微调模型
- LAMBADA 达到 86.4，远超之前的 63
- 展示了惊人的少样本学习能力

### 涌现能力的发现

最令人兴奋的是**涌现能力**（Emergent Abilities）：

```
任务：三 digit 加法

125M 模型：234 + 567 = ? → "791" (错)
350M 模型：234 + 567 = ? → "791" (错)
760M 模型：234 + 567 = ? → "801" (接近)
1.3B 模型：234 + 567 = ? → "801" (接近)
6.7B 模型：234 + 567 = ? → "801" (对！)
175B 模型：234 + 567 = ? → "801" (对！)

能力不是逐渐提升，而是在某个规模阈值后突然涌现！
```

---

## 层 3：深度精读

### 开场：一个失败的尝试

**2019 年，OpenAI 办公室**

研究人员正在分析 GPT-2（1.5B 参数）在少样本学习上的表现：

```
任务：情感分类

示例：
"这部电影太棒了！" → 正面
"剧情很无聊。" → 负面
"演员表演精彩。" → ???

GPT-2 输出："精彩"
预期输出："正面"

问题：GPT-2 无法从少量示例中理解任务模式
```

**分析**：
- GPT-2 看到示例后，继续生成"合理"的文本
- 但它没有理解这是在做一个分类任务
- 1.5B 参数不足以支持"元学习"能力

**关键洞察**：
> "也许问题不在于架构或算法，而在于规模不够大？"

### 第一章：GPT-3 架构设计

#### 1.1 模型配置

GPT-3 使用标准 Transformer decoder-only 架构，但在规模上做了系统性探索：

```
┌─────────────────────────────────────────────────────────┐
│  GPT-3 模型系列配置                                      │
├──────────┬─────────┬───────┬────────┬──────────────────┤
│ 模型     │ 参数    │ 层数  │ 注意力 │ 每层前馈维度      │
├──────────┼─────────┼───────┼────────┼──────────────────┤
│ 125M     │ 125M    │ 12    │ 12     │ 4×hidden         │
│ 350M     │ 350M    │ 24    │ 16     │ 4×hidden         │
│ 760M     │ 760M    │ 24    │ 16     │ 4×hidden         │
│ 1.3B     │ 1.3B    │ 24    │ 16     │ 4×hidden         │
│ 2.7B     │ 2.7B    │ 32    │ 16     │ 4×hidden         │
│ 6.7B     │ 6.7B    │ 32    │ 16     │ 4×hidden         │
│ 13B      │ 13B     │ 40    │ 20     │ 4×hidden         │
│ 175B     │ 175B    │ 96    │ 12     │ 4×hidden         │
└──────────┴─────────┴───────┴────────┴──────────────────┘

注意：175B 模型使用更少的注意力头（12 个），但每头维度更大 (d_k = 128)
```

**为什么 175B 参数的配置特殊**？

```
175B 模型：
- hidden_dim = 12288
- num_heads = 12
- d_k = 12288 / 12 = 1024（每头维度很大）

vs 标准配置：
- num_heads = 96（如果按 4×hidden 比例）
- d_k = 128（每头维度小）

设计理由：
- 更大的每头维度，增强表达能力
- 减少注意力头的通信开销
- 经验验证效果更好
```

#### 1.2 位置编码

GPT-3 使用**绝对位置编码**（ learned positional embeddings）：

```python
# 与原始 Transformer 不同，GPT-3 使用可学习的位置编码
class GPT3Embeddings(nn.Module):
    def __init__(self, vocab_size, max_position, hidden_dim):
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_position, hidden_dim)
        # 可学习的位置编码

    def forward(self, input_ids):
        positions = torch.arange(input_ids.size(1)).to(input_ids.device)
        return self.token_embedding(input_ids) + self.position_embedding(positions)
```

**为什么不用正弦位置编码**？
- 可学习的编码更灵活
- 可以适应不同任务的需求
- 实验效果略好

#### 1.3 稀疏注意力？

**重要发现**：GPT-3 **没有**使用稀疏注意力！

```
探索过的稀疏注意力模式：
┌────────────────────────────────────────┐
│ 1. 固定稀疏（每个 token 关注局部 + 全局） │
│ 2. 滑动窗口（只关注附近 token）          │
│ 3. 分层稀疏（底层局部，高层全局）        │
└────────────────────────────────────────┘

结果：所有稀疏注意力都损害了性能！

结论：
- 对于 few-shot learning，全局注意力至关重要
- 稀疏性可能丢失关键的上下文信息
- 175B 模型必须使用标准稠密注意力
```

### 第二章：训练数据与配置

#### 2.1 训练数据集

GPT-3 的训练数据是**5 个数据集的混合**：

```
┌─────────────────────────────────────────────────────────┐
│  训练数据组成（约 300B tokens）                          │
├─────────────────────────────────────────────────────────┤
│  Common Crawl (2016-2019)    ~60%  (~180B tokens)      │
│  ↓                                                      │
│  - 41TB 压缩文本                                         │
│  - 质量较低，需要过滤                                   │
│  - 覆盖互联网多样化内容                                 │
├─────────────────────────────────────────────────────────┤
│  WebText2                        ~22%  (~66B tokens)   │
│  ↓                                                      │
│  - Reddit 高质量链接                                    │
│  - 比 Common Crawl 质量更高                             │
├─────────────────────────────────────────────────────────┤
│  Books1                         ~8%  (~24B tokens)     │
│  ↓                                                      │
│  - Project Gutenberg 等书籍                            │
│  - 长文本，连贯性强                                    │
├─────────────────────────────────────────────────────────┤
│  Books2                        ~8%  (~24B tokens)      │
│  ↓                                                      │
│  - 额外书籍语料                                         │
├─────────────────────────────────────────────────────────┤
│  Wikipedia                     ~3%  (~9B tokens)       │
│  ↓                                                      │
│  - 英文 Wikipedia 全文                                 │
│  - 事实性知识丰富                                     │
└─────────────────────────────────────────────────────────┘
```

**数据处理**：

```python
# Common Crawl 过滤策略
def filter_common_crawl(text):
    # 1. 基于启发式规则过滤低质量
    if len(text) < 200:  # 太短
        return False
    if text.count('.') < 3:  # 不是完整句子
        return False

    # 2. 与高质量语料相似度检查
    if similarity_to_wikipedia(text) > 0.9:  # 重复
        return False

    # 3. 分类器过滤
    if quality_classifier.predict(text) < 0.5:  # 低质量
        return False

    return True

# 结果：从 41TB 压缩到约 10TB 高质量文本
```

#### 2.2 训练优化

**分布式训练配置**：

```
硬件：
- 数百个 NVIDIA V100 GPU（16GB/32GB）
- 使用模型并行 + 数据并行

175B 模型训练细节：
- 模型并行度：8-way（模型切分到 8 个 GPU）
- 数据并行度：多组模型并行副本
- 总 GPU 数：约 300-400 个

训练时间：
- 约 3-4 个月
- 使用混合精度训练（FP16）
```

**优化器配置**：

```python
# Adam 优化器
optimizer = Adam(
    lr=6e-4,          # 学习率
    betas=(0.9, 0.95), # β1, β2
    eps=1e-8,
    weight_decay=0.1   # 权重衰减
)

# 学习率调度
# 前 375M tokens：线性预热
# 之后：余弦退火到 0
```

**关键技巧**：

1. **梯度检查点**（Gradient Checkpointing）：
   - 节省显存，以计算换内存
   - 只保存部分激活值，需要时重计算

2. **ZeRO 优化**（数据并行）：
   - 优化器状态分片
   - 梯度分片
   - 参数分片（可选）

3. **流水线并行**（模型并行）：
   - 将 Transformer 层切分到不同 GPU
   - 层间流水线执行

### 第三章：上下文学习（In-Context Learning）

#### 3.1 什么是上下文学习？

**定义**：
> 模型通过输入上下文中的示例，在没有梯度更新的情况下学会执行新任务。

**与传统微调对比**：

```
┌─────────────────────────────────────────────────────────┐
│  传统微调                                                │
│  ─────────────────────────────────────────────────────   │
│  1. 收集任务特定数据（数千样本）                         │
│  2. 初始化预训练模型                                    │
│  3. 梯度下降更新参数                                    │
│  4. 得到任务特定模型                                    │
│                                                         │
│  缺点：每个任务需要一个模型，无法泛化到新任务            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  上下文学习（GPT-3）                                     │
│  ─────────────────────────────────────────────────────   │
│  1. 准备几个示例（0-10 个）                               │
│  2. 将示例放入输入提示（prompt）                         │
│  3. 模型直接生成答案                                    │
│  4. 参数不变，无需训练                                  │
│                                                         │
│  优点：一个模型处理所有任务，即时适应                    │
└─────────────────────────────────────────────────────────┘
```

#### 3.2 上下文学习的机制

**问题**：GPT-3 是如何从上下文"学会"任务的？

**理论 1：隐式贝叶斯推理**

```
模型看到示例：
"Hello → Bonjour"
"Good morning → Bonjour la matinée"

模型推理：
"这看起来像翻译任务，源语言是英语，目标语言是法语"
"下一个输入应该是英语，我应该输出法语"

形式化：
P(任务 | 示例) ∝ P(示例 | 任务) × P(任务)

模型从示例中推断出潜在的任务类型
```

**理论 2：模式匹配**

```
训练时，模型见过大量类似模式：
"英语→法语：X → Y"

测试时，看到相似模式：
"英语→法语：A → ?"

模型匹配到训练时的模式，生成类似响应
```

**理论 3：元学习**

```
在预训练中，模型学会：
- 如何快速从示例中提取规则
- 如何识别任务类型
- 如何应用学到的规则

这是一种"学习如何学习"的元能力
```

#### 3.3 实验：Few-Shot vs Fine-Tuning

**表格 3.6 深度解读**（部分任务）：

| 任务 | GPT-3 Few-Shot | 微调 SOTA | 差距 |
|------|---------------|-----------|------|
| **TriviaQA** | 64.3 (12-shot) | 55.0 (T5+SSM) | **+9.3** ✓ |
| **Natural Questions** | 41.5 (64-shot) | 36.6 (T5+SSM) | **+4.9** ✓ |
| SQuAD 2.0 | 50.6 (0-shot) | 63.0 | -12.4 |
| **LAMBADA** | **86.4 **(0-shot) | 63.0 | **+23.4** ✓ |
| Winogrande | 77.5 (5-shot) | 79.0 | -1.5 |
| QuAC | 52 (1-shot) | 59.0 | -7 |
| DROP | 52 (5-shot) | 56.0 | -4 |

**关键洞察**：
- 知识密集型任务（TriviaQA、NQ）：GPT-3 显著超越
- 阅读理解任务（SQuAD）：仍有差距
- 上下文理解（LAMBADA）：巨大优势

**为什么知识任务表现好**？
```
GPT-3 训练数据包含：
- Wikipedia 全文
- 大量书籍和网页
- 覆盖广泛的事实知识

Few-shot 时，模型可以直接"回忆"训练过的知识
而微调模型参数量小，知识容量有限
```

### 第四章：涌现能力

#### 4.1 什么是涌现能力？

**定义**：
> 当模型规模达到某个阈值时，突然获得的能力，小模型完全不具备这种能力。

**经典例子：三位数加法**

```
任务：计算 234 + 567

125M 模型：输出 "791"（完全错误）
350M 模型：输出 "791"（完全错误）
760M 模型：输出 "801"（接近）
1.3B 模型：输出 "801"（接近）
6.7B 模型：输出 "801"（正确！）
175B 模型：输出 "801"（正确！）

准确率随规模变化：
100% │                  ●●●
     │              ●
 50% │          ●
     │      ●
  0% │  ●
     └────────────────────────→ 参数量
       125M  1B    10B   100B

能力不是线性增长，而是在某个点后突然涌现！
```

#### 4.2 其他涌现能力

**1. 词性标注**（POS Tagging）
```
输入："The cat sat on the mat"
输出：DET NOUN VERB PREP DET NOUN

小模型：完全无法理解任务
大模型：few-shot 下准确率 > 90%
```

**2. 命名实体识别**（NER）
```
输入："Elon Musk founded SpaceX in 2002"
输出：Elon Musk (PERSON), SpaceX (ORG), 2002 (DATE)

小模型：输出随机
大模型：准确识别实体类型
```

**3. 翻译**（低资源语言对）
```
输入：" translate to Swahili: Hello"
小模型：输出无意义内容
大模型：输出正确的 Swahili 翻译 "Jambo"
```

#### 4.3 涌现的理论解释

**解释 1：临界质量假说**

```
模型需要达到一定的"知识密度"才能执行复杂推理：

小模型：
- 知识稀疏，难以建立关联
- "猫"和"动物"的概念可能不连通

大模型：
- 知识密集，概念网络完整
- 可以通过多跳推理得到答案
```

**解释 2：电路假说**

```
Transformer 中的某些子网络形成"电路"，负责特定能力：

小模型：
- 电路不完整，信号无法传导

大模型：
- 电路完整，可以执行复杂操作
- 类似数字电路中的逻辑门组合
```

### 第五章：偏见与风险

#### 5.1 偏见问题

GPT-3 在训练数据中吸收了社会偏见：

```
测试：职业 - 性别关联

输入："The doctor said that"
输出："she" vs "he" 的概率

结果：
P("he") = 0.85
P("she") = 0.15

偏见：医生默认是男性

输入："The nurse said that"
输出：
P("she") = 0.80
P("he") = 0.20

偏见：护士默认是女性
```

**其他偏见**：
- 种族刻板印象
- 宗教偏见
- 政治倾向

#### 5.2 缓解措施

论文讨论了但未完全解决偏见问题：

```
尝试过的方法：
1. 过滤训练数据中的偏见内容
   → 难以定义什么是"偏见"

2. 对抗性训练
   → 效果有限

3. 后处理过滤
   → 可能过滤掉合理内容

结论：偏见缓解仍是开放问题
```

#### 5.3 滥用风险

**潜在滥用**：
- 生成虚假信息
- 垃圾内容自动化
- 钓鱼邮件
- 学术不端

**OpenAI 的应对**：
- 不公开完整模型（仅提供 API）
- 监控滥用行为
- 内容过滤

### 第六章：局限性

#### 6.1 知识更新问题

```
GPT-3 训练数据截止时间：2019 年 10 月

输入："2023 年诺贝尔文学奖得主是谁？"
输出：（错误或胡说）

问题：模型无法获取训练后的新知识
```

**解决方向**：
- 检索增强（RAG）
- 定期重新训练
- 参数高效微调

#### 6.2 推理能力有限

```
逻辑推理任务：

输入："如果 A 在 B 左边，B 在 C 左边，那么 A 在 C 哪边？"
GPT-3：有时答对，有时答错

问题：模型不是真正的逻辑推理，而是模式匹配
```

#### 6.3 长程一致性

```
生成长文本时：

前文："John 是个素食主义者"
后文："John 点了一份牛排"

问题：模型忘记或忽视前面的信息
```

### 第七章：代码示例

#### 7.1 GPT-3 最小实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPT3Config:
    def __init__(self, vocab_size=50257, hidden_dim=12288,
                 num_layers=96, num_heads=12, max_position=2048):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_position = max_position
        self.d_k = hidden_dim // num_heads  # 1024

class GPT3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 嵌入层
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_position, config.hidden_dim)

        # Transformer 层
        self.layers = nn.ModuleList([
            GPT3Block(config) for _ in range(config.num_layers)
        ])

        # 层归一化
        self.ln_f = nn.LayerNorm(config.hidden_dim)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 位置编码
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        positions = self.position_embedding(positions)

        # token 嵌入
        x = self.token_embedding(input_ids) + positions

        # 因果掩码（防止看到未来）
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Transformer 层
        for layer in self.layers:
            x = layer(x, causal_mask)

        # 最终层归一化
        x = self.ln_f(x)

        return x  # (batch, seq_len, hidden_dim)

class GPT3Block(nn.Module):
    """单个 Transformer 层"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_dim)
        self.attn = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim)
        )

    def forward(self, x, mask):
        # 自注意力 + 残差
        attn_output, _ = self.attn(
            self.ln_1(x), self.ln_1(x), self.ln_1(x),
            attn_mask=mask
        )
        x = x + attn_output

        # MLP + 残差
        mlp_output = self.mlp(self.ln_2(x))
        x = x + mlp_output

        return x

class GPT3ForCausalLM(nn.Module):
    """GPT-3 语言模型"""
    def __init__(self, config):
        super().__init__()
        self.transformer = GPT3Model(config)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # 权重共享（嵌入和输出层）
        self.lm_head.weight = self.transformer.token_embedding.weight

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Transformer 编码
        hidden_states = self.transformer(input_ids, attention_mask)

        # 语言模型头
        logits = self.lm_head(hidden_states)

        # 计算损失（如果提供 labels）
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

        return logits, loss

    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=None):
        """自回归生成"""
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                # 前向传播
                logits, _ = self(input_ids)
                next_token_logits = logits[:, -1, :] / temperature

                # Top-K 采样
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # 采样
                next_token = torch.multinomial(
                    F.softmax(next_token_logits, dim=-1), num_samples=1
                )

                # 拼接到输入
                input_ids = torch.cat([input_ids, next_token], dim=-1)

            return input_ids
```

#### 7.2 Few-Shot Prompting

```python
def create_few_shot_prompt(task, examples, test_input):
    """构建 few-shot prompt"""
    prompt = ""

    for example in examples:
        prompt += f"Input: {example['input']}\n"
        prompt += f"Output: {example['output']}\n\n"

    prompt += f"Input: {test_input}\n"
    prompt += "Output:"

    return prompt

# 示例：情感分类
examples = [
    {"input": "这部电影太棒了！", "output": "正面"},
    {"input": "剧情很无聊。", "output": "负面"},
    {"input": "演员表演精彩。", "output": "正面"},
]

test_input = "特效令人惊叹，但故事薄弱。"
prompt = create_few_shot_prompt("情感分类", examples, test_input)

print(prompt)
# 输出：
# Input: 这部电影太棒了！
# Output: 正面
# ...
# Input: 特效令人惊叹，但故事薄弱。
# Output:
```

### 第八章：与其他方法对比

#### 8.1 与 T5 对比

| 方面 | T5 (2020) | GPT-3 (2020) |
|------|-----------|--------------|
| 架构 | Encoder-Decoder | Decoder-only |
| 最大模型 | 11B | 175B |
| 训练目标 | 去噪自编码 | 因果语言建模 |
| 任务适应 | 必须微调 | Few-shot/Zero-shot |
| 知识能力 | 有限 | 强大 |

#### 8.2 与 BERT 对比

| 方面 | BERT (2018) | GPT-3 (2020) |
|------|-------------|--------------|
| 架构 | Encoder-only | Decoder-only |
| 注意力 | 双向 | 单向（因果） |
| 训练目标 | Masked LM | Causal LM |
| 任务适应 | 必须微调 | Few-shot/Zero-shot |
| 生成能力 | 无 | 强 |

### 第九章：历史定位

```
时间线：
─────────────────────────────────────────────────────────→
2018      2019      2020      2021      2022      2023
  │         │         │         │         │         │
  │         │         │         │         │         └── GPT-4
  │         │         │         │         └── ChatGPT, InstructGPT
  │         │         │         └── GPT-3 API 开放
  │         │         │
  │         │         └── GPT-3 (本文)
  │         │
  │         └── GPT-2 (1.5B)
  │
  └── BERT, GPT

影响：
- 开创了大语言模型时代
- 展示了规模带来的涌现能力
- Few-shot learning 成为标准范式
- 为 ChatGPT 和 GPT-4 奠定基础
```

---

## 结语

**Language Models are Few-Shot Learners** 是大语言模型历史上的里程碑。它证明了：

> **规模本身就是能力**（Scale is an Ability）

GPT-3 的核心贡献：

1. **系统性探索语言模型规模**
   - 从 125M 到 175B 的 8 个模型
   - 揭示了规模与能力的关系

2. **发现 Few-Shot Learning**
   - 无需梯度更新，从上下文学习
   - 一个模型处理所有任务

3. **观察到涌现能力**
   - 超过阈值后突然获得的能力
   - 小模型完全无法企及

4. **展示了知识密集型任务的优势**
   - TriviaQA 等任务超越微调模型
   - 证明了大规模预训练的价值

**历史意义**：

GPT-3 开启了"大模型时代"，后续的 ChatGPT、GPT-4、Claude、LLaMA 等都建立在 GPT-3 的基础之上。

当你今天使用 ChatGPT 时，它的能力——从对话理解到代码生成——都可以追溯到 GPT-3 展示的核心洞察：

> **训练一个足够大的语言模型，它就能学会做任何事情**。

这，就是 GPT-3 的革命性。
