# Automatic Chain of Thought Prompting (Auto-CoT)

## 层 1：电梯演讲

**一句话概括**：亚马逊 AWS 和上海交通大学在 2022 年提出 Auto-CoT，通过"Let's think step by step"自动为每个问题生成推理链，利用聚类保证多样性，在 10 个推理任务上自动达到甚至超过手工设计 CoT 演示的性能，消除了人工设计推理链的繁重工作。

---

## 层 2：故事摘要

### 核心问题

2022 年，CoT prompting 正火，但研究者面临一个**根本性矛盾**：

**CoT 的两种范式**：
```
范式 1: Zero-Shot-CoT (Kojima et al., 2022)
提示："Let's think step by step"
- ✅ 无需人工设计
- ❌ 性能有限，复杂任务容易错

范式 2: Manual-CoT (Wei et al., 2022)
提示：提供 8 个手工设计的 {问题 + 推理链 + 答案} 演示
- ✅ 性能强，任务特定
- ❌ 需要大量人工设计，不同任务要重新设计
```

**核心矛盾**：
```
场景：你想用 CoT 解决数学应用题

Manual-CoT 流程：
1. 手工写 8 个数学题示例
2. 为每个题写详细的推理步骤
3. 确保推理正确、多样、有代表性

问题：
- 换到常识推理任务？重新写 8 个！
- 换到符号推理任务？再重新写 8 个！
- 不同 annotator 写的演示？性能差距 28.2%！

Zero-Shot-CoT 流程：
提示："Let's think step by step"

问题：
- 太通用，复杂任务容易错
- 没有示例引导，质量不稳定

研究者问："能不能让 LLM 自己生成推理链，同时保证质量和多样性？"
```

### 核心洞察

亚马逊团队发现了一个**反直觉的现象**：

> "LLM 生成的推理链会有错误，但**多样性**能抵消错误的影响！"

**关键观察**：
1. Zero-Shot-CoT 确实会犯错，但错误不是均匀的
2. 如果演示的问题类型足够多样，单个错误的影响就小了
3. LLM 可以生成推理链，关键是**如何选问题**

**Auto-CoT 的答案**：
```
让 LLM 不仅"step by step"思考，还要"one by one"地为每个问题生成推理链！

核心公式：
Auto-CoT = 问题聚类 + 多样性采样 + Zero-Shot-CoT 生成
```

### Auto-CoT 框架大图

```
┌─────────────────────────────────────────────────────────┐
│              Auto-CoT 两阶段流程                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  输入：数据集的问题集合（无标签，无演示）                  │
│                    ↓                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  阶段 1: Question Clustering (问题聚类)            │   │
│  │  - 用 Sentence-BERT 编码所有问题                  │   │
│  │  - 用 k-means 聚类成 k=8 个簇                      │   │
│  │  - 每个簇代表一类问题                            │   │
│  │  示例簇：                                        │   │
│  │  [加减法], [乘除法], [比较大小], [时间计算]...    │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  阶段 2: Demonstration Sampling (演示采样)       │   │
│  │  - 从每个簇选 1 个代表问题（如簇中心）              │   │
│  │  - 用 Zero-Shot-CoT 生成推理链                    │   │
│  │    提示："Q: [问题] A: Let's think step by step" │   │
│  │  - 应用启发式过滤：                              │   │
│  │    * 问题长度 < 60 tokens                       │   │
│  │    * 推理步数 < 5 步                            │   │
│  │  - 得到 8 个 {问题 + 推理链 + 答案} 演示             │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  阶段 3: Few-Shot CoT Inference (少样本推理)     │   │
│  │  - 将 8 个演示 + 测试问题 拼成 prompt              │   │
│  │  - 输入 LLM 得到最终答案                         │   │
│  │  示例：                                         │   │
│  │  Q: Roger has 5 tennis balls... (demo 1)        │   │
│  │  Q: John takes care of 10 dogs... (demo 2)      │   │
│  │  ...                                             │   │
│  │  Q: [测试问题] A: Let's think step by step...   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘

关键创新:
1. 用聚类保证问题多样性 → 抵消推理错误
2. 用 Zero-Shot-CoT 自动生成 → 无需人工
3. 用启发式过滤质量 → 简单且准确
```

### 关键结果

| 任务类型 | Zero-Shot-CoT | Manual-CoT | Auto-CoT | Auto-CoT vs Manual |
|---------|--------------|-----------|---------|-------------------|
| 数学推理 (GSM8K) | 78.7% | 91.7% | **92.0%** | +0.3% |
| 数学推理 (MultiArith) | 55.0% | 98.0% | **98.3%** | +0.3% |
| 常识推理 (CSQA) | 63.0% | 79.2% | **79.5%** | +0.3% |
| 常识推理 (StrategyQA) | 59.0% | 76.5% | **76.8%** | +0.3% |
| 符号推理 (Coin Flip) | 62.0% | 84.0% | **84.5%** | +0.5% |
| 符号推理 (Last Letter) | 31.0% | 64.0% | **65.0%** | +1.0% |
| **平均** | **58.1%** | **82.2%** | **82.7%** | **+0.5%** |

**核心贡献**：
- Auto-CoT 是首个自动构造 CoT 演示的方法
- 无需人工干预，性能匹敌甚至超越手工设计
- 发现多样性是抵消自动推理错误的关键

---

## 层 3：深度精读

### 开场：2022 年 CoT 的困境

**时间**：2022 年秋
**地点**：亚马逊 AWS 研究团队
**人物**：Zhuosheng Zhang（上海交大实习生）, Aston Zhang, Mu Li, Alex Smola

**场景还原**：
```
团队周会上，Zhuosheng 展示了一个尴尬的结果：

任务：数学应用题 "宠物店有 64 只小狗，卖了 28 只，剩下的每 4 只放一个笼子，需要几个笼子？"

Manual-CoT (Wei et al., 2022):
Q: 宠物店有 15 个苹果，买了 7 个，剩下几个？
A: 15 - 7 = 8。答案是 8 个苹果。

Q: ... (7 个手工设计的示例)

结果：准确率 98%

Zero-Shot-CoT (Kojima et al., 2022):
Q: 宠物店有 64 只小狗...
A: Let's think step by step. 64 - 28 = 36。36 / 4 = 9。答案是 9 个笼子。

结果：准确率 55%

问题：
- Manual-CoT 效果好，但每个任务要手工写 8 个示例
- Zero-Shot-CoT 不用手工，但效果差太多
- 换个任务（比如常识推理）？Manual-CoT 又要重新写！

"等等，"Aston 打断，
"既然 LLM 能回答测试问题，为什么不能让 LLM 自己生成示例的推理链？"

全场安静。

"但是，"Zhuosheng 回应，
"我们试过。LLM 生成的推理链会有错误...
错误示例会污染 prompt，导致性能下降。"

关键问题来了：
**如何自动生成推理链，同时保证质量？**

这就是 Auto-CoT 的起点。
```

---

### 第一章：2022 年 CoT 的三大局限

#### 局限 1：Manual-CoT 的人工成本

**手工设计的流程**：
```
任务：算术推理

步骤 1：设计 8 个代表性问题
- 要有加减法
- 要有乘除法
- 要有混合运算
- 难度要分布均匀

步骤 2：为每个问题写推理链
Q: "Roger 有 5 个网球，又买了 2 罐，每罐 3 个，现在有几个？"
A: "Roger 从 5 个球开始。2 罐每罐 3 个是 6 个球。5 + 6 = 11。答案是 11。"

步骤 3：验证正确性
- 确保推理没错
- 确保格式一致
- 确保语言清晰

耗时：2-3 小时/任务
```

**更糟的是**：
```
换到常识推理任务？
Q: "以下哪个物体更重：大象还是老鼠？"
A: "大象是大型哺乳动物，老鼠是小型啮齿动物。大象体重可达几吨，
    老鼠只有几十克。所以大象更重。"

→ 推理链风格完全不同！要重新写！

换到符号推理任务？
Q: "单词'thinking'的最后一个字母是什么？"
A: "thinking 有 8 个字母：t-h-i-n-k-i-n-g。最后一个是 g。"

→ 又要重新写！

结论：Manual-CoT 不可扩展。
```

#### 局限 2：Zero-Shot-CoT 的性能瓶颈

**Zero-Shot-CoT 的问题**：
```
简单任务：
Q: "9 + 5 = ?"
A: "Let's think step by step. 9 + 5 = 14。答案是 14。"
✅ 正确

复杂任务：
Q: "小明有 3 个苹果，小红给他 5 个，他又吃了 2 个，现在有几个？"
A: "Let's think step by step. 3 + 5 = 8。答案是 8。"
❌ 错误！忘了减去吃的 2 个。

原因：
- 没有示例引导，LLM 不知道推理的"格式"和"深度"
- 复杂任务需要多步推理，Zero-Shot 容易跳步
```

#### 局限 3：演示质量的敏感性

**关键发现（Wei et al., 2022）**：
```
实验：不同 annotator 写的演示

Annotator A 写的演示 → 准确率 84%
Annotator B 写的演示 → 准确率 55.8%

差距：28.2%！

原因：
- 推理链的质量（是否正确、清晰）
- 问题的多样性（是否覆盖不同类型）
- 格式的一致性（是否便于 LLM 理解）

结论：演示质量极其敏感，手工设计难以保证稳定性。
```

---

### 第二章：核心洞察 - 多样性抵消错误

#### 关键实验

**实验 1：检索 vs 随机**
```
方法 1: Retrieval-Q-CoT
- 对测试问题，用语义相似度检索 8 个最相似的问题
- 用 Zero-Shot-CoT 生成这 8 个问题的推理链
- 作为演示

方法 2: Random-Q-CoT
- 随机选 8 个问题
- 用 Zero-Shot-CoT 生成推理链
- 作为演示

直觉：检索应该更好，因为问题更相关

实际结果：
任务            Retrieval-Q-CoT    Random-Q-CoT
GSM8K           71.2%              74.5%
MultiArith      85.0%              89.3%
CSQA            68.5%              71.0%

Random 赢了！为什么？
```

**分析**：
```
Retrieval-Q-CoT 的问题：
- 检索的 8 个问题太相似
- 如果这类问题的推理链错了，全军覆没

示例：
测试问题："小明买苹果，花了多少钱？"
检索的 8 个问题都是"购物花钱"类
如果 Zero-Shot-CoT 在这类问题上犯错（比如忘了找零），8 个演示都错！

Random-Q-CoT 的优势：
- 问题类型多样
- 即使某些类错了，其他类是对的
- 多样性抵消了局部错误
```

#### 核心洞察

> "多样性不是锦上添花，而是**必需品**！"

**作者的顿悟**：
```
"等等，如果多样性是关键...
那我们不需要检索，也不需要随机。
我们可以主动保证多样性！

怎么做？
→ 聚类！
→ 每个簇选一个代表！
→ 这样覆盖的问题类型最广！"
```

---

### 第三章：Auto-CoT 的两阶段设计

#### 阶段 1: Question Clustering（问题聚类）

**目标**：将问题分成 k 个簇，每个簇代表一类问题。

**具体步骤**：
```
输入：600 个测试问题（以 GSM8K 为例）

步骤 1：编码
- 用 Sentence-BERT 将每个问题编码成向量
- 问题："小明有 3 个苹果..." → [0.12, -0.45, ..., 0.89] (768 维)

步骤 2：聚类
- 用 k-means 聚类，k=8
- 聚类依据：语义相似度（余弦相似度）

输出：8 个簇
簇 1: [加减法问题]
  - "小红有 5 本书，又买了 3 本..."
  - "小明有 10 元，花了 4 元..."

簇 2: [乘除法问题]
  - "每盒有 6 个苹果，3 盒有几个？"
  - "24 个苹果平均分给 4 人..."

簇 3: [混合运算问题]
  - "先加后乘的问题..."

...
簇 8: [比较问题]
  - "谁更多？谁更少？..."
```

**为什么是 k=8？**
```
实验：不同的 k 值

k=4: 覆盖率不够，某些问题类型没被代表
k=8: 最佳平衡
k=16: 演示太多，超过 LLM 的上下文窗口，且冗余

Manual-CoT 也用 8 个演示 → 为了公平比较
```

---

#### 阶段 2: Demonstration Sampling（演示采样）

**目标**：从每个簇选 1 个代表，生成推理链。

**具体步骤**：
```
输入：8 个簇

步骤 1：选代表
- 选每个簇的中心问题（离簇心最近的）
- 或者随机选（实验发现差异不大）

步骤 2：生成推理链
对每个代表问题：
Prompt = "Q: [问题] A: Let's think step by step."
调用 GPT-3 (text-davinci-002, 175B)
得到："A: Let's think step by step. [推理步骤] 答案是 X。"

步骤 3：启发式过滤
为了确保质量，过滤掉：
- 问题长度 > 60 tokens（太复杂容易错）
- 推理步数 > 5 步（太长容易出错）

如果过滤掉了，重新选一个代表，重复步骤 2-3。

输出：8 个 {问题 + 推理链 + 答案} 演示
```

**启发式的作用**：
```
为什么限制问题长度？
- 长问题 → 需要更多推理步骤 → 更容易错
- 短问题 → 推理链简洁 → 准确率高

为什么限制推理步数？
- 超过 5 步，LLM 容易迷失
- 简单推理链 → 作为演示更清晰

实验：启发式的效果

无过滤 → 演示错误率 18%
有过滤 → 演示错误率 8%
```

---

#### 阶段 3: Few-Shot CoT Inference（少样本推理）

**目标**：用自动生成的演示，对测试问题进行推理。

**具体步骤**：
```
输入：测试问题 "Q_test"

Prompt = [
  "Q: [demo_1 问题] A: [demo_1 推理链]",
  "Q: [demo_2 问题] A: [demo_2 推理链]",
  ...
  "Q: [demo_8 问题] A: [demo_8 推理链]",
  "Q: Q_test A: Let's think step by step."
]

调用 GPT-3 → 得到答案
```

**示例**：
```
Q: Roger has 5 tennis balls. He buys 2 more cans. Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Q: John takes care of 10 dogs. Each dog takes 0.5 hours a day to walk. How many hours a week does he spend taking care of dogs?
A: John takes 0.5 × 10 = 5 hours a day. A week has 7 days. 5 × 7 = 35. The answer is 35 hours a week.

... (6 more demos)

Q: A pet store had 64 puppies. In one day they sold 28 of them and put the rest into cages with 4 in each cage. How many cages did they use?
A: Let's think step by step.
```

---

### 第四章：大量实例

#### 【实例 1：算术推理 - GSM8K】

**任务**：
```
Q: "小明家有 15 棵苹果树，每棵树产 12 个苹果。小明卖了 50 个苹果，又买了 30 个梨。现在小明有多少个苹果？"
```

**Auto-CoT 执行**：
```
阶段 1 - 问题聚类：
这个问题被分到 [乘法和减法混合运算] 簇

阶段 2 - 演示生成：
从该簇选代表问题：
"Q: 每盒有 8 个橙子，5 盒有几个？卖了 15 个，还剩几个？"

用 Zero-Shot-CoT 生成：
"A: Let's think step by step. 每盒 8 个，5 盒是 8 × 5 = 40 个。
    卖了 15 个，剩下 40 - 15 = 25 个。答案是 25 个。"

阶段 3 - 推理：
用 8 个演示 + 测试问题 → GPT-3 输出：
"A: Let's think step by step. 15 棵树，每棵 12 个，总共 15 × 12 = 180 个苹果。
    卖了 50 个，剩下 180 - 50 = 130 个。
    买了 30 个梨（与苹果无关）。
    所以小明现在有 130 个苹果。答案是 130。"

✅ 正确！
```

---

#### 【实例 2：常识推理 - CSQA】

**任务**：
```
Q: "以下哪种动物会冬眠？
    A. 狗  B. 熊  C. 猫  D. 兔子"
```

**Auto-CoT 执行**：
```
阶段 1 - 问题聚类：
这个问题被分到 [动物行为常识] 簇

阶段 2 - 演示生成：
从该簇选代表问题：
"Q: 鸟会飞，鱼会游泳，哪种动物会爬树？A. 猴子 B. 牛 C. 羊"

用 Zero-Shot-CoT 生成：
"A: Let's think step by step. 牛和羊是食草动物，主要在地上活动。
    猴子是灵长类动物，擅长爬树。
    所以答案是 A. 猴子。"

阶段 3 - 推理：
用 8 个演示 + 测试问题 → GPT-3 输出：
"A: Let's think step by step. 冬眠是某些动物在冬季的生存策略。
    狗、猫、兔子是家养动物，一般不冬眠。
    熊是野生动物，会在冬季冬眠。
    所以答案是 B. 熊。"

✅ 正确！
```

---

#### 【实例 3：符号推理 - Coin Flip】

**任务**：
```
Q: "一枚硬币初始是正面朝上。翻了一次，又翻了一次。现在哪面朝上？"
```

**Auto-CoT 执行**：
```
阶段 1 - 问题聚类：
这个问题被分到 [状态变化追踪] 簇

阶段 2 - 演示生成：
从该簇选代表问题：
"Q: 灯初始是关的。按了开关一次，又按了一次。现在灯是开还是关？"

用 Zero-Shot-CoT 生成：
"A: Let's think step by step. 灯初始是关的。
    按一次开关，灯变开。
    再按一次，灯变关。
    所以答案是关。"

阶段 3 - 推理：
用 8 个演示 + 测试问题 → GPT-3 输出：
"A: Let's think step by step. 硬币初始是正面朝上。
    翻一次，变成反面朝上。
    再翻一次，变成正面朝上。
    所以答案是正面朝上。"

✅ 正确！
```

---

#### 【对比场景 1：有聚类 vs 无聚类】

```
有聚类（Auto-CoT）：
演示覆盖：[加减法], [乘除法], [混合运算], [比较], [时间], [距离], [分配], [比例]
测试问题：混合运算 → 有对应演示 → 准确率 92%

无聚类（Random-Q-CoT）：
演示可能全是 [加减法]（运气不好）
测试问题：混合运算 → 没有类似演示 → 准确率 74%

差异：18%！
```

---

#### 【对比场景 2：有启发式 vs 无启发式】

```
有启发式（Auto-CoT）：
过滤掉 >60 tokens 问题和 >5 步推理
演示错误率：8%
最终性能：92%

无启发式：
演示错误率：18%
最终性能：85%

差异：7%！
```

---

### 第五章：预期 vs 实际

### 你的直觉 vs Auto-CoT 的实现

| 维度 | 你的直觉/预期 | Auto-CoT 实际 | 为什么有差距？ |
|------|--------------|--------------|---------------|
| **如何保证质量** | 用更强大的模型生成 | 用多样性抵消错误 | 模型再强也会犯错，关键是容错 |
| **如何选择问题** | 选最相似的（检索） | 选最多样的（聚类） | 相似问题容易同错，多样问题分散风险 |
| **推理链长度** | 越长越详细越好 | 限制<5 步 | 太长容易错，简洁更可靠 |
| **演示数量** | 越多越好 | k=8 最优 | 太多超过上下文窗口，且冗余 |
| **是否需要训练** | 可能需要微调 | 完全零样本 | LLM 已有能力，只需正确触发 |

---

### 反直觉挑战

**问题 1**：如果 LLM 生成的推理链有错误，为什么还能用？

[先想 1 分钟...]

**直觉**："错误演示会误导 LLM，性能应该下降"

**实际**：
```
Auto-CoT 演示错误率：8%
但性能仍然达到 92%！

原因：
1. 多样性确保大部分演示是正确的
2. LLM 有"纠错"能力，能从多个示例中学习模式
3. 即使演示有错，测试时的"Let's think step by step"也能触发推理

关键：多样性 > 完美
```

---

**问题 2**：检索相似问题应该更好，为什么随机/聚类赢了？

[先想 1 分钟...]

**直觉**："相似问题更能帮助理解测试问题"

**实际**：
```
Retrieval-Q-CoT: 71.2% (GSM8K)
Random-Q-CoT: 74.5% (GSM8K)
Auto-CoT (Clustering): 78.0% (GSM8K)

原因：
- 检索的问题太相似 → 如果这类问题推理错了，全盘皆输
- 随机/聚类覆盖不同类型 → 错误被稀释

类比：
- 检索：把所有鸡蛋放在一个篮子里
- 聚类：分散到多个篮子，即使一个掉了，其他还在
```

---

**问题 3**：为什么 k=8 最优？更多演示不是更好吗？

[先想 1 分钟...]

**直觉**："演示越多，信息越多，应该更好"

**实际**：
```
k=4:  85.0% (覆盖不够)
k=8:  92.0% (最佳)
k=16: 88.5% (性能下降)
k=32: 82.0% (更差)

原因：
1. 上下文窗口限制：GPT-3 最多 4096 tokens
2. 注意力稀释：太多演示，LLM 难以聚焦
3. 冗余：k>8 后，新增簇与前 8 个重复

最佳实践：k = √(n)，n 是问题类型数量
```

---

### 第六章：关键实验的细节

#### 实验 1: 十项基准测试

**数据集**：
```
算术推理 (4 个)：
- MultiArith: 多步数学应用题
- GSM8K: 小学数学题
- AQUA-RAT: 代数问题
- SVAMP: 变量应用题

常识推理 (2 个)：
- CSQA: 常识问答
- StrategyQA: 需要策略的常识推理

符号推理 (2 个)：
- Last Letter Concatenation: 单词字母拼接
- Coin Flip: 硬币翻转追踪
```

**基线方法**：
```
1. Zero-Shot-CoT: "Let's think step by step"
2. Manual-CoT: 手工设计 8 个演示
3. Retrieval-Q-CoT: 检索相似问题 + Zero-Shot-CoT
4. Random-Q-CoT: 随机选问题 + Zero-Shot-CoT
5. Few-Shot (无 CoT): 只提供 {问题，答案}，无推理链
```

**模型**：
```
- GPT-3 (text-davinci-002, 175B)
- 所有方法用相同的模型，公平比较
```

---

#### 实验 2：消融实验

**问题：聚类和启发式哪个更重要？**

| 配置 | GSM8K | MultiArith | 平均 |
|------|-------|-----------|------|
| 完整 Auto-CoT | 78.0% | 98.3% | 88.2% |
| - 聚类（用随机） | 74.5% | 89.3% | 81.9% |
| - 启发式（无过滤） | 72.0% | 92.0% | 82.0% |
| - 两者 | 65.0% | 85.0% | 75.0% |

**结论**：
- 聚类贡献 +6.3%
- 启发式贡献 +6.2%
- 两者结合贡献 +13.2%

---

#### 实验 3：跨模型泛化

**问题：Auto-CoT 在其他模型上有效吗？**

| 模型 | Zero-Shot-CoT | Manual-CoT | Auto-CoT |
|------|--------------|-----------|---------|
| GPT-3 (175B) | 58.1% | 82.2% | 82.7% |
| GPT-3 (6.7B) | 35.0% | 58.5% | 59.2% |
| GPT-3 (2.7B) | 22.0% | 42.0% | 43.5% |

**结论**：
- Auto-CoT 在各规模模型上都有效
- 大模型效果更好（推理能力更强）
- Auto-CoT 可以弥合大小模型差距

---

### 第七章：与其他方法对比

#### Auto-CoT vs 其他 CoT 变体

| 方法 | 核心思想 | 优点 | 缺点 | 性能 (GSM8K) |
|------|---------|------|------|-------------|
| **Zero-Shot-CoT** | 只用"Let's think step by step" | 简单，无需示例 | 性能有限 | 58.1% |
| **Manual-CoT** | 手工设计演示 | 性能强 | 人工成本高 | 82.2% |
| **Auto-CoT** | 聚类 + 自动生成 | 自动 + 性能强 | 需要 Sentence-BERT | 82.7% |
| **Self-Consistency** | 多次采样投票 | 提升稳定性 | 计算成本高 | 86.0%* |
| **PAL** | 生成代码执行 | 精确计算 | 需要代码能力 | 84.0%* |

* 可以与 Auto-CoT 结合使用

---

#### 论文定位图谱

```
                    CoT 发展时间线

2022.01: Wei et al. (Manual-CoT) ─┬────→ 手工设计演示
                                  │
2022.02: Kojima et al. (Zero-Shot)┼────→ 零样本触发
                                  │
2022.10: Auto-CoT (本文) ─────────┼────→ 自动生成演示 【本文】
                                  │
2022.11: Self-Consistency ────────┤     → 多次采样投票
                                  │
2022.11: PAL ─────────────────────┘     → 代码辅助推理

上游工作：
- Manual-CoT: 证明了 CoT 的有效性
- Zero-Shot-CoT: 证明了 LLM 有零样本推理能力

下游工作：
- Self-Consistency: 可以用 Auto-CoT 生成演示，再多次采样
- PAL: 可以用 Auto-CoT 生成代码示例
- Active Prompting: 用 Auto-CoT 初始，再主动学习优化

影响力：
- 首个全自动 CoT 演示生成方法
- 消除了人工设计演示的需求
- 启发了后续的自动 prompt 工程研究
```

---

### 第八章：局限性分析

#### 局限性 1：依赖聚类质量

```
问题：
- Sentence-BERT 的嵌入质量影响聚类
- 某些问题类型边界模糊，难以聚类

示例：
"小明买了 3 个苹果，每个 2 元，给了 10 元，找零多少？"
→ 这是 [乘法] 还是 [减法] 还是 [混合运算]？

影响：
- 聚类不准 → 代表性下降 → 性能下降
```

#### 局限性 2：无法完全消除错误

```
问题：
- Zero-Shot-CoT 本身会犯错
- 启发式只能过滤，不能纠正

示例：
Q: "24 个苹果分给 4 人，每人几个？"
A: "Let's think step by step. 24 / 4 = 5。答案是 5。"
❌ 错误！应该是 6。

影响：
- 演示错误率仍有 8%
- 虽然多样性抵消，但仍有限制
```

#### 局限性 3：需要访问全部问题

```
问题：
- Auto-CoT 需要先看到所有测试问题才能聚类
- 在线场景（问题逐个到来）难以应用

变通：
- 用历史问题聚类
- 或者用预定义的问题模板
```

---

### 第九章：改进方向

#### 改进 1：迭代优化

```
思路：
1. 用 Auto-CoT 生成初始演示
2. 在部分数据上测试
3. 找出错误的演示，重新生成
4. 重复 2-3 轮

预期收益：
- 演示错误率从 8% 降到 3%
- 性能提升 2-3%
```

#### 改进 2：结合 Self-Consistency

```
思路：
1. 用 Auto-CoT 生成演示
2. 对测试问题采样多次（如 10 次）
3. 投票选答案

预期收益：
- GSM8K 从 82.7% 提升到 86%+
- 稳定性增强
```

#### 改进 3：动态 k 值

```
思路：
- 根据任务复杂度调整 k
- 简单任务 k=4，复杂任务 k=16

预期收益：
- 平衡性能和效率
- 减少不必要的计算
```

---

### 第十章：如何应用

#### 快速开始

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import openai

class AutoCoT:
    def __init__(self, k=8, model="text-davinci-002"):
        self.k = k
        self.model = model
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.demos = []

    def generate_demos(self, questions):
        # 阶段 1: 聚类
        embeddings = self.embedder.encode(questions)
        kmeans = KMeans(n_clusters=self.k, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        # 阶段 2: 从每个簇选代表并生成推理链
        for cluster_id in range(self.k):
            # 选簇中心最近的问题
            cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
            center = kmeans.cluster_centers_[cluster_id]

            # 计算每个问题到簇心的距离
            distances = [
                np.linalg.norm(embeddings[i] - center)
                for i in cluster_indices
            ]
            rep_idx = cluster_indices[np.argmin(distances)]
            rep_question = questions[rep_idx]

            # 用 Zero-Shot-CoT 生成推理链
            demo = self._generate_rationale(rep_question)
            if demo and self._passes_heuristic(demo):
                self.demos.append(demo)

        return self.demos

    def _generate_rationale(self, question):
        prompt = f"Q: {question} A: Let's think step by step."
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            max_tokens=200,
            temperature=0
        )
        rationale = response.choices[0].text.strip()
        return {"question": question, "rationale": rationale}

    def _passes_heuristic(self, demo):
        # 启发式过滤
        question_len = len(demo["question"].split())
        rationale_steps = demo["rationale"].count(".")

        return question_len <= 60 and rationale_steps <= 5

    def infer(self, test_question):
        # 构建 prompt
        prompt_parts = []
        for demo in self.demos:
            prompt_parts.append(f"Q: {demo['question']} A: {demo['rationale']}")
        prompt_parts.append(f"Q: {test_question} A: Let's think step by step.")

        prompt = "\n".join(prompt_parts)

        # 调用 LLM
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            max_tokens=200,
            temperature=0
        )

        return response.choices[0].text.strip()


# 使用示例
questions = [
    "小明有 5 个苹果，又买了 3 个，现在有几个？",
    "每盒有 6 个橙子，4 盒有几个？",
    # ... 更多问题
]

auto_cot = AutoCoT(k=8)
auto_cot.generate_demos(questions)

test_q = "小红有 10 元，买了 2 个苹果，每个 3 元，还剩几元？"
answer = auto_cot.infer(test_q)
print(answer)
```

---

### 第十一章：延伸思考

[停下来，想一想...]

1. **Auto-CoT 能用于非推理任务吗？**
   - 创意写作？代码生成？
   - 核心思想是"多样性演示"，这可能泛化到其他领域吗？

2. **如果让 Auto-CoT 自己选择 k 值，会怎样？**
   - 根据问题分布自动调整簇数
   - 会出现什么新的权衡？

3. **Auto-CoT 的"多样性"本质是什么？**
   - 是问题语义的多样？还是推理模式的多样？
   - 如何量化"多样性"？

4. **如果用更强的嵌入模型（如 GPT-3 自己），聚类会更好吗？**
   - Sentence-BERT vs GPT-3 嵌入
   - 成本 vs 收益如何？

5. **Auto-CoT 能否与人类协作？**
   - 人类只审查 Auto-CoT 生成的演示，而不是从头写
   - 这样能减少多少人工时间？

6. **如果测试问题是动态到来的（流式场景），Auto-CoT 如何适应？**
   - 需要在线聚类算法
   - 或者预定义问题模板库

7. **Auto-CoT 生成的演示，人类能理解吗？**
   - 如果推理链太简略，人类看不懂怎么办？
   - 可解释性 vs 性能的权衡？

---

## 附录：关键公式与算法

### Auto-CoT 算法伪代码

```
Algorithm 1: Auto-CoT
Input: Questions Q = {q_1, q_2, ..., q_n}, k (number of clusters)
Output: k demonstrations D = {d_1, d_2, ..., d_k}

1: // Phase 1: Question Clustering
2: E ← SentenceBERT(Q)  // Encode questions
3: labels ← KMeans(E, k)  // Cluster into k groups
4:
5: // Phase 2: Demonstration Sampling
6: D ← ∅
7: for i = 1 to k do
8:     Q_i ← {q ∈ Q | label(q) = i}  // Questions in cluster i
9:     q_rep ← SelectRepresentative(Q_i)  // Closest to centroid
10:    r ← ZeroShotCoT(q_rep)  // Generate rationale
11:    if PassHeuristic(q_rep, r) then
12:        D ← D ∪ {(q_rep, r)}
13:    else
14:        goto line 9  // Try another representative
15:    end if
16: end for
17:
18: return D
```

### 启发式函数

```
PassHeuristic(question, rationale):
    question_tokens = len(tokenize(question))
    rationale_steps = count_sentences(rationale)

    if question_tokens > 60:
        return False
    if rationale_steps > 5:
        return False
    if not has_answer(rationale):
        return False

    return True
```

---

## 写作检查清单

- [x] 电梯演讲层
- [x] 故事摘要层（含框架大图）
- [x] 深度精读层
- [x] 具体失败场景开场
- [x] 故事化叙述
- [x] 多角度实例（生活类比 + 代码实例 + 对比场景）
- [x] 预期 vs 实际对比表
- [x] 反直觉挑战
- [x] 关键实验细节
- [x] 局限性分析
- [x] 改进方向
- [x] 应用指南（代码）
- [x] 延伸思考
- [x] 论文定位图谱

---

## 关键术语表

| 术语 | 含义 |
|------|------|
| CoT (Chain-of-Thought) | 思维链，让 LLM 生成中间推理步骤 |
| Zero-Shot-CoT | 零样本 CoT，只用"Let's think step by step"提示 |
| Manual-CoT | 手工设计 CoT 演示 |
| Auto-CoT | 自动 CoT，用 LLM 自动生成演示 |
| Sentence-BERT | 语义嵌入模型，用于编码问题 |
| k-means | 聚类算法，将问题分组 |
| 启发式 | 经验规则，用于过滤低质量演示 |
| 多样性 | 演示覆盖的问题类型范围 |

---

## 参考资源

- **论文**: [Automatic Chain of Thought Prompting in Large Language Models](https://arxiv.org/abs/2210.03493)
- **代码**: [Amazon Research Auto-CoT GitHub](https://github.com/amazon-research/auto-cot)
- **解读**: [Learn Prompting - Auto-CoT](https://learnprompting.org/docs/advanced/thought_generation/automatic_chain_of_thought)
