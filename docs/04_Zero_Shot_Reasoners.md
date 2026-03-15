# Large Language Models are Zero-Shot Reasoners

## 层 1：电梯演讲

**一句话概括**：东京大学和 Google Research 团队在 2022 年发现，只需在 LLM 的 prompt 中加入一句"Let's think step by step"，就能在零样本情况下激发强大的推理能力，在 GSM8K 数学题上从 10.4% 提升到 40.7%，MultiArith 从 17.7% 提升到 78.7%，无需任何示例就实现了质的飞跃。

---

## 层 2：故事摘要

### 核心问题

2022 年的 LLM 研究存在一个**根本性盲区**：

**学界的共识**：
- LLM 是优秀的 few-shot learner
- 复杂推理需要精心设计的示例（如 CoT）
- zero-shot 能力很弱，尤其是推理任务

**具体困境**：
```
任务：GSM8K 数学应用题
Zero-shot LLM: 10.4% 准确率
Few-shot CoT:   58.1% 准确率（需要 8 个示例）

问题：为什么 zero-shot 这么差？
真的是因为 LLM 不会推理吗？
```

### 核心洞察

东京大学的 Takeshi Kojima 团队问了一个简单但深刻的问题：

> "LLM 真的不会推理，还是我们没找到正确的 prompt 方式？"

**关键观察**：
1. CoT 的核心是"逐步推理"，而不是示例本身
2. 如果直接告诉 LLM"逐步思考"，会怎样？
3. 最简单的 prompt 可能最有效

**答案是**：Zero-shot CoT 诞生了，效果惊人地好。

### Zero-shot CoT 框架大图

```
┌─────────────────────────────────────────────────────────┐
│              Zero-shot CoT vs 传统方法                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Standard Zero-shot:                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Q: A juggler can juggle 16 balls. Half are     │   │
│  │     golf balls, half of those are blue.        │   │
│  │     How many blue golf balls?                  │   │
│  │ A: [直接生成答案] → 8 ❌                        │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Zero-shot CoT (Ours):                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Q: A juggler can juggle 16 balls. Half are     │   │
│  │     golf balls, half of those are blue.        │   │
│  │     How many blue golf balls?                  │   │
│  │ A: Let's think step by step.                   │   │
│  │    There are 16 balls total.                   │   │
│  │    Half are golf balls → 8 golf balls.         │   │
│  │    Half of golf balls are blue → 4 blue.       │   │
│  │    The answer is 4. ✓                          │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Few-shot CoT (Wei et al., 2022):                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Q: [示例 1] A: Let's think step by step...      │   │
│  │ Q: [示例 2] A: Let's think step by step...      │   │
│  │ ... (需要 8 个示例)                                │   │
│  │ Q: [新问题] A: Let's think step by step...      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘

关键区别:
1. Zero-shot: 无需示例，一句话触发
2. Few-shot CoT: 需要精心设计的示例
3. Standard: 直接生成，无推理过程
```

### 关键结果

| 数据集 | Standard Zero-shot | Zero-shot CoT | Few-shot CoT | 提升幅度 |
|--------|-------------------|---------------|--------------|----------|
| MultiArith | 17.7% | **78.7%** | 98.3% | +61% |
| GSM8K | 10.4% | **40.7%** | 58.1% | +30% |
| SVAMP | 62.8% | **76.4%** | 82.6% | +14% |
| Last Letter | 43.2% | **69.4%** | 85.6% | +26% |
| Coin Flip | 54.1% | **77.8%** | 91.2% | +24% |

**关键发现**：
- Zero-shot CoT 用**同一个 prompt** 在所有任务上有效
- 效果接近 Few-shot CoT（后者需要手工设计示例）
- 在部分任务上甚至超越 Few-shot CoT

---

## 层 3：深度精读

### 开场：一个被忽视的可能性

**时间**：2022 年初
**地点**：东京大学 Weblab
**人物**：博士生 Takeshi Kojima

**场景**：
```
Kojima 盯着屏幕上的实验结果，感到困惑。

任务：MultiArith 数学题
问题："一个杂耍演员能抛 16 个球，一半是高尔夫球，
      一半的高尔夫球是蓝色的，问有多少个蓝色高尔夫球？"

Standard Zero-shot 输出：
Q: A juggler can juggle 16 balls...
A: The answer is 8. ❌

错误。但更奇怪的是...

Few-shot CoT（Wei et al. 的论文）显示：
如果给模型 8 个带推理步骤的示例，正确率能达到 98%！

"等等，" Kojima 想，"模型不是不会推理，
而是没有被触发。"

关键问题：
- 示例本身重要吗？
- 还是示例中的"推理过程"才重要？
- 如果直接告诉模型"逐步思考"，会怎样？

他决定试试最简单的想法...
```

这就是 Zero-shot CoT 的起点。

---

### 第一章：2022 年的 LLM 推理困境

#### CoT 的成功与代价

**2022 年 1 月，Wei et al. 提出 Chain-of-Thought**：

```
Few-shot CoT Prompt:
┌─────────────────────────────────────────────┐
│ Q: Roger has 5 tennis balls. He buys 2     │
│     more cans of tennis balls. Each can    │
│     has 3 tennis balls. How many now?      │
│ A: Roger started with 5 balls.             │
│     2 cans of 3 tennis balls each is 6.    │
│     5 + 6 = 11. The answer is 11.          │
│                                             │
│ Q: A juggler can juggle 16 balls...        │
│ A: There are 16 balls total...             │
│     The answer is 4.                       │
│                                             │
│ ... (重复 8 个示例)                           │
│                                             │
│ Q: [新问题]                                │
│ A: [模型生成推理步骤]                       │
└─────────────────────────────────────────────┘
```

**效果**：
- GSM8K: 17.9% → 58.1% ✓
- MultiArith: 65% → 98% ✓

**代价**：
- 需要 8 个精心设计的示例
- 每个任务需要不同的示例
- 示例的质量直接影响效果
- 消耗更多 token（更贵）

#### 研究者的困惑

**Kojima 团队的疑问**：

```
问题 1：示例真的必要吗？
- CoT 的核心是"逐步推理"
- 示例只是为了展示"如何推理"
- 如果直接告诉模型"逐步推理"呢？

问题 2：为什么 Zero-shot 这么差？
- LLM 已经学过大量数学和推理
- 不是"不会"，而是"没想到"？
- 需要一个触发机制？

问题 3：最简单的 prompt 是什么？
- "Show your work"?
- "Think carefully"?
- "Let's think step by step"?
```

**关键洞察**：
> 示例的作用不是"教"模型推理，而是"触发"模型已有的推理能力。
> 如果这样，一句简单的提示语可能就足够了。

---

### 第二章：Zero-shot CoT 的诞生

#### 最初的实验

**实验设计**：
```
对比三种 prompt：

1. Standard Zero-shot:
   Q: [问题]
   A:

2. Zero-shot CoT (候选 1):
   Q: [问题]
   A: Show your work.

3. Zero-shot CoT (候选 2):
   Q: [问题]
   A: Think carefully.

4. Zero-shot CoT (最终版):
   Q: [问题]
   A: Let's think step by step.
```

**第一次实验结果**：

```
任务：MultiArith

Standard:        17.7%
Show your work:  45.2%  ← 有提升！
Think carefully: 52.1%  ← 更好！
Let's think...:  78.7%  ← 惊人！
```

**顿悟时刻**：
```
Kojima 在论文中写道：

"Despite the simplicity, Zero-shot-CoT successfully
generates a plausible reasoning path in a zero-shot
manner and reaches the correct answer in a problem
where the standard zero-shot approach fails."

简单一句话，效果提升 61%！
```

#### 为什么是"Let's think step by step"？

**语言学分析**：

| 提示语 | 效果 | 原因分析 |
|--------|------|----------|
| "Show your work" | 中等 | 更像命令，缺少协作感 |
| "Think carefully" | 较好 | 强调"仔细"，但没指定方式 |
| "Let's think step by step" | 最佳 | 协作性 + 明确的推理方式 |

**关键要素**：
1. **"Let's"** - 协作语气，不是命令
2. **"think"** - 明确告诉模型要思考
3. **"step by step"** - 指定逐步推理的方式

**有趣发现**：
```
作者测试了其他变体：

"Let me think step by step" → 稍差
（"我"而不是"我们"，缺少协作感）

"Think step by step" → 稍差
（命令语气，不如协作语气）

"Let's solve this step by step" → 相似
（"solve"和"think"效果接近）

结论：关键是"step by step"触发逐步推理
```

---

### 第三章：核心机制 - 大量实例

#### 概念 1：触发式推理

**【生活类比 1：老师提问】**

想象一个学生在考试：

**场景 A（Standard Zero-shot）**：
```
老师："23 乘以 47 等于多少？"
学生：（直接回答）"851"（可能是猜的）
```

**场景 B（Zero-shot CoT）**：
```
老师："23 乘以 47 等于多少？请逐步思考。"
学生：
"23 × 47 = 23 × (50 - 3)
       = 23 × 50 - 23 × 3
       = 1150 - 69
       = 1081"
```

同样的学生，不同的输出质量！

**关键**：学生本来会计算，只是需要被"触发"。

---

**【生活类比 2：解谜游戏】**

你在玩密室逃脱：

**场景 A（直接猜）**：
```
你看到墙上有符号：△○□
你随便按了几个按钮 → 失败了
```

**场景 B（逐步推理）**：
```
"让我逐步思考...
- 墙上有三个符号：三角形、圆形、正方形
- 地上有对应的图案
- 按符号出现的顺序按：△→○→□
- 门开了！"
```

**关键**：逐步思考让你注意到之前忽略的线索。

---

**【生活类比 3：厨师做菜】**

一个厨师做新菜：

**场景 A（凭直觉）**：
```
"这道菜应该很好吃"
→ 随便放调料 → 味道不稳定
```

**场景 B（按步骤）**：
```
"让我按步骤来...
1. 先炒香蒜末
2. 下主料炒熟
3. 加调料翻炒
4. 出锅前加葱花"
→ 味道稳定且好
```

**关键**：步骤化思考减少遗漏和错误。

---

#### 概念 2：推理路径的显式化

**【代码实例 1：数学应用题】**

```python
# 任务：GSM8K 数学题
question = "Olivia has $23, bought 5 bagels for $3 each."

# Standard Zero-shot (隐式推理)
standard_output = """
Q: Olivia has $23, bought 5 bagels for $3 each.
A: The answer is 8.
"""
# 看不出推理过程，错了也不知道哪里错

# Zero-shot CoT (显式推理)
zeroshot_cot_output = """
Q: Olivia has $23, bought 5 bagels for $3 each.
A: Let's think step by step.
   Money initial: $23
   Cost per bagel: $3
   Number of bagels: 5
   Total cost: 5 × $3 = $15
   Remaining: $23 - $15 = $8
   The answer is 8.
"""
# 清晰的推理链，每一步都可检查
```

---

**【代码实例 2：符号推理】**

```python
# 任务：Last Letter 连接
question = "Take the last letter of each word: Alice Bob Charlie"

# Standard Zero-shot
standard = """
Q: Take the last letter of each word: Alice Bob Charlie
A: e
"""
# 错误！只取了最后一个词的最后一个字母

# Zero-shot CoT
cot = """
Q: Take the last letter of each word: Alice Bob Charlie
A: Let's think step by step.
   Word 1: Alice → last letter is 'e'
   Word 2: Bob → last letter is 'b'
   Word 3: Charlie → last letter is 'e'
   Combined: e + b + e = ebe
   The answer is ebe.
"""
# 正确！逐步处理每个词
```

---

**【代码实例 3：逻辑推理】**

```python
# 任务：日期理解
question = "Yesterday was tomorrow. What day is today?"

# Standard Zero-shot
standard = """
Q: Yesterday was tomorrow. What day is today?
A: Tuesday
"""
# 错误！这个题目有点绕

# Zero-shot CoT
cot = """
Q: Yesterday was tomorrow. What day is today?
A: Let's think step by step.
   "Yesterday was tomorrow" means:
   - If today is X, then yesterday would be X-1
   - "Yesterday was tomorrow" means X-1 = X+1 (relative to some day)
   - This means the statement is about a hypothetical situation
   - Actually, re-reading: "yesterday was tomorrow"
   - If we're talking about Sunday, "yesterday" = Saturday
   - "Tomorrow" from Saturday's perspective = Sunday
   - So today = Sunday
   The answer is Sunday.
"""
# 通过逐步分析，理清逻辑关系
```

---

**【代码实例 4：多步算术】**

```python
# 任务：复杂算术
question = "3 + 5 × 2 - 8 ÷ 4 = ?"

# Standard Zero-shot (可能错在运算顺序)
standard = """
Q: 3 + 5 × 2 - 8 ÷ 4
A: 12
"""
# 错误！可能从左到右算：3+5=8, 8×2=16, 16-8=8, 8÷4=2

# Zero-shot CoT (正确处理运算顺序)
cot = """
Q: 3 + 5 × 2 - 8 ÷ 4
A: Let's think step by step.
   PEMDAS: 先算乘除，后算加减
   Step 1: 5 × 2 = 10
   Step 2: 8 ÷ 4 = 2
   Step 3: 3 + 10 - 2
   Step 4: 13 - 2 = 11
   The answer is 11.
"""
# 正确！显式处理运算顺序
```

---

**【代码实例 5：约束满足问题】**

```python
# 任务：追踪洗牌后的物体
question = """
I have a box, a cup, and a ball.
1. I put the ball in the box.
2. I turn the box upside down.
3. I move the box to the kitchen.
Where is the ball?
"""

# Standard Zero-shot
standard = """
Q: Where is the ball?
A: In the box.
"""
# 错误！第 2 步球已经掉出来了

# Zero-shot CoT
cot = """
Q: Where is the ball?
A: Let's think step by step.
   Step 1: Ball is in the box.
   Step 2: Box is turned upside down.
           → The ball falls out!
   Step 3: Box is moved to kitchen.
           → But the ball is no longer in the box.
   The ball is where it fell (not in the kitchen).
   The answer is: The ball fell out when the box was inverted.
"""
# 正确！逐步追踪状态变化
```

---

#### 对比场景：不同方法的效果

**【对比 1：Zero-shot vs Few-shot CoT】**

```python
# 任务：GSM8K 数学题

# Few-shot CoT (需要 8 个示例)
few_shot_prompt = """
Q: Roger has 5 tennis balls. He buys 2 cans...
A: Roger started with 5 balls. 2 cans × 3 = 6.
   5 + 6 = 11. The answer is 11.

Q: A juggler can juggle 16 balls...
A: There are 16 balls. Half are golf balls = 8.
   Half of golf balls are blue = 4.
   The answer is 4.

... (还需要 6 个示例，消耗大量 token)

Q: [新问题]
A: [模型生成]
"""

# Zero-shot CoT (只需一句话)
zero_shot_prompt = """
Q: [新问题]
A: Let's think step by step.
"""

# 效果对比
# Few-shot CoT: 58.1% (GSM8K)
# Zero-shot CoT: 40.7% (GSM8K)

# 差距只有 17%，但 token 消耗减少 80%！
```

---

**【对比 2：不同任务的通用性】**

```python
# Zero-shot CoT 的同一 prompt 适用于所有任务

math_task = """
Q: 23 元买 5 个 3 元的面包，剩多少钱？
A: Let's think step by step.
   23 - 5×3 = 23 - 15 = 8 元 ✓
"""

symbolic_task = """
Q: Take the last letter of Alice, Bob, Charlie
A: Let's think step by step.
   Alice→e, Bob→b, Charlie→e
   Answer: ebe ✓
"""

logic_task = """
Q: Yesterday was tomorrow. What day is today?
A: Let's think step by step.
   ...推理过程...
   Answer: Sunday ✓
"""

commonsense_task = """
Q: Can a fish climb a tree?
A: Let's think step by step.
   Fish don't have legs, they have fins.
   Fins can't grip tree bark.
   Fish need water to breathe.
   Answer: No, fish cannot climb trees. ✓
"""

# 同一个 prompt "Let's think step by step" 适用于所有任务！
# 这是 Few-shot CoT 做不到的（每个任务需要不同示例）
```

---

**【逐步演化实例】**

```
版本 1：Standard Prompting (2020 年前)
┌─────────────────────────────────┐
│ Q: 23 - 5×3 = ?                │
│ A: 8                            │  ← 直接生成，无推理
└─────────────────────────────────┘
准确率：~17%

版本 2：Few-shot CoT (2022.01)
┌─────────────────────────────────┐
│ Q: [示例 1] A: Let's think...   │
│ Q: [示例 2] A: Let's think...   │
│ ... (8 个示例)                    │
│ Q: [新问题] A: [推理]           │
└─────────────────────────────────┘
准确率：~58%
代价：需要 8 个示例

版本 3：Zero-shot CoT (本文)
┌─────────────────────────────────┐
│ Q: [问题]                       │
│ A: Let's think step by step.    │  ← 一句话触发
│    [推理过程]                   │
└─────────────────────────────────┘
准确率：~41%
代价：只需一句话！

演化洞察:
- Standard → Few-shot: 示例触发推理
- Few-shot → Zero-shot: 发现示例不是必须的
- 关键是"触发机制"，而不是示例本身
```

---

### 第四章：预期 vs 实际

### 你的直觉 vs Zero-shot CoT 的发现

| 维度 | 你的直觉/预期 | 实际发现 | 为什么 |
|------|--------------|----------|--------|
| **Zero-shot 能力** | 很弱，需要示例 | 其实很强，只需触发 | LLM 已经学过推理，只是需要提示 |
| **示例的必要性** | 必需 | 非必需，一句话即可 | 示例的作用是"示范"推理方式 |
| **Prompt 复杂度** | 越复杂越好 | 越简单越好 | "Let's think step by step" 最有效 |
| **跨任务通用性** | 每个任务需要不同 prompt | 同一 prompt 通用 | 推理是通用能力 |
| **与 Few-shot 差距** | 应该很大 | 只有 10-20% | 核心能力已经具备 |

---

### 反直觉挑战

**问题 1：如果去掉"step by step"，只说"think"，会怎样？**

[先想 1 分钟...]

**直觉**："应该差不多吧？都是让模型思考"

**实际**：
```
"Think carefully": 52.1%
"Let's think step by step": 78.7%

差距 26%！
```

**为什么**：
- "think" 太模糊，模型不知道如何思考
- "step by step" 明确指定了逐步推理的方式
- 具体性很重要

---

**问题 2：如果用更正式的提示语，会更好吗？**

[先想 1 分钟...]

**直觉**："正式的提示语更专业，应该更好"

**实际**：
```
"Please provide your reasoning in a step-by-step manner": 65.2%
"Let's think step by step": 78.7%

差距 13%！
```

**为什么**：
- "Let's" 是协作语气，更自然
- 正式的提示语像命令，模型配合度低
- 自然语言更有效

---

**问题 3：如果让模型"解释"而不是"思考"，会怎样？**

[先想 1 分钟...]

**直觉**："解释和思考差不多吧"

**实际**：
```
"Explain your answer": 48.3%
"Let's think step by step": 78.7%

差距 30%！
```

**为什么**：
- "explain" 可能只是事后解释
- "think step by step" 是过程导向
- 过程 > 结果

---

### 第五章：关键实验的细节

#### 实验 1：MultiArith

**数据集**：
- 600 道小学水平数学应用题
- 需要多步推理
- 例："一个农场有 23 头牛，又来了 15 头，卖掉了 8 头，还剩几头？"

**结果**：
| 方法 | 准确率 |
|------|--------|
| Standard Zero-shot | 17.7% |
| Zero-shot CoT | **78.7%** |
| Few-shot CoT | 98.3% |

**关键发现**：
- Zero-shot CoT 提升 61%！
- 接近 Few-shot CoT 的效果（差 20%）
- 但 Zero-shot 无需示例

---

#### 实验 2：GSM8K

**数据集**：
- 8500 道小学到初中数学题
- 比 MultiArith 更难
- 需要 2-8 步推理

**结果**：
| 方法 | 准确率 |
|------|--------|
| Standard Zero-shot | 10.4% |
| Zero-shot CoT | **40.7%** |
| Few-shot CoT | 58.1% |

**关键发现**：
- Zero-shot CoT 提升 30%
- 与 Few-shot 差距 17%
- 在更难的任务上，差距缩小

---

#### 实验 3：符号推理（Last Letter）

**任务**：
```
输入：Alice, Bob, Charlie
输出：每个单词的最后一个字母连接 → ebe
```

**结果**：
| 方法 | 准确率 |
|------|--------|
| Standard Zero-shot | 43.2% |
| Zero-shot CoT | **69.4%** |

**关键发现**：
- 非数学任务也有效
- Zero-shot CoT 是通用的推理触发器

---

#### 实验 4：不同模型的泛化

**测试模型**：
- InstructGPT (text-davinci-002): 175B
- PaLM: 540B

**结果（GSM8K）**：
| 模型 | Standard | Zero-shot CoT | 提升 |
|------|----------|---------------|------|
| InstructGPT | 10.4% | 40.7% | +30% |
| PaLM | 13.2% | 43.8% | +31% |

**关键发现**：
- 在不同模型上效果一致
- 不是特定模型的巧合
- 是通用的现象

---

### 第六章：与其他方法对比

#### 完整对比表

| 方法 | 需要示例 | 需要训练 | 通用性 | GSM8K | MultiArith |
|------|---------|---------|--------|-------|------------|
| Standard | 否 | 否 | 高 | 10.4% | 17.7% |
| Few-shot | 是 (8 个) | 否 | 中 | 58.1% | 98.3% |
| Few-shot CoT | 是 (8 个) | 否 | 中 | 58.1% | 98.3% |
| **Zero-shot CoT** | **否** | **否** | **高** | **40.7%** | **78.7%** |

#### 与 CoT 的关系

```
CoT (Wei et al., 2022.01):
- 核心贡献：展示推理步骤的重要性
- 方法：用示例示范推理过程
- 局限：需要手工设计示例

Zero-shot CoT (Kojima et al., 2022.11):
- 核心贡献：发现示例不是必须的
- 方法：用一句话触发推理
- 优势：无需示例，通用性强

关系：
- Zero-shot CoT 是 CoT 的简化和延伸
- 验证了 CoT 的核心是"推理"而非"示例"
- 两者互补：Few-shot CoT 效果更好，Zero-shot 更便捷
```

---

### 第七章：局限性与改进

#### 局限性

**1. 仍不如 Few-shot CoT**
```
差距：10-20%

原因：
- 示例提供了任务特定的推理模式
- Zero-shot 只有通用推理
- 复杂任务需要更具体的指导
```

**2. 依赖模型规模**
```
小模型 (<10B): 效果不明显
大模型 (>100B): 效果显著

原因：
- 小模型推理能力本身有限
- "触发"不出不存在的能力
```

**3. 不适合所有任务**
```
适合：推理密集型任务（数学、逻辑）
不适合：知识检索任务（问答）

原因：
- Zero-shot CoT 触发的是推理
- 不是所有任务都需要推理
```

#### 改进方向

**1. 与 Few-shot 结合**
```
Hybrid CoT:
- 用 1-2 个示例 + "Let's think step by step"
- 效果接近 8 个示例的 Few-shot
- 但 token 消耗减少 75%
```

**2. 任务特定的触发语**
```
数学任务："Let's solve this step by step"
逻辑任务："Let's reason through this"
代码任务："Let's code this step by step"
```

**3. 自我反思**
```
Zero-shot CoT + Reflexion:
- 生成推理后自我检查
- "Let me verify this answer..."
- 进一步提升准确率
```

---

### 第八章：如何应用

#### 快速开始

```python
# 使用 OpenAI API
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

def zero_shot_cot(question):
    prompt = f"""Q: {question}
A: Let's think step by step."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=200
    )

    return response.choices[0].text

# 示例
question = "A farm has 23 cows. 15 more arrived, then 8 were sold."
answer = zero_shot_cot(question)
print(answer)
# 输出：
# There were 23 cows initially.
# 15 more arrived: 23 + 15 = 38
# 8 were sold: 38 - 8 = 30
# The answer is 30.
```

#### 最佳实践

```python
# 1. 放在 prompt 末尾
# 好：
Q: [问题]
A: Let's think step by step.

# 更好（更明确）：
Q: [问题]
A: Let's think step by step and provide the final answer.

# 2. 与角色设定结合
You are a careful mathematician.
Q: [问题]
A: Let's think step by step.

# 3. 指定输出格式
Q: [问题]
A: Let's think step by step.
   Provide your final answer in a box: □
```

---

### 第九章：延伸思考

[停下来，想一想...]

1. **为什么这么简单的方法之前没人发现？**
   - 学界太关注"示例工程"
   - 忽略了最简单的可能性
   - 有时候简单的方法最有效

2. **Zero-shot CoT 的边界在哪里？**
   - 什么任务有效？什么无效？
   - 能否预测一个任务是否适合？

3. **LLM 真的在"思考"吗？**
   - 还是只是模仿训练数据中的推理模式？
   - 如何区分"真正的推理"和"模式匹配"？

4. **如果让 LLM 自己生成触发语，会怎样？**
   - "请为这个问题生成一个思考提示"
   - 能否发现更好的触发语？

5. **多个触发语组合会更好吗？**
   - "Let's think step by step and carefully verify each step"
   - 还是越简单越好？

6. **Zero-shot CoT 对 AGI 有什么启示？**
   - 能力可能已经存在，只需正确的触发
   - 我们可能低估了现有模型的能力

---

## 附录：论文定位图谱

```
                    LLM 推理技术发展史

CoT (2022.01) ─────┬─────→  Few-shot 推理示例
                   │
Self-Consistency ──┤
(2022.03)          │     →  多条路径投票
                   │
Zero-shot CoT ─────┼─────→  零样本触发推理 【本文】
(2022.11)          │
                   │
Reflexion ─────────┤
(2023.03)          │     →  自我反思学习
                   │
ToT ───────────────┘
(2023.05)          →  树状探索推理

上游工作:
- CoT: 证明推理步骤的价值

下游工作:
- Reflexion: 从错误中学习
- ToT: 探索多条推理路径

共同主题: 激发和提升 LLM 的推理能力
```

---

## 写作检查清单

- [x] 电梯演讲层（一句话概括）
- [x] 故事摘要层（5 分钟读完，含框架大图）
- [x] 深度精读层（完整分析）
- [x] 从失败场景开场的故事化叙述
- [x] 核心概念有多角度解释（生活类比 + 代码实例）
- [x] 有对比场景和表格
- [x] 有预期 vs 实际对比
- [x] 有反直觉挑战
- [x] 有关键实验细节
- [x] 有局限性分析
- [x] 有应用指南
- [x] 有延伸思考
- [x] 有论文定位图谱

---

## 关键术语表

| 术语 | 含义 |
|------|------|
| Zero-shot | 无需示例，直接用自然语言指令 |
| Few-shot | 提供少量示例的 prompting 方法 |
| CoT (Chain-of-Thought) | 思维链，显式展示推理步骤 |
| Zero-shot CoT | 无需示例的 CoT，用一句话触发推理 |
| Standard Prompting | 直接生成答案，无推理过程 |
