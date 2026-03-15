# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

**论文信息**: Wei et al., NeurIPS 2022, arXiv:2201.11903
**作者**: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, Denny Zhou (Google Research, Brain Team)

---

## 层 1：电梯演讲（30 秒）

**一句话概括**：通过在 few-shot prompt 中提供包含中间推理步骤的示例（思维链），大语言模型（~100B+ 参数）能够被激发出强大的推理能力，在 GSM8K 数学推理任务上 PaLM 540B 达到 56.9% 准确率，超越经过微调的 GPT-3 + verifier。

---

## 层 2：故事摘要（5 分钟）

### 核心问题

2022 年，大语言模型面临一个尴尬的局面：
- **Scaling 有效但有限**：增大模型参数能提升性能，但在算术推理、常识推理、符号推理等复杂任务上，单纯增大模型效果平平
- **微调成本高**：要让模型学会推理，需要大量标注中间步骤的训练数据，标注成本极高
- **标准 Prompting 失效**：Brown et al. (2020) 的 few-shot prompting 在简单 QA 任务上有效，但在需要多步推理的任务上几乎不工作

### 核心洞察

研究者的直觉很简单：**人类解决复杂问题时，会把问题分解成中间步骤，一步步思考后再给出答案**。

> "After Jane gives 2 flowers to her mom she has 10... then after she gives 3 to her dad she will have 7... so the answer is 7."

如果让模型也这样思考呢？

### Chain-of-Thought Prompting 方法

```
Standard Prompting:
Q: Roger has 5 tennis balls. He buys 2 more cans...
A: The answer is 11.

Chain-of-Thought Prompting:
Q: Roger has 5 tennis balls. He buys 2 more cans...
A: Roger started with 5 balls. 2 cans of 3 tennis balls
   each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
```

**关键区别**：在 prompt 示例中包含 `⟨input, chain of thought, output⟩` 三元组，而不是 `⟨input, output⟩` 二元组。

### 研究框架大图

```
┌─────────────────────────────────────────────────────────┐
│                    问题定义                              │
│  大模型在复杂推理任务上表现差，标准 prompting 失效          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                    核心假设                              │
│  如果在 prompt 中展示"思考过程"，模型能学会推理            │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│               Chain-of-Thought Prompting                │
│  • 输入：⟨question, reasoning steps, answer⟩示例        │
│  • 输出：模型生成推理步骤 → 最终答案                       │
│  • 关键：不需要微调，只需要 prompt 设计                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                    实验验证                              │
│  • 算术推理：GSM8K, SVAMP, ASDiv, AQuA, MAWPS           │
│  • 常识推理：CSQA, StrategyQA, Date Understanding...    │
│  • 符号推理：Last Letter Concatenation, Coin Flip       │
│  • 模型：GPT-3, LaMDA, PaLM (8B → 540B)                 │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│                    核心发现                              │
│  • Emergent Ability: 只在 100B+ 模型上有效               │
│  • GSM8K: PaLM 540B 达到 56.9%, SOTA                    │
│  • 复杂任务收益更大，简单任务收益小                      │
│  • 对 annotator 风格、示例选择相对鲁棒                   │
└─────────────────────────────────────────────────────────┘
```

### 关键结果速览

| 任务 | 模型 | Standard | CoT | 提升 |
|------|------|----------|-----|------|
| GSM8K | PaLM 540B | 17.9% | **56.9%** | +39.0% |
| GSM8K | GPT-3 175B | 15.6% | **46.9%** | +31.3% |
| SVAMP | PaLM 540B | 69.4% | **79.0%** | +9.6% |
| StrategyQA | PaLM 540B | 68.6% | **77.8%** | +9.2% |

**核心发现**：CoT 是一种 **emergent ability** —— 小模型用 CoT 反而更差，只有 100B+ 的模型才能产生逻辑正确的推理链。

---

## 层 3：深度精读

### 开场：一个失败的场景

时间回到 2021-2022 年，Google Brain 团队的研究者们盯着屏幕上的输出，陷入了困惑。

**任务**（GSM8K 数学应用题）：
> "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"

**PaLM 540B 的标准 Prompting 输出**：
> "The answer is 27."

**错误**！正确答案是 9（23 - 20 = 3, 3 + 6 = 9）。

但更让人沮丧的是：这个错误太低级了，任何一个小学生都能做对。为什么 5400 亿参数的模型连这个都不会？

研究者们尝试了各种方法：
- 增加训练数据？效果有限
- 调整 prompt 措辞？几乎没变化
- 增大模型？性能曲线是平的

**核心困境**：模型有知识，但无法在推理任务上"调用"这些知识。

---

### 第一章：研究者的困境

#### 当时学界的两大主流方向

**方向 1：Rationale-augmented Training**
- Ling et al. (2017): 训练模型生成自然语言推理步骤
- Cobbe et al. (2021): 微调预训练模型，添加 GSM8K 标注数据
- **局限**：需要大量标注数据，每个任务都要单独微调

**方向 2：Few-shot Prompting**
- Brown et al. (2020): 通过 prompt 示例让模型学会任务
- **局限**：在简单 QA 上有效，在推理任务上几乎不工作

**关键观察**：
> Rae et al. (2021) 发现：单纯增大模型规模，在推理任务上的性能曲线是平的 —— 从 1B 到 100B，GSM8K 准确率几乎没有提升。

研究团队开始思考：**能不能结合这两个方向的优点，同时避免它们的局限**？

---

### 第二章：试错的旅程

#### 最初的直觉

Jason Wei 团队有一个简单的想法：
> "人类解决复杂问题时，会在心里默默计算：'先减 20，再加 6...'。如果让模型也把这个过程说出来，会怎样？"

#### 第一次尝试

他们设计了这样的 prompt：

```
Q: There are 15 trees in the grove. After workers plant
   more trees, there will be 21 trees. How many trees
   did workers plant?
A: There are 15 trees originally. Then there were 21
   trees after some more were planted. So there must
   have been 21 - 15 = 6. The answer is 6.
```

**结果**：第一次跑实验时，团队震惊了 —— PaLM 540B 居然真的能生成类似的推理步骤！

#### 但问题还没完

**新问题 1**：小模型用 CoT 会怎样？

实验结果让团队意外：
- LaMDA 8B: CoT 性能 **低于** Standard Prompting
- LaMDA 68B: CoT 开始略有提升
- LaMDA 137B: CoT 性能大幅提升

**发现**：CoT 是一种 **emergent ability** —— 只有足够大的模型（~100B+）才能产生逻辑正确的推理链。小模型会生成"流畅但无逻辑"的胡言乱语。

**新问题 2**：为什么 CoT 有效？

团队设计了三个 ablation 实验来验证假设：

| 假设 | 验证方法 | 结果 |
|------|----------|------|
| CoT 只是让模型输出方程？ | "Equation only" prompt | 方程本身不够，需要自然语言推理 |
| CoT 只是增加计算量？ | 输出等量的 "..." | 无效，关键是推理内容 |
| CoT 只是激活预训练知识？ | 把推理放在答案之后 | 无效，推理必须在答案之前 |

**结论**：CoT 有效的核心原因是 **sequential reasoning** —— 通过中间步骤，模型能够分解复杂问题，一步步推导。

---

### 第三章：核心概念 - 大量实例

#### 什么是 Chain-of-Thought？

**定义**：Chain of Thought 是一系列中间的自然语言推理步骤，引导模型到达最终答案。

**核心思想**：让模型"展示它的工作过程"（Show Your Work）。

---

#### 实例 1：算术推理（Math Word Problems）

```
Q: Roger has 5 tennis balls. He buys 2 more cans of
   tennis balls. Each can has 3 tennis balls. How many
   tennis balls does he have now?

CoT Output:
Roger started with 5 balls.
2 cans of 3 tennis balls each is 6 tennis balls.
5 + 6 = 11.
The answer is 11.
```

**推理步骤分解**：
1. 提取初始状态：5 个球
2. 理解"2 cans"的含义：2 罐 × 3 个/罐 = 6 个
3. 执行加法：5 + 6 = 11
4. 输出答案

---

#### 实例 2：常识推理（Commonsense Reasoning）

```
Q: Sammy wanted to go to where the people were.
   Where might he go?
   Options: (a) race track (b) populated areas
            (c) desert (d) apartment (e) roadblock

CoT Output:
The answer must be a place with a lot of people.
Race tracks, desert, apartments, and roadblocks
don't have a lot of people, but populated areas do.
So the answer is (b).
```

**推理步骤分解**：
1. 理解问题核心：找"人多的地方"
2. 逐个排除选项
3. 确认正确答案

---

#### 实例 3：符号推理（Symbolic Reasoning）

```
Q: Take the last letters of the words in "Lady Gaga"
   and concatenate them.

CoT Output:
The last letter of "Lady" is "y".
The last letter of "Gaga" is "a".
Concatenating them is "ya".
So the answer is ya.
```

---

#### 实例 4：日期推理（Date Understanding）

```
Q: The concert was scheduled to be on 06/01/1943,
   but was delayed by one day to today.
   What is the date 10 days ago in MM/DD/YYYY?

CoT Output:
One day after 06/01/1943 is 06/02/1943,
so today is 06/02/1943.
10 days before today is 05/23/1943.
So the answer is 05/23/1943.
```

---

#### 实例 5：体育常识推理（Sports Understanding）

```
Q: Is the following sentence plausible?
   "Joao Moutinho caught the screen pass in the
   NFC championship."

CoT Output:
Joao Moutinho is a soccer player.
The NFC championship is part of American football,
not soccer.
So the answer is no.
```

---

#### 实例 6：机器人规划（SayCan）

```
Human: How would you bring me something that isn't a fruit?

CoT Output:
Explanation: the user wants something to eat that isn't a
fruit. An energy bar is not a fruit, so I will bring the
user an energy bar.
Plan: 1. find(energy bar) 2. pick(energy bar)
      3. find(user) 4. put(energy bar) 5. done().
```

---

#### 对比场景 1：有 CoT vs 无 CoT

| 维度 | Standard Prompting | Chain-of-Thought |
|------|-------------------|------------------|
| 问题分解 | ❌ 无法分解 | ✅ 自动分解 |
| 可解释性 | ❌ 黑盒 | ✅ 可见推理过程 |
| 调试难度 | ❌ 难以定位错误 | ✅ 可追踪错误步骤 |
| GSM8K 性能 | 17.9% | 56.9% |

---

#### 对比场景 2：密集思考 vs 稀疏思考

论文还发现一个有趣现象：

**密集思考**（每个行动前都思考）：
```
Thought: 我需要搜索 X...
Action: search[X]
Thought: 现在我有 Y 信息...
Action: process[Y]
...
```

**稀疏思考**（只在关键时刻思考）：
```
Action: search[X]
Observation: ...
Thought: 等等，这个信息不对，我需要...
Action: search[Z]
...
```

**发现**：稀疏思考策略更高效 —— 只在遇到异常或需要调整计划时才思考。这个发现后来被 ReAct 论文进一步发展。

---

#### 逐步演化实例

**版本 1：纯 CoT（原始版本）**
- 优点：能推理
- 缺点：无法行动（没有工具调用）
- 典型任务：数学题、常识 QA

**版本 2：CoT + 工具调用**
- 优点：能推理 + 能行动
- 缺点：推理和行动割裂
- 典型任务：Web 搜索、计算器

**版本 3：ReAct（后续发展）**
- 优点：Thought + Action + Observation 循环
- 缺点：无法从失败中学习
- 典型任务：ALFWorld、HotpotQA

**版本 4：Self-Consistency（后续发展）**
- 优点：对同一问题采样多条推理链，取多数答案
- 缺点：计算成本高
- 典型任务：所有 CoT 适用任务

---

### 第四章：预期 vs 实际

#### 你的直觉 vs CoT 的实现

| 维度 | 你的直觉/预期 | CoT 实际实现 | 为什么有差距？ |
|------|--------------|-------------|---------------|
| 什么时候有效？ | 所有模型 | 只有 100B+ 模型 | 小模型无法产生逻辑推理 |
| 示例需要精心设计吗？ | 需要大量 prompt 工程 | 相对鲁棒 | 不同 annotator 风格都有效 |
| 推理链必须完美吗？ | 必须完全正确 | 允许小错误 | 模型能容忍 minor mistakes |
| 只对数学题有效？ | 可能只适合计算 | 适用于常识、符号推理 | 核心是 sequential reasoning |
| 需要微调吗？ | 可能需要训练 | 完全不需要 | pure prompting |

---

#### 反直觉问题

**问题 1**：如果去掉所有的 Thought，只保留答案，会怎样？

**直觉预测**："只是少了一些输出，应该差不多吧？"

**实际结果**：
- GSM8K: 从 56.9% 掉到 17.9%（PaLM 540B）
- 下降超过 3 倍！

**为什么**？因为没有 Thought，模型就无法：
- 分解多步问题
- 追踪中间状态
- 整合语义信息

---

**问题 2**：如果让小模型用 CoT，会怎样？

**直觉预测**："至少不会更差吧？"

**实际结果**：
- LaMDA 8B: CoT **低于** Standard Prompting
- LaMDA 420M: CoT 只有 0.4%，Standard 有 2.6%

**为什么**？因为小模型生成的推理链是"流畅的胡言乱语"——语法正确但逻辑不通。

---

**问题 3**：Equation only（只输出方程）够吗？

**直觉预测**："方程才是核心，自然语言只是装饰吧？"

**实际结果**（GSM8K）：
- Equation only: 5.4%
- CoT (full): 14.3%
- Standard: 6.5%

**为什么**？因为 GSM8K 的语义太复杂，模型无法直接从问题跳到方程。需要自然语言作为"中间表示"来桥接语义和数学。

**例子**：
```
Question: Mike plays ping pong for 40 minutes. In the first
20 minutes, he scores 4 points. In the second 20 minutes,
he scores 25% more points. How many total points did he score?

Equation Only (Wrong): (4 + 20 * 0.25) = 6
Chain of Thought (Correct):
- Mike played for 40 minutes
- First 20 minutes: 4 points
- Second 20 minutes: 25% more = 4 * 1.25 = 5 points
- Total: 4 + 5 = 9 points
```

---

### 第五章：关键实验的细节

#### 实验设置

**Benchmarks**：
1. **GSM8K**: 1319 道小学数学应用题（multi-step）
2. **SVAMP**: 1000 道结构变化的数学题
3. **ASDiv**: 2096 道多样化数学题
4. **AQuA**: 254 道代数选择题
5. **MAWPS**: 数学题仓库（分难度子集）

**模型**：
- GPT-3 (350M, 1.3B, 6.7B, 175B)
- LaMDA (420M, 2B, 8B, 68B, 137B)
- PaLM (8B, 62B, 540B)
- UL2 (20B)
- Codex (code-davinci-002)

**Prompt 设计**：
- 8 个 CoT 示例（人工编写）
- 所有数学任务用同一套示例（除了 AQuA）
- 不需要 prompt 工程，示例未经过调优

---

#### 核心结果

**算术推理（Arithmetic Reasoning）**：

| 模型 | Standard | CoT | 提升 |
|------|----------|-----|------|
| **GSM8K** |
| GPT-3 175B | 15.6% | 46.9% | +31.3% |
| PaLM 540B | 17.9% | 56.9% | +39.0% |
| **SVAMP** |
| GPT-3 175B | 65.7% | 68.9% | +3.2% |
| PaLM 540B | 69.4% | 79.0% | +9.6% |
| **AQuA** |
| PaLM 540B | 25.2% | 35.8% | +10.6% |

**关键发现**：
1. CoT 是 emergent ability（小模型无效）
2. 任务越复杂，收益越大（GSM8K > SVAMP）
3. PaLM 540B 达到 SOTA，超过微调的 GPT-3 + verifier

---

**常识推理（Commonsense Reasoning）**：

| 任务 | Standard (PaLM 540B) | CoT | 提升 |
|------|---------------------|-----|------|
| CSQA | 78.1% | 79.9% | +1.8% |
| StrategyQA | 68.6% | 77.8% | +9.2% |
| Date Understanding | 49.0% | 65.3% | +16.3% |
| Sports Understanding | 80.5% | 95.4% | +14.9% |
| SayCan (Robot) | 80.8% | 91.7% | +10.9% |

**关键发现**：CoT 在需要多步推理的任务上收益最大（StrategyQA, Date Understanding）。

---

**符号推理（Symbolic Reasoning）**：

**任务 1：Last Letter Concatenation**
```
Q: Take the last letters of "Lady Gaga" and concatenate.
A: "ya"
```

**任务 2：Coin Flip**
```
Q: A coin is heads up. Maybelle flips it.
   Shalonda does not flip. Is it still heads up?
A: No (flipped 1 time = odd → tails)
```

**Out-of-Distribution 测试**：
- In-domain: 示例和测试都是 2 步
- OOD: 示例 2 步，测试 3-4 步

| 任务 | Setting | Standard | CoT |
|------|---------|----------|-----|
| Coin Flip | 2步 (in-domain) | 98.1% | 100% |
| Coin Flip | 3步 (OOD) | 49.3% | 98.6% |
| Coin Flip | 4步 (OOD) | 54.8% | 90.2% |
| Letter Concat | 3步 (OOD) | 0.2% | 94.8% |
| Letter Concat | 4步 (OOD) | 0.0% | 63.0% |

**关键发现**：CoT 能促进 **length generalization** —— 模型能泛化到比示例更长的序列。

---

#### 错误分析

**正确回答的推理链分析**（LaMDA 137B, GSM8K, 50 个样本）：
- 49/50: 推理完全正确
- 1/50: 偶然猜对（推理错误但答案对）

**错误回答的推理链分析**（50 个样本）：
| 错误类型 | 比例 | 说明 |
|----------|------|------|
| Calculator error | 8% | 推理正确，计算错误 |
| Symbol mapping error | 16% | 数字映射错误 |
| One step missing | 22% | 缺少一个推理步骤 |
| Semantic understanding | 54% | 语义理解错误或逻辑不连贯 |

**Scaling 修复错误**（PaLM 62B → 540B）：
- 62B 的 20 个语义错误 → 540B 修复 6 个
- 62B 的 18 个缺失步骤错误 → 540B 修复 12 个
- 62B 的 7 个其他错误 → 540B 修复 4 个

**关键发现**：Scaling 主要修复 **语义理解**和**缺失步骤**错误。

---

#### 鲁棒性分析

**不同 Annotator**（3 位作者独立编写 CoT）：

| Annotator | GSM8K | MAWPS |
|-----------|-------|-------|
| A (原始) | 14.3% | 57.9% |
| B | 15.5% | 58.2% |
| C | 17.6% | 60.1% |
| Concise style | 11.1% | 59.6% |
| GSM8K 训练集示例 (α) | 12.6% | 53.9% |
| GSM8K 训练集示例 (β) | 12.7% | 60.9% |
| GSM8K 训练集示例 (γ) | 12.6% | 54.2% |

**关键发现**：
- 不同 annotator 风格有方差，但都远超 baseline
- 即使使用 crowd-sourced 标注（GSM8K 训练集），效果依然好
- 示例不需要和测试集同分布

**不同示例数量**：

| 示例数 | GSM8K | Sports | Coin Flip |
|--------|-------|--------|-----------|
| 1 | ~8% | ~60% | ~80% |
| 4 | ~12% | ~75% | ~95% |
| 8 | ~14% | ~85% | ~99% |

**关键发现**：CoT 的收益对不同示例数量相对鲁棒。

---

### 第六章：与其他方法对比

#### CoT vs 其他推理方法

| 方法 | 需要微调 | 需要标注数据 | 多任务能力 | GSM8K 性能 |
|------|----------|-------------|-----------|-----------|
| Standard Prompting | ❌ | ❌ | ✅ | 17.9% |
| CoT Prompting | ❌ | ❌ | ✅ | 56.9% |
| Finetuned GPT-3 + Verifier | ✅ | ✅ | ❌ | 34% |
| Neuro-symbolic | ✅ | ✅ | ❌ | ~50% |

**CoT 的优势**：
1. 不需要微调（single checkpoint 处理多任务）
2. 不需要大量标注数据
3. 性能达到或超过 SOTA

---

#### 局限性分析

**CoT 的局限**：

1. **Emergent 特性的双面性**
   - 只在 100B+ 模型上有效
   - 小模型用 CoT 反而更差
   - 实际应用成本高

2. **推理链不一定可靠**
   - 正确答案可能有错误推理
   - 模型会"自信地胡说八道"
   - 事实性无法保证

3. **标注成本（如果要微调）**
   - 人工编写 CoT 成本高
   - 虽然 prompting 不需要标注，但 few-shot 示例需要
   - 可能用 synthetic data 解决

4. **长序列任务**
   - 没有记忆机制
   - 无法跨 trial 学习
   - 复杂任务可能迷失

---

#### 改进方向（后续论文）

| 改进方向 | 论文 | 核心思想 |
|----------|------|----------|
| Self-Consistency | Wang et al. (2022) | 采样多条推理链，取多数答案 |
| ReAct | Yao et al. (2022) | Thought + Action + Observation |
| Reflexion | Shinn et al. (2023) | 从失败中学习，自我反思 |
| STaR | Zelikman et al. (2022) | 用模型生成的 CoT 自举训练 |
| Auto-CoT | Zhang et al. (2022) | 自动生成 CoT 示例 |

---

### 第七章：如何应用

#### 适用场景

根据论文分析，CoT 最有帮助的场景满足三个条件：
1. **任务具有挑战性**，需要多步推理
2. **使用大语言模型**（100B+ 参数）
3. **Standard Prompting 的 scaling curve 相对平坦**

**推荐场景**：
- ✅ 数学应用题（GSM8K 类型）
- ✅ 多跳常识推理（StrategyQA 类型）
- ✅ 符号操作（状态追踪、字符串处理）
- ✅ 日期/时间推理
- ✅ 机器人任务规划

**不推荐场景**：
- ❌ 单步简单问题（收益小）
- ❌ 小模型（< 10B 参数）
- ❌ 标准 Prompting 已经很好的任务

---

#### 实现指南

**Step 1: 准备 CoT 示例**

```python
# 8 个示例足够，不需要太多
examples = [
    """Q: There are 15 trees in the grove. Grove workers will
         plant trees in the grove today. After they are done,
         there will be 21 trees. How many trees did the grove
         workers plant today?
       A: There are 15 trees originally. Then there were 21
         trees after some more were planted. So there must have
         been 21 - 15 = 6. The answer is 6.""",
    # ... 更多示例
]
```

**Step 2: 构建 Prompt**

```python
def build_cot_prompt(question, examples):
    prompt = "\n\n".join(examples)
    prompt += f"\n\nQ: {question}\nA:"
    return prompt
```

**Step 3: 生成答案**

```python
prompt = build_cot_prompt(my_question, examples)
response = model.generate(prompt, max_tokens=512)
# 从 response 中提取最终答案
```

**Step 4 (Optional): 添加外部计算器**

```python
# 提取推理链中的方程，用 Python eval 执行
# 可以修复 calculator error
```

---

#### 不适用场景

**不要在这些场景用 CoT**：
1. **简单分类任务**（情感分析、垃圾邮件检测）
   - 标准 prompting 已经足够
   - CoT 增加延迟但无收益

2. **小模型部署**
   - < 10B 参数的模型
   - CoT 会产生逻辑错误的输出

3. **需要低延迟的场景**
   - CoT 生成更长的输出
   - 推理时间增加 2-5 倍

4. **事实性要求极高的场景**
   - CoT 推理链可能包含错误信息
   - 医疗、法律等高风险场景需要额外验证

---

### 第八章：延伸思考

#### 深度问题

1. **为什么 CoT 只在 大模型上有效？**
   - 小模型缺少什么能力？
   - 是语义理解不足，还是逻辑推理不足？
   - 能否通过训练让小模型获得 CoT 能力？

2. **CoT 真的在"推理"吗？**
   - 还是只是在模仿推理的形式？
   - 模型理解它生成的推理链吗？
   - 如何验证模型的"理解"？

3. **推理链的忠实性（Faithfulness）问题**
   - 模型生成的推理链是否反映了真实的计算过程？
   - 还是只是"事后合理化"？
   - 如何确保推理链是真实的？

4. **CoT 能否泛化到更多任务？**
   - 机器翻译能用 CoT 吗？
   - 文本摘要能用 CoT 吗？
   - 什么样的任务适合 CoT？

5. **如何自动化生成 CoT 示例？**
   - 能否用大模型自己生成示例？
   - 如何保证生成质量？
   - 能否完全消除人工标注？

6. **CoT 与人类推理的相似性**
   - CoT 真的模仿了人类思考吗？
   - 人类的推理是 serial 还是 parallel？
   - 能否设计更接近人类的推理方式？

7. **CoT 的理论基础**
   - 为什么 intermediate tokens 能帮助推理？
   - 有没有理论框架解释 CoT？
   - CoT 的计算复杂性如何？

---

### 第九章：与其他论文的联系

#### 直接影响（下游工作）

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| **Self-Consistency** | 2022.03 | 对同一问题采样多条 CoT，取多数答案 |
| **ReAct** | 2022.10 | 结合 Reasoning 和 Acting |
| **STaR** | 2022.03 | 用 CoT 自举训练模型 |
| **Reflexion** | 2023 | 从失败中学习，添加自我反思 |
| **Auto-CoT** | 2022 | 自动生成 CoT 示例 |

#### 并行工作

| 论文 | 年份 | 核心思想 |
|------|------|----------|
| **Scratchpads** | 2021.12 | 用中间步骤执行 Python 程序 |
| **Show Your Work** | 2020 | 在答案前生成解释 |

#### 上游基础

| 论文 | 年份 | 核心贡献 |
|------|------|----------|
| **GPT-3** | 2020 | Few-shot prompting 基础 |
| **Ling et al. (2017)** | 2017 | 用自然语言 rationale 解数学题 |
| **Cobbe et al. (2021)** | 2021 | GSM8K 数据集 + 微调方法 |

---

### 第十章：关键引用

论文中的经典语句：

> "Chain-of-thought reasoning is an emergent property of model scale."

> "Standard prompting only provides a lower bound on the capabilities of large language models."

> "The ability to perform abstract manipulations on unseen symbols only arises at the scale of 100B model parameters."

---

## 附录：完整 Prompt 示例

### GSM8K CoT Prompt（8 个示例）

```
Q: There are 15 trees in the grove. Grove workers will plant
   trees in the grove today. After they are done, there will
   be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees
   after some more were planted. So there must have been
   21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive,
   how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.
   The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35,
   how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in
   total they had 32 + 42 = 74. After eating 35, they had
   74 - 35 = 39. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now
   Jason has 12 lollipops. How many lollipops did Jason give
   to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving
   some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from
   his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got 2 toys each from his mom
   and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.

Q: There were nine computers in the server room. Five more computers
   were installed each day, from monday to thursday. How many
   computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more
   computers were added. So 5 * 4 = 20 computers were added.
   9 + 20 is 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls.
   On wednesday, he lost 2 more. How many golf balls did he have
   at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday,
   he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33
   golf balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much
   money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be
   5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15
   is 8. The answer is 8.

Q: {your_question}
A:
```

---

## 总结

**Chain-of-Thought Prompting** 的核心贡献：

1. **发现**：CoT 是一种 emergent ability，只在 100B+ 模型上有效
2. **方法**：在 prompt 中包含推理示例，无需微调即可激发推理能力
3. **效果**：在 GSM8K、StrategyQA 等任务上达到 SOTA
4. **影响**：开启了后续大量研究（Self-Consistency、ReAct、Reflexion 等）

**核心洞察**：
> "Standard prompting only provides a lower bound on the capabilities of large language models."

CoT 告诉我们：**大模型的真正能力，需要正确的 prompting 方法才能释放**。

---

*文档生成时间：2026 年 3 月*
*基于论文：arXiv:2201.11903v6 (Jan 2023)*
