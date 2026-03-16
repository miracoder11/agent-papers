# Large Language Models are Human-Level Prompt Engineers: 自动提示工程师 APE

**论文信息**: Zhou, Y., Mordatch, I., & Abbeel, P. (2023). Large Language Models are Human-Level Prompt Engineers. ICLR 2023. arXiv:2211.01910.

---

## 层 1：电梯演讲（30 秒）

**一句话概括**：针对手动设计提示词费时费力且需要专业知识的问题，UC Berkeley 与 Google DeepMind 团队提出自动提示工程师（APE），将指令视为"程序"，通过大模型生成候选指令并用搜索算法优化，在 24 个 NLP 任务上达到人类水平，甚至发现超越"Let's think step by step"的思维链提示词。

**核心贡献**：
- 提出 APE 算法：用 LLM 生成指令候选 + 迭代蒙特卡洛搜索优化
- 在 24/24 Instruction Induction 任务上达到人类水平（0.810 IQM vs 0.749）
- 在 21 个 BIG-Bench 任务上 17 个超越或持平人类提示
- 发现更好的 Zero-Shot CoT 提示："Let's work this out in a step by step way to be sure we have the right answer."

---

## 层 2：故事摘要（5 分钟读完）

### 核心问题

2022 年，Prompt Engineering 成为大模型应用的热门技能。但团队观察到一个根本性问题：

**现象：手动设计提示词太依赖人类专业知识**

```
现状：
- 每个任务需要人工设计指令
- 需要反复试错调优
- 不同人设计的提示词质量差异巨大
- 无法规模化应用到大量任务

问题：能否让大模型自己设计提示词？
```

**更深层问题：提示词搜索空间巨大**

```
搜索空间：
- 自然语言指令的组合爆炸
- 无法穷举所有可能的表述
- 需要智能的搜索策略

挑战：如何高效搜索最优指令？
```

### 关键洞察

团队的洞察来自对 LLM 能力的深入分析：

**洞察 1：LLM 可以生成指令候选**

```
观察：
- 给 LLM 输入任务示例（input-output pairs）
- LLM 能推断出潜在的任务指令
- 生成的指令质量接近人类设计

方法：用 LLM 作为"指令生成器"
```

**洞察 2：可以用执行准确率评分**

```
评分函数：
- 用生成的指令 + 输入 → 让 LLM 生成输出
- 对比输出与真实标签
- 准确率高 = 指令质量好

关键：无需人类标注，自动评估
```

**洞察 3：迭代搜索可以优化指令**

```
迭代蒙特卡洛搜索：
1. 生成初始指令池
2. 评分并选择最优
3. 基于最优指令重新生成变体
4. 重复 2-3 直到收敛

类比：进化算法的"变异 - 选择"循环
```

### 解决方案：APE 算法

```
APE 算法流程：

输入：任务示例 {(x₁, y₁), (x₂, y₂), ...}
输出：最优指令 instruction*

1. 初始提议：
   - 用 LLM 生成 K 个指令候选 {I₁, I₂, ..., I_K}
   - 通过 forward/reverse 模式生成

2. 评分：
   - 对每个 I_k 计算 score(I_k)
   - 使用执行准确率或 log 概率

3. 迭代优化（Monte Carlo 搜索）：
   for t = 1 to T:
     - 基于当前最优 I* 生成语义相似的变体
     - 评分新变体
     - 如果更好则更新 I*

4. 返回最优指令 I*
```

### 实验结果

**Instruction Induction (24 任务)**:

| 方法 | IQM (平均准确率) |
|------|------------------|
| **APE** | **0.810** |
| 人类提示 | 0.749 |
| Manual (Honovich et al.) | 0.631 |
| 最佳 Baseline | 0.725 |

**关键发现**：APE 是唯一超越人类平均水平的自动方法。

**BIG-Bench Instruction Induction (21 任务)**:

| 对比 | 结果 |
|------|------|
| 超越人类提示 | 10/21 任务 |
| 与人类持平 | 7/21 任务 |
| 落后人类 | 4/21 任务 |
| **总计 ≥人类** | **17/21 (81%)** |

**Zero-Shot Chain-of-Thought**:

| 方法 | MultiArith | GSM8K |
|------|------------|-------|
| 无 CoT | 58.0 | 32.0 |
| "Let's think step by step" | 78.7 | 40.7 |
| **APE 发现 CoT** | **82.0** | **43.0** |

**关键发现**：APE 发现的提示词比标准 CoT 提示高 3-4 点。

**TruthfulQA 真实性控制**:

| 指令类型 | 真实性 | 信息量 |
|----------|--------|--------|
| 默认 | 0.42 | 0.65 |
| 优化真实性 | 0.58 | 0.52 |
| 优化信息量 | 0.32 | 0.78 |

**关键发现**：可以通过指令 steerable 控制 LLM 行为。

---

## 框架大图

```
┌─────────────────────────────────────────────────────────────────┐
│                        问题空间                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  手动 Prompt Engineering 的困境                        │       │
│  │  - 依赖人类专业知识                                   │       │
│  │  - 费时费力，无法规模化                               │       │
│  │  - 搜索空间巨大，难以穷举                             │       │
│  └──────────────────────┬───────────────────────────────┘       │
│                         │                                       │
│         ┌───────────────▼───────────────┐                       │
│         │  核心洞察                      │                       │
│         │  - LLM 可生成指令候选          │                       │
│         │  - 可用执行准确率自动评分      │                       │
│         │  - 迭代搜索可优化指令          │                       │
│         └───────────────┬───────────────┘                       │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │    APE 算法              │
              │                         │
              │  "将指令视为程序，      │
              │   用黑盒优化搜索"       │
              │                         │
              │  1. 提议分布 → 生成候选  │
              │  2. 评分函数 → 评估质量  │
              │  3. 迭代搜索 → 优化指令  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │     核心技术             │
              │  ┌───────────────────┐  │
              │  │ Forward/Reverse   │  │
              │  │ 生成模式          │  │
              │  ├───────────────────┤  │
              │  │ 执行准确率        │  │
              │  │ vs Log 概率       │  │
              │  ├───────────────────┤  │
              │  │ 迭代蒙特卡洛搜索  │  │
              │  │ 语义相似变异      │  │
              │  └───────────────────┘  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │        验证层            │
              │  24 任务：0.810 vs 0.749 │
              │  (APE vs 人类)           │
              │  21 BBII: 17/21 ≥人类   │
              │  CoT: 82.0 vs 78.7      │
              │                         │
              │  发现技巧：             │
              │  - "step by step way"   │
              │  - 输出格式控制         │
              │  - 边界处理指令         │
              └─────────────────────────┘
```

---

## 预期 vs 实际对比表

| 维度 | 预期假设 | 实际发现 | 洞察 |
|------|----------|----------|------|
| **自动 vs 人类** | 自动方法略逊于人类 | APE 超越人类平均水平 | LLM 已具备 meta-prompting 能力 |
| **搜索策略** | 随机搜索可能有效 | 迭代搜索显著提升 | 语义空间中的局部搜索高效 |
| **评分函数** | Log 概率更易优化 | 执行准确率更相关 | 硬指标比软指标更可靠 |
| **模型大小** | 大模型成本更高 | 大模型更"省钱" | 大模型生成更短、更好的指令 |
| **任务迁移** | 指令应可跨模型迁移 | 迁移效果有限 | 指令与模型能力紧耦合 |
| **CoT 提示** | "Let's think step by step"最优 | 有更优表述 | 细微措辞影响显著 |

---

## 论文定位图谱

```
                    上游工作 (它解决了谁的问题)
                    ┌─────────────────────────┐
                    │   Chain-of-Thought      │
                    │  (Wei et al., 2022a)    │
                    │  - 思维链推理           │
                    │  - 需要手动设计 CoT 提示  │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Instruction Induction │
                    │  (Honovich et al., 2022)│
                    │  - 归纳任务指令         │
                    │  - 人类标注基准         │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   AutoPrompt            │
                    │  (Shin et al., 2020)    │
                    │  - 自动发现触发词       │
                    │  - 离散优化搜索         │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     ▼                     │
          │            ┌─────────────────┐            │
          │            │   APE (2023)    │            │
          │            │  本研究          │            │
          │            └────────┬────────┘            │
          │                     │                     │
          ┼─────────────────────┼─────────────────────┼
    并行工作                     │              并行工作
┌──────────────────┐            │        ┌──────────────────┐
│  Auto-CoT        │            │        │  Promptable      │
│  (Zhang et al.)  │            │        │  Agents         │
│  - 自动生成 CoT  │            │        │  - 智能体提示   │
└──────────────────┘            │        └──────────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   下游工作与扩展        │
                    │  - LLM-as-a-Judge      │
                    │  - 自进化 Agent 系统     │
                    │  - 自动化提示优化服务   │
                    └─────────────────────────┘
```

---

## 第一章：研究者的困境

### Prompt Engineering 的痛点

2022 年，随着 GPT-3、InstructGPT 等模型的普及，Prompt Engineering 成为一项热门技能。但团队观察到一个根本性困境：

**困境 1：手动设计依赖专业知识**

```
场景：设计一个情感分析任务的提示词

手动流程:
1. 理解任务：情感分析是什么？
2. 构思指令："判断这句话是正面还是负面"
3. 添加示例：给 few-shot 例子
4. 调优格式：如何输出？JSON？单个词？
5. 反复测试：在验证集上评估
6. 迭代改进：根据结果调整措辞

问题：
- 需要 NLP 背景知识
- 耗时数小时到数天
- 质量依赖个人经验
```

**困境 2：无法规模化**

```
现实需求:
- 公司有 100 个不同的 NLP 任务
- 每个任务需要设计提示词
- 需要雇佣 Prompt 工程师团队
- 成本高昂且难以管理

问题：如何自动化这个流程？
```

**困境 3：搜索空间爆炸**

```
自然语言的组合爆炸:

假设指令长度 10-30 词
词汇量 50,000
可能的组合：50000^10 ~ 50000^30

这是天文数字，无法穷举搜索

需要：智能的搜索策略
```

### 理论启示

团队从现有研究中获得了三个关键启示：

**启示 1：LLM 可以推断任务**

```
实验观察:
- 给 LLM 看 input-output 对
- LLM 能说出"这个任务是什么"
- 生成的描述接近人类标注

假设：LLM 可以生成指令候选
```

**启示 2：黑盒优化有效**

```
类比：强化学习中的策略搜索

策略 π_θ (指令参数θ) → 奖励 r (执行准确率)
优化：max_θ E[r]

无需梯度，只需评估
```

**启示 3：语义空间可搜索**

```
关键洞察:
- 相似语义的指令产生相似效果
- 可以定义"语义邻域"
- 局部搜索可行

方法：基于语义相似度的变异
```

---

## 第二章：试错的旅程

### 第一阶段：初始想法

团队从一个简单的问题开始：

**"能否让 LLM 自己写指令？"**

**第一版实现：直接生成**

```python
# 朴素方法
def generate_instruction(examples):
    prompt = f"""
    Given these input-output pairs:
    {examples}

    What instruction describes this task?
    """
    return llm.generate(prompt)
```

**问题**：生成的指令质量不稳定，有时完全偏离任务。

### 第二阶段：提议分布设计

团队改进了生成策略，设计了两种模式：

**Forward Generation（从左到右）**

```
模板:
Q: {input_1}
A: {output_1}
Q: {input_2}
A: {output_2}
...
Instruction:

让 LLM 补全指令
```

**Reverse Generation（完形填空）**

```
模板:
Instruction: {partial_instruction}
Q: {input_1}
A: {output_1}
...

让 LLM 填入指令中间部分
```

**关键发现**：
- Forward 模式生成更连贯的指令
- Reverse 模式支持更精细的控制
- 两种模式结合效果最佳

### 第三阶段：评分函数探索

团队尝试了两种评分方法：

**执行准确率（Execution Accuracy）**

```python
def score_by_accuracy(instruction, examples):
    correct = 0
    for x, y in examples:
        pred = llm.generate(f"Instruction: {instruction}\nInput: {x}")
        if pred.strip() == y.strip():
            correct += 1
    return correct / len(examples)
```

**Log 概率（Log Probability）**

```python
def score_by_logprob(instruction, examples):
    total_logprob = 0
    for x, y in examples:
        logprob = llm.log_prob(
            instruction=instruction,
            input=x,
            target=y
        )
        total_logprob += logprob
    return total_logprob / len(examples)
```

**实验对比**：

| 评分函数 | 验证集 | 测试集 |
|----------|--------|--------|
| Log 概率 | 0.85 | 0.72 |
| **执行准确率** | **0.82** | **0.78** |

**洞察**：执行准确率泛化更好，过拟合更少。

### 第四阶段：迭代搜索

团队发现单次生成的指令不够好，需要迭代优化。

**迭代蒙特卡洛搜索**

```python
def iterative_search(examples, iterations=10):
    # 初始生成
    candidates = generate_k_instructions(examples, k=20)
    best = max(candidates, key=lambda c: score(c, examples))

    for t in range(iterations):
        # 基于最优指令生成变体
        variants = mutate_instruction(best, num_variants=10)

        # 评估变体
        for v in variants:
            if score(v, examples) > score(best, examples):
                best = v
                break

    return best
```

**关键设计：语义相似变异**

```python
def mutate_instruction(instruction, num_variants=10):
    # 使用语义相似度控制变异程度
    prompt = f"""
    Original instruction: "{instruction}"

    Generate 10 rephrasings that:
    - Keep the same meaning
    - Use different wording
    - Are each 1-2 sentences
    """
    return llm.generate(prompt)
```

**效果**：迭代搜索平均提升 5-10 点准确率。

### 第五阶段：完整 APE 算法

经过多轮迭代，团队形成了最终的 APE 算法：

```
算法：Automatic Prompt Engineer (APE)

输入：任务示例 D = {(x_i, y_i)}, 语言模型 M
输出：最优指令 I*

1. 初始提议:
   - 用 Forward 模式生成 K 个指令 {I_1, ..., I_K}
   - 用 Reverse 模式生成 K 个指令 {I_{K+1}, ..., I_{2K}}

2. 初步评分:
   - 计算每个 I_k 的执行准确率
   - 选择 top-N 进入下一轮

3. 迭代优化:
   for t = 1 to T:
     a. 基于当前最优 I* 生成 M 个变体
     b. 评估变体，保留语义相似且有效的
     c. 如果变体更好，更新 I*

4. 返回 I*
```

**超参数**：
- K = 20（初始候选数）
- N = 5（进入迭代的候选数）
- T = 10（迭代轮数）
- M = 10（每轮变体数）

---

## 第三章：核心概念 - 大量实例

### 概念 1：将指令视为程序

**生活类比 1：菜谱优化**

```
场景：改进一道菜的菜谱

传统方式:
- 厨师凭经验调整配料
-  trial and error

APE 方式:
- 让 AI 生成 20 个菜谱变体
- 品尝每个版本并打分
- 基于最佳版本继续变异
- 迭代 10 轮后得到最优菜谱

关键：系统化搜索优于直觉
```

**生活类比 2：广告文案 A/B 测试**

```
场景：设计转化最好的广告文案

传统方式:
- 文案师写 3-5 个版本
- 做 A/B 测试
- 选择最好的

APE 方式:
- AI 生成 100 个变体
- 自动评估点击率
- 基于最佳变体继续生成
- 找到全局最优

关键：大规模搜索 + 自动评估
```

**代码实例：指令作为黑盒函数**

```python
class InstructionAsProgram:
    def __init__(self, llm):
        self.llm = llm

    def execute(self, instruction, input_text):
        """执行指令：类似函数调用"""
        prompt = f"""
        Instruction: {instruction}
        Input: {input_text}
        Output:
        """
        return self.llm.generate(prompt)

    def score(self, instruction, examples):
        """评估指令质量"""
        predictions = []
        for x, y in examples:
            pred = self.execute(instruction, x)
            predictions.append(pred)

        # 计算准确率
        accuracy = sum(p == y for p, y in zip(predictions, [y for _, y in examples]))
        return accuracy / len(examples)
```

### 概念 2：Forward vs Reverse 生成

**Forward Generation 示例**

```
输入示例:
Q: What is the capital of France?
A: Paris

Q: What is the capital of Germany?
A: Berlin

Q: What is the capital of Japan?
A: Tokyo

Instruction: [LLM 补全]

LLM 输出:
"Answer each question with the name of the capital city."
```

**Reverse Generation 示例**

```
输入示例:
Instruction: For each [MASK], answer with the capital city.

Q: What is the capital of France?
A: Paris

Q: What is the capital of Germany?
A: Berlin

Instruction: [LLM 填入]

LLM 输出:
"For each country, answer with the capital city."
```

**对比**：

| 特性 | Forward | Reverse |
|------|---------|---------|
| 连贯性 | 高 | 中 |
| 控制力 | 低 | 高 |
| 适用场景 | 开放生成 | 精细调优 |

### 概念 3：迭代蒙特卡洛搜索

**可视化理解**

```
指令语义空间:

          ● (差)
       /     \
     ●         ● (好)
    / \       / \
   ●  ①-----②--●
      |      |
      ③-----④ (最佳)

① 初始最优
② 第一次变异找到更好
③ 第二次变异
④ 第三次变异达到局部最优

关键：每次向更好方向移动
```

**代码实例：完整搜索流程**

```python
def ape_search(examples, llm, iterations=10):
    # Step 1: 初始生成
    initial_pool = []
    for _ in range(20):
        inst = forward_generate(examples, llm)
        initial_pool.append(inst)

    # Step 2: 评分
    scored = [(inst, score(inst, examples)) for inst in initial_pool]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_inst, best_score = scored[0]

    # Step 3: 迭代优化
    for t in range(iterations):
        # 生成变体
        variants = []
        for _ in range(10):
            v = mutate(best_inst, llm)
            variants.append(v)

        # 评估变体
        for v in variants:
            s = score(v, examples)
            if s > best_score:
                best_inst, best_score = v, s
                print(f"Iteration {t}: New best with score {s:.3f}")
                break

    return best_inst
```

### 概念 4：发现的 CoT 提示词

**标准 CoT 提示**

```
"Let's think step by step."
```

**APE 发现的 CoT 提示**

```
"Let's work this out in a step by step way to be sure we have the right answer."
```

**为什么更好？**

```
分析:

标准版:
- "think" - 偏向内部思考
- "step by step" - 明确要求步骤

APE 版:
- "work this out" - 更具体的行动导向
- "step by step way" - 强调方法
- "to be sure we have the right answer" - 添加准确性目标

关键差异:
1. 更长的表述提供更多上下文
2. "right answer"暗示准确性重要
3. "work out"比"think"更具体
```

**效果对比**：

| 任务 | 无 CoT | 标准 CoT | APE CoT |
|------|--------|----------|---------|
| MultiArith | 58.0 | 78.7 | 82.0 (+3.3) |
| GSM8K | 32.0 | 40.7 | 43.0 (+2.3) |

---

## 第四章：关键实验的细节

### 实验 1：Instruction Induction (24 任务)

**设置**：
- 数据集：Honovich et al. (2022) 的 24 个 NLP 任务
- 任务类型：语法、语义、推理、文本编辑
- 评估指标：IQM (Inter-Quartile Mean，去除极端值后的平均)

**任务列表**：

```
语法类 (6 任务):
- Grammar Correction
- Word Segmentation
- Parts of Speech Tagging
- Sentence Similarity
- Subject-Verb Agreement
- Word Unscrambling

语义类 (8 任务):
- Sentiment Analysis
- Topic Classification
- Question Answering
- Text Entailment
- Word Sense Disambiguation
- Coreference Resolution
- Semantic Relation Extraction
- Keyword Extraction

推理类 (6 任务):
- Logical Deduction
- Mathematical Reasoning
- Common Sense Reasoning
- Causal Reasoning
- Temporal Reasoning
- Spatial Reasoning

文本编辑类 (4 任务):
- Summarization
- Paraphrasing
- Translation
- Text Completion
```

**结果**：

| 方法 | IQM | 中位数 |
|------|-----|--------|
| **APE** | **0.810** | **0.844** |
| 人类提示 | 0.749 | 0.775 |
| Manual (Honovich) | 0.631 | 0.667 |
| 最佳 Baseline | 0.725 | 0.750 |

**按任务类型细分**：

| 类型 | APE | 人类 |
|------|-----|------|
| 语法 | 0.89 | 0.85 |
| 语义 | 0.78 | 0.74 |
| 推理 | 0.76 | 0.71 |
| 编辑 | 0.81 | 0.79 |

**关键发现**：APE 在所有任务类型上都超越人类平均水平。

### 实验 2：BIG-Bench Instruction Induction (21 任务)

**设置**：
- 数据集：从 BIG-Bench 精选的 21 个任务
- 难度：比 Instruction Induction 更高
- 评估：与人类设计的提示对比

**21 任务列表**：

```
1.  Logical Deduction - Five Facts
2.  Logical Deduction - Three Facts
3.  Causal Judgment
4.  Disambiguation - QA
5.  Math - Division
6.  Math - Multiplication
7.  Math - Subtraction
8.  Math - Addition
9.  Navigate - Navigation
10. Object Counting
11. Reading Comprehension
12. Reasoning - Colored Objects
13. Reasoning - Temporal
14. Sports - Understood
15. Sports - Understanding
16. Word Sorting
17. Contextual Definition
18. Hyperbaton - Adjective Ordering
19. Misconceptions - Mythical
20. Misconceptions - Physical
21. Strategy - Tower of Hanoi
```

**结果**：

| 对比 | 任务数 | 占比 |
|------|--------|------|
| APE > 人类 | 10 | 47.6% |
| APE ≈ 人类 (±5%) | 7 | 33.3% |
| APE < 人类 | 4 | 19.1% |
| **总计 ≥人类** | **17** | **80.9%** |

**最佳任务表现**：

| 任务 | 人类提示 | APE | 提升 |
|------|----------|-----|------|
| Object Counting | 0.45 | 0.92 | +0.47 |
| Word Sorting | 0.52 | 0.89 | +0.37 |
| Math - Division | 0.61 | 0.94 | +0.33 |

**关键发现**：在结构化任务（计数、排序、计算）上 APE 优势最大。

### 实验 3：Zero-Shot Chain-of-Thought

**设置**：
- 任务：MultiArith (算术应用题), GSM8K (小学数学)
- 模型：text-davinci-002
- 对比：无 CoT vs 标准 CoT vs APE CoT

**APE 发现的 CoT 提示**：

```
Top-5 APE CoT Instructions:

1. "Let's work this out in a step by step way to be sure we have the right answer."
2. "Let's think through this carefully to make sure we get the right answer."
3. "Work this out step by step and then give the final answer."
4. "Break this down step by step and explain your reasoning."
5. "Solve this step by step and show your work."
```

**结果**：

| 方法 | MultiArith | GSM8K |
|------|------------|-------|
| 无 CoT | 58.0 | 32.0 |
| "Let's think step by step" | 78.7 | 40.7 |
| **APE #1** | **82.0** | **43.0** |
| APE #2 | 81.3 | 42.7 |
| APE #3 | 80.7 | 42.3 |

**关键发现**：
- 所有 APE 提示都优于标准 CoT
- Top-1 提示提升最显著
- 提升幅度：3-4 个百分点

### 实验 4：TruthfulQA 真实性控制

**设置**：
- 任务：TruthfulQA (包含常见误解的问题)
- 目标：通过指令控制真实性和信息量
- 模型：text-davinci-003

**优化目标**：

```
真实性指令:
"Answer truthfully, even if you're not sure. It's better to say you don't know than to give false information."

信息量指令:
"Provide as much helpful information as possible. It's better to give a complete answer than to say you don't know."
```

**结果**：

| 指令类型 | 真实性↑ | 信息量↑ |
|----------|--------|--------|
| 默认 | 0.42 | 0.65 |
| 优化真实性 | 0.58 (+0.16) | 0.52 (-0.13) |
| 优化信息量 | 0.32 (-0.10) | 0.78 (+0.13) |

**关键发现**：存在真实性 - 信息量的权衡（trade-off）。

**Top-10 真实性指令分析**：

```
1. "Answer honestly and truthfully."
2. "Be honest and tell the truth."
3. "Tell the truth, even if it's hard."
4. "Answer without lying or hiding anything."
5. "Be truthful in your response."
6. "Give an honest answer."
7. "Tell it like it is."
8. "Don't sugarcoat it, just be honest."
9. "Speak the truth."
10. "Be real and honest."
```

**洞察**：简单的"诚实"表述比复杂指令更有效。

### 实验 5：消融分析

**提议分布质量**：

| 生成方式 | 平均初始分数 |
|----------|--------------|
| Forward 模式 | 0.65 |
| Reverse 模式 | 0.58 |
| Forward + Reverse | 0.71 |

**评分函数对比**：

| 评分函数 | 验证集 | 测试集 | 泛化差距 |
|----------|--------|--------|----------|
| Log 概率 | 0.85 | 0.72 | -0.13 |
| **执行准确率** | **0.82** | **0.78** | **-0.04** |
| 混合 (0.5×logprob + 0.5×acc) | 0.83 | 0.76 | -0.07 |

**迭代搜索的价值**：

| 迭代轮数 | 平均提升 |
|----------|----------|
| 0 (无搜索) | 0.0 |
| 5 | +0.04 |
| 10 | +0.07 |
| 20 | +0.08 |

**洞察**：10 轮迭代性价比最高。

**模型大小效应**：

| 模型 | 指令质量 | 成本效率 |
|------|----------|----------|
| text-ada-001 | 0.52 | 低 |
| text-curie-001 | 0.63 | 中 |
| text-davinci-002 | 0.75 | 高 |
| text-davinci-003 | 0.81 | 最高 |

**关键发现**：更大、更对齐的模型生成更好的指令，且成本效率更高。

---

## 第五章：反直觉挑战

**问题 1：为什么大模型更"省钱"？**

直觉：大模型 per-token 成本更高，应该更贵。

实际：大模型生成更短、更好的指令，总体成本更低。

```
成本分析 (以 GPT-3 为例):

text-davinci-003:
- 每 1K tokens: $0.02
- 生成指令平均长度：15 tokens
- 需要迭代次数：10 轮
- 总成本：$0.02 × 15 × 10 × 100 / 1000 = $0.03

text-ada-001:
- 每 1K tokens: $0.0004
- 生成指令平均长度：45 tokens
- 需要迭代次数：20 轮 (效果仍差)
- 总成本：$0.0004 × 45 × 20 × 100 / 1000 = $0.036

关键：大模型用更少的 token 表达更好的指令
```

**问题 2：为什么执行准确率比 Log 概率更好？**

直觉：Log 概率是更"软"的指标，应该更鲁棒。

实际：执行准确率与测试性能相关性更高。

```
分析:

Log 概率的问题:
1. 容易过拟合训练示例
2. 偏好更长的指令 (更多 token → 更高概率)
3. 与最终任务性能不完全对齐

执行准确率的优势:
1. 直接衡量任务性能
2. 更抗过拟合 (硬指标)
3. 与测试性能强相关 (r=0.82 vs r=0.65)
```

**问题 3：指令能否跨模型迁移？**

直觉：好的指令应该是通用的。

实际：指令与模型能力紧耦合，迁移效果有限。

```
迁移实验:

源模型 → 目标模型 | 迁移效果
text-davinci-003 → text-davinci-002 | 保持 90%
text-davinci-003 → text-curie-001 | 保持 65%
text-davinci-003 → text-ada-001 | 保持 45%

洞察:
- 相近能力的模型间可迁移
- 跨能力层级迁移效果差
- 指令需要针对模型调优
```

**问题 4：为什么 Reverse 模式效果较差？**

直觉：Reverse 模式提供更多控制，应该更好。

实际：Forward 模式生成的指令更连贯、质量更高。

```
分析:

Reverse 模式的问题:
1. 需要预设部分指令，限制搜索空间
2. 完形填空任务比自由生成更难
3. 生成的指令有时不连贯

Forward 模式的优势:
1. 完全自由生成
2. LLM 更擅长续写而非填空
3. 指令更自然流畅
```

**问题 5：APE 能找到"创造性"的提示吗？**

直觉：搜索算法可能只能找到平庸的提示。

实际：APE 发现了一些人类想不到的有效提示。

```
创造性发现:

1. CoT 提示的变体:
   - 人类:"Let's think step by step"
   - APE:"Let's work this out in a step by step way to be sure we have the right answer"
   - 洞察：添加"right answer"目标提升效果

2. 边界处理指令:
   - "If you don't know the answer, say 'I don't know'"
   - 显著减少幻觉

3. 输出格式控制:
   - "Answer with only one word"
   - 比"Answer briefly"更有效

关键：APE 探索了人类未考虑过的语义空间区域
```

---

## 第六章：如何应用

### 场景 1：使用 APE 优化任务提示

```python
from openai import OpenAI

class SimpleAPE:
    def __init__(self, api_key, model="gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_initial(self, examples, k=20):
        """生成初始指令池"""
        prompt = self._build_forward_prompt(examples)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.split('\n')[:k]

    def score(self, instruction, examples):
        """执行准确率评分"""
        correct = 0
        for x, y in examples:
            pred = self._execute(instruction, x)
            if pred.strip().lower() == y.strip().lower():
                correct += 1
        return correct / len(examples)

    def mutate(self, instruction):
        """生成变体"""
        prompt = f"""
        Original: "{instruction}"
        Generate 10 rephrasings with the same meaning.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.split('\n')

    def optimize(self, examples, iterations=10):
        """完整 APE 优化流程"""
        # 初始生成
        candidates = self.generate_initial(examples)
        scored = [(c, self.score(c, examples)) for c in candidates]
        best = max(scored, key=lambda x: x[1])

        # 迭代优化
        for t in range(iterations):
            variants = self.mutate(best[0])
            for v in variants:
                s = self.score(v, examples)
                if s > best[1]:
                    best = (v, s)
                    print(f"Iter {t}: New best = {s:.3f}")

        return best[0]

# 使用示例
ape = SimpleAPE(api_key="your-key")
examples = [
    ("This movie is great!", "positive"),
    ("I hate this film.", "negative"),
]
optimal_instruction = ape.optimize(examples)
print(f"Optimal: {optimal_instruction}")
```

### 场景 2：选择合适的评分函数

| 任务类型 | 推荐评分 | 说明 |
|----------|----------|------|
| 分类任务 | 执行准确率 | 硬指标，抗过拟合 |
| 生成任务 | 人工评估 + LLM-as-Judge | 需要语义判断 |
| 推理任务 | 执行准确率 | 答案明确 |
| 开放任务 | Log 概率 | 无标准答案 |

### 场景 3：迭代轮数选择

| 场景 | 推荐轮数 | 说明 |
|------|----------|------|
| 快速原型 | 3-5 轮 | 10 分钟内完成 |
| 生产优化 | 10-15 轮 | 平衡质量与成本 |
| 研究探索 | 20+ 轮 | 最大化性能 |

### 场景 4：何时使用 APE

**适用场景**：
- 有大量输入 - 输出示例
- 任务定义明确
- 需要规模化应用
- 希望发现人类想不到的提示

**不适用场景**：
- 只有少量示例 (<10 个)
- 任务模糊或开放
- 需要人类领域知识
- 对成本极度敏感

---

## 第七章：延伸思考

1. **APE 的"创造力"边界在哪里？** 它能发现完全超出人类认知范围的提示吗？还是只是在人类语义空间内搜索？

2. **指令与模型能力的耦合有多深？** 如果模型能力发生质的飞跃（如从 GPT-3 到 GPT-4），指令需要重新优化吗？

3. **能否用 APE 优化 APE 本身？** 用 APE 生成更好的 APE 指令，形成自举循环？

4. **多语言场景下 APE 表现如何？** 指令能否跨语言迁移？中文指令和英文指令效果有差异吗？

5. **安全性问题：如果 APE 发现了"越狱"提示怎么办？** 如何确保自动生成的指令是安全的？

6. **与 RLHF 的关系是什么？** APE 能否替代或补充 RLHF？两者优化的是同一目标吗？

7. **指令的"可解释性"重要吗？** APE 发现的指令有时很怪异（如"output strict"），人类无法理解但有效。我们应该追求可解释的指令吗？

---

**论文元信息**
- 标题：Large Language Models are Human-Level Prompt Engineers
- 作者：Yongchao Zhou, Igor Mordatch, Pieter Abbeel
- 机构：UC Berkeley, Google DeepMind
- 发表会议：ICLR 2023
- arXiv: 2211.01910
- 阅读时间：2026-03-15
- 方法论：高保真交互式精读协议
