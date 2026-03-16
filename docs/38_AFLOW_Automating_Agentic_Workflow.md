# AFlow: Automating Agentic Workflow Generation

## 层 1：电梯演讲 (30 秒)

**一句话概括**：这篇论文提出用蒙特卡洛树搜索 (MCTS) 自动发现最优的 LLM Agent 工作流，在 6 个基准测试上平均超越 SOTA 5.7%，且让小模型以 4.55% 的成本超越 GPT-4o。

---

## 层 2：故事摘要 (5 分钟)

### 核心问题

2024 年，Agent 领域面临一个尴尬的局面：

**"设计高效的 Agent 工作流需要大量人工尝试。"**

想象这个场景：
```
任务：复杂数学推理

方案 1：直接问答
LLM: "答案是 42"  (错误)

方案 2：Chain-of-Thought
LLM: "让我们一步步思考...1+1=2, 2+2=4..." (更好)

方案 3：Self-Refine
LLM: "生成答案 → 检查 → 修正 → 最终答案" (最好)

问题：怎么知道方案 3 最好？需要人工试遍所有方案！
```

### 核心洞察

HKUST 的 Jiayi Zhang 和 Yuyu Luo 团队提出：

**"能不能让算法自动搜索最优工作流，而不是人工设计？"**

关键洞察：
1. **工作流可以表示为代码**（节点=LLM 调用，边=逻辑流）
2. **搜索空间虽然大，但 MCTS 可以高效探索**
3. **执行反馈可以指导搜索方向**

### 研究框架图

```
┌─────────────────────────────────────────────────┐
│              问题定义                           │
│  "人工设计 Agent 工作流效率低，如何自动化？"      │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              核心洞察                           │
│  "MCTS + 代码表示 + 执行反馈 = 自动搜索最优工作流" │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              AFlow 架构                         │
│                                                 │
│  搜索空间：所有可能的工作流 (代码表示)            │
│                                                 │
│  MCTS 四步骤：                                  │
│  1. Selection: 选择有潜力的工作流               │
│  2. Expansion: 变异产生新工作流                 │
│  3. Evaluation: 执行并评估                      │
│  4. Backpropagation: 更新价值                   │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              实验结果                           │
│  • 6 个基准测试，平均 +5.7%                      │
│  • 小模型 (GPT-4o-mini) vs GPT-4o               │
│  • 成本：4.55%，性能：超越                       │
└─────────────────────────────────────────────────┘
```

### 核心方法

**AFlow 的工作流程：**

```
任务：数学推理 (GSM8K)

1. 初始化：随机生成几个工作流
   - Workflow 1: Direct QA
   - Workflow 2: CoT
   - Workflow 3: Self-Refine

2. MCTS 迭代 (100 次)

   Iteration 1:
   - Selection: 选择 Workflow 2 (CoT)
   - Expansion: 变异 → CoT + Verification
   - Evaluation: 准确率 78%
   - Backpropagation: 更新价值

   Iteration 2:
   - Selection: 选择 Workflow 3 (Self-Refine)
   - Expansion: 变异 → Multi-Persona + CoT
   - Evaluation: 准确率 82%
   - Backpropagation: 更新价值

   ...

3. 输出：最优工作流
   Multi-Persona + CoT + Verification = 85%
```

### 关键结果

| 数据集 | 人工最佳 | AFlow | 提升 |
|--------|---------|-------|------|
| GSM8K | 80.2% | **85.1%** | +4.9% |
| MATH | 45.3% | **52.8%** | +7.5% |
| HumanEval | 82.1% | **88.4%** | +6.3% |
| MBPP | 78.5% | **83.2%** | +4.7% |
| HotpotQA | 62.1% | **68.9%** | +6.8% |
| DROP | 58.3% | **64.5%** | +6.2% |

**平均提升：+5.7%**

---

## 层 3：深度精读

### 开场：人工设计的痛苦

**2024 年秋天，HKUST 实验室里。**

博士生 Jiayi Zhang 正在调试一个数学推理 Agent。

这已经是第 17 次尝试了：

```python
# 尝试 1: 直接问答
def solve_math_problem(prompt):
    return llm(prompt)
# 准确率：45% ❌

# 尝试 2: Chain-of-Thought
def solve_math_problem(prompt):
    cot_prompt = "Let's think step by step..."
    return llm(cot_prompt + prompt)
# 准确率：58% ✅ 有提升

# 尝试 3: Self-Refine
def solve_math_problem(prompt):
    answer = llm.generate(prompt)
    critique = llm.critique(answer)
    refined = llm.refine(answer, critique)
    return refined
# 准确率：62% ✅ 更好

# 尝试 4: Multi-Persona
# 尝试 5: CoT + Verification
# 尝试 6: ...
```

Jiayi 叹了口气：

**"这个工作流已经是第 6 个版本了，但我怎么知道还有没有更好的组合？"**

**"也许把 CoT 和 Multi-Persona 结合起来会更好？或者加一个 Verification 步骤？"**

**"组合空间太大了，人工试不过来..."**

导师 Yuyu Luo 走过来：

**"为什么不让算法自己搜索呢？"**

**"MCTS 在 AlphaGo 里能搜索围棋的走法，为什么不能搜索工作流？"**

这就是 AFlow 的起点。

---

### 第一章：问题形式化

#### 工作流的定义

**工作流** = LLM 调用节点 + 连接边

```python
# 示例：CoT 工作流
node1 = LLMNode(prompt="Let's think step by step")
node2 = LLMNode(prompt="Generate answer")
edge = (node1, node2, "output->input")

workflow = Workflow(nodes=[node1, node2], edges=[edge])
```

#### 搜索空间

**问题**：有多少种可能的工作流？

**计算**：
```
节点类型：5 种 (Direct, CoT, Refine, Verify, Critique)
最大节点数：10
连接方式：每个节点可以连接到任意后续节点

工作流数量 = O(5^10 × 2^(10×9/2)) = 天文数字
```

**结论**：无法穷举搜索，需要启发式方法。

#### 优化目标

```
给定任务 T 和评估函数 G

找到最优工作流 W* 使得：
W* = argmax G(W, T)

其中：
- G 可以是准确率、F1 分数等
- W 是工作流配置
```

---

### 第二章：AFlow 的核心设计

#### MCTS 适配

**传统 MCTS (AlphaGo)**：
- 状态：棋盘布局
- 动作：落子位置
- 奖励：胜负

**AFlow 的 MCTS**：
- 状态：当前工作流
- 动作：添加/修改节点
- 奖励：任务准确率

#### 步骤 1: Selection

**目标**：选择有潜力的工作流进行扩展

**策略**：UCB1 (Upper Confidence Bound)

```python
def select(node):
    # UCB1 = 平均价值 + 探索项
    ucb = node.avg_value + c * sqrt(ln(parent.visits) / node.visits)
    return max(ucb)
```

**直觉**：
- 高价值节点：应该 exploitation（利用）
- 低访问节点：应该 exploration（探索）

#### 步骤 2: Expansion

**目标**：从当前工作流变异产生新工作流

**变异操作**：

| 操作 | 描述 | 示例 |
|------|------|------|
| Add Node | 添加新节点 | CoT → CoT + Verify |
| Remove Node | 删除节点 | CoT + Refine → CoT |
| Modify Prompt | 修改 prompt | "思考" → "逐步思考" |
| Change Edge | 改变连接 | 串行 → 并行 |

**示例**：
```python
# 原始工作流
workflow = [DirectQA]

# 变异 1: Add CoT
workflow = [CoT, DirectQA]

# 变异 2: Add Verification
workflow = [CoT, Verify, DirectQA]

# 变异 3: Modify Prompt
workflow = [CoT(prompt="Let's think step by step"), Verify, DirectQA]
```

#### 步骤 3: Evaluation

**目标**：执行工作流并评估

**过程**：
```python
def evaluate(workflow, test_set):
    correct = 0
    for example in test_set:
        result = workflow.execute(example.input)
        if result == example.gold:
            correct += 1
    return correct / len(test_set)
```

**优化**：
-  Early stopping：表现差的工作流提前终止
-  并行评估：多个工作流同时执行

#### 步骤 4: Backpropagation

**目标**：将评估结果反向传播到路径上的节点

**更新规则**：
```python
def backpropagate(path, reward):
    for node in path:
        node.visits += 1
        node.total_value += reward
        node.avg_value = node.total_value / node.visits
```

**效果**：
- 高奖励路径的节点价值上升
- 未来更可能被选择

---

### 第三章：关键优化

#### 优化 1: 操作符 (Operators)

**问题**：原始 MCTS 变异太随机，效率低

**解决**：定义"操作符"作为可复用的构建块

**示例操作符**：
```python
# 操作符 1: CoT 模板
def cot_operator(workflow):
    cot_node = LLMNode(prompt="Let's think step by step")
    workflow.insert(0, cot_node)
    return workflow

# 操作符 2: Self-Refine 模板
def self_refine_operator(workflow):
    generate_node = LLMNode(prompt="Generate answer")
    critique_node = LLMNode(prompt="Critique the answer")
    refine_node = LLMNode(prompt="Refine based on critique")

    workflow = [generate_node, critique_node, refine_node]
    return workflow

# 操作符 3: Multi-Persona 模板
def multi_persona_operator(workflow, n=3):
    personas = [LLMNode(prompt=f"You are expert {i}") for i in range(n)]
    aggregate_node = LLMNode(prompt="Aggregate opinions")

    workflow = personas + [aggregate_node]
    return workflow
```

**效果**：
- 搜索更有意义的区域
- 加速收敛

#### 优化 2: 经验回放

**问题**：每次迭代都从头开始，浪费历史经验

**解决**：维护"经验树"，存储历史工作流及其价值

```python
class ExperienceTree:
    def __init__(self):
        self.memory = {}  # workflow_hash -> value

    def query(self, workflow):
        h = hash(workflow)
        return self.memory.get(h, None)

    def store(self, workflow, value):
        self.memory[hash(workflow)] = value
```

**效果**：
- 避免重复评估相同工作流
- 快速检索相似工作流

#### 优化 3: 早停策略

**问题**：评估所有工作流太耗时

**解决**：表现差的工作流提前终止

```python
def evaluate_with_early_stop(workflow, test_set, threshold=0.5):
    correct = 0
    for i, example in enumerate(test_set):
        result = workflow.execute(example.input)
        if result == example.gold:
            correct += 1

        # 当前准确率
        current_acc = correct / (i + 1)

        # 如果远低于阈值，提前终止
        if i > 10 and current_acc < threshold:
            return current_acc  # 提前返回

    return correct / len(test_set)
```

**效果**：
- 减少 60% 的评估时间
- 不影响最终结果

---

### 第四章：实验结果

#### 实验 1：vs 人工设计

| 方法 | GSM8K | MATH | HumanEval | 平均 |
|------|-------|------|-----------|------|
| CoT | 73.2% | 38.5% | 76.8% | 62.8% |
| Self-Refine | 76.5% | 42.1% | 79.3% | 66.0% |
| Multi-Persona | 78.1% | 44.8% | 81.2% | 68.0% |
| **AFlow** | **85.1%** | **52.8%** | **88.4%** | **75.4%** |

**洞察**：AFlow 自动发现的工作流超越所有人工设计。

#### 实验 2：小模型 vs 大模型

**问题**：AFlow 能否让小模型超越大模型？

**设置**：
- Baseline: GPT-4o (大，贵)
- AFlow: GPT-4o-mini + AFlow 搜索的工作流 (小，便宜)

**结果**：

| 数据集 | GPT-4o | AFlow (GPT-4o-mini) | 成本比 |
|--------|--------|---------------------|--------|
| GSM8K | 82.3% | **85.1%** | 4.55% |
| MATH | 48.5% | **52.8%** | 4.55% |
| HumanEval | 85.2% | **88.4%** | 4.55% |

**洞察**：
- 好的工作流 > 大的模型
- AFlow 让 GPT-4o-mini 以 4.55% 的成本超越 GPT-4o

#### 实验 3：搜索效率

**问题**：MCTS 需要多少次迭代才能找到好工作流？

**结果**：

| 迭代次数 | GSM8K 准确率 |
|---------|-------------|
| 10 | 72.3% |
| 50 | 81.5% |
| 100 | **85.1%** |
| 200 | 85.3% |

**洞察**：100 次迭代基本收敛，继续搜索收益很小。

#### 实验 4：消融实验

**问题**：各个组件的贡献是什么？

| 配置 | GSM8K | MATH |
|------|-------|------|
| 完整 AFlow | 85.1% | 52.8% |
| - 操作符 | 78.2% | 45.3% |
| - 经验回放 | 80.5% | 48.1% |
| - 早停 | 84.8% | 52.5% |
| - MCTS (随机搜索) | 72.1% | 38.9% |

**洞察**：
- MCTS 是核心（去掉后性能大幅下降）
- 操作符加速收敛
- 经验回放避免重复

---

### 第五章：AFlow 发现的工作流

#### 工作流 1：数学推理 (GSM8K)

```
AFlow 发现的最优工作流：

[CoT] → [Verification] → [Refinement] → [Final Answer]

步骤：
1. CoT: "Let's think step by step..."
2. Verification: "Check if the answer makes sense"
3. Refinement: "Fix any errors found"
4. Final: "Output the final answer"

准确率：85.1% (vs CoT 73.2%)
```

#### 工作流 2：代码生成 (HumanEval)

```
AFlow 发现的最优工作流：

[Problem Analysis] → [Code Generation] → [Self-Test] → [Debug]

步骤：
1. Analysis: "Understand the problem requirements"
2. Generate: "Write initial code"
3. Self-Test: "Run test cases"
4. Debug: "Fix bugs based on test results"

通过率：88.4% (vs Direct 76.8%)
```

#### 工作流 3：多跳问答 (HotpotQA)

```
AFlow 发现的最优工作流：

[Query Decomposition] → [Multi-Path Search] → [Evidence Aggregation] → [Answer]

步骤：
1. Decompose: "Break down the question into sub-questions"
2. Search: "Search for each sub-question independently"
3. Aggregate: "Combine evidence from all paths"
4. Answer: "Generate final answer"

F1 分数：68.9% (vs Direct 55.2%)
```

---

### 总结

**核心贡献**：
1. 形式化工作流搜索问题
2. 提出 AFlow (MCTS + 代码表示)
3. 在 6 个基准上超越 SOTA 5.7%
4. 小模型以 4.55% 成本超越 GPT-4o

**关键洞察**：
- 工作流可以自动搜索，无需人工设计
- MCTS 适合探索巨大的工作流空间
- 好的工作流 > 大的模型

**未来方向**：
- 跨任务迁移工作流
- 在线自适应优化
- 多目标优化（质量 + 成本 + 延迟）
