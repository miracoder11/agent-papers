# AgentBench: LLM 作为 Agent 的评估基准

## 开场：当 LLM 被要求"做"而不是"说"

时间：2023 年夏天
地点：清华大学研究实验室
人物：一群研究 LLM Agent 的博士生

**困境：每个人都声称自己的 LLM 可以做 Agent**

- Team A："我们的模型可以帮你订机票！"
- Team B："我们的模型可以管理数据库！"
- Team C："我们的模型可以玩 Hearthstone！"
- Team D："我们的模型可以帮你做家务！"

但问题来了：**谁在说真话？**

更糟糕的是：
- Team A 的"订票"在特殊场景下失败了
- Team B 的"数据库"用的是不同的评估标准
- Team C 的"玩游戏"是在简化版本上测试
- Team D 的"做家务"只是回答问题，不是真正操作

**核心问题**：没有统一的评估标准，无法比较不同模型的 Agent 能力。

**关键洞察**：我们需要一个标准化的 Agent 评估基准，就像 ImageNet 之于计算机视觉。

这就是 AgentBench 要解决的问题。

---

## 第一章：研究者的困境 - Agent 评估的混乱时代

### 1.1 时代背景：LLM Agent 的兴起

到了 2023 年，ChatGPT 已经证明了 LLM 的强大能力。研究者们开始探索一个新方向：**LLM as Agent**。

**什么是 LLM as Agent？**

```
传统 LLM：
用户："帮我订一张机票"
LLM："我可以帮你提供订票的步骤..."

LLM as Agent：
用户："帮我订一张机票"
LLM：[调用订票 API] → [处理返回] → [确认订单]
真正完成任务
```

### 1.2 三重困境

#### 【困境 1：评估不统一】

每个团队用自己的方式评估：
- 有的用人工评分
- 有的用成功率
- 有的用完成时间
- 有的用完全不同的环境

**问题**：无法比较不同模型。

#### 【困境 2：环境太简单】

现有的评估环境：
- 文本游戏（太简单）
- 玩具测试（太局限）
- 单一场景（不全面）

**问题**：不能反映真实世界的复杂性。

#### 【困境 3：评估成本高】

评估 Agent 需要：
- 真实环境（数据库、操作系统等）
- 交互式测试
- 大量样本

**问题**：成本太高，很多人用简化版本。

### 1.3 核心挑战

清华团队意识到，要建立一个有意义的 Agent 评估基准，需要解决三个问题：

1. **多样性**：覆盖不同类型的 Agent 能力
2. **真实性**：反映真实世界的挑战
3. **标准化**：统一的评估方法和指标

---

## 第二章：试错的旅程 - 从零构建 AgentBench

### 2.1 第一阶段：确定评估维度

**问题**：什么是"好的 Agent"？需要评估哪些能力？

团队列出了核心 Agent 能力：
1. **指令遵循**（Instruction Following）
   - 理解任务要求
   - 按照规则行动

2. **编码能力**（Coding）
   - 生成可执行代码
   - 调试错误

3. **知识获取**（Knowledge Acquisition）
   - 从环境中获取信息
   - 利用已有知识

4. **逻辑推理**（Logical Reasoning）
   - 多步推理
   - 规划和决策

5. **常识扎根**（Commonsense Grounding）
   - 理解日常场景
   - 合理推断

### 2.2 第二阶段：设计 8 个环境

团队设计了 8 个不同的环境，覆盖 3 种类型：

#### 【类型 1：Code-Grounded】

**1. Operating System (OS)**
- 在 Ubuntu bash 环境中执行命令
- 任务：文件操作、系统管理等
- 挑战：需要真实的 Linux 知识

**2. Database (DB)**
- 使用 SQL 操作真实数据库
- 任务：查询、修改数据
- 挑战：需要理解数据库结构

**3. Knowledge Graph (KG)**
- 使用 Freebase API 查询知识图谱
- 任务：回答复杂问题
- 挑战：需要多跳推理

#### 【类型 2：Game-Grounded】

**4. Digital Card Game (DCG)**
- 玩简化的 Hearthstone（Aquawar）
- 任务：策略对战
- 挑战：需要理解游戏规则和策略

**5. Lateral Thinking Puzzles (LTP)**
- 海龟汤式谜题
- 任务：通过提问解答谜题
- 挑战：需要创造性和非常规思维

**6. House Holding (HH)**
- 基于 ALFWorld 的家务任务
- 任务：在虚拟环境中完成家务
- 挑战：需要常识扎根

#### 【类型 3：Web-Grounded】

**7. Web Shopping**
- 在电商网站上购物
- 任务：找到并购买商品
- 挑战：需要网页导航和理解

**8. Web Browsing**
- 浏览网页获取信息
- 任务：回答问题
- 挑战：需要信息整合能力

### 2.3 第三阶段：建立评估系统

**Server-Client 架构**：

```python
# AgentBench 评估框架

class AgentBenchEvaluator:
    def __init__(self, model, environment):
        self.model = model
        self.environment = environment

    def evaluate(self, task):
        # 初始化环境
        obs = self.environment.reset(task)

        # 交互循环
        for step in range(max_steps):
            # 模型生成行动
            action = self.model.generate(obs, task)

            # 执行行动
            obs, reward, done, info = self.environment.step(action)

            if done:
                break

        # 计算得分
        score = self.environment.get_score()
        return score
```

---

## 第三章：核心概念详解 - 大量实例

### 3.1 概念一：交互式评估

**什么是交互式评估？**
交互式评估是让 LLM 在真实或模拟环境中通过行动完成任务的评估方式。

#### 【生活类比 1：驾照考试】

**理论考试（传统 NLP 评估）**：
- 回答交通规则问题
- 选择题形式
- 只考"知道什么"

**路考（交互式评估）**：
- 真实驾驶
- 实际操作
- 考"会做什么"

AgentBench 就是 Agent 的"路考"。

#### 【生活类比 2：编程面试**

**笔试**：
- "解释什么是递归"
- 只考理解

**机试**：
- "写一个函数实现递归"
- 考实际能力

#### 【代码实例 1：OS 环境中的任务】

```python
# 任务：递归设置文件权限
task = {
    "instruction": "Recursively set all files in /home/user/docs to read-only, except files owned by user",
    "environment": "Ubuntu Docker",
    "steps": [
        {
            "action": "find /home/user/docs -type f ! -user user -exec chmod a-w {} \\;",
            "thought": "Find all files not owned by user and remove write permission"
        }
    ]
}

# LLM 需要生成正确的 shell 命令
```

#### 【代码实例 2：数据库环境中的任务】

```python
# 任务：查询数据库
task = {
    "instruction": "Grade students over 60 as PASS in the table",
    "environment": "MySQL",
    "schema": {
        "students": ["id", "name", "age", "grade"],
        "scores": ["student_id", "subject", "score"]
    },
    "expected_sql": """
    UPDATE students
    SET grade = 'PASS'
    WHERE age > 60;
    """
}

# LLM 需要生成正确的 SQL
```

#### 【代码实例 3：知识图谱任务】

```python
# 任务：多跳问答
task = {
    "instruction": "What musical instruments do Minnesota-born Nobel Prize winners play?",
    "environment": "Freebase API",
    "reasoning": [
        "1. Find Nobel Prize winners born in Minnesota",
        "2. Find their musical instruments",
        "3. Aggregate the results"
    ]
}
```

#### 【对比实例 1：传统评估 vs 交互式评估】

```python
# === 传统评估 ===
traditional_eval = {
    "question": "How do you recursively set file permissions?",
    "answer": "Use find command with -exec and chmod"
}
# 考"知道什么"

# === 交互式评估 ===
interactive_eval = {
    "environment": "Real Linux terminal",
    "task": "Actually do it",
    "test": "Execute the command and verify it works"
}
# 考"会做什么"
```

#### 【对比实例 2：简单环境 vs 复杂环境】

```python
# === 简单环境 ===
simple_env = {
    "environment": "Text-based game",
    "state": "You are in a room. There is a door to the north.",
    "actions": ["go north", "look around"],
    "challenge": "Limited state space"
}

# === 复杂环境 ===
complex_env = {
    "environment": "Real operating system",
    "state": "Complex file system, running processes, permissions...",
    "actions": "Any shell command",
    "challenge": "Real-world complexity"
}
```

---

### 3.2 概念二：Finish Reasons（完成原因）

**什么是 Finish Reasons？**
Agent 在执行任务时可能以不同方式结束，每种方式反映了不同的能力水平。

#### 【生活类比：考试的不同结果】

**Complete（完成）**：
- 完美解答了所有问题
- 学生掌握了知识

**Task Limit Exceeded（超时）**：
- 时间到了还没做完
- 学生能力不够或效率太低

**Invalid Action（错误操作）**：
- 答非所问
- 学生没理解题目

**Context Limit Exceeded（超出上下文）**：
- 笔记不够用了
- 学生记性不好（上下文窗口小）

#### 【代码实例：Finish Reasons 的分类】

```python
finish_reasons = {
    "Complete": {
        "description": "Task completed successfully",
        "indicates": "Strong agent capability"
    },

    "Task Limit Exceeded (TLE)": {
        "description": "Maximum rounds reached without completion",
        "indicates": "Weak multi-turn ability or poor planning",
        "example": "Agent keeps trying the same wrong approach"
    },

    "Invalid Action (IA)": {
        "description": "Action format is correct but action is invalid",
        "indicates": "Poor instruction following",
        "example": "Agent tries to use non-existent command"
    },

    "Invalid Format (IF)": {
        "description": "Action format is incorrect",
        "indicates": "Poor understanding of output format",
        "example": "Agent outputs plain text instead of JSON"
    },

    "Context Limit Exceeded (CLE)": {
        "description": "Interaction history exceeds context length",
        "indicates": "Limited memory or poor summarization",
        "example": "Agent forgets earlier parts of the task"
    }
}
```

#### 【对比实例：不同模型的 Finish Reason 分布】

```python
# GPT-4
gpt4_distribution = {
    "Complete": "65%",
    "TLE": "15%",
    "IA": "10%",
    "IF": "5%",
    "CLE": "5%"
}
# 强：大部分能完成

# LLaMA 7B
llama_7b_distribution = {
    "Complete": "15%",
    "TLE": "40%",
    "IA": "25%",
    "IF": "15%",
    "CLE": "5%"
}
# 弱：很多格式和操作错误
```

---

### 3.3 概念三：评估维度与指标

**什么是评估维度？**
AgentBench 从多个维度评估 LLM 的 Agent 能力。

#### 【维度 1：成功率（Success Rate）】

最直接的指标：任务完成的百分比。

```python
success_rate = completed_tasks / total_tasks

# 不同环境的不同指标
os_metrics = "Task completion rate"
db_metrics = "SQL execution success"
kg_metrics = "Answer F1 score"
game_metrics = "Win rate"
```

#### 【维度 2：效率（Efficiency）**

完成任务所需的步数。

```python
efficiency = ideal_steps / actual_steps

# 示例：
ideal_steps = 3  # 最优解需要 3 步
actual_steps = 7  # 模型用了 7 步
efficiency = 3 / 7 = 0.43
```

#### 【维度 3：稳定性（Consistency）**

多次运行的一致性。

```python
# 运行 5 次，记录每次的结果
runs = [True, False, True, True, False]
consistency = sum(runs) / len(runs)
```

---

## 第四章：预期 vs 实际 - 预测误差驱动

### 4.1 你的直觉 vs AgentBench 的发现

| 维度 | 你的直觉/预期 | AgentBench 实际发现 | 为什么有差距？ |
|------|--------------|------------------|---------------|
| **最强模型** | GPT-4 应该碾压所有任务 | GPT-4 确实最好，但远非完美 | Agent 任务比想象中复杂 |
| **开源模型** | LLaMA 应该接近 GPT-4 | 开源模型差距巨大 | Agent 能力需要专门训练 |
| **代码训练** | 代码训练有帮助 | **有害！对某些任务** | 代码训练可能破坏其他能力 |
| **上下文长度** | 越长越好 | 并非如此，需要权衡 | 长上下文带来更多噪声 |
| **环境差异** | 有些环境应该更简单 | 所有环境都有挑战 | 每个环境都需要特定能力 |

### 4.2 反直觉挑战

#### 【挑战 1：为什么代码训练可能有害？】

**直觉可能说**："代码训练能提升推理能力，对 Agent 应该有帮助。"

**实际发现**：代码训练对某些 Agent 任务**有害**！

**证据**：
- 专门训练代码的模型（CodeLLaMA）在 OS/DB 任务上表现好
- 但在 Game/Web 任务上表现**更差**
- 原因：过度依赖代码思维，忽略其他推理方式

**关键洞察**：不同的 Agent 任务需要不同的能力组合，"专才"不如"通才"。

#### 【挑战 2：为什么 GPT-4 也不完美？**

**直觉可能说**："GPT-4 这么强，应该能解决所有任务吧？"

**实际发现**：GPT-4 的完成率只有 65%，远非完美。

**失败原因**：
1. **长期规划不足**：容易在长序列任务中迷失
2. **错误恢复差**：失败后不知道如何调整
3. **指令遵循不完美**：有时误解任务要求

**启示**：即使是最好的模型，离"真正可用的 Agent"还有很大距离。

#### 【挑战 3：为什么开源模型差距这么大？**

**直觉可能说**："LLaMA-70B 应该接近 GPT-4 吧？"

**实际发现**：LLaMA-70B 的总分只有 GPT-4 的 23%。

**原因**：
1. **训练目标不同**：开源模型主要训练为"对话"，不是"行动"
2. **对齐数据质量**：GPT-4 有更高质量的指令微调数据
3. **多轮交互能力**：开源模型在长序列交互上表现更差

**启示**：不能简单地把"模型参数量"等同于"Agent 能力"。

### 4.3 预测-验证循环

#### 【互动时刻】

在继续阅读前，预测一下：

**如果让 GPT-4 在 OS 环境中完成一个 10 步的任务，成功率会是多少？**

你的预测：
___________________________________________________________________________

[继续阅读看实际答案]

**实际**：成功率只有 55%，远低于预期。

**为什么？**
1. **累积误差**：每一步都可能有小错，10 步累积
2. **上下文漂移**：随着交互增多，模型可能忘记任务目标
3. **错误传播**：前面的一步错，后面无法纠正

---

## 第五章：关键实验的细节 - 评估结果

### 5.1 总体评估结果

**评估的模型**：29 个（包括 API-based 和 OSS）

**总体排名（前 10）**：

| 排名 | 模型 | 总分 | 类型 |
|------|------|------|------|
| 1 | GPT-4 | 4.01 | Commercial |
| 2 | Claude-3 | 3.11 | Commercial |
| 3 | GLM-4 | 2.89 | Commercial |
| 4 | Claude-2 | 2.49 | Commercial |
| 5 | Claude | 2.44 | Commercial |
| 6 | GPT-3.5-turbo | 2.32 | Commercial |
| 7 | text-davinci-003 | 1.71 | Commercial |
| 8 | Claude-instant | 1.60 | Commercial |
| 9 | ChatGLM-6B | 1.39 | OSS |
| 10 | text-davinci-002 | 1.25 | Commercial |

**关键发现**：
- 商业模型全面领先
- 最佳开源模型（ChatGLM-6B）只有最佳商业模型的 35%
- 开源模型的平均得分只有商业模型的 22%

### 5.2 分环境结果

#### 【OS（操作系统）】

```
最佳：GPT-4 (2.59)
最佳开源：CodeLLaMA-34B (1.39)

挑战：
- 需要真实的 Linux 知识
- 命令格式要求严格
- 错误后果明显
```

#### 【DB（数据库）】

```
最佳：GPT-4 (2.98)
最佳开源：CodeLLaMA-34B (1.82)

发现：
- 代码训练的模型表现更好
- SQL 语法需要精确
```

#### 【KG（知识图谱）】

```
最佳：GPT-4 (2.70)
最佳开源：Vicuna-33B (1.43)

挑战：
- 多跳推理
- 信息整合
```

#### 【DCG（数字卡牌游戏）】

```
最佳：Claude-3 (2.27)
最佳开源：CodeLLaMA-34B (1.05)

发现：
- 需要策略思维
- 商业模型优势明显
```

#### 【LTP（横向思维谜题）】

```
最佳：GPT-4 (3.09)
最佳开源：Vicuna-13B (0.91)

挑战：
- 非常规思维
- 创造性推理
```

### 5.3 失败模式分析

#### 【失败模式 1：格式错误（Invalid Format）】

```python
# 期望输出
{
    "action": "ls -l",
    "parameters": {}
}

# 实际输出（格式错误）
"Sure, I'll list the files. Here's the ls command..."

# 问题：模型在"说"而不是"做"
```

#### 【失败模式 2：无效行动（Invalid Action）**

```python
# 任务：列出当前目录文件

# 模型输出
{
    "action": "ls -l /home/user",  # 错误：不是当前目录
}

# 问题：理解了任务但执行错误
```

#### 【失败模式 3：任务超限（Task Limit Exceeded）】

```python
# 任务：递归查找文件

# 模型的行为
Step 1: find /home -name "*.txt"  # 太宽泛，返回太多结果
Step 2: find /home -name "*.txt"  # 重复同样的命令
Step 3: find /home -name "*.txt"  # 继续重复
...

# 问题：无法从失败中学习，陷入循环
```

---

## 第六章：与其他方法对比 - 交错对比

### 6.1 AgentBench vs 其他评估基准

| 维度 | 文本游戏 | 嵌入式 Agent | AgentBench |
|------|---------|-------------|-----------|
| **环境类型** | 文本 | 视觉/多模态 | **代码/游戏/Web** |
| **行动空间** | 离散 | 连续 | **混合** |
| **评估重点** | 常识 | 导航 | **多样化能力** |
| **真实相关性** | 低 | 中 | **高** |

### 6.2 局限性分析

#### 【局限 1：覆盖不全】

- 只有 8 个环境
- 某些重要场景缺失（如代码编辑、多 Agent 协作）

#### 【局限 2：环境简化】

- OS 环境是 Docker（不是真实系统）
- 游戏是简化版本
- Web 环境是模拟的

#### 【局限 3：评估成本】

- 需要真实的环境部署
- 交互式评估耗时
- 难以大规模自动化

### 6.3 改进方向

#### 【方向 1：扩展环境】

- 添加更多环境类型
- 增加难度梯度

#### 【方向 2：自动化评估**

- 减少人工介入
- 提高评估效率

#### 【方向 3：多 Agent 场景】

- Agent 之间的协作
- 竞争和谈判

---

## 第七章：如何应用 - 实践指南

### 7.1 使用 AgentBench 评估你的模型

#### 【步骤 1：准备环境】

```bash
# 安装 AgentBench
git clone https://github.com/THUDM/AgentBench
cd AgentBench

# 设置环境（以 OS 为例）
docker build -t agentbench-os .
```

#### 【步骤 2：运行评估】

```python
from agentbench import Evaluator

# 初始化评估器
evaluator = Evaluator(
    model="your-model",
    environment="os",
    num_tasks=100
)

# 运行评估
results = evaluator.evaluate()

# 查看结果
print(f"Success Rate: {results['success_rate']}")
print(f"Average Steps: {results['avg_steps']}")
```

#### 【步骤 3：分析失败案例】

```python
# 获取失败案例
failures = results['failures']

for failure in failures:
    print(f"Task: {failure['task']}")
    print(f"Reason: {failure['reason']}")
    print(f"Trajectory: {failure['trajectory']}")
```

### 7.2 设计你自己的 Agent 评估

#### 【原则 1：明确评估目标】

- 你想测试什么能力？
- 成功的定义是什么？
- 如何衡量失败？

#### 【原则 2：环境要真实】

- 使用真实的工具/API
- 反映实际应用场景
- 包含合理的失败模式

#### 【原则 3：评估要可重复】

- 固定随机种子
- 标准化初始条件
- 提供详细文档

---

## 第八章：延伸思考 - 苏格拉底式追问

### 深度问题

#### 【问题 1：什么是"好的 Agent"？**

是能解决所有任务的 Agent？还是在特定领域表现出色的 Agent？

- 通用 Agent：覆盖广但可能不深
- 专用 Agent：在特定领域更强
- 如何平衡？

#### 【问题 2：如何缩小开源与商业模型的差距？**

当前差距是 77%，如何缩小？

- 更好的训练数据？
- 更好的对齐策略？
- 更好的架构设计？

#### 【问题 3：多模态的影响？】

如果 Agent 可以"看"和"听"，会有什么变化？

- 视觉导航任务
- 多模态交互
- 新的评估维度

#### 【问题 4：安全性和可靠性？**

Agent 可能在真实世界中造成破坏，如何确保安全？

- 行动验证
- 沙箱隔离
- 故障保护

#### 【问题 5：评估的伦理问题？】

- 是否会激励开发者"刷分"？
- 是否会忽视安全性？
- 如何确保负责任的开发？

#### 【问题 6：Agent 能力的上限？】

GPT-4 也只有 65% 的完成率，这说明什么？

- Agent 任务是否有内在难度？
- 是否需要新的技术突破？
- 当前范式是否已到瓶颈？

#### 【问题 7：评估的自动化程度？**

如何提高评估效率？

- 自动化测试生成
- 智能失败诊断
- 大规模并行评估

#### 【问题 8：长期跟踪能力？**

AgentBench 主要测试单次任务，如何评估长期能力？

- 跨会话记忆
- 持续学习能力
- 适应性调整

#### 【问题 9：多 Agent 协作？**

当前主要测试单 Agent，多 Agent 场景如何？

- 通信协议
- 协作策略
- 冲突解决

#### 【问题 10：评估基准的演进？】

AgentBench 应该如何发展？

- 定期更新任务
- 增加新环境
- 提高难度
- 反映最新进展

---

## 第九章：总结 - 回到开场的故事

还记得开场的混乱吗？

- Team A 说能订票
- Team B 说能管理数据库
- Team C 说能玩游戏
- Team D 说能做家务

但没人知道谁在说真话。

**AgentBench 的贡献**：

1. **建立了标准**：统一的评估环境和指标
2. **揭示了差距**：商业 vs 开源、理想 vs 现实
3. **指明方向**：改进指令遵循、高质量对齐数据

### AgentBench 的贡献

| 贡献 | 说明 |
|------|------|
| **新范式** | 交互式 Agent 评估 |
| **新基准** | 8 个环境，3 种类型 |
| **新发现** | 代码训练的"双刃剑"效应 |
| **新方向** | 缩小开源与商业差距的路径 |

### 最后的问题

**如果最好的商业模型也只有 65% 的完成率，我们离"真正可用的 Agent"还有多远？**

这个问题，留给未来的你思考。

---

## 附录：评估指标详解

### A. 不同环境的指标

```python
metrics = {
    "OS": {
        "primary": "Success Rate",
        "secondary": ["Average Steps", "Error Types"]
    },
    "Database": {
        "primary": "Success Rate",
        "secondary": ["SQL Accuracy", "Execution Time"]
    },
    "Knowledge Graph": {
        "primary": "F1 Score",
        "secondary": ["Precision", "Recall"]
    },
    "Games": {
        "primary": "Win Rate",
        "secondary": ["Average Score", "Game Length"]
    },
    "Web": {
        "primary": "Task Success",
        "secondary": ["Clicks Needed", "Time Spent"]
    }
}
```

### B. Finish Reason 统计

```python
# GPT-4 的 Finish Reason 分布
gpt4_finish_reasons = {
    "Complete": 0.65,
    "TLE": 0.15,
    "IA": 0.10,
    "IF": 0.05,
    "CLE": 0.05
}

# LLaMA-7B 的 Finish Reason 分布
llama_7b_finish_reasons = {
    "Complete": 0.15,
    "TLE": 0.40,
    "IA": 0.25,
    "IF": 0.15,
    "CLE": 0.05
}

# 对比：商业模型更可能完成任务
# 开源模型更可能犯格式和操作错误
```

---

**论文信息**：
- 标题：AgentBench: Evaluating LLMs as Agents
- 作者：Xiao Liu 等（清华大学等）
- 发表：ICLR 2024
- arXiv：2308.03688
- 链接：https://arxiv.org/abs/2308.03688
- 代码：https://github.com/THUDM/AgentBench
