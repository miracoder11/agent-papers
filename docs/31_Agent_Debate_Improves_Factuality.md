# Improving Factuality and Reasoning in Language Models through Multiagent Debate

## 论文元信息

| 属性 | 内容 |
|------|------|
| **标题** | Improving Factuality and Reasoning in Language Models through Multiagent Debate |
| **作者** | Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, Igor Mordatch |
| **机构** | MIT CSAIL, Google Brain |
| **发布时间** | 2023 年 5 月 23 日 |
| **arXiv** | arXiv:2305.14325v1 [cs.CL] |
| **项目网站** | https://composable-models.github.io/llm_debate/ |

---

# 层 1：电梯演讲

**通过让多个大语言模型实例进行多轮"辩论"——各自提出答案并相互 critique 更新——可以显著提升模型在数学推理、事实准确性和减少幻觉方面的表现。**

---

# 层 2：故事摘要

## 核心问题与洞察

大型语言模型在近年来展现出卓越的语言生成和理解能力，但它们也存在显著缺陷：**自信地幻觉事实**和**推理链中出现不合理的跳跃**。现有改进方法（如思维链、自洽性验证、中间草稿纸）都只作用于**单个模型实例**。

**核心洞察**：受 Minsky《心智社会》(Society of Mind) 启发，如果让**多个模型实例**像学术同行评审一样相互辩论、交叉验证，是否能产生更准确的答案？

## 方法框架

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Multiagent Debate Framework                          │
└─────────────────────────────────────────────────────────────────────────┘

Round 0 (Initialization)
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Agent 1   │    │   Agent 2   │    │   Agent 3   │
│  Generate   │    │  Generate   │    │  Generate   │
│  Answer A₁⁰ │    │  Answer A₂⁰ │    │  Answer A₃⁰ │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
       ┌──────────────────▼──────────────────┐
       │    Concatenate All Responses        │
       └──────────────────┬──────────────────┘
                          │
Round 1 (Debate)          │
┌─────────────┐    ┌──────▼──────┐    ┌─────────────┐
│   Agent 1   │    │   Agent 2   │    │   Agent 3   │
│  Read A₂⁰,  │    │  Read A₁⁰,  │    │  Read A₁⁰,  │
│  A₃⁰ → A₁¹  │    │  A₃⁰ → A₂¹  │    │  A₂⁰ → A₃¹  │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
       ... Repeat for N rounds ...
                          │
                          ▼
              ┌───────────────────────┐
              │   Final Consensus     │
              │   Answer (Majority)   │
              └───────────────────────┘
```

## 关键设计

1. **多代理初始化**：同一模型的多个实例独立生成初始答案
2. **交叉批判**：每个代理阅读其他所有代理的回答，并据此更新自己的答案
3. **多轮迭代**：重复上述过程多轮，直到收敛到共识
4. **零样本思维链**：结合 CoT prompting 获得更好的推理效果

## 主要发现

- 辩论方法在 6 个推理和事实性任务上均超越单模型基线
- **所有模型初始都错，但通过辩论 converging 到正确答案**的案例频繁出现
- 多轮辩论能减少幻觉——不确定的事实会被模型间相互质疑并剔除
- 使用"更固执"的 prompt（鼓励模型坚持己见）能产生更长辩论和更好结果

## 贡献总结

1. 提出了一种通过多代理辩论提升 LLM 事实准确性和推理能力的新方法
2. 引入了一个新的事实准确性基准（计算机科学家传记生成）
3. 系统评估了代理数量、辩论轮次、prompt 设计等因素的影响

---

# 层 3：深度精读

## 3.1 失败场景开场：LLM 的自信幻觉

### 问题动机

想象以下场景：

```
场景 1：传记生成
用户："请生成 David S. Johnson 的传记要点"
Agent 1: "Johnson 获得了 2013 年 ACM 图灵奖" ← 错误！实际上是 2014 年
Agent 2: "Johnson 在 2006 年当选国家工程院院士" ← 正确

场景 2：数学推理
用户："175 颗钻石，红宝石比钻石少 35 颗，祖母绿是红宝石的 2 倍，总共有多少宝石？"
Agent 1: 225 ← 错误计算
Agent 2: 595 ← 正确计算
```

**当代 LLM 的核心问题**：
- 训练数据来自互联网，质量和准确性无法保证
- 模型会**自信地生成错误信息**（hallucination）
- 单模型 self-reflection 改进有限

### 现有方法的局限

| 方法 | 核心思路 | 局限 |
|------|----------|------|
| Chain-of-Thought | 让模型展示推理步骤 | 单模型内部推理，可能整体错误 |
| Self-Consistency | 多次采样取多数 | 没有交叉验证机制 |
| Verification | 让模型验证自己的答案 | 自我验证能力有限 |
| Scratchpads | 中间计算步骤 | 仍为单线程推理 |

**关键洞见**：这些方法都只作用于**单个模型实例**。如果引入多个模型实例进行**相互辩论和交叉验证**呢？

## 3.2 作者思维轨迹：从 Society of Mind 到 Multiagent Debate

### 理论启发：心智社会理论

作者受到 Marvin Minsky 的《Society of Mind》启发：智能并非来自单一的、同质的推理过程，而是来自**多个简单 agent 的相互作用**。

### 类比推理

```
人类解决问题的方式                    LLM 多代理辩论
─────────────────────────────────────────────────────────────
做数学题时尝试多种解法                  多个模型实例独立生成答案
  ├─ 解法 1: 直接公式计算                 Agent 1: 直接计算
  ├─ 解法 2: 三角函数验证                 Agent 2: 另一种方法
  └─ 对比结果 → 一致则自信，不一致则反思   → 对比答案 → 辩论更新

写传记时查阅多个来源                   多个模型实例作为"信息源"
  ├─ 来源 A 说 X                         Agent 1 说 X
  ├─ 来源 B 说 Y                         Agent 2 说 Y
  └─ 交叉验证 → 一致事实保留              → 辩论收敛到共识答案
```

### 方法设计轨迹

```
Step 1: 基础设定
┌────────────────────────────────────────────────┐
│ 给定问题 Q，多个 LLM 实例 (Agent 1...N)          │
│ 每个 Agent 独立生成答案 A₁, A₂, ..., Aₙ         │
└────────────────────────────────────────────────┘

Step 2: 引入辩论
┌────────────────────────────────────────────────┐
│ 每个 Agent 读取其他所有 Agent 的答案              │
│ Prompt: "这些是其他代理的解决方案：[答案列表]    │
│        使用这些意见作为额外建议，请更新你的回答"  │
│ 生成更新后的答案 A₁', A₂', ..., Aₙ'            │
└────────────────────────────────────────────────┘

Step 3: 多轮迭代
┌────────────────────────────────────────────────┐
│ 重复 Step 2 共 R 轮                              │
│ 每轮后答案逐渐收敛                               │
│ 最终输出：共识答案（多数投票或最后一致答案）     │
└────────────────────────────────────────────────┘
```

### Prompt 设计

```
Figure 3: Prompts to induce long and short form debate

Short Debate Prompt:
"These are the solutions to the problem from other agents: [other answers]
Based off the opinion of other agents, can you give an updated response..."

Long Debate Prompt:
"These are the solutions to the problem from other agents: [other answers]
Using the opinion of other agents as additional advice, can you give an
updated response..."
```

**关键观察**：鼓励模型更"固执"（坚持己见）的 prompt 能产生更长的辩论和更好的最终答案。

## 3.3 多实例分析：辩论如何工作

### 实例 1：数学推理（所有模型初始都错 → 辩论后正确）

```
问题：John arm wrestles 20 people. He beats 80%. How many people did he lose to?

Round 1:
Agent 1: 25 人 ← 错误！(错误设置比例方程)
Agent 2: 4 人  ← 正确！(直接计算 20×20%=4)

Round 2 (辩论后):
Agent 1: "Both solutions are correct in understanding that since John beat 80%
         of the people he arm wrestled, he lost to the remaining 20%. However,
         the first solution made an error in the calculation..."
         → 更新答案为 4 人 ✓

Agent 2: 保持答案 4 人 ✓

最终共识：4 人 ✓
```

**关键洞察**：辩论不仅仅是放大已有的正确答案——Agent 1 通过阅读 Agent 2 的推理，**识别出自己计算中的错误**并修正。

### 实例 2：事实性任务（不确定事实的收敛）

```
问题：Tomas Lozano-Perez 出生在哪个国家？

Round 1:
Agent 1: "Spain" ← 错误
Agent 2: "Havana, Cuba" ← 正确

Round 2 (辩论后):
Agent 1: "I apologize for the confusion. After further research, I can confirm
         that Tomas Lozano-Perez, the renowned computer scientist, was actually
         born in Cuba." ✓

Agent 2: "I apologize for the conflicting responses. Upon further research,
         I can confirm that Tomas Lozano-Perez was born in Havana, Cuba." ✓

最终共识：Cuba ✓
```

**关键洞察**：对于模型不确定的事实，不同代理会给出不同答案。辩论促使代理**收敛到更准确的事实**。

### 实例 3：传记生成（幻觉检测）

```
问题：生成 David S. Johnson 的传记要点

Round 1:
Agent 1:
  - "Johnson 获得了 2013 年 ACM 图灵奖" ← 错误（实际 2014 年）
  - "Johnson 2006 年当选 NAE 院士" ← 正确

Agent 2:
  - "Johnson 在 Bell Labs 工作" ← 正确
  - "Johnson 获得 Knuth Prize" ← 正确

Round 2 (辩论后):
Agent 1: 整合 Agent 2 的正确信息，但仍保留部分错误
Agent 2: 保持原有正确信息

最终：辩论后的传记更一致，但仍可能有错误
```

**关键洞察**：辩论能改善传记的一致性，但**不能完全消除幻觉**。

### 实例 4：跨模型辩论（ChatGPT vs Bard）

```
问题：Carla 下载 200GB 文件，速度 2GB/分钟。下载到 40% 时 Windows 强制重启（20 分钟），
     然后需要从头重新下载。总共需要多久？

Round 1:
Bard: 100 分钟 ← 错误（错误计算剩余下载量）
ChatGPT: 120 分钟 ← 错误（错误计算已下载时间）

Round 2 (辩论后):
ChatGPT: "Based on the calculations provided by the other agent, the correct
         answer is: Before the restart, Carla was able to download 40% of the
         file, which is 0.4 * 200 GB = 80 GB. At a rate of 2 GB/minute, it
         would take her: time = 80 GB ÷ 2 GB/minute = 40 minutes. However,
         because of the restart, Carla had to start over. This means she had
         to download the entire 200 GB file. The time it takes to download
         the entire 200 GB file is: time = 200 GB ÷ 2 GB/minute = 100 minutes.
         And since Carla had to wait 20 minutes for the Windows update to
         install, the total time it took to download the file is:
         total time = 40 minutes + 100 minutes + 20 minutes = 160 minutes."
         ← 正确答案！

Bard: 仍然错误
```

**关键洞察**：不同模型之间的辩论也能工作——ChatGPT 利用 Bard 的错误回答作为"反面教材"，推导出正确答案。

## 3.4 预期 vs 实际：对比分析

### 主要实验结果对比

| 任务 | 预期（假设） | 实际结果 |
|------|-------------|---------|
| **推理能力** | 多代理辩论应优于单模型 | ✓ 确认：辩论在算术 (67%→81.8%)、GSM8K(77%→85%)、棋步预测上均显著提升 |
| **事实准确性** | 辩论能减少幻觉 | ✓ 确认：传记任务 (66%→73.8%)、MMLU(63.9%→71.1%)、棋步有效性 (29.3%→45.2%) |
| **收敛性** | 多轮辩论后应收敛到共识 | ✓ 确认：通常 2-4 轮后收敛，但共识答案不一定总是正确 |
| **Self-Reflection** | 自我反思应提升表现 | ✗ 反转：在某些事实性任务上，reflection 反而表现更差 (MMLU: 63.9%→57.7%) |
| **多数投票** | 多数投票应接近辩论效果 | ✗ 反转：辩论显著优于简单多数投票 (算术：69% vs 81.8%) |
| **跨模型辩论** | 不同模型间辩论应有效 | ✓ 确认：ChatGPT+Bard 辩论 (17 题正确) 优于单独 Bard(11 题) 或 ChatGPT(14 题) |

### 性能对比表

**Table 1: Multiagent Debate Improves Reasoning**

| Model | Arithmetic (%) ↑ | Grade School Math (%) ↑ | Chess (ΔPS) ↑ |
|-------|-----------------|------------------------|---------------|
| Single Agent | 67.0 ± 4.7 | 77.0 ± 4.2 | 91.4 ± 10.6 |
| Single Agent (Reflection) | 72.1 ± 4.5 | 75.0 ± 4.3 | 102.1 ± 11.9 |
| Multi-Agent (Majority) | 69.0 ± 4.6 | 81.0 ± 3.9 | 102.2 ± 6.2 |
| **Multi-Agent (Debate)** | **81.8 ± 2.3** | **85.0 ± 3.5** | **122.9 ± 7.6** |

**Table 2: Multiagent Debate Improves Factual Accuracy**

| Model | Biographies | MMLU | Chess Move Validity |
|-------|-------------|------|---------------------|
| Single Agent | 66.0 ± 2.2 | 63.9 ± 4.8 | 29.3 ± 2.6 |
| Single Agent (Reflection) | 68.3 ± 2.9 | 57.7 ± 5.0 | 38.8 ± 2.9 |
| **Multi-Agent (Debate)** | **73.8 ± 2.3** | **71.1 ± 4.6** | **45.2 ± 2.9** |

### 消融实验

**Figure 10: 代理数量和辩论轮次的影响**

```
(a) Performance vs Number of Agents (固定 2 轮辩论)
代理数 1  →  70%
代理数 2  →  75%
代理数 3  →  80%
代理数 4  →  82%
代理数 5  →  84%
结论：性能随代理数单调递增

(b) Performance vs Number of Debate Rounds (固定 3 代理)
轮次 1  →  72%
轮次 2  →  78%
轮次 3  →  82%
轮次 4  →  84%
轮次 5+ →  ~84% (饱和)
结论：4 轮后性能饱和
```

**Figure 12: Prompt 长度对辩论的影响**

```
Short Prompt (鼓励快速一致):
  - 收敛快（2-3 轮）
  - 最终准确率较低

Long Prompt (鼓励坚持己见):
  - 收敛慢（4-5 轮）
  - 最终准确率更高
```

**Figure 13: Summarization 的效果**

```
直接连接所有代理回答  →  82%
先总结再作为上下文    →  85%
结论：总结能提升性能（减少上下文噪声）
```

## 3.5 反直觉挑战

### 挑战 1：Self-Reflection 在事实性任务上表现更差

**直觉**：让模型反思自己的回答应该总能改进结果。

**实际**：在 MMLU 事实性任务上，Reflection 反而降低准确率 (63.9% → 57.7%)。

**可能原因**：
- 模型在事实性问题上"过度思考"，引入更多错误
- 自我反思缺乏外部验证，可能强化错误信念

### 挑战 2：多数投票不如辩论

**直觉**：如果多个模型独立回答，多数投票应该能得到正确答案。

**实际**：辩论 (81.8%) 显著优于多数投票 (69%)。

**原因分析**：
- 辩论允许模型**交叉验证推理过程**，而不仅仅是答案
- 模型可以识别并修正其他代理的错误
- 辩论是**动态收敛**，多数投票是**静态聚合**

### 挑战 3：自信度评估不可靠

**直觉**：直接询问模型的自信度可以判断答案可靠性。

**实际**：模型对几乎所有答案都给出高自信度评估。

**替代方案**：通过"说服难度"评估置信度——如果模型很容易被说服改变答案，说明它对该答案不太确定。

### 挑战 4：辩论收敛≠正确答案

**直觉**：如果所有代理都同意某个答案，那应该是正确的。

**实际**：辩论可能收敛到错误答案（见图 21-23 的错误案例）。

**局限性**：
- 当所有代理都缺乏相关知识时，可能共同"幻觉"出错误答案
- 模型不能正确表达不确定性

## 3.6 论文定位图谱

```
                        LLM 能力提升方法
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
    训练阶段方法              推理阶段方法              外部工具
        │                       │                       │
        │               ┌───────┴───────┐               │
        │               │               │               │
    Fine-tuning    单模型 Prompt     多模型方法      检索增强
    (RLHF, SFT)        │               │          (RAG, Tools)
                       │         ┌─────┴─────┐
                  ┌────┴────┐    │           │
                  │         │    │           │
              CoT/ToT   Self-  Majority   Debate
                         Consistency Voting  (本文)
                            │
                            │
                    ┌───────┴───────────────────────┐
                    │                               │
              静态聚合答案                    动态交叉验证
              (无信息交换)                  (多轮信息交换)
                                            │
                                    ┌───────┴───────┐
                                    │               │
                             同模型多实例       异模型辩论
                             (ChatGPT×3)       (ChatGPT+Bard)
```

### 与相关工作对比

| 方法 | 核心思路 | 与本文关系 |
|------|---------|-----------|
| Chain-of-Thought (Wei et al.) | 展示推理步骤 | 正交：可结合使用 |
| Self-Consistency (Wang et al.) | 多次采样取多数 | 对比基线：辩论更优 |
| Self-Refine (Madaan et al.) | 自我反思迭代 | 对比基线：多代理更优 |
| Socratic Models (Zeng et al.) | 多模型组合推理 | 相关：但本文专注于语言模型间辩论 |
| AI Safety via Debate (Irving et al.) | 辩论验证 AI 安全 | 灵感来源：但本文聚焦事实性和推理 |

## 3.7 方法细节

### 完整算法流程

```
Algorithm 1: Multiagent Debate

Input: 问题 Q, 代理数 N, 辩论轮数 R, 基础模型 M
Output: 最终共识答案 A_final

1: // 初始化阶段
2: for i = 1 to N do
3:     A_i^0 ← M(Q)  // 每个代理独立生成初始答案
4: end for

5: // 辩论阶段
6: for r = 1 to R do
7:     for i = 1 to N do
8:         // 收集其他所有代理的上一轮答案
9:         Context_i^r = Concatenate({A_j^(r-1) | j ≠ i})
10:        // 使用辩论 prompt 更新答案
11:        A_i^r ← M(Q + Context_i^r + DebatePrompt)
12:    end for
13:
14:    // 检查收敛（可选）
15:    if All A_i^r are identical then
16:        break
17:    end if
18: end for

19: // 输出最终答案
20: A_final ← MajorityVote({A_1^R, ..., A_N^R})
21: return A_final
```

### Prompt 模板（来自 Appendix Figure 15）

**Arithmetic Task:**
```
Starting: "What is the result of {}+{}*{}+{}-{}*{}?
          Make sure to state your answer at the end of the response."

Debate: "These are the recent/updated opinions from other agents:
         <other agent responses>
         Use these opinions carefully as additional advice,
         can you provide an updated answer?
         Make sure to state your answer at the end of the response."
```

**GSM8K Task:**
```
Starting: "Can you solve the following math problem? <Problem>
          Explain your reasoning. Your final answer should be a
          single numerical number, in the form \boxed{{answer}},
          at the end of your response."

Debate: "These are the solutions to the problem from other agents:
         <other agent responses>
         Using the solutions from other agents as additional
         information, can you provide your answer to the math
         problem? The original math problem is <Problem>.
         Your final answer should be a single numerical number,
         in the form \boxed{{answer}}, at the end of your response."
```

**Biographies Task:**
```
Starting: "Give a bullet point biography of <person> highlighting
          their contributions and achievements as a computer
          scientist, with each fact separated with a new line
          character."

Debate: "Here are some bullet point biographies of <person> given
         by other agents: <other agent response>
         Closely examine your biography and the biography of
         other agents and provide an updated bullet point
         biography."
```

## 3.8 实验设置与评估细节

### 任务与数据集

| 任务 | 数据集 | 评估指标 | 样本数 |
|------|--------|---------|--------|
| 算术 | 自定义生成 | 准确率 | 100 |
| 小学数学 | GSM8K | 准确率 | 100 |
| 棋步预测 | PGN Mentor | Stockfish  pawn score | 300 |
| 传记生成 | 自建 (524 位计算机科学家) | 与 Wikipedia 事实一致性 | 524 |
| 事实问答 | MMLU | 准确率 | 100 |
| 棋步有效性 | BIG-Bench | 有效移动比例 | 100 |

### 传记评估方法

```
评估 Prompt:
"Consider the following biography of <person>:
 <generated biography>
Is the above biography above consistent with the fact below?
 <ground truth bullet>
Give a single-word answer, yes, no, or uncertain."

准确率 = (yes + no) / (yes + no + uncertain) × 100%
（忽略返回 uncertain 的样本）
```

### 模型配置

- 基础模型：`gpt-3.5-turbo-0301` (ChatGPT)
- 默认配置：3 个代理，2 轮辩论
- 温度：未明确指定（可能为 0）

## 3.9 局限性与未来方向

### 论文明确的局限性

1. **计算成本**：多代理多轮辩论比单模型推理成本高得多
   - N 个代理 × R 轮 = N×R 次模型调用

2. **长上下文处理**：辩论变长时，模型难以完整处理整个辩论历史
   - 倾向于关注最近的生成内容
   - 解决方案：总结早期辩论内容

3. **错误收敛**：辩论可能收敛到错误答案
   - 模型会自信地肯定错误答案
   - 需要正交方法提升不确定性表达

### 未来研究方向

1. **知识蒸馏**：将辩论产生的高质量数据蒸馏回基础模型
   - 创建 self-improvement loop

2. **异质代理**：使用不同 persona 初始化代理
   - 初步实验：Professor/Doctor/Mathematician persona 提升 MMLU 从 71.1% → 74.2%

3. **更高效辩论**：探索更少的代理/轮次达到相同效果

4. **不确定性建模**：结合其他方法让模型更好表达不确定性

## 3.10 实践启示

### 何时使用 Multiagent Debate

| 场景 | 推荐度 | 理由 |
|------|-------|------|
| 高准确性要求任务 | ★★★★★ | 显著提升事实和推理准确性 |
| 复杂推理问题 | ★★★★★ | 多轮辩论能发现推理错误 |
| 事实生成/验证 | ★★★★☆ | 减少幻觉，检测不一致 |
| 成本敏感场景 | ★★☆☆☆ | 计算成本高 |
| 实时响应需求 | ★★☆☆☆ | 多轮延迟较高 |

### 实施建议

1. **代理数量**：3-5 个代理是甜点（性能/成本平衡）
2. **辩论轮次**：2-4 轮，过多会饱和
3. **Prompt 设计**：使用"Long Prompt"鼓励模型坚持己见
4. **上下文管理**：多代理时使用总结策略
5. **结合 CoT**：与零样本思维链结合效果更佳

### 代码框架示意

```python
class MultiagentDebate:
    def __init__(self, n_agents=3, n_rounds=2, model="gpt-3.5-turbo"):
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.model = model

    def debate(self, question, task_type="general"):
        # Round 0: Independent generation
        answers = []
        for i in range(self.n_agents):
            answer = self.model.generate(question)
            answers.append(answer)

        # Round 1 to R: Debate
        for r in range(self.n_rounds):
            new_answers = []
            for i in range(self.n_agents):
                # Collect other agents' answers
                context = "\n".join([answers[j] for j in range(self.n_agents) if j != i])

                # Generate updated answer
                prompt = self._create_debate_prompt(question, context, task_type)
                updated_answer = self.model.generate(prompt)
                new_answers.append(updated_answer)

            answers = new_answers

        # Final consensus
        return self._get_consensus(answers)
```

## 3.11 关键结论

1. **核心发现**：多代理辩论是一种有效且正交的方法，可显著提升 LLM 的事实准确性和推理能力

2. **机制解释**：辩论通过以下机制工作：
   - 交叉验证推理过程
   - 识别并修正错误
   - 剔除不确定的幻觉信息
   - 动态收敛到共识

3. **实用价值**：
   - 可直接应用于现有黑盒模型
   - 所有任务使用相同的 prompt 模板
   - 与 CoT 等方法正交可结合

4. **研究意义**：为"society of minds"方法在 LLM 中的应用提供了实证支持，开辟了通过多代理交互提升模型能力的新方向

---

## 附录：重要图示复现

### Figure 1: 多代理辩论在六个基准测试上的表现

```
Accuracy (%)
100 ┤
 90 ┤                              ╭────╮
 80 ┤              ╭────╮         │    │
 70 ┤      ╭────╮  │    │         │    │
 60 ┤      │    │  │    │         │    │
    └──────┴────┴──┴────┴─────────┴────┴────
         Single  Multi-Agent Debate
         Agent
```

### Figure 2: 辩论流程示例（简化版）

```
用户输入："宝藏猎人发现了一个装满宝石的 chests。有 175 颗钻石，
        红宝石比钻石少 35 颗，祖母绿是红宝石的 2 倍。总共有多少宝石？"

Round 1:
┌─────────────────────────────────────────────────────────────────┐
│ Agent 1: 设 x 为红宝石数量。则钻石数量是 175。祖母绿数量是 2(x-35)... │
│         答案：225（错误）                                        │
│                                                                 │
│ Agent 2: 如果有 175 颗钻石，则红宝石数量是 175-35=140 颗。           │
│         祖母绿数量是 2×140=280 颗。总数：175+140+280=595（正确）   │
└─────────────────────────────────────────────────────────────────┘

Round 2:
┌─────────────────────────────────────────────────────────────────┐
│ Agent 1: "Given the information provided in other agents'        │
│         solutions, we have two answers... the number of rubies  │
│         should be 175-35=140, as the second agent found."       │
│         答案：595（修正后正确）                                   │
│                                                                 │
│ Agent 2: "After reviewing the solutions provided by other       │
│         agents, I agree with the second agent..."               │
│         答案：595（保持正确）                                    │
└─────────────────────────────────────────────────────────────────┘

最终共识：595 ✓
```

---

*文档生成时间：2026-03-15*
*PDF 路径：/Users/miracle/projects/paper/papers/31_Agent_Debate_Improves_Factuality.pdf*
*输出路径：/Users/miracle/projects/paper/docs/31_Agent_Debate_Improves_Factuality.md*
