# ReAct: Synergizing Reasoning and Acting in Language Models

**论文信息**: Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023.
**arXiv**: [2210.03629](https://arxiv.org/abs/2210.03629)

---

## 层 1：电梯演讲（30 秒）

**一句话概括**：针对大语言模型推理（CoT）和行动（工具调用）被割裂研究的问题，普林斯顿 & Google Brain 提出 ReAct 范式——** interleaved 生成推理 trace 和任务特定 action**，在问答/事实验证任务上克服幻觉问题，在决策任务上超越 SOTA 34%（ALFWorld）和 10%（WebShop），仅需 1-2 个 in-context 示例。

---

## 层 2：故事摘要（5 分钟读完）

### 核心问题

2022 年，LLM 研究面临一个奇怪的分裂：
- **推理派**（CoT）：模型能一步步推理，但无法落地执行（无法调用 API、搜索网络）
- **行动派**（工具调用）：模型能调用工具，但不会思考（陷入循环、无法处理异常）

姚顺雨团队发现：**人类解决问题时，是边思考边行动的**。

### 核心洞察

**ReAct 范式**：Thought → Action → Observation 循环

```
Thought: 我需要搜索 X 的信息...
Action: search_wikipedia("X")
Observation: [搜索结果]
Thought: 根据结果，我还需要查 Y...
Action: search_wikipedia("Y")
Observation: [搜索结果]
Thought: 现在我有足够信息了，答案是...
```

### 研究框架图

```
┌─────────────────────────────────────────────────────────────────┐
│                        问题空间                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  LLM 能力的割裂                                       │       │
│  │  - Chain-of-Thought: 能推理但不能行动                 │       │
│  │  - Tool Use: 能行动但不能推理                         │       │
│  │  - 幻觉问题：自信地输出错误答案                       │       │
│  │  - 循环问题：重复同样的失败动作                       │       │
│  └──────────────────────┬───────────────────────────────┘       │
│                         │                                       │
│         ┌───────────────▼───────────────┐                       │
│         │  人类的解决方式                │                       │
│         │  - 自言自语："抽屉没有，试试柜子"│                     │
│         │  - 思考和行动交织               │                     │
│         └───────────────┬───────────────┘                       │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   ReAct 核心洞察         │
              │                         │
              │  "让 LLM  interleaved   │
              │   生成 Thought + Action │
              │   会怎样？"             │
              │                         │
              │  关键优势：             │
              │  - Thought 处理异常     │
              │  - Action 获取信息      │
              │  - Observation 更新信念 │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │      ReAct 范式          │
              │  ┌───────────────────┐  │
              │  │ Thought           │  │
              │  │   ↓               │  │
              │  │ Action            │  │
              │  │   ↓               │  │
              │  │ Observation       │  │
              │  │   ↓               │  │
              │  │ (循环)            │  │
              │  └───────────────────┘  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │        验证层            │
              │  HotpotQA: 27.4% → 41%  │
              │  Fever: 60.9% → 66%     │
              │  ALFWorld: 45% → 71%    │
              │  WebShop: 37% → 47%     │
              └─────────────────────────┘
```

### 关键结果

| 任务 | 基线 | ReAct | 提升 |
|------|------|-------|------|
| **HotpotQA**（多跳问答） | CoT: 29.4% | 41.0% | +11.6% |
| **Fever**（事实验证） | CoT: 60.9% | 66.0% | +5.1% |
| **ALFWorld**（决策） | BC: 45% | 71% | +26% |
| **WebShop**（购物） | BC: 37% | 47% | +10% |

---

## 层 3：深度精读

---

## 开场：一个令人沮丧的失败场景

2022 年秋天，普林斯顿大学的实验室里，博士生姚顺雨盯着一组令人困惑的实验数据。

他正在训练一个大语言模型来完成一个简单的任务：**回答多跳问题**。

**任务示例**：
> "电影《盗梦空间》的导演还执导过哪部关于间谍的电影？"

模型 confidently 输出了推理链：
```
《盗梦空间》是一部科幻电影 → 导演是克里斯托弗·诺兰 → 诺兰还执导过《信条》
```

**问题**：《信条》（Tenet）确实关于间谍，但模型完全不知道这个答案是否正确——它没有验证信息的能力。

姚顺雨尝试了 Chain-of-Thought  prompting，效果有所提升，但幻觉问题依然存在。

然后他尝试了工具调用——让模型搜索 Wikipedia。但新的问题出现了：模型机械地调用 API，却不会思考下一步该做什么。

"有没有可能，"姚顺雨想，"问题不在于推理或行动本身，而在于它们被割裂了？"

---

## 第一章：研究者的困境

### 2022 年的 LLM Landscape

在 ReAct 出现之前，LLM 的能力发展呈现一种奇怪的分裂：

**阵营 1：推理派（CoT）**
```
Q: "小明比小红高，小红比小华高，谁最高？"
A: "小明比小红高，小红比小华高 → 小明最高"
```
- **优点**：能进行多步推理，可解释
- **缺点**：无法获取外部信息，容易产生幻觉

**阵营 2：行动派（工具调用）**
```
Q: "搜索今天的天气"
A: call_weather_api() → "晴天"
```
- **优点**：能与环境交互，获取实时信息
- **缺点**：不会思考，容易陷入循环，无法处理异常

### 两个致命问题

**问题 1：幻觉（Hallucination）**
```
Q: "X 和 Y 在哪部电影中合作？"
CoT 输出：
- X 和 Y 都是著名演员 → 他们可能合作过多次 →
  最知名的合作是《电影 Z》→ 答案是《电影 Z》

实际：完全错误，但模型看起来很自信
```

**问题 2：循环（Looping）**
```
Task: "找到胡椒瓶并放到抽屉里"
Agent:
- go to cabinet 1 → No pepper here
- go to cabinet 1 → No pepper here  (重复!)
- go to cabinet 1 → No pepper here  (无限循环...)
```

姚顺雨和团队陷入了一个奇怪的境地：
- CoT 能推理但不能落地
- 工具调用能落地但不会思考
- 似乎没有两全其美的方案

---

## 第二章：试错的旅程

### 第一阶段：最初的直觉

"人类是怎么解决问题的？"姚顺雨在白板上画了一个简单的图。

**场景**：你在厨房找盐
```
1. 想："可能在第一个抽屉"
2. 打开第一个抽屉 → 没有
3. 想："那可能在柜子上面"
4. 打开柜子 → 找到了！
```

**关键洞察**：人类的思考（Thought）和行动（Action）是**交织在一起**的。

"如果让 LLM 也 interleaved 输出 Thought 和 Action，会怎样？"

### 第二阶段：第一次实验

团队设计了 ReAct 的基本范式：

```
Thought: 我需要先搜索 X 的信息
Action: search_wikipedia("X")
Observation: [维基百科关于 X 的信息]
Thought: 根据搜索结果，X 和 Y 有关系，我需要进一步查 Y
Action: search_wikipedia("Y")
Observation: [维基百科关于 Y 的信息]
Thought: 现在我有足够信息了，答案是 Z
```

**第一次实验结果**：
- HotpotQA：成功率从 29% 提升到 41%！
- ALFWorld：成功率从 45% 提升到 71%！

但团队发现了新问题。

### 第三阶段：稀疏思考的发现

**问题**：密集思考（每步都 Thought）太慢了。

```
密集思考：
Thought: 我要去柜子
Action: go to cabinet
Thought: 我到了柜子
Thought: 我要打开柜子
Action: open cabinet
Thought: 柜子打开了
...

这太啰嗦了！
```

姚顺雨观察到：**人类并不是每步都思考**。很多时候是自动行动的。

**稀疏思考策略**：
```
Action: go to cabinet 1
Obs: Nothing here
Thought: 这里没有，试试其他地方...  （只在关键时刻思考）
Action: go to counter 3
```

**关键洞察**：只在以下情况触发思考：
1. 遇到异常（东西不在预期位置）
2. 需要整合信息（多跳推理）
3. 计划调整（原路不通，换策略）

### 第四阶段：验证与完善

经过数月的迭代，ReAct 的最终方案确定了：

1. **Prompt 设计**：few-shot 示例展示 Thought-Action-Observation 格式
2. **稀疏思考**：只在关键时刻触发 Thought，减少冗余
3. **灵活格式**：不同任务可以有不同的 Action space

---

## 第三章：核心概念 - 大量实例

### 概念 1：ReAct 范式

**生活类比 1：厨房找东西**
```
想象你在厨房找搅拌机：
1. 想："可能在下面的柜子"  （Thought）
2. 打开柜子                （Action）
3. 观察：没有搅拌机         （Observation）
4. 想："那可能在台面上"     （Thought）
5. 走到台面                （Action）
6. 观察：找到了！           （Observation）

思考和行动是交织的——这就是 ReAct。
```

**生活类比 2：开车导航**
```
你开车去一个陌生地方：
1. 看导航："前方右转"        （Observation）
2. 想："这条路好像不对"      （Thought）
3. 打方向盘调整路线          （Action）
4. 看新路标："目的地 500m"    （Observation）
5. 想："快到了"              （Thought）
6. 继续直行                  （Action）

你不断在观察 → 思考 → 行动之间循环。
```

**代码实例 1：ReAct 基础循环**
```python
def react_loop(llm, task, tools, max_steps=10):
    history = []

    for step in range(max_steps):
        # 生成 Thought
        thought = llm.generate(
            prompt=build_prompt(task, history, mode="thought")
        )

        # 生成 Action
        action = llm.generate(
            prompt=build_prompt(task, history + [thought], mode="action")
        )

        # 执行 Action，获取 Observation
        observation = tools.execute(action)

        # 更新历史
        history.extend([thought, action, observation])

        # 检查是否完成
        if is_done(thought, action):
            break

    return extract_answer(history)
```

**任务实例 1：HotpotQA 多跳问答**
```
Question: "X Men 的导演还执导过哪部关于时间旅行的电影？"

ReAct 输出：
Thought 1: 我需要先搜索 X Men 的导演是谁
Action 1: search_wikipedia("X Men film")
Observation 1: X-Men (2000) is a superhero film directed by Bryan Singer.

Thought 2: 导演是 Bryan Singer，我需要查他是否执导过时间旅行电影
Action 2: search_wikipedia("Bryan Singer filmography")
Observation 2: Singer directed The Usual Suspects, X-Men series, and Superman Returns.
               None are explicitly about time travel...

Thought 3: 让我确认一下问题，可能是问其他 X Men 电影
Action 3: search_wikipedia("X-Men Days of Future Past director")
Observation 3: X-Men: Days of Future Past (2014) is directed by Bryan Singer.
               The film features time travel plot.

Thought 4: 找到了，答案是《X-Men: Days of Future Past》
Answer: X-Men: Days of Future Past
```

**任务实例 2：ALFWorld 决策任务**
```
Task: "Put pepper in drawer"

ReAct 输出：
Thought 1: 我需要先找到 pepper shaker 在哪里
Action 1: go to cabinet 1
Obs 1: On the cabinet 1, you see nothing.

Thought 2: cabinet 1 没有，试试 cabinet 2
Action 2: go to cabinet 2
Obs 2: On the cabinet 2, you see nothing.

Thought 3: 也许在 countertop 上
Action 3: go to countertop 1
Obs 3: On the countertop 1, you see a pepper shaker.

Thought 4: 找到了！现在需要 take 它
Action 4: take pepper shaker from countertop 1
Obs 4: You pick up the pepper shaker.

Thought 5: 现在去 drawer
Action 5: go to drawer 1
Obs 5: On the drawer 1, you see nothing.

Thought 6: 把 pepper 放进去
Action 6: put pepper shaker in/on drawer 1
Obs 6: You put the pepper shaker in/on drawer 1.

Thought 7: 任务完成！
```

### 概念 2：稀疏思考（Sparse Thought）

**生活类比：熟练工 vs 新手**
```
新手开车（密集思考）：
- 想："我要踩油门" → 踩油门
- 想："我要看后视镜" → 看后视镜
- 想："我要打方向盘" → 打方向盘
→ 开得很慢，很紧张

熟练司机（稀疏思考）：
- 正常行驶（自动）
- 看到红灯 → 想："要刹车了" → 刹车
- 看到路口 → 想："要转弯了" → 打方向
→ 开得快，轻松

稀疏思考就像"熟练模式"——只在关键时刻思考。
```

**代码实例：稀疏思考触发条件**
```python
def should_generate_thought(observation, history):
    """判断是否需要生成 Thought"""

    # 条件 1: 遇到异常/失败
    if observation_contains_failure(observation):
        return True

    # 条件 2: 信息不完整（多跳推理需要）
    if need_more_info(history):
        return True

    # 条件 3: 需要调整计划
    if plan_adjustment_needed(history):
        return True

    # 其他情况：直接行动，不生成 Thought
    return False
```

**对比场景：密集 vs 稀疏**
```
密集思考（慢但稳）：
Thought: 我要去 cabinet 1
Action: go to cabinet 1
Thought: 我到了 cabinet 1
Thought: 我要检查有什么
Action: examine cabinet 1
Thought: 我看到 nothing
Thought: 那我去 cabinet 2
...
步数：15 步，其中 8 步是 Thought

稀疏思考（快且聪明）：
Action: go to cabinet 1
Obs: Nothing here
Thought: 这里没有，试试其他地方
Action: go to cabinet 2
Obs: Nothing here
Thought: 也不在，可能在 countertop
Action: go to countertop 1
Obs: Found pepper shaker!
Action: take pepper shaker
...
步数：8 步，其中 3 步是 Thought

效率提升：约 50%
```

### 概念 3：Thought 的作用

**Thought 的三大功能**：

1. **异常处理**
```
Action: go to cabinet 1
Obs: Nothing here
Thought: 这里没有，但任务说一定有 pepper，让我想想其他可能的位置...
       （识别异常，调整策略）
```

2. **计划更新**
```
Obs: 找到 pepper shaker，但它被其他东西挡住了
Thought: 我需要先移开障碍物，才能拿到 pepper
         （更新计划：添加前置步骤）
```

3. **信息整合**
```
Obs 1: X Men 导演是 Bryan Singer
Obs 2: Bryan Singer 的作品列表...
Thought: 等等，Days of Future Past 是时间旅行题材的！
         （整合两条信息，得出结论）
```

**对比场景：有 Thought vs 无 Thought**
```
无 Thought（纯行动）：
Action: go to cabinet 1
Obs: Nothing here
Action: go to cabinet 1  (重复！)
Obs: Nothing here
Action: go to cabinet 1  (无限循环...)

有 Thought：
Action: go to cabinet 1
Obs: Nothing here
Thought: 这里没有，试试 cabinet 2
Action: go to cabinet 2
Obs: Nothing here
Thought: 也不在，可能在 countertop
Action: go to countertop 1
Obs: Found it!
```

---

## 第四章：预期 vs 实际

### 预期 vs 实际对比表

| 维度 | 你的直觉/预期 | ReAct 实际实现 | 为什么有差距？ |
|------|--------------|---------------|---------------|
| **什么时候思考？** | 每步都思考，越多越好 | 只在关键时刻思考 | 密集思考太慢，很多行动是自动的 |
| **Thought 的作用** | 推理下一步行动 | 异常处理 + 计划调整 | 关键价值是从失败中恢复 |
| **能否预规划？** | 先完整规划再执行 | 边做边调整 | 环境未知，无法预规划 |
| **Few-shot 数量** | 需要很多示例 | 1-2 个就够 | ReAct 格式本身就有提示作用 |
| **Action 空间** | 需要复杂工具 | 简单 API 就够 | 关键是 Thought-Action 协同 |

### 反直觉的事实

**问题 1：ReAct 的 Thought 真的是在"推理"吗？**

直觉可能说："Thought 就是在推理下一步怎么做吧？"

实际：**Thought 更多是在处理异常，而不是规划**。

原论文分析：
```
成功轨迹中：
- 60% 的 Thought 出现在失败/异常之后
- 只有 20% 的 Thought 是纯规划
- 其余是信息整合

Thought 的核心价值：从失败中恢复
```

**问题 2：ReAct 一定需要复杂的工具吗？**

直觉可能说："要完成任务，肯定需要强大的 API 吧？"

实际：**简单的 Wikipedia search 就足够解决 HotpotQA**。

```
ReAct 在 HotpotQA 上用的工具：
- search_wikipedia(query): 搜索页面
- lookup_wikipedia(entity): 查看页面内容

就这两个简单工具，超越了所有基线！
```

---

## 第五章：反直觉挑战

### 挑战 1：去掉所有的 Thought，只保留 Action，会怎样？

**预测**：可能只是少了一些输出，性能差不多吧？

**实际**：成功率大幅下降！

| 任务 | ReAct（有 Thought） | Act-only（无 Thought） | 下降 |
|------|---------------------|------------------------|------|
| **ALFWorld** | 71% | 45% | -26% |
| **WebShop** | 47% | 37% | -10% |
| **HotpotQA** | 41% | 25% | -16% |

**为什么？**
- 没有 Thought，Agent 无法识别" Cabinet1 没有"是异常情况
- 没有 Thought，Agent 不会调整策略（"试试 Counter3"）
- 没有 Thought，Agent 容易陷入无限循环

### 挑战 2：如果用纯 CoT（只有 Thought，没有 Action），会怎样？

**预测**：CoT 能推理，应该比 Act-only 好吧？

**实际**：在需要外部信息的任务上，CoT 表现很差！

| 任务 | CoT（纯推理） | ReAct | 差距 |
|------|--------------|-------|------|
| **HotpotQA** | 29% | 41% | -12% |
| **Fever** | 61% | 66% | -5% |

**为什么？**
- CoT 无法获取外部信息，只能靠模型内部知识
- 当问题涉及训练数据之外的信息，CoT 只能"编造"
- 这就是幻觉问题的根源

### 挑战 3：如果让 ReAct 在每步都思考（密集思考），会怎样？

**预测**：思考越多，效果应该越好吧？

**实际**：效果差不多，但速度慢很多！

```
密集思考：
- 每步都生成 Thought
- Token 消耗：+80%
- 推理时间：+100%
- 成功率：+1%（几乎没提升）

稀疏思考：
- 只在关键时刻生成 Thought
- Token 消耗：基准
- 推理时间：基准
- 成功率：基准

结论：稀疏思考是"性价比"最优的选择
```

---

## 第六章：关键实验的细节

### 实验 1：HotpotQA 多跳问答

**任务设置**：
- 数据：HotpotQA dev 集（5-10 跳问题）
- 工具：Wikipedia search API
- 评估：精确匹配（EM）准确率

**结果**：
```
| 方法 | 准确率 |
|------|--------|
| Standard Prompting | 18.5% |
| Chain-of-Thought | 29.4% |
| Act-only | 25.7% |
| ReAct | 41.0% |
| ReAct + Self-Consistency | 47.5% |
```

**关键洞察**：
- ReAct 超越 CoT 11.6%
- ReAct 通过搜索外部信息，减少了幻觉
- 结合 Self-Consistency 可以进一步提升

### 实验 2：ALFWorld 决策

**任务设置**：
- 环境：TextWorld 模拟家居环境
- 任务：找物品并放置（如"Put pepper in drawer"）
- 评估：任务成功率

**结果**：
```
| 方法 | 成功率 |
|------|--------|
| Behavior Cloning (BC) | 45% |
| Reinforcement Learning (RL) | 52% |
| Act-only | 48% |
| ReAct | 71% |
```

**关键洞察**：
- ReAct 超越 BC 26%（绝对提升）
- Thought 帮助 Agent 从失败中恢复
- 仅需 2 个 few-shot 示例

### 实验 3：WebShop 购物

**任务设置**：
- 环境：模拟电商网站
- 任务：根据指令购买商品（如"找一件$50 以下的蓝色衬衫"）
- 评估：任务成功率 + 商品匹配度

**结果**：
```
| 方法 | 成功率 |
|------|--------|
| Behavior Cloning | 37% |
| RL | 42% |
| ReAct | 47% |
```

**关键洞察**：
- WebShop 需要多步决策（搜索 → 筛选 → 选择 → 购买）
- ReAct 的 Thought 帮助整合商品信息和用户指令

### 实验 4：Fever 事实验证

**任务设置**：
- 数据：Fever 数据集（验证陈述真伪）
- 工具：Wikipedia search API
- 评估：验证准确率

**结果**：
```
| 方法 | 准确率 |
|------|--------|
| Standard Prompting | 52.0% |
| Chain-of-Thought | 60.9% |
| ReAct | 66.0% |
```

**关键洞察**：
- Fever 需要搜索证据来验证陈述
- ReAct 的 Thought 帮助选择搜索关键词

---

## 第七章：与其他方法对比

### 论文定位图谱

```
                    上游工作（它解决了谁的问题）
                    ┌─────────────────────────┐
                    │   Chain-of-Thought      │
                    │  (Wei et al., 2022)     │
                    │  - 能推理但不能行动      │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Tool Use / API Call   │
                    │  - 能行动但不能推理      │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     ▼                     │
          │            ┌─────────────────┐            │
          │            │     ReAct       │            │
          │            │  (Yao et al.,   │            │
          │            │   ICLR 2023)    │            │
          │            └────────┬────────┘            │
          │                     │                     │
          ┼─────────────────────┼─────────────────────┼
    并行工作                     │              并行工作
┌──────────────────┐            │        ┌──────────────────┐
│  Reflexion       │            │        │  HuggingGPT      │
│  (Shinn et al.)  │            │        │  (Shen et al.)   │
│  - 从失败中学习  │            │        │  - 工具编排     │
└──────────────────┘            │        └──────────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Agent 框架            │
                    │  - AutoGen              │
                    │  - LangChain Agents     │
                    │  - CAMEL                │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Toolformer            │
                    │  - 自学习工具使用       │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   现代 Agent 系统        │
                    │  - OpenAI Assistants API│
                    │  - Claude Function Call │
                    └─────────────────────────┘

         下游工作（谁解决了它的问题/扩展了它）
```

### 详细对比表

| 方法 | 推理 | 行动 | 异常处理 | 可解释性 | 适用场景 |
|------|------|------|----------|----------|----------|
| **Standard** | ❌ | ❌ | ❌ | ❌ | 简单任务 |
| **CoT** | ✅ | ❌ | ❌ | ✅ | 推理任务 |
| **Act-only** | ❌ | ✅ | ❌ | ❌ | 简单工具调用 |
| **ReAct** | ✅ | ✅ | ✅ | ✅ | 复杂任务 |
| **Reflexion** | ✅ | ✅ | ✅✅ | ✅ | 需要学习的任务 |

### 局限性分析

ReAct 并非完美，存在以下局限：

1. **无法跨 Trial 学习**
   - 每次执行都是独立的，第二次不会比第一次好
   - 需要结合 RL 或 Reflexion 来实现持续学习

2. **长序列任务容易迷失**
   - 没有记忆机制，无法记住之前的尝试
   - 对于需要 20+ 步的任务，表现下降

3. **依赖 Prompt 设计**
   - Few-shot 示例的质量影响很大
   - 不同任务需要不同的 Action space 设计

### 改进方向

1. **Reflexion (2023)**
   - 改进：增加自我反思机制，从失败中学习
   - 效果：跨 trial 性能提升

2. **Memory 机制**
   - 改进：长期记忆存储和检索
   - 效果：处理长序列任务

3. **多 Agent 协作**
   - 改进：多个 Agent 分工合作
   - 效果：解决更复杂任务

4. **Toolformer / Gorilla**
   - 改进：自学习工具使用
   - 效果：减少 few-shot 依赖

---

## 第八章：如何应用

### 推荐配置

**基础 ReAct Prompt 模板**：
```
Task: {task_description}

Here are some examples of how to solve this task:

Example 1:
Thought: {example_thought_1}
Action: {example_action_1}
Observation: {example_observation_1}
Thought: {example_thought_2}
Action: {example_action_2}
Observation: {example_observation_2}
...

Now solve the following task:
Task: {actual_task}
```

**稀疏思考触发条件**：
```python
def should_think(observation, context):
    # 触发思考的情况
    if observation_contains_failure(observation):
        return True  # 失败/异常
    if observation_is_surprising(observation, context):
        return True  # 意外发现
    if need_to_integrate_info(context):
        return True  # 需要整合信息
    if plan_adjustment_needed(context):
        return True  # 需要调整计划

    # 其他情况：直接行动
    return False
```

### 实战代码

**PyTorch + HuggingFace 实现**：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

class ReActAgent:
    def __init__(self, model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tools = {}  # 注册工具

    def register_tool(self, name, func):
        self.tools[name] = func

    def parse_action(self, action_text):
        # 解析 "search_wikipedia("query")" 格式
        match = re.match(r'(\w+)\("([^"]+)"\)', action_text)
        if match:
            tool_name, query = match.groups()
            return tool_name, query
        return None, None

    def react_loop(self, task, max_steps=10):
        history = []
        prompt = self.build_prompt(task)

        for step in range(max_steps):
            # 生成 Thought
            thought_prompt = prompt + f"Thought:"
            thought = self.generate(thought_prompt, stop=["\n"])

            # 生成 Action
            action_prompt = prompt + f"Thought: {thought}\nAction:"
            action = self.generate(action_prompt, stop=["\n"])

            # 执行 Action
            tool_name, query = self.parse_action(action)
            if tool_name and tool_name in self.tools:
                observation = self.tools[tool_name](query)
            else:
                observation = "Invalid action"

            # 更新历史
            history.append((thought, action, observation))
            prompt += f"Thought: {thought}\nAction: {action}\nObservation: {observation}\n"

            # 检查是否完成
            if "Answer:" in thought or "done" in action.lower():
                break

        return self.extract_answer(history)
```

### 避坑指南

**常见错误 1：Thought 太冗长**
```python
# ❌ 错误：Thought 写成一篇文章
Thought: 好的，让我仔细想想这个问题。首先，我需要理解任务的要求...

# ✅ 正确：Thought 简洁直接
Thought: 需要搜索 X Men 导演
```

**常见错误 2：Action 格式不统一**
```python
# ❌ 错误：混用多种格式
Action: search_wikipedia("X")
Action: call search_wikipedia with X
Action: {tool: "search_wikipedia", args: ["X"]}

# ✅ 正确：统一格式
Action: search_wikipedia("X")
```

**常见错误 3：忘记处理无效 Action**
```python
# ❌ 错误：假设所有 Action 都有效
observation = tools[action_name](action_args)  # 可能 KeyError!

# ✅ 正确：处理无效 Action
if action_name in self.tools:
    observation = self.tools[action_name](action_args)
else:
    observation = "Invalid action. Available tools: " + str(list(self.tools.keys()))
```

---

## 第九章：延伸思考

### 深度问题

1. **为什么 ReAct 的 Thought 更多出现在失败之后，而不是规划时？**
   - 提示：从认知科学角度思考——人类什么时候最"需要"思考？

2. **ReAct 能否应用于非语言任务（如机器人控制）？**
   - 提示：考虑 Thought 如何表示，Action 如何执行

3. **如果让 ReAct 自己学会何时思考（而不是规则触发），会怎样？**
   - 提示：结合 RL 或 meta-learning

4. **ReAct 的 Thought 真的是"推理"吗？还是只是模式匹配？**
   - 提示：思考 LLM 的本质能力

5. **多 Agent 协作时，ReAct 应该如何修改？**
   - 提示：多个 Agent 的 Thought 如何同步？

6. **ReAct 与 System 2 思维（慢思考）有什么关系？**
   - 提示：参考 Kahneman 的 Thinking, Fast and Slow

7. **如果去掉 Observation，只保留 Thought-Action，会怎样？**
   - 提示：Observation 的作用是什么？

### 实践挑战

1. **复现 ReAct 实验**
   - 在 HotpotQA 子集上复现 ReAct vs CoT 对比
   - 验证原论文结论

2. **实现稀疏思考优化**
   - 基于规则/分类器判断何时思考
   - 对比密集思考的性能/效率

3. **扩展 Action space**
   - 为特定任务（如编程、数据分析）设计专用工具
   - 评估 ReAct 的表现

---

## 总结

ReAct 通过 interleaved 生成 Thought 和 Action，成功将推理和行动统一到一个框架中。

**核心贡献**：
1. **Thought-Action-Observation 循环** - 模拟人类问题解决方式
2. **稀疏思考策略** - 只在关键时刻思考，提升效率
3. **Few-shot 通用性** - 1-2 个示例即可适配新任务

**历史地位**：
- ICLR 2023 Oral
- 开启 Agent 研究新方向
- 影响后续 AutoGen、LangChain 等框架

**一句话总结**：ReAct 让 LLM 学会"边想边做"——不再是只会推理的书呆子，也不会是只会行动的莽夫。

---

**参考文献**
1. Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023.
2. Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS 2022.
3. Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS 2023.
4. Shen, Y., et al. (2023). HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face. NeurIPS 2023.
