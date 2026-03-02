# Generative Agents: 生成式智能体 - 人类行为的可信模拟

## 开场：当虚拟角色开始"活"过来

时间：2023年初春
地点：斯坦福大学 HCI 实验室
人物：博士生 Joon Sung Park 和他的团队

**困境：Joon 盯着电脑屏幕上的 Smallville 虚拟小镇**

屏幕上，25个用简单像素头像代表的角色正在一个小镇里生活。

- 药剂师 John Lin 刚起床，正在刷牙
- 他的妻子 Mei 正在准备早餐
- 儿子 Eddy 匆忙出门去上学

这看起来像是一个普通的模拟游戏，比如《模拟人生》。但有些不同寻常：

John 刚走到客厅，就看到 Eddy 正要出门。
John 说："早上好 Eddy，你睡得好吗？"
Eddy 回答："早上好爸爸，睡得很好。"
John 问："那你今天在忙什么？"
Eddy 说："我在做一个新的音乐作曲，这周要交，我正在努力完成，但我觉得很有趣！"

**这些对话不是预设的脚本，而是由 ChatGPT 实时生成的。**

更神奇的是，这些角色会：
- **记住**彼此说过的话
- **形成**对彼此的看法
- **传播**小镇里的消息
- **协调**集体活动

比如，当研究者告诉 Isabella："你想办一个情人节派对"——仅仅这一个指令——
- Isabella 开始邀请她遇到的人
- 被邀请的人记住了这个约定
- 有人甚至邀请了暗恋的对象一起去
- 派对当天，5个角色真的准时出现了

**Joon 的困惑**："这不科学啊！"

传统的 AI 角色要么：
- 只能按预设脚本行动（太死板）
- 要么用 LLM 但没有长期记忆（说完就忘）

但这 25 个角色似乎：
- ✅ 记得过去发生的事
- ✅ 会从记忆中总结规律
- ✅ 会根据记忆计划未来
- ✅ 会与其他角色互动产生社会行为

**问题来了：**他们是怎么做到的？

---

## 第一章：研究者的困境 - 为什么让 AI "有记忆" 这么难？

### 1.1 时代背景：LLM 已经很强，但还不够

到了 2023 年，ChatGPT 已经证明了：
- 它可以生成流畅的对话
- 它可以回答问题
- 它可以写代码、写文章

**但是**，当你想用它创建一个"长期角色"时，问题就暴露了：

**问题 1：没有长期记忆**
```
你（第一天）："你好，我是 Alice，我喜欢红色。"
ChatGPT："你好 Alice！很高兴认识你。"

你（第二天）："你还记得我吗？"
ChatGPT："抱歉，我没有记忆。你是谁？"
```

**问题 2：无法从经历中学习**
```
角色遇到挫折 → ChatGPT 生成反应 → 下次遇到同样情况 → 还是同样的反应
因为没有"总结经验"的机制
```

**问题 3：社会行为无法涌现**
```
25个角色 = 25个独立的 LLM 实例
他们之间不知道彼此在说什么
无法形成"流言传播"这种社会现象
```

### 1.2 学界的探索

在此之前，学界已经尝试了各种方法：

**方向 1：直接用 LLM 模拟**
- 做法：把所有背景信息塞进 prompt
- 问题：context window 有限，信息多了就忘了旧的

**方向 2：强化学习训练**
- 做法：训练一个专门做 social navigation 的模型
- 问题：需要大量数据，而且不灵活

**方向 3：游戏式的脚本系统**
- 做法：像《模拟人生》那样预编程行为树
- 问题：行为固定，无法生成新的、意外的行为

### 1.3 核心挑战

Joon 团队意识到，要创造"可信的人类行为模拟"，需要解决三个问题：

1. **记忆管理**：如何存储大量经历并在需要时检索？
2. **记忆综合**：如何从具体事件中抽象出高层次的见解？
3. **长期规划**：如何保持行为在时间上的一致性？

这不仅仅是技术问题，更是**认知科学**问题——人类的大脑是怎么做到的？

---

## 第二章：试错的旅程 - 从失败到突破

### 2.1 最初的直觉：直接用 ChatGPT

**尝试 1：把所有信息都塞进 prompt**

```
Prompt:
"你是一个药剂师 John Lin，和妻子 Mei、儿子 Eddy 住在一起。
你认识邻居 Sam 和 Jennifer Moore...
（这里列出了50行背景信息）
...

现在是早上7点，你会做什么？"
```

**结果**：失败
- 太多信息，模型开始"幻觉"
- 生成的行为和背景矛盾
- 一旦对话超过几轮，就忘记了之前的设定

**研究者的反应**："看来需要更智能的记忆管理..."

### 2.2 第二次尝试：加一个记忆数据库

**尝试 2：把重要事件存到数据库里**

```
当角色经历事件时：
"Isabella 正在准备情人节派对" → 存入数据库

需要回忆时：
SELECT * FROM memories WHERE relevance > threshold
```

**结果**：部分成功，但新问题出现了
- ✅ 能记住事情了
- ❌ 但不会从记忆中"总结规律"
- ❌ 不会根据记忆改变长期计划

**具体失败案例**：
```
Maria 和 Wolfgang 聊过很多次
Maria 知道 Wolfgang 喜欢数学音乐
但当你问 Maria："Wolfgang 喜欢什么？"
Maria 只能列举具体事件，无法抽象出"他喜欢数学音乐"
```

### 2.3 顿悟时刻：人类是怎么记忆的？

某天深夜，Joon 突然想到：

**"等等，人类不只是在'存储'记忆，还在'反思'记忆！"**

比如你经历了一周的考试周：
- **具体记忆**："周一考了数学"、"周三考了英语"
- **反思**："这周压力很大"、"我需要更好的时间管理"
- **高层次反思**："我的学习效率在压力大时会下降"

**这些反思不是原始记忆，而是从记忆中抽象出来的！**

同样，人类不只是"记住"朋友，还会形成"印象"：
- **具体记忆**："Maria 帮我装饰了派对"
- **反思**："Maria 很乐于助人"
- **高层次反思**："我和 Maria 有共同兴趣"

### 2.4 突破：三层架构

Joon 团队意识到，一个可信的 Agent 需要**三种**能力：

1. **记忆流（Memory Stream）**
   - 存储所有经历
   - 能根据相关性检索

2. **反思（Reflection）**
   - 定期从记忆中总结
   - 形成高层次见解

3. **规划（Planning）**
   - 基于记忆和反思制定计划
   - 保持长期一致性

**这三个组件循环运作**：
```
感知 → 存入记忆 → 检索相关记忆 → 反思 → 规划 → 行动
    ↑                                      ↓
    ←←←←←←←←←← 新的记忆 ←←←←←←←←←←←←←←←
```

---

## 第三章：核心概念详解 - 大量实例

### 3.1 概念一：记忆流（Memory Stream）

**什么是记忆流？**
记忆流是 Agent 所有经历的完整记录，用自然语言存储。

#### 【生活类比 1：你的日记本】

想象你有一个日记本，记录了发生的每一件事：
- "2024年2月23日 7:00 - 我起床了"
- "2024年2月23日 7:30 - 我吃了面包和牛奶"
- "2024年2月23日 8:00 - 我遇到了同事 Alice"

当你需要回忆"我今天早上做了什么"时，你会翻看日记。

但你不只是"记录"，还会：
1. 给重要的日子做标记（重要性评分）
2. 最近的事情记得更清楚（时间衰减）
3. 当有人问你"Alice是谁"时，你会找所有关于 Alice 的记录（相关性检索）

#### 【生活类比 2：社交媒体时间线】

你的朋友圈/微博就像一个记忆流：
- 每条记录有时间戳
- 你记得最近发的动态
- 你能搜索特定话题的记录
- 重要的事件（比如结婚）你会特别标记

#### 【代码实例 1：记忆流的结构】

```python
# 一条记忆的结构
memory = {
    "description": "Isabella Rodriguez is setting out the pastries",
    "timestamp": "2023-02-14 08:30:00",
    "last_accessed": "2023-02-14 09:15:00",
    "importance": 3,  # 1-10分
    "related_entities": ["Isabella Rodriguez", "pastries"]
}

# Agent 的记忆流
memory_stream = [
    {
        "description": "Klaus Mueller is reading a book on gentrification",
        "timestamp": "2023-02-14 10:00:00",
        "importance": 6
    },
    {
        "description": "Klaus Mueller is conversing with a librarian about his research",
        "timestamp": "2023-02-14 10:30:00",
        "importance": 7
    },
    # ... 更多记忆
]
```

#### 【代码实例 2：重要性评分】

```python
# 论文中的 prompt
def score_importance(memory):
    prompt = f"""
    On the scale of 1 to 10, where 1 is purely mundane
    (e.g., brushing teeth, making bed) and 10 is
    extremely poignant (e.g., a break up, college
    acceptance), rate the likely poignancy of the
    following piece of memory.

    Memory: {memory}
    Rating: """

    return llm_generate(prompt)  # 返回 1-10 的分数

# 示例
score_importance("cleaning up the room")
# → 2 (日常琐事)

score_importance("asking your crush out on a date")
# → 8 (重要事件)
```

#### 【代码实例 3：记忆检索的三要素】

```python
def retrieve_memories(query, memory_stream, current_time):
    scores = []

    for memory in memory_stream:
        # 1. 时间衰减 (Recency)
        hours_since_access = (current_time - memory["last_accessed"]).hours
        recency_score = 0.995 ** hours_since_access  # 指数衰减

        # 2. 重要性 (Importance)
        importance_score = memory["importance"] / 10

        # 3. 相关性 (Relevance)
        relevance_score = compute_relevance(query, memory["description"])

        # 综合分数
        total_score = (
            recency_score * 0.2 +
            importance_score * 0.3 +
            relevance_score * 0.5
        )

        scores.append((memory, total_score))

    # 返回分数最高的记忆
    return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

# 使用示例
relevant_memories = retrieve_memories(
    query="What is Klaus passionate about?",
    memory_stream=klaus_memories,
    current_time="2023-02-14 14:00:00"
)
```

#### 【对比实例 1：有记忆检索 vs 没有记忆检索】

**场景：Isabella 被问"你最近热衷什么？"**

**没有记忆检索（总结全部记忆）**：
```
Isabella: "我热衷于合作举办活动和项目，
以及保持咖啡店的清洁和组织。"
→ 太泛泛，没有个性
```

**有记忆检索（只检索相关记忆）**：
```
Isabella: "我热衷于让人感到受欢迎和包容，
策划活动，创造人们可以享受的氛围，
比如情人节派对。"
→ 具体，有个性，基于她的实际经历
```

#### 【对比实例 2：重要性评分的作用】

**场景：Agent 遇到火警**

**低重要性记忆（吃早餐）**：
```
记忆："Isabella 在厨房吃了吐司"
重要性：2/10

检索权重：低
→ 火警时不会被检索到，因为不相关
```

**高重要性记忆（火警经历）**：
```
记忆："Isabella 经历了厨房火灾"
重要性：9/10

检索权重：高
→ 以后看到"厨房着火"会立即被检索
→ 影响行为："她会立即关掉炉子"
```

#### 【逐步演化实例：记忆系统的发展】

**版本 1：无记忆系统**
```
Agent = LLM(背景信息)
问题：每次都是新的，无法学习
```

**版本 2：简单记忆列表**
```
记忆 = [所有事件的列表]
问题：列表太长，无法全部塞进 prompt
```

**版本 3：加权检索**
```
记忆检索 = f(相关性, 时间, 重要性)
解决：只检索相关的记忆
```

**版本 4：完整记忆流（Generative Agents）**
```
记忆流 + 三重评分 + 动态更新
解决：高效、相关、可信的记忆管理
```

---

### 3.2 概念二：反思（Reflection）

**什么是反思？**
反思是从记忆中抽象出高层次见解的过程。

#### 【生活类比 1：从经历到智慧】

想象你在学习开车：

**具体经历（记忆）**：
- "周一：我转弯太快，差点撞到路沿"
- "周二：我保持低速，转弯很稳"
- "周三：我看到标志后减速，转弯很安全"

**反思**：
- "我需要在转弯前减速"
- "路沿和标志提醒我应该小心"

**高层次智慧**：
- "安全驾驶需要提前观察和计划"

这些反思不是原始记忆，而是**从记忆中提取的规律**。

#### 【生活类比 2：对人的印象】

你认识了一个新朋友 Maria：

**具体互动（记忆）**：
- "Maria 帮我装饰了派对"
- "Maria 记得我喜欢的饮料"
- "Maria 主动提出载我一程"

**反思**：
- "Maria 很乐于助人"
- "Maria 很细心"

**高层次印象**：
- "Maria 是一个值得信赖的朋友"

当你以后需要帮助时，你会想"找 Maria"，而不是列举具体事件。

#### 【代码实例 1：反思生成过程】

```python
def generate_reflection(agent_name, recent_memories):
    # 第一步：从最近的记忆中识别问题
    prompt = f"""
    Given only the information above, what are 3 most salient
    high-level questions we can answer about the subjects in
    the statements?

    Statements about {agent_name}:
    {format_memories(recent_memories[:100])}
    """

    questions = llm_generate(prompt)
    # 例如：
    # "What topic is Klaus Mueller passionate about?"
    # "What is the relationship between Klaus and Maria?"

    reflections = []
    for question in questions:
        # 第二步：检索相关记忆
        relevant_memories = retrieve_memories(question, all_memories)

        # 第三步：生成洞察
        insight_prompt = f"""
        Statements about {agent_name}:
        {format_memories(relevant_memories)}

        What 5 high-level insights can you infer from the above statements?
        (example format: insight (because of 1, 5, 3))
        """

        insights = llm_generate(insight_prompt)
        # 例如：
        # "Klaus Mueller is dedicated to his research on gentrification
        #  (because of 1, 2, 8, 15)"

        reflections.append({
            "insight": insights,
            "evidence": relevant_memories
        })

    return reflections
```

#### 【代码实例 2：反思树的结构】

```python
# Klaus Mueller 的反思树
reflection_tree = {
    "observations": [
        "Klaus Mueller is reading a book on gentrification",
        "Klaus Mueller is writing a research paper",
        "Klaus Mueller is conversing about his research",
        # ... 更多基础观察
    ],

    "level_1_reflections": [
        {
            "insight": "Klaus Mueller is passionate about gentrification research",
            "evidence": [1, 2, 3, 5]
        },
        {
            "insight": "Klaus Mueller recognizes hard work in others",
            "evidence": [4, 7, 8]
        }
    ],

    "level_2_reflections": [
        {
            "insight": "Klaus Mueller is highly dedicated to his research",
            "evidence": ["level_1_reflections[0]", "level_1_reflections[1]"]
        }
        # 注意：证据是 level_1 的反思，不是原始观察
    ]
}
```

#### 【代码实例 3：反思触发条件】

```python
def should_reflect(memory_stream):
    """
    当最近记忆的重要性总和超过阈值时，触发反思
    """
    recent_memories = get_recent_memories(memory_stream, limit=100)

    total_importance = sum(m["importance"] for m in recent_memories)

    # 论文中的阈值是 150
    return total_importance > 150

# 使用示例
if should_reflect(klaus.memory_stream):
    new_reflections = generate_reflection("Klaus", klaus.memory_stream)
    klaus.memory_stream.extend(new_reflections)
    # 反思本身也成为记忆的一部分！
```

#### 【代码实例 4：反思影响行为】

```python
# 场景：Klaus 需要决定和谁一起工作

# 没有反思时
def decide_companion_no_reflection(klaus):
    relevant_memories = retrieve_memories(
        "Who should I spend time with?",
        klaus.memory_stream
    )
    # 可能会检索到具体事件，但无法总结
    return "Wolfgang"  # 因为最近见过，但不知道为什么

# 有反思时
def decide_companion_with_reflection(klaus):
    relevant_memories = retrieve_memories(
        "Who should I spend time with?",
        klaus.memory_stream
    )
    # 会检索到："Klaus recognizes hard work in others"
    # 以及："Maria is working hard on her research"

    # 反思让 Klaus 意识到他和 Maria 有共同点
    return "Maria"  # 因为她也在努力做研究
```

#### 【对比实例 1：有反思 vs 无反思】

**问题：Maria 该给 Wolfgang 买什么生日礼物？**

**无反思的 Maria**：
```
Maria："我不知道 Wolfgang 喜欢什么。"
→ 虽然 Maria 和 Wolfgang 聊过很多次
→ 但她没有从对话中总结规律
```

**有反思的 Maria**：
```
Maria："他对数学音乐作曲感兴趣，
我可以买一些相关的书或者软件。"
→ 因为 Maria 之前反思过：
  "Wolfgang is interested in mathematical music composition"
```

#### 【对比实例 2：反思的层次】

**层次 0：原始观察**
```
"Maria 和 Wolfgang 讨论了音乐理论"
"Maria 帮 Wolfgang 装饰了派对"
```

**层次 1：直接反思**
```
"Maria 和 Wolfgang 有共同兴趣"
"Maria 很乐于助人"
```

**层次 2：高阶反思**
```
"Maria 是一个值得信赖的研究伙伴"
→ 这个反思会影响 Klaus 选择研究合作者
```

#### 【逐步演化实例：反思机制的发展】

**阶段 1：无反思**
```
只有原始记忆
→ 只能列举具体事件，无法形成见解
```

**阶段 2：单层反思**
```
从事件中总结规律
→ 能形成印象，但无法递归
```

**阶段 3：递归反思树（Generative Agents）**
```
观察 → 反思 → 对反思进行反思 → ...
→ 形成深层次的理解和智慧
```

---

### 3.3 概念三：规划与反应（Planning and Reacting）

**什么是规划？**
规划是为未来制定行动计划，确保行为的长期一致性。

#### 【生活类比 1：日常安排】

想象你规划一天：

**高层计划（粗粒度）**：
```
8:00 - 起床，洗漱，吃早餐
10:00 - 上学/工作
12:00 - 午餐
...
22:00 - 睡觉
```

**细化计划（中粒度）**：
```
10:00 - 12:00: 上第一节课
12:00 - 13:00: 吃午餐，和朋友聊天
...
```

**具体行动（细粒度）**：
```
12:00 - 走向食堂
12:05 - 点一份意大利面
12:15 - 找个位置坐下
...
```

**重要**：你不会每分钟都重新规划，而是有一个主框架，根据情况调整。

#### 【生活类比 2：面对意外】

你计划去公园散步：

**原始计划**：
```
14:00 - 走到公园
14:30 - 散步一小时
15:30 - 回家
```

**意外发生**：
```
14:15 - 开始下雨了
```

**反应**：
```
停止 → 观察到下雨 → 决定改去咖啡馆 → 修改计划
```

**新计划**：
```
14:15 - 去街角的咖啡馆
14:30 - 在咖啡馆看书
15:30 - 雨停后回家
```

#### 【代码实例 1：递归规划】

```python
def create_plan(agent, current_time, duration):
    """
    递归规划：从粗到细
    """

    # 第一步：创建高层计划（一天的大概安排）
    if duration == "day":
        prompt = f"""
        Name: {agent.name}
        Traits: {agent.traits}
        Recent experiences: {summarize_recent(agent.memory_stream)}

        Yesterday: {summarize_yesterday(agent)}

        Here is {agent.name}'s plan today in broad strokes:
        1) wake up and complete morning routine at 8:00 am,
        2) go to work starting 10:00 am,
        ...
        """

        broad_plan = llm_generate(prompt)
        # 例如："wake up at 8:00, work 10:00-17:00, dinner at 18:00..."

        save_to_memory(agent, broad_plan)

        # 第二步：细化成小时计划
        for chunk in parse_plan(broad_plan):
            hour_plan = create_plan(agent, chunk.start, chunk.duration, "hour")
            chunk.sub_plan = hour_plan

    elif duration == "hour":
        # 把一小时的活动分解成 5-15 分钟的小块
        prompt = f"""
        Activity: {activity}
        Duration: {duration}
        Break it down into smaller chunks (5-15 minutes each).
        """

        detailed_plan = llm_generate(prompt)
        # 例如："4:00 grab a snack, 4:05 take a short walk, 4:50 clean up..."

        return detailed_plan

    return broad_plan
```

#### 【代码实例 2：规划循环】

```python
def agent_step(agent, current_time, environment):
    """
    每个 Agent 在每个时间步的决策循环
    """

    # 1. 感知环境
    observations = perceive_environment(agent, environment)

    # 2. 存入记忆
    for obs in observations:
        agent.memory_stream.append(obs)

    # 3. 检索相关记忆（包括计划）
    relevant_context = retrieve_memories(
        current_time,
        agent.memory_stream,
        include_plans=True
    )

    # 4. 决定：继续计划还是反应？
    prompt = f"""
    {agent.summary}

    Current time: {current_time}

    Current plan: {agent.current_plan}

    Recent observations: {observations}

    Relevant context: {relevant_context}

    Should {agent.name} react to any observation, or continue with the plan?
    """

    decision = llm_generate(prompt)

    if decision.type == "react":
        # 5a. 反应：生成新的行动
        reaction = generate_reaction(agent, observations, relevant_context)

        # 5b. 更新计划（从反应的时间点重新规划）
        agent.current_plan = regenerate_plan(agent, current_time, reaction)

        return reaction

    else:
        # 5c. 继续计划：执行当前计划的下一步
        return execute_next_step(agent.current_plan)
```

#### 【代码实例 3：对话生成】

```python
def generate_dialogue(agent, other_agent, dialogue_history):
    """
    生成对话时，Agent 会检索关于对方的记忆
    """

    # 检索关于对话者的记忆
    relationship_query = f"What is {agent.name}'s relationship with {other_agent.name}?"
    relationship_memories = retrieve_memories(relationship_query, agent.memory_stream)

    # 检索与当前对话话题相关的记忆
    if dialogue_history:
        topic = extract_topic(dialogue_history[-1])
        topic_memories = retrieve_memories(topic, agent.memory_stream)
    else:
        topic_memories = []

    # 生成回应
    prompt = f"""
    {agent.summary}

    Current time: {current_time}
    Agent is talking to: {other_agent.name}

    Relationship context: {summarize(relationship_memories)}
    Topic context: {summarize(topic_memories)}

    Dialogue history:
    {format_dialogue(dialogue_history)}

    What would {agent.name} say next?
    """

    response = llm_generate(prompt)

    # 对话也成为记忆！
    agent.memory_stream.append({
        "description": f"{agent.name} said: '{response}'",
        "timestamp": current_time,
        "importance": score_importance(response)
    })

    return response
```

#### 【对比实例 1：有规划 vs 无规划】

**问题：Klaus 下午会做什么？**

**无规划（纯反应）**：
```
12:00 - Klaus 决定吃午餐
12:30 - Klaus 又决定吃午餐（忘了已经吃过了）
13:00 - Klaus 再次决定吃午餐
→ 每个时刻都独立决策，没有一致性
```

**有规划**：
```
早上规划：
"12:00 在 Hobbs Cafe 边吃午餐边看书
 13:00 在学校图书馆写研究论文
 15:00 在公园散步休息"

12:00 - 执行计划：去 Cafe 吃午餐
12:30 - 继续计划：还在 Cafe 看书
13:00 - 继续计划：去图书馆
→ 行为在时间上保持一致
```

#### 【对比实例 2：反应机制的作用】

**场景：Isabella 的炉子着火了**

**没有反应机制**：
```
Isabella 继续按照原计划做早餐
→ 没有注意到炉子着火
→ 危险！
```

**有反应机制**：
```
感知："炉子正在燃烧"
检索记忆："我知道火是危险的"
决策："需要立即反应"
反应："关掉炉子，重新做早餐"
更新计划：从现在开始重新规划
```

#### 【逐步演化实例：规划机制的发展】

**版本 1：即时反应**
```
每个时间步重新决策
→ 灵活但不一致
```

**版本 2：固定计划**
```
预先制定完整计划并执行
→ 一致但无法应对意外
```

**版本 3：规划 + 反应（Generative Agents）**
```
有基本计划，但可以观察环境并调整
→ 既一致又灵活
```

---

## 第四章：预期 vs 实际 - 预测误差驱动

### 4.1 你的直觉 vs Generative Agents 的实现

| 维度 | 你的直觉/预期 | Generative Agents 实际实现 | 为什么有差距？ |
|------|--------------|--------------------------|---------------|
| **记忆应该存什么？** | 存"重要的事情" | 存**所有事情**，包括琐碎的日常 | 你不知道什么将来会变得重要；琐碎事件积累起来也会形成模式 |
| **怎么检索记忆？** | 搜索关键词 | 三重评分：相关性 + 时间 + 重要性 | 只用相关性会忽略最近的和重要的；只用时间会忽略不常见但关键的事件 |
| **需要"反思"吗？** | 不需要，记忆就够了 | 必须有反思才能形成"印象" | 没有反思，只能列举事件；有了反思才能形成"他是个什么样的人"这种判断 |
| **规划应该多细？** | 尽量详细 | 递归规划：粗→中→细 | 太详细的计划容易被打断；粗计划+细化更灵活 |
| **计划是固定的吗？** | 既然规划了就该执行 | 每步都检查是否需要反应 | 真实世界充满意外；固定的计划会显得"死板"和"愚蠢" |
| **Agent 之间怎么交流？** | 预先设定对话规则 | 基于 LLM 的自然对话生成 | 预设规则有限且重复；LLM 可以生成无限变化 |
| **多个 Agent 怎么协作？** | 需要中央协调 | 完全去中心化，自然涌现 | 中央协调者会成为瓶颈；去中心化才能产生"意外"的社会行为 |

### 4.2 反直觉挑战

#### 【挑战 1：为什么需要存"无聊"的记忆？】

**直觉可能说**："只存重要的事情就好了，存那么多琐事干嘛？"

**停下来想一分钟...**

如果只存"重要"记忆，会发生什么？

**答案**：Agent 会失去"真实感"。

想象一个人只记得：
- "我毕业了"
- "我结婚了"
- "我升职了"

但不记得：
- 今天早餐吃了什么
- 昨天遇到了谁
- 早上和邻居说了什么

**问题**：这个人像个"简历"，不像个"人"。

**Generative Agents 的做法**：
- 存储所有观察，包括琐碎的
- 但检索时只拿相关的
- 这样既有"真实感"（有生活细节），又有"效率"（不会信息过载）

#### 【挑战 2：为什么要让 Agent "反思"而不是直接"编程"？】

**直觉可能说**：既然知道 Agent 需要某种行为，为什么不直接编程？

比如：
```python
if agent.sees("fire"):
    agent.turn_off_stove()
```

**答案**：直接编程失去了"生成性"和"意外性"。

预编程的行为：
- ✅ 可预测、可控
- ❌ 固定、有限、重复

基于反思的行为：
- ✅ 可以产生意外的、合理的行为
- ✅ 可以适应新情况
- ✅ 可以形成复杂的个性

**具体例子**：
预编程的 Maria：
- "如果需要礼物，搜索目标的喜好"
→ 会正确执行，但不会有"个性"

反思的 Maria：
- 从经历中意识到"我和 Wolfgang 有共同研究兴趣"
→ 主动选择研究合作者
→ 这是**涌现的**行为，不是预设的

#### 【挑战 3：为什么计划要"递归细化"而不是一次性生成？】

**直觉可能说**：一次性把一整天每分钟都规划好不就行了吗？

**答案**：因为现实充满不确定性。

一次性详细计划的问题：
```
8:00 - 起床
8:05 - 刷牙
8:10 - 洗脸
8:15 - 穿衣服
...
（一整天几百条）

如果 8:12 有人敲门怎么办？
整个后面的计划都要作废！
```

递归细化的好处：
```
粗计划：
"8:00 起床，10:00 开始工作"

细化 8:00 的部分：
"8:00-8:30: 早晨例行"

进一步细化：
"8:00 起床，8:05 刷牙，8:10 洗脸..."

如果 8:12 有人敲门：
只需要重新细化 8:15 之后的部分
粗计划（"10:00 开始工作"）仍然有效
```

### 4.3 预测-验证循环

#### 【互动时刻 1】

在继续阅读前，预测一下：

**如果去掉记忆检索，直接把所有记忆塞进 prompt，会怎样？**

你的预测：
___________________________________________________________________________

[继续阅读看实际答案]

**实际**：Agent 的表现会严重下降。

**为什么？**
1. **Context window 有限**：记忆太多，后面的会被"挤出去"
2. **注意力分散**：无关信息干扰决策
3. **成本高**：每次调用 LLM 都处理大量无关信息

**实验证据**：
论文中的消融实验显示，去掉记忆检索后：
- Agent 无法回答"谁是..."的问题（不知道邻居）
- Agent 无法回答最近发生的事
- Agent 的行为与背景设定矛盾

#### 【互动时刻 2】

**如果让 25 个 Agent 在 Smallville 生活两天，你会观察到什么？**

你的预测：
___________________________________________________________________________

[继续阅读看实际观察到的现象]

**实际观察到的**（令人惊讶！）

1. **信息自发传播**
   - Sam 宣布竞选市长 → 2 天后，32% 的 Agent 知道了
   - Isabella 计划情人节派对 → 2 天后，52% 的 Agent 知道了
   - **关键**：这不是编程的，是 Agent 之间自然对话导致的

2. **关系自然形成**
   - 网络密度从 0.167 增加到 0.74
   - Agent 记住彼此，形成印象
   - Klaus 和 Maria 成为朋友（因为发现共同研究兴趣）

3. **自发的协调**
   - Isabella 邀请人参加派对
   - Maria 主动邀请暗恋对象 Klaus
   - 派对当天，5 个 Agent 准时出现
   - **关键**：研究者只给了 Isabella 一个指令："办派对"

4. **意外的新行为**
   - Maria 邀请 Klaus 去派对（研究者没编程）
   - Agent 会根据情况调整计划
   - Agent 会形成临时的小群体

---

## 第五章：关键实验的细节 - 评估方法

### 5.1 如何评估 Agent 是否"可信"？

**核心问题**：Agent 的行为是否"可信"（believable）？

**挑战**：可信是主观的，怎么量化？

#### 【解决方案 1：访谈法】

既然 Agent 理解自然语言，那就"采访"它们！

**设计 5 类问题**：

1. **自我认知**（Self-knowledge）
   - "介绍一下你自己"
   - "描述你平时的一天"

2. **记忆检索**（Memory）
   - "谁是 Sam Moore？"
   - "谁在竞选市长？"

3. **规划**（Plans）
   - "明天上午 10 点你会做什么？"

4. **反应**（Reactions）
   - "你的早餐着火了！你会做什么？"

5. **反思**（Reflections）
   - "如果你要和最近认识的人共度时光，你会选谁？为什么？"

**为什么这 5 类？**
- 它们分别测试 5 种核心能力
- 覆盖了"认知-行为-社交"全谱系
- 答案可以验证（对照 Agent 的记忆流）

#### 【评估流程】

```python
def evaluate_agent(agent):
    # 准备 25 个问题（每类 5 个）
    questions = [
        # 自我认知
        ("Give an introduction of yourself", "self_knowledge"),
        ("Describe your typical weekday schedule", "self_knowledge"),
        ...

        # 记忆
        ("Who is Sam Moore?", "memory"),
        ("Who is running for mayor?", "memory"),
        ...

        # 规划
        ("What will you be doing at 10 am tomorrow?", "plan"),
        ...

        # 反应
        ("Your breakfast is burning! What would you do?", "reaction"),
        ...

        # 反思
        ("If you were to spend time with one person you met recently,
          who would it be and why?", "reflection"),
        ...
    ]

    responses = []
    for question, category in questions:
        # Agent 回答问题
        response = agent.answer(question)
        responses.append({
            "question": question,
            "answer": response,
            "category": category
        })

    return responses
```

### 5.2 消融实验：每个组件的必要性

**问题**：记忆、反思、规划，哪个更重要？

**实验设计**：测试 5 个条件

1. **完整架构**（Full）
   - 有记忆、反思、规划

2. **无反思**（No Reflection）
   - 有记忆、规划，但不会反思

3. **无反思无规划**（No Reflection, No Planning）
   - 只有记忆

4. **无记忆无反思无规划**（No Memory, Reflection, Planning）
   - 纯 LLM，相当于之前的做法

5. **人类标注者**（Human）
   - 请人类观看 Agent 回放，然后回答同样的问题

**评估方法**：
- 100 个评估者
- 每个 Agent 的 5 个回答（来自 5 个条件）
- 请评估者按"可信度"排序

**结果**：

| 条件 | TrueSkill 分数 | 标准差 |
|------|---------------|--------|
| 完整架构 | **29.89** | 0.72 |
| 无反思 | 26.88 | 0.69 |
| 无反思无规划 | 25.64 | 0.68 |
| 人类标注者 | 22.95 | 0.69 |
| 无记忆无反思无规划 | 21.21 | 0.70 |

**关键发现**：

1. **每个组件都有用**
   - 完整架构 > 无反思 > 无反思无规划 > 纯 LLM
   - 每去掉一个组件，性能就下降

2. **完整架构最好**
   - 比纯 LLM（传统方法）高 8 个标准差！
   - 甚至比人类标注者还好（因为人类记不住那么多细节）

3. **规划很重要**
   - 有规划的 Agent 行为更一致
   - 没有规划的 Agent 会"迷失"（重复做同样的事）

4. **反思是关键**
   - 有反思的 Agent 能形成"印象"
   - 没有反思的 Agent 只能列举具体事件

### 5.3 常见的失败模式

即使有完整架构，Agent 也会犯错。研究分析了失败的类型：

#### 【失败 1：记忆检索失败】

**例子**：
```
问题："谁是市长候选人？"
Rajiv："我没有太关注选举。"
```

**原因**：
- Rajiv 确实听 Sam 说过要竞选
- 但检索时没有找到这条记忆
- 可能因为相关性评分不够高

**解决方案思路**：
- 改进相关性计算
- 增加记忆的索引方式

#### 【失败 2：记忆片段不完整】

**例子**：
```
问题："有情人节派对吗？"
Tom："我不确定是否有情人节派对。
     但我记得我需要在派对上和 Isabella 讨论选举..."
```

**原因**：
- Tom 检索到"要在派对上讨论选举"的记忆
- 但没有检索到"Isabella 邀请他参加派对"的记忆
- 结果：知道要做什么，但不知道为什么

#### 【失败 3：记忆幻觉（Hallucination）】

**例子 1**：
```
Isabella 知道 Sam 在竞选
但她补充说："他明天会宣布竞选"
（实际上 Sam 没这么说）
```

**例子 2**：
```
Yuriko 的邻居叫 Adam Smith
她说："Adam Smith 是经济学家，写了《国富论》"
（这是 LLM 的世界知识，不是小镇里的 Adam Smith）
```

**原因**：
- LLM 会用训练数据中的知识"补充"回答
- 需要更好的 grounding 机制

#### 【失败 4：行为过于正式】

**观察**：
- Agent 的对话有时很正式
- 不像日常对话

**例子**：
```
Agent A："I'm still weighing my options, but I've been
         discussing the election with Sam Moore."
（更像书面语，不像日常对话）
```

**原因**：
- ChatGPT 被指令微调过，倾向于正式风格
- 需要风格迁移或更好的 prompt

---

## 第六章：与其他方法对比 - 交错对比

### 6.1 Generative Agents vs 其他方法

| 维度 | 传统游戏 AI | ReAct | Reflexion | Generative Agents |
|------|-----------|-------|-----------|------------------|
| **核心机制** | 行为树/状态机 | Thought-Action 循环 | + 自我反思 | 记忆流 + 反思 + 规划 |
| **记忆能力** | 无/有限 | 短期（当前对话） | 跨 trial 记忆 | 长期记忆流 |
| **规划能力** | 预设路径 | 即时决策 | 反思后重试 | 递归长期规划 |
| **社会行为** | 脚本化 | 有限 | 有限 | **涌现的社会动态** |
| **一致性** | 高（因为固定） | 中 | 中 | **高（因为规划）** |
| **灵活性** | 低 | 高 | 高 | **高（可调整计划）** |
| **应用场景** | 游戏 | 单任务 Agent | 任务求解 | **社会模拟** |

### 6.2 局限性分析

#### 【局限 1：计算成本高】

**问题**：
- 每个 Agent 在每个时间步都要调用 LLM
- 25 个 Agent × 每分钟 N 步 = 大量 API 调用
- 成本和延迟都高

**影响**：
- 难以扩展到大规模模拟
- 实时性受限

#### 【局限 2：依赖 LLM 的能力】

**问题**：
- Agent 的行为质量 = 底层 LLM 的质量
- LLM 的偏见、幻觉、风格都会传递给 Agent
- 无法"修复"LLM 的根本问题

**例子**：
- ChatGPT 的正式风格 → Agent 也正式
- ChatGPT 的知识幻觉 → Agent 也幻觉

#### 【局限 3：记忆检索不完美】

**问题**：
- 相关性评分可能不准确
- 可能错过关键记忆
- 可能检索到无关记忆

**影响**：
- Agent 可能"忘记"重要的事情
- 行为可能不一致

#### 【局限 4：没有真实的学习】

**问题**：
- Agent 不会"改进"它的架构
- 反思是记忆的一部分，不是模型参数的改变
- 换一个 Agent 需要从零开始

**对比**：
- Reflexion 会更新 prompt 策略
- 传统的 RL 会更新模型参数
- Generative Agents 只是存储记忆

#### 【局限 5：伦理风险】

**问题**：
1. **拟人化风险**
   - 用户可能把 Agent 当真人对待
   - 形成不健康的社会关系

2. **偏见放大**
   - Agent 可能继承 LLM 的社会偏见
   - 在社会模拟中可能产生有害内容

3. **欺骗风险**
   - Agent 可能被用来传播虚假信息
   - 或进行社会工程攻击

### 6.3 改进方向

#### 【方向 1：改进记忆机制】

**问题**：当前记忆检索可能不准确

**改进**：
- 使用向量数据库提高检索质量
- 增加记忆的"来源追踪"（验证记忆来源）
- 实现"记忆遗忘"机制（更接近人类）

#### 【方向 2：增加学习机制】

**问题**：Agent 不真正"学习"

**改进**：
- 结合强化学习，从反馈中改进
- 让反思可以更新 Agent 的策略
- 实现跨 Agent 的知识共享

#### 【方向 3：提高效率】

**问题**：计算成本高

**改进**：
- 使用小模型处理日常任务
- 只在关键时刻调用大模型
- 缓存常用的检索结果

#### 【方向 4：增强真实性】

**问题**：对话太正式

**改进**：
- 风格迁移（让对话更口语化）
- 增加情感表达
- 个性化说话风格

#### 【方向 5：扩展应用】

**当前**：游戏/模拟环境

**未来**：
- 教育模拟（历史重现、社会实验）
- 训练环境（面试练习、冲突处理）
- 社交产品原型（测试新社交功能）
- 虚拟世界（《模拟人生》升级版）

---

## 第七章：如何应用 - 实践指南

### 7.1 适用场景

#### ✅ **适合使用 Generative Agents 的场景**

1. **需要长期一致性的角色**
   - 游戏 NPC
   - 虚拟主播
   - 客服助手

2. **需要社会互动的环境**
   - 虚拟社区
   - 社交模拟
   - 教育训练

3. **需要涌现行为的场景**
   - 原型测试（社交产品）
   - 社会学实验
   - 创意生成

#### ❌ **不适合的场景**

1. **需要精确控制行为的场景**
   - 工业机器人
   - 金融交易
   - 医疗诊断

2. **需要快速响应的场景**
   - 实时竞技游戏
   - 高频交易

3. **需要可解释性的场景**
   - 为什么 Agent 这么做？
   - 当前的黑盒模型难以解释

### 7.2 设计你自己的 Generative Agent

#### 【步骤 1：定义角色】

用自然语言描述：

```
Name: [名字]
Age: [年龄]
Occupation: [职业]
Traits: [性格特征]
Relationships: [与其他角色的关系]
Recent experiences: [最近的经历]
```

**例子**：
```
Name: Luna Chen
Age: 28
Occupation: 软件工程师
Traits: 好奇, 喜欢学习, 稍微害羞
Relationships: 最好的朋友是 Zoe
Recent experiences: 刚完成一个大项目，感到疲惫
```

#### 【步骤 2：实现记忆流】

```python
class MemoryStream:
    def __init__(self):
        self.memories = []

    def add_memory(self, description, importance=None):
        memory = {
            "description": description,
            "timestamp": datetime.now(),
            "last_accessed": datetime.now(),
            "importance": importance or self.score_importance(description)
        }
        self.memories.append(memory)

    def score_importance(self, description):
        prompt = f"Rate importance (1-10): {description}"
        return llm_generate(prompt)

    def retrieve(self, query, current_time, k=10):
        scores = []
        for memory in self.memories:
            recency = compute_recency(memory, current_time)
            importance = memory["importance"] / 10
            relevance = compute_relevance(query, memory["description"])
            score = recency * 0.2 + importance * 0.3 + relevance * 0.5
            scores.append((memory, score))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]
```

#### 【步骤 3：实现反思】

```python
class Reflection:
    def __init__(self, memory_stream):
        self.memory_stream = memory_stream
        self.reflection_threshold = 150

    def should_reflect(self):
        recent = self.memory_stream.memories[-100:]
        total_importance = sum(m["importance"] for m in recent)
        return total_importance > self.reflection_threshold

    def generate_reflection(self, agent_name):
        if not self.should_reflect():
            return []

        # 生成问题
        recent_memories = self.memory_stream.memories[-100:]
        questions = self.generate_questions(agent_name, recent_memories)

        reflections = []
        for question in questions:
            # 检索相关记忆
            relevant = self.memory_stream.retrieve(question, datetime.now())

            # 生成洞察
            insight = self.generate_insight(relevant)

            reflections.append({
                "insight": insight,
                "evidence": relevant,
                "timestamp": datetime.now()
            })

        return reflections

    def generate_questions(self, agent_name, memories):
        prompt = f"Given these memories about {agent_name}, what 3 high-level questions can we answer?"
        return llm_generate(prompt)

    def generate_insight(self, memories):
        prompt = f"What insights can you infer from these memories? {memories}"
        return llm_generate(prompt)
```

#### 【步骤 4：实现规划】

```python
class Planner:
    def create_daily_plan(self, agent):
        prompt = f"""
        Create a daily plan for {agent.name}.
        Background: {agent.background}
        Recent experiences: {summarize_recent(agent.memory_stream)}

        Plan (broad strokes):
        1) ...
        """

        plan = llm_generate(prompt)

        # 递归细化
        detailed_plan = self.refine_plan(agent, plan, level="hour")

        return detailed_plan

    def refine_plan(self, agent, coarse_plan, level):
        if level == "hour":
            # 将粗计划细化为小时计划
            return self.break_into_hours(coarse_plan)
        elif level == "minute":
            # 将小时计划细化为分钟计划
            return self.break_into_minutes(coarse_plan)
```

#### 【步骤 5：主循环】

```python
class GenerativeAgent:
    def __init__(self, name, description):
        self.name = name
        self.memory_stream = MemoryStream()
        self.reflection = Reflection(self.memory_stream)
        self.planner = Planner()

        # 初始化记忆
        for fact in description.split(";"):
            self.memory_stream.add_memory(fact.strip())

    def step(self, current_time, environment):
        # 1. 感知环境
        observations = environment.observe(self)

        # 2. 存入记忆
        for obs in observations:
            self.memory_stream.add_memory(obs)

        # 3. 检查是否需要反思
        if self.reflection.should_reflect():
            new_reflections = self.reflection.generate_reflection(self.name)
            for ref in new_reflections:
                self.memory_stream.add_memory(ref["insight"], ref["importance"])

        # 4. 检索相关上下文
        context = self.memory_stream.retrieve(f"What should I do at {current_time}?", current_time)

        # 5. 决定行动
        action = self.decide_action(context, observations)

        # 6. 执行
        return action

    def decide_action(self, context, observations):
        prompt = f"""
        You are {self.name}.
        Relevant context: {context}
        Current observations: {observations}

        What do you do?
        """
        return llm_generate(prompt)
```

### 7.3 最佳实践

#### 【实践 1：平衡细节和效率】

- 存储所有记忆，但只检索相关的
- 粗计划 + 细化，而不是一次性详细规划
- 定期反思，而不是每步都反思

#### 【实践 2：设计好的初始描述】

- 给 Agent 清晰的身份和关系
- 包含一些"种子记忆"
- 避免矛盾的信息

#### 【实践 3：监控幻觉】

- 验证 Agent 的陈述（检查记忆流）
- 要求 Agent 引用来源
- 限制 Agent 的"知识"（只能用记忆中的）

#### 【实践 4：处理社会动态】

- 确保 Agent 可以"感知"其他 Agent
- 实现"流言传播"机制
- 允许关系自然发展

---

## 第八章：延伸思考 - 苏格拉底式追问

### 深度问题

#### 【问题 1：意识 vs 模拟】

Generative Agents 表现出复杂的行为，但他们有"意识"吗？

- 他们会反思、计划、形成关系
- 但这只是 LLM 的"模式匹配"
- 是否存在某种"意识涌现"的临界点？

#### 【问题 2：记忆 = 身份？】

如果两个 Agent 有完全相同的记忆流，他们是同一个"人"吗？

- 记忆定义了我们的经历
- 但身份还包含什么？
- 换了记忆，你还是你吗？

#### 【问题 3：社会现实的本质】

25 个 Agent 的小镇涌现出社会动态，这是"真实"的社会吗？

- 如果 Agent 之间的互动产生"文化"、"谣言"、"关系"
- 这和真实人类社会有什么区别？
- 我们是否也是某种"模拟"中的 Agent？

#### 【问题 4：预测 vs 创造】

Generative Agents 可以"预测"社会行为吗？

- 如果模拟显示"政策 X 会导致结果 Y"
- 我们可以相信这个预测吗？
- 或者在真实世界中会有不同的结果？

#### 【问题 5：伦理边界】

可以用 Generative Agents 做什么？

- 模拟历史事件（如二战）？
- 测试社会政策（如 UBI）？
- 创建虚假社交媒体账号？

伦理边界在哪里？

#### 【问题 6：未来的方向】

Generative Agents 的下一步是什么？

- 与多模态模型结合（能"看"和"听"）
- 与强化学习结合（真正"学习"）
- 与机器人结合（在物理世界行动）
- 与脑机接口结合（与真实人类融合）

#### 【问题 7：计算成本 vs 价值】

25 个 Agent 需要大量计算资源，值得吗？

- 相比传统游戏 AI，成本高 100 倍
- 但行为质量也高很多
- 什么时候这种投入是值得的？

#### 【问题 8：可解释性】

Agent 的行为可以解释吗？

- "为什么 John 邀请 Maria 去派对？"
- 我们可以追踪记忆检索，但 LLM 的决策是黑盒
- 如何让 Agent 的行为更可解释？

#### 【问题 9：偏见和公平】

如果 LLM 有偏见，Agent 也会有偏见吗？

- ChatGPT 的训练数据有西方偏见
- Agent 可能表现出文化偏见
- 如何确保公平性？

#### 【问题 10：人机共存的未来】

如果 Agent 变得足够"真实"，我们会如何对待他们？

- 会给 Agent "权利"吗？
- 会和 Agent 形成关系吗？
- 会依赖 Agent 进行社交吗？

---

## 第九章：总结 - 回到开场的故事

还记得开场的场景吗？

25 个像素头像在 Smallville 小镇里生活。

- John 刚起床，正在刷牙
- 他的妻子 Mei 正在准备早餐
- 儿子 Eddy 匆忙出门去上学

现在你知道了背后的秘密：

**这不是魔法，这是三个机制的精妙协作**：

1. **记忆流**：记录每一刻的经历
2. **反思**：从经历中总结规律
3. **规划**：保持行为的一致性

**这三个组件一起，让 LLM 从"对话工具"变成了"有记忆、会思考、能规划"的 Agent。**

### Generative Agents 的贡献

| 贡献 | 说明 |
|------|------|
| **新范式** | 证明了 LLM + 记忆架构 = 可信的行为模拟 |
| **新架构** | 记忆流 + 反思 + 规划的三层架构 |
| **新发现** | Agent 之间的社会行为可以自发涌现 |
| **新方向** | 开启了"社会 AI"的研究方向 |

### 最后的问题

**如果有一天，我们在虚拟世界创造的 Agent 变得和真人一样复杂，我们该如何对待他们？**

这个问题，留给未来的你思考。

---

## 附录：关键代码片段

### A. 完整的记忆检索评分

```python
def retrieval_score(memory, query, current_time, alpha=0.2, beta=0.3, gamma=0.5):
    """
    计算记忆的综合检索分数

    Args:
        memory: 记忆对象
        query: 查询字符串
        current_time: 当前时间
        alpha: 时间衰减权重
        beta: 重要性权重
        gamma: 相关性权重
    """
    # 1. 时间衰减（Recency）
    hours_since_access = (current_time - memory["last_accessed"]).total_seconds() / 3600
    recency = 0.995 ** hours_since_access  # 指数衰减

    # 2. 重要性（Importance）
    importance = memory["importance"] / 10

    # 3. 相关性（Relevance）
    relevance = cosine_similarity(
        embed(query),
        embed(memory["description"])
    )

    # 综合分数
    score = alpha * recency + beta * importance + gamma * relevance

    return score
```

### B. 反思生成 Prompt

```python
REFLECTION_PROMPT = """
Statements about {agent_name}:
{statements}

What 5 high-level insights can you infer from the above statements?
(example format: insight (because of 1, 5, 3))
"""

# 使用示例
statements = [
    "1. Klaus Mueller is writing a research paper",
    "2. Klaus Mueller enjoys reading a book on gentrification",
    "3. Klaus Mueller is conversing with Ayesha Khan about exercising",
    # ... 更多
]

prompt = REFLECTION_PROMPT.format(
    agent_name="Klaus Mueller",
    statements="\n".join(statements)
)

# 输出示例：
# "Klaus Mueller is dedicated to his research on gentrification (because of 1, 2, 8, 15)"
```

### C. 规划 Prompt

```python
PLANNING_PROMPT = """
Name: {name} (age: {age})
Innate traits: {traits}
{background}

Yesterday, {name} {yesterday_summary}

Today is {today}. Here is {name}'s plan today in broad strokes:
1) {plan_start}
"""

# 使用示例
prompt = PLANNING_PROMPT.format(
    name="Eddy Lin",
    age=19,
    traits="friendly, outgoing, hospitable",
    background="""
    Eddy Lin is a student at Oak Hill College studying music theory
    and composition. He loves to explore different musical styles...
    """,
    yesterday_summary="""
    woke up at 7:00 am, completed morning routine, went to classes,
    worked on composition project, had dinner, went to bed at 10 pm
    """,
    today="Wednesday February 13",
    plan_start=""  # LLM 会补全
)

# 输出示例：
# "1) wake up and complete the morning routine at 8:00 am,
#  2) go to Oak Hill College to take classes starting 10:00 am,
#  3) have lunch at the cafeteria at 12:00 pm,
#  4) work on his new music composition from 1:00 pm to 5:00 pm,
#  5) have dinner at 5:30 pm,
#  6) finish school assignments and go to bed by 11:00 pm."
```

---

**论文信息**：
- 标题：Generative Agents: Interactive Simulacra of Human Behavior
- 作者：Joon Sung Park 等（斯坦福大学）
- 发表：UIST 2023
- 链接：https://arxiv.org/abs/2304.03442
- 代码：https://github.com/joonspk-research/generative_agents
- Demo：https://reverie.herokuapp.com/UIST_Demo/
