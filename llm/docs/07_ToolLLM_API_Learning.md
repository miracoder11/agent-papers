# ToolLLM: 让开源 LLM 掌握 16000+ 真实世界 API

## 层 1：电梯演讲（30 秒）

**一句话概括**：针对开源 LLM（如 LLaMA）工具使用能力远落后于 ChatGPT 的问题，清华大学团队构建了包含 16,464 个真实 API 的 ToolBench 数据集，并提出 DFSDT（深度优先搜索决策树）推理算法，使 ToolLLaMA 在工具使用任务上达到与 ChatGPT 相当的水平。

---

## 层 2：故事摘要（5 分钟读完）

**核心问题**：2023 年，开源 LLM（LLaMA、Vicuna、Alpaca）在对话任务上表现出色，但在工具使用（调用 API 完成复杂任务）方面几乎为零——而 ChatGPT 已经能熟练调用各种工具。

**关键洞察**：现有工具学习数据集有三大局限：(1) API 数量太少（<100 个），(2) 只有单工具场景，(3) 使用 CoT/ReAct 推理能力不足。团队决定构建一个大规模、多工具、复杂推理的数据集。

**解决方案**：ToolBench——三阶段自动构建：
1. **API 收集**：从 RapidAPI 爬取 16,464 个真实 API，覆盖 49 个类别
2. **指令生成**：用 ChatGPT 生成单工具/多工具指令，共 20 万+ 样本
3. **解法标注**：提出 DFSDT 算法，让 ChatGPT 搜索有效解法路径

**验证结果**：ToolLLaMA（LLaMA-2 7B 微调）在 ToolBench 上 pass rate 66.7%，win rate 60.0%，超越 Claude-2 和 Text-Davinci-003，与 ChatGPT（64.8% pass）相当；在 OOD 数据集 APIBench 上超越专门训练的 Gorilla。

---

## 框架大图

```
┌─────────────────────────────────────────────────────────────┐
│                    问题空间                                  │
│  ┌───────────────────────────────────────────────────┐      │
│  │  开源 LLM 工具使用困境                              │      │
│  │  - Vicuna/Alpaca pass rate = 0%                   │      │
│  │  - 只会对话，不会调用 API 完成任务                  │      │
│  │  - ChatGPT 已能熟练使用工具，但闭源                 │      │
│  └───────────────────────┬───────────────────────────┘      │
│                          │                                  │
│         ┌────────────────▼────────────────┐                 │
│         │  现有数据集局限                  │                 │
│         │  - API 数量少 (<100)             │                 │
│         │  - 只有单工具场景                │                 │
│         │  - ReAct 推理能力不足            │                 │
│         └────────────────┬────────────────┘                 │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           ▼
               ┌─────────────────────────┐
               │   ToolLLM 核心洞察       │
               │                         │
               │  "用 ChatGPT 构建数据，   │
               │   训练出媲美 ChatGPT 的   │
               │   开源模型"              │
               │                         │
               │  关键策略：             │
               │  - 大规模真实 API        │
               │  - 多工具复杂场景        │
               │  - DFSDT 增强推理        │
               └────────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │   ToolBench 三阶段构建   │
               │  ┌───────────────────┐  │
               │  │ 1. API 收集        │  │
               │  │ 16,464 个 API      │  │
               │  │ 49 个类别          │  │
               │  └───────────────────┘  │
               │  ┌───────────────────┐  │
               │  │ 2. 指令生成        │  │
               │  │ 单工具 + 多工具    │  │
               │  │ 20 万+ 样本        │  │
               │  └───────────────────┘  │
               │  ┌───────────────────┐  │
               │  │ 3. 解法标注        │  │
               │  │ DFSDT 算法        │  │
               │  │ 126,486 解法路径   │  │
               │  └───────────────────┘  │
               └────────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │      关键创新：DFSDT     │
               │  ┌───────────────────┐  │
               │  │ vs ReAct          │  │
               │  │ - 支持回退        │  │
               │  │ - 多路径探索      │  │
               │  │ - 63.8% vs 35.3%  │  │
               │  └───────────────────┘  │
               └────────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │        验证结果          │
               │  ToolLLaMA (LLaMA-7B):  │
               │  - Pass Rate: 66.7%     │
               │  - Win Rate: 60.0%      │
               │  - vs ChatGPT: 相当     │
               │  - vs Claude-2: 超越    │
               │                         │
               │  OOD 泛化 (APIBench):   │
               │  - 超越 Gorilla         │
               │  - 无需训练直接迁移     │
               └─────────────────────────┘
```

---

## 层 3：深度精读

### 开场：一个失败的实验

2023 年夏天，清华大学计算机系的实验室里，博士生秦雨佳盯着一组令人沮丧的数据。

她和团队正在测试当时最火的开源 LLM——Vicuna 和 Alpaca。这些模型在对话任务上表现出色，能写诗、能编程、能聊天。但当一个简单的工具使用任务摆在他们面前时，这些模型集体"哑火"了。

**任务**："帮我查一下北京明天的天气，然后推荐附近适合周末去的餐厅。"

**Vicuna 输出**："我很乐意帮助你！北京明天..." 然后开始胡编乱造天气信息。

**Alpaca 输出**：直接拒绝，或者假装自己已经"查询"了。

"pass rate 是 0%，"秦雨佳对同事说，"这两个模型完全不会调用 API。"

与此同时，ChatGPT 已经能熟练调用各种工具——搜索 API、计算器、代码解释器。但 ChatGPT 是闭源的，社区无法在其基础上进行改进和创新。

"为什么开源模型在工具使用上这么落后？"团队成员问。

答案很简单：**现有的指令微调数据几乎都集中在基础语言任务上，完全忽略了工具使用领域**。

即使有少量工具学习数据集，也存在三大问题：
1. API 数量太少（几十个），远不足以让模型学会泛化
2. 只有单工具场景，而真实任务往往需要多工具协作
3. 推理方法停留在 CoT/ReAct，复杂任务容易失败

"我们要不，"秦雨佳说，"自己构建一个大规模的工具学习数据集，然后用它训练一个开源模型，看看能不能追上 ChatGPT？"

这个想法，最终诞生了 ToolLLM。

---

### 第一章：研究者的困境

#### 2023 年工具学习的 Landscape

在 ToolLLM 之前，已有一些工具学习数据集和框架：

| 数据集 | API 数量 | 多工具 | 真实 API 调用 | 推理步数 |
|--------|----------|--------|-------------|----------|
| APIBench | 1,645 | ❌ | ❌ | 1.0 |
| API-Bank | 53 | ❌ | ✅ | 2.1 |
| ToolAlpaca | 400 | ❌ | ❌ | 1.0 |
| **ToolBench** | **16,464** | ✅ | ✅ | **4.0** |

**问题 1：API 数量有限**

之前的数据集最多包含几百个 API。这意味着：
- 模型只能学会调用这些特定 API
- 无法泛化到未见过的 API
- 真实世界中，用户可能想用自己的 API

**问题 2：只有单工具场景**

真实任务往往是这样的：
```
"帮我规划一个去日本的旅行：
1. 查一下东京的天气预报
2. 找几家附近的酒店
3. 看看当地有什么景点推荐
4. 计算一下总预算"
```
这需要调用天气 API、酒店 API、景点 API、货币转换 API——多工具协作。

但现有数据集只有"调用天气 API 查天气"这种单工具任务。

**问题 3：推理能力不足**

CoT（Chain-of-Thought）和 ReAct（Reasoning + Acting）是当时的主流推理方法。但它们有致命缺陷：
- **错误传播**：一步错，步步错
- **探索有限**：只走一条路，遇到死胡同就卡住

秦雨佳在论文中写道：
> "即使是 GPT-4，在复杂人类指令上的 pass rate 也很低，这使得标注效率极低。"

这意味着：即使用最强的模型，也很难自动生成有效的解法路径。

---

### 第二章：试错的旅程

#### 第一阶段：最初的直觉

"既然 ChatGPT 能做工具使用，"团队想，"那我们让 ChatGPT 帮我们生成训练数据不就行了？"

这听起来很直接。但具体怎么做？

**尝试 1：从指令出发**

先想一堆指令，然后找对应的 API。
```
指令："查天气"
→ 需要天气 API
```
问题：这样只能覆盖已知的指令，泛化性差。

**尝试 2：从 API 出发**

先收集 API，然后让 ChatGPT 根据 API 生成指令。
```
API：天气查询 API
→ 指令："明天北京下雨吗？"
→ 指令："这周末适合去爬山吗？"
```
这个方向对了！团队决定采用这种思路。

#### 第二阶段：API 收集的苦战

团队从 RapidAPI Hub 开始爬取 API。RapidAPI 是一个 API 市场，连接开发者和成千上万的真实 API。

**第一版**：爬了 53,190 个 API（10,853 个工具）。

"太多了，"团队发现，"很多 API 根本不能用。"

问题包括：
- 404 错误
- 服务器内部错误
- 响应时间过长
- 返回 HTML 错误信息而非有效数据

**过滤过程**：
1. 测试基本功能：API 是否能正常调用
2. 评估响应质量：返回的是有效数据还是错误信息
3. 检查响应时间：过慢的 API 直接淘汰

**最终结果**：3,451 个工具（16,464 个 API），覆盖 49 个大类：
- 金融、天气、电商、社交媒体、电影、美食...

每个 API 都有完整的文档：
- 功能描述
- 必需参数
- 可选参数
- 代码示例
- 响应示例

#### 第三阶段：指令生成的设计

有了 API，下一步是让 ChatGPT 生成指令。

**单工具指令**：
```
API：电影搜索 API
指令："帮我找一下诺兰导演的电影"
```

**多工具指令**是难点。团队尝试了两种方式：

**方式 1：随机采样工具组合**
```
随机选 5 个工具：天气 + 计算器 + 新闻 + 地图 + 翻译
→ 生成指令
```
问题：这些工具可能完全不相关，很难写出自然的指令。

**方式 2：利用 RapidAPI 层级结构**
```
同一类别的工具（如"电影"类别）：
- 电影搜索 API
- 电影评分 API
- 演员信息 API
→ 指令："帮我找一下最近评分最高的科幻电影，并告诉我主演的其他作品"
```
这个思路成功了！团队定义了两种多工具场景：
- **类别内多工具**：同一类别的多个工具
- **集合内多工具**：同一集合的多个工具（更细粒度）

最终生成了近 20 万条（指令，相关 API）对：
- I1（单工具）：87,413 条
- I2（类别内多工具）：84,815 条
- I3（集合内多工具）：25,251 条

#### 第四阶段：DFSDT 的诞生

最关键的挑战来了：如何为每条指令标注有效的解法路径？

**问题场景**：
```
指令："帮我规划一个生日惊喜，我朋友最喜欢的演员是 Hailee Steinfeld"

ReAct 输出：
Thought: 我需要搜索 Hailee Steinfeld 的信息
Action: search("Hailee Steinfeld")
Observation: {年龄：28, 最近电影：["蜘蛛侠：纵横宇宙", ...]}
Thought: 我需要推荐礼物
Action: search("gift ideas")
Observation: {...}
...
```

但 ReAct 有问题：
- 如果某一步调用失败（API 返回错误），模型可能陷入死循环
- 只探索一条路径，可能错过更好的解法

秦雨佳和团队陷入了焦虑。

**顿悟时刻**：

某天深夜，秦雨佳在想：
"人类是怎么解决复杂问题的？"

想象你在厨房做饭，发现没有盐了：
- 你会想："酱油能代替吗？不行，味道不对。那我去楼下便利店买？"
- 如果便利店关门了："那算了，这道菜先不放盐，下次再试"

人类会：
1. 评估多个选项
2. 选择最有希望的
3. 如果失败，回退并尝试其他选项

"这就是深度优先搜索！"秦雨佳意识到。

团队设计了 **DFSDT（Depth-First Search based Decision Tree）**：
```
指令
├─ 尝试路径 1
│  ├─ Thought 1 → Action 1 → Observation 1
│  ├─ Thought 2 → Action 2 → Observation 2
│  └─ ... → 成功/失败
├─ 尝试路径 2（如果路径 1 失败）
│  ├─ Thought 1' → Action 1' → ...
│  └─ ...
└─ 尝试路径 3（如果路径 2 失败）
   └─ ...
```

关键设计：
- **支持回退**：可以调用 "Finish by Giving Up" 放弃当前路径
- **多路径探索**：显式鼓励生成不同的子节点
- **深度优先**：只要找到一条有效路径就停止（节省成本）

**效果**：
| 方法 | I1 | I2 | I3 | 平均 |
|------|----|----|----|------|
| ReACT | 37.8 | 40.6 | 27.6 | 35.3 |
| ReACT@N（多次尝试） | 49.4 | 49.4 | 34.6 | 44.5 |
| **DFSDT** | **58.0** | **70.6** | **62.8** | **63.8** |

DFSDT 显著优于 ReACT，尤其在复杂任务（I2、I3）上提升更大。

最终生成了 126,486 条（指令，解法路径）对。

---

### 第三章：核心概念 - 大量实例

#### 概念 1：ToolBench 数据集三阶段构建

**生活类比 1：教学生使用工具**

想象你要教一个学生（LLM）如何使用各种工具完成任务：

**阶段 1：准备工具清单**
```
你去五金店收集工具：
- 锤子、螺丝刀、扳手、电钻...
- 每个工具都有说明书：用途、使用方法、注意事项
```
对应 API 收集：16,464 个 API + 完整文档

**阶段 2：设计练习题**
```
根据工具设计练习：
- 单工具："用锤子钉钉子"
- 多工具："用螺丝刀和扳手组装这个家具"
- 复杂任务："装修这个房间，需要用到所有工具"
```
对应指令生成：单工具/多工具指令

**阶段 3：写参考答案**
```
为每道题写解法：
- 第一步：做什么
- 第二步：如果失败了怎么办
- 第三步：如何检查结果
```
对应 DFSDT 标注解法路径

**生活类比 2：美食菜谱**

```
阶段 1（API 收集）：
收集 16,000+ 种食材和厨具，每种都有详细说明

阶段 2（指令生成）：
- 单工具："用烤箱烤面包"
- 多工具："用烤箱和搅拌机做蛋糕"
- 复杂任务："准备一顿完整的晚餐"

阶段 3（解法标注）：
为每道菜写详细菜谱，包括：
- 第一步：预热烤箱到 180 度
- 第二步：如果烤箱坏了，用平底锅代替
- ...
```

**代码实例：ToolBench 构建流程**
```python
# 阶段 1：API 收集
def collect_apis(rapidapi_url):
    tools = crawl_rapidapi(rapidapi_url)
    filtered_tools = []
    for tool in tools:
        if test_api(tool) and evaluate_response(tool):
            filtered_tools.append(tool)
    return filtered_tools  # 3,451 tools, 16,464 APIs

# 阶段 2：指令生成
def generate_instructions(sampled_apis):
    prompt = build_prompt(sampled_apis, seed_examples)
    response = chatgpt(prompt)
    return parse_instructions(response)  # 200k+ pairs

# 阶段 3：DFSDT 解法标注
def dfsdt_annotate(instruction, apis):
    root = Node(instruction, apis)
    stack = [root]

    while stack:
        node = stack.pop()

        # 生成动作
        thought, action, params = chatgpt_generate_action(
            instruction, node.history, apis
        )

        # 执行 API 调用
        observation = call_api(action, params)

        # 检查结果
        if is_success(observation):
            return node.path  # 找到解法
        elif should_give_up(node.history):
            continue  # 放弃当前路径，回溯
        else:
            # 扩展子节点
            child = Node(observation, node)
            stack.append(child)

    return None  # 无解

# 最终数据集
toolbench = []
for instruction, apis in generated_instructions:
    solution_path = dfsdt_annotate(instruction, apis)
    if solution_path:
        toolbench.append((instruction, solution_path))
# 126,486 条
```

**任务实例：单工具 vs 多工具**
```
【单工具指令 I1】
API：电影搜索 API
指令："帮我找一下 2023 年上映的科幻电影"
解法：
  Thought: 我需要搜索 2023 年的科幻电影
  Action: search_movies(year=2023, genre="sci-fi")
  Observation: [{title: "星际穿越 2", ...}, ...]
  Thought: 我已经找到结果
  Action: Finish with Final Answer
  Final Answer: "2023 年上映的科幻电影有：..."

【类别内多工具指令 I2】
API：电影搜索 + 电影评分 + 演员信息
指令："帮我找一下最近评分最高的科幻电影，并告诉我主演的其他作品"
解法：
  Thought: 我需要先搜索科幻电影
  Action: search_movies(genre="sci-fi", sort_by="rating")
  Observation: [{title: "电影 A", rating: 8.5, cast: ["演员 X"]}, ...]
  Thought: 找到最高评分的电影，现在查演员信息
  Action: get_actor_info(actor="演员 X")
  Observation: {filmography: ["电影 B", "电影 C", ...]}
  Thought: 整合信息
  Action: Finish with Final Answer
  Final Answer: "最高评分的科幻电影是《电影 A》(8.5 分)，主演 X 的其他作品包括..."

【集合内多工具指令 I3】
API：天气 + 酒店 + 景点 + 货币转换（跨类别）
指令："我下个月要去东京旅行 5 天，帮我规划一下行程和预算"
解法：
  Thought: 我需要先查东京下个月的天气
  Action: get_weather(city="Tokyo", month="next")
  Observation: {temp: "15-22°C", condition: "mostly sunny"}
  Thought: 天气不错，现在找酒店
  Action: search_hotels(city="Tokyo", nights=5)
  Observation: [{name: "酒店 A", price: 8000, ...}, ...]
  Thought: 找一些景点推荐
  Action: get_attractions(city="Tokyo")
  Observation: [{name: "浅草寺", ...}, ...]
  Thought: 计算总预算（假设从美国出发，需要货币转换）
  Action: convert_currency(amount=1000, from="USD", to="JPY")
  Observation: {result: 150000}
  Thought: 整合所有信息
  Action: Finish with Final Answer
  Final Answer: "东京 5 天行程建议：... 总预算约..."
```

#### 概念 2：DFSDT vs ReAct

**生活类比 1：迷宫寻宝**

想象你在一个迷宫里找宝藏：

**ReAct 方式**：
```
- 选择一条路，一直走
- 遇到死胡同：卡住，或者原路返回重新开始
- 只探索过一条路径
```

**DFSDT 方式**：
```
- 在岔路口标记所有选项
- 选择一条路，一直走
- 遇到死胡同：回到最近的岔路口，选另一条路
- 继续探索，直到找到宝藏
```

**生活类比 2：Debug 代码**

你的代码报错了，你怎么 debug：

**ReAct（类似一条路走到黑）**：
```
假设 1：可能是变量 X 的问题
→ 改 X → 还是错 → 卡住
```

**DFSDT（系统排查）**：
```
假设 1：可能是变量 X 的问题
→ 改 X → 还是错 → 回退
假设 2：可能是函数 Y 的问题
→ 改 Y → 还是错 → 回退
假设 3：可能是配置 Z 的问题
→ 改 Z → 成功了！
```

**代码实例：ReAct vs DFSDT**
```python
# ReAct：线性推理
def react_reasoning(instruction, apis):
    history = []
    for step in range(max_steps):
        thought, action, params = llm(instruction, history, apis)
        observation = call_api(action, params)
        history.append((thought, action, observation))

        if is_final_answer(action):
            return history
        # 如果出错，只能硬着头皮继续或完全重启

    return None  # 失败

# DFSDT：树形搜索
def dfsdt_reasoning(instruction, apis):
    def dfs(node, depth):
        if depth > max_depth:
            return None

        # 生成动作
        thought, action, params = llm_with_diversity(
            instruction, node.history, apis, node.siblings
        )
        observation = call_api(action, params)

        new_node = Node(history=node.history + [(thought, action, observation)])

        # 检查是否成功
        if is_final_answer(action) and is_valid(observation):
            return new_node.path

        # 检查是否需要放弃
        if should_give_up(observation):
            return None  # 回退，让父节点尝试其他子节点

        # 继续深度探索
        result = dfs(new_node, depth + 1)
        if result:
            return result

        return None  # 回退

    root = Node(history=[])
    return dfs(root, 0)
```

**对比场景：API 调用失败时**
```
任务："搜索电影《Batman》，然后获取主演信息"

【ReAct 处理】
Step 1: search_movie("Batman")
        → 成功：[{title: "Batman", id: "123"}]
Step 2: get_actor(movie_id="123")
        → API 错误：服务器 500
Step 3: get_actor(movie_id="123")  # 重试，还是错
Step 4: get_actor(movie_id="123")  # 继续重试...
结果：陷入循环，最终失败

【DFSDT 处理】
路径 1:
  Step 1: search_movie("Batman")
          → 成功：[{title: "Batman", id: "123"}]
  Step 2: get_actor(movie_id="123")
          → API 错误：服务器 500
  Step 3: Give Up（放弃当前路径）
路径 2（回溯后）:
  Step 1: search_movie("Batman")
          → 成功：[{title: "Batman", id: "123"}, {title: "Batman 2", id: "456"}]
  Step 2: get_actor(movie_id="456")  # 尝试第二部电影
          → 成功：{actors: [...]}
  Step 3: Finish with Final Answer
结果：成功
```

#### 概念 3：API Retriever

**生活类比：图书管理员**

想象你在一个有 16,000 本书的图书馆里找书：

**问题**：用户说"我想看关于二战的科幻小说"

**直接方法**：把 16,000 本书全部给模型看
- 问题：上下文太长，模型处理不了

**Retriever 方法**：图书管理员先帮你找几本相关的
- "这是《1984》、《高堡奇人》、《时间机器》"
- 模型只需要看这几本书

**代码实例：API Retriever**
```python
from sentence_transformers import SentenceTransformer, InputExample
from torch.utils.data import DataLoader
from sentence_transformers import losses

# 1. 加载预训练模型
model = SentenceTransformer('bert-base-uncased')

# 2. 准备训练数据
train_examples = []
for instruction, relevant_apis in toolbench:
    # 正样本：相关 API
    for api in relevant_apis:
        train_examples.append(InputExample(
            texts=[instruction, api.documentation],
            label=1.0
        ))
    # 负样本：随机其他 API
    negative_apis = sample_negative_apis(api_pool, relevant_apis)
    for api in negative_apis:
        train_examples.append(InputExample(
            texts=[instruction, api.documentation],
            label=0.0
        ))

# 3. 训练
train_dataloader = DataLoader(train_examples, batch_size=64)
loss = losses.ContrastiveLoss(model=model)
model.fit(train_objectives=[(train_dataloader, loss)], epochs=3)

# 4. 推理
def retrieve_apis(instruction, api_pool, top_k=5):
    instruction_emb = model.encode(instruction)
    api_embs = model.encode([api.doc for api in api_pool])

    # 计算相似度
    scores = cosine_similarity([instruction_emb], api_embs)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]

    return [api_pool[i] for i in top_indices]

# 使用
relevant_apis = retrieve_apis(user_instruction, all_apis, top_k=5)
```

**任务实例：Retriever 效果**
```
指令："帮我找一下最近上映的科幻电影，然后查查主演的其他作品"

所有 API 池：16,464 个 API

Retriever 输出（Top 5）：
1. TMDb 电影搜索 API（相关）✓
2. TMDb 演员信息 API（相关）✓
3. TMDb 电影评分 API（相关）✓
4. IMDb 搜索 API（相关）✓
5. 天气 API（不相关）✗

Ground Truth API：
1. TMDb 电影搜索 API
2. TMDb 演员信息 API

NDCG@1 = 1.0（第一个就命中）
NDCG@5 = 0.92（前 5 个里大部分相关）
```

---

### 第四章：预期 vs 实际

#### 预期 vs 实际对比表

| 维度 | 你的直觉/预期 | ToolLLM 实际实现 | 为什么有差距？ |
|------|--------------|-----------------|---------------|
| **数据构建方式** | 人工标注 | 完全用 ChatGPT 自动生成 | 人工标注 12 万条成本太高 |
| **API 数量** | 几百个就够 | 16,464 个 | 数量少无法泛化到新 API |
| **推理方法** | ReAct 应该够了 | 需要 DFSDT | ReAct 在复杂任务上失败率太高 |
| **训练模型大小** | 越大越好 | LLaMA-2 7B 就够 | 数据质量比模型大小更重要 |
| **泛化能力** | 只能用于训练过的 API | 可以处理未见过的 API | 学会的是"读文档用 API"的能力 |

#### 反直觉的事实

**事实 1：Vicuna 和 Alpaca 的 pass rate 是 0%**

直觉可能说："这些模型不是很强吗？怎么连一个任务都完成不了？"

实际：
```
Vicuna + ReACT/DFSDT: Pass Rate = 0%, Win Rate = 0%
Alpaca + ReACT/DFSDT: Pass Rate = 0%, Win Rate = 0%
```

原因：这些模型只在对话数据上微调，完全没有学过工具使用。就像让一个只学过聊天的人去开飞机——完全不会。

**事实 2：用 Retriever 推荐 API 比 Ground Truth 更好**

直觉可能说："Ground Truth 应该是最优的吧？"

实际（Table 4）：
```
ToolLLaMA-DFSDT（用 Ground Truth API）: Pass 66.7%, Win 60.0%
ToolLLaMA-DFSDT-Retriever（用推荐 API）: Pass 67.3%, Win 63.1%
```

原因：Ground Truth 里的某些 API 可能不是最优选择，Retriever 找到的替代 API 可能功能更好。

**事实 3：DFSDT 比 ReAct 贵，但值得**

DFSDT 需要更多 API 调用（探索多条路径），成本更高。但效果提升显著：
```
ChatGPT + ReACT: Pass 40.2%
ChatGPT + DFSDT: Pass 64.8%
```
提升 24.6%！

---

### 第五章：反直觉挑战

#### 挑战 1：如果只用单工具数据训练，模型能处理多工具任务吗？

**预测**：应该不行吧？

**实际**：可以，但效果会打折。

论文中的泛化实验（Table 4）：
```
I1-Inst.（单工具，未见过的指令）: Pass 57.0%
I1-Tool（单工具，未见过的工具）: Pass 61.0%
I1-Cat.（单工具，未见过的类别）: Pass 62.0%

I2-Inst.（多工具，未见过的指令）: Pass 77.0%
I2-Cat.（多工具，未见过的类别）: Pass 77.0%

I3-Inst.（复杂多工具）: Pass 66.0%
```

关键洞察：**多工具数据是必须的**。只在单工具数据上训练，模型学不会多工具协作。

#### 挑战 2：如果去掉 DFSDT 中的"回退"机制，会怎样？

**预测**：影响不大？

**实际**：会退化成 ReAct，效果大幅下降。

DFSDT 的核心是：
1. 可以放弃当前路径（Give Up）
2. 回溯并尝试其他路径

去掉回退 = 一条路走到黑 = ReAct。

从 Table 3 看：
```
ReACT: 35.3%
DFSDT: 63.8%
```
回退机制贡献了 28.5% 的提升！

#### 挑战 3：ToolLLaMA 能泛化到完全没见过的 API 领域吗？

**预测**：应该不行，领域差距太大了。

**实际**：可以！

在 APIBench 上的 OOD 泛化实验（Table 5）：
```
APIBench 领域：TorchHub, TensorHub, HuggingFace（机器学习 API）
ToolBench 领域：天气、电影、金融、新闻...（生活类 API）

ToolLLaMA + Our Retriever:
- HuggingFace: AST 16.77%, Hallucination 10.60%
- TorchHub: AST 51.16%, Hallucination 15.70%
- TensorHub: AST 40.59%, Hallucination 6.48%

Gorilla-ZS + BM25（专门在 APIBench 上训练）:
- HuggingFace: AST 10.51%, Hallucination 46.90%
- TorchHub: AST 44.62%, Hallucination 17.20%
- TensorHub: AST 34.31%, Hallucination 20.58%
```

ToolLLaMA 超越 Gorilla，尽管完全没在 APIBench 上训练过！

原因：ToolLLaMA 学会的是**阅读 API 文档并调用 API 的通用能力**，而不是死记硬背特定 API。

---

### 第六章：关键实验的细节

#### 实验 1：主实验结果

**设置**：
- 测试集：三种指令（I1, I2, I3）
- 泛化层级：未见过的指令/工具/类别
- Baselines：Vicuna, Alpaca, ChatGPT, Claude-2, GPT-4
- 指标：Pass Rate（完成率），Win Rate（胜率）

**结果（Table 4 简化）**：

| 模型 | 方法 | I1-Inst | I1-Cat | I2-Inst | I3-Inst | 平均 Pass | 平均 Win |
|------|------|---------|--------|---------|---------|-----------|----------|
| Vicuna | ReACT/DFSDT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Alpaca | ReACT/DFSDT | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ChatGPT | ReACT | 41.5 | 44.5 | 42.5 | 22.0 | 40.2 | - |
| ChatGPT | DFSDT | 54.5 | 60.5 | 75.0 | 62.0 | **64.8** | **64.3** |
| Claude-2 | DFSDT | 20.5 | 18.5 | 17.0 | 28.0 | 22.6 | 43.5 |
| Davinci-003 | DFSDT | 43.5 | 46.0 | 37.0 | 46.0 | 43.1 | 46.3 |
| GPT-4 | DFSDT | 60.0 | 67.0 | 79.5 | 71.0 | **71.1** | **70.4** |
| **ToolLLaMA** | **DFSDT** | **57.0** | **62.0** | **77.0** | **66.0** | **66.7** | **60.0** |

**关键观察**：
1. ToolLLaMA（7B）pass rate 超越 Claude-2 和 Davinci-003，接近 ChatGPT
2. 多工具任务（I2, I3）上提升最明显
3. GPT-4 仍然是最强，但 ToolLLaMA 缩小了差距

#### 实验 2：DFSDT 有效性验证

**设置**：
- 只用 ChatGPT（不训练模型）
- 比较 ReACT、ReACT@N（多次尝试）、DFSDT
- 指标：Pass Rate

**结果（Table 3）**：

| 方法 | I1 | I2 | I3 | 平均 |
|------|----|----|----|------|
| ReACT | 37.8 | 40.6 | 27.6 | 35.3 |
| ReACT@N | 49.4 | 49.4 | 34.6 | 44.5 |
| **DFSDT** | **58.0** | **70.6** | **62.8** | **63.8** |

**洞察**：
- DFSDT 在复杂任务（I2, I3）上提升最大
- ReACT@N（花同样成本多次尝试）只提升 9.2%，DFSDT 提升 28.5%
- 说明 DFSDT 的"智能回退"比"盲目重试"更有效

#### 实验 3：API Retriever 效果

**设置**：
- Baselines：BM25（传统检索），Ada（OpenAI embedding）
- 指标：NDCG@1, NDCG@5

**结果（Table 2）**：

| 方法 | I1 NDCG@1 | I1 NDCG@5 | I2 NDCG@1 | I2 NDCG@5 | I3 NDCG@1 | I3 NDCG@5 | 平均 NDCG@1 | 平均 NDCG@5 |
|------|-----------|-----------|-----------|-----------|-----------|-----------|-------------|-------------|
| BM25 | 18.4 | 19.7 | 12.0 | 11.0 | 25.2 | 20.4 | 18.5 | 17.0 |
| Ada | 57.5 | 58.8 | 36.8 | 30.7 | 54.6 | 46.8 | 49.6 | 45.4 |
| **Ours** | **84.2** | **89.7** | **68.2** | **77.9** | **81.7** | **87.1** | **78.0** | **84.9** |

**洞察**：
- 在单工具检索（I1）上表现最好
- 多工具检索（I2, I3）更难，但仍有 75-85 NDCG
- 显著优于 OpenAI 的 text-embedding-ada-002

#### 实验 4：OOD 泛化到 APIBench

**设置**：
- APIBench 领域：TorchHub, TensorHub, HuggingFace（ML API）
- 比较对象：Gorilla（专门在 APIBench 上训练）
- 指标：AST 准确率，幻觉率

**结果（Table 5）**：

| 方法 | HuggingFace AST | HuggingFace Hallu. | TorchHub AST | TorchHub Hallu. | TensorHub AST | TensorHub Hallu. |
|------|-----------------|--------------------|--------------|-----------------|---------------|------------------|
| ToolLLaMA + Our Retriever | **16.77** | **10.60** | **51.16** | **15.70** | **40.59** | **6.48** |
| Gorilla-ZS + BM25 | 10.51 | 46.90 | 44.62 | 17.20 | 34.31 | 20.58 |
| Gorilla-RS + BM25 | 15.71 | 6.42 | 50.00 | 5.91 | 41.90 | 2.77 |
| ToolLLaMA + Oracle | **88.80** | **8.66** | **85.88** | **14.12** | **88.62** | **7.44** |
| Gorilla-ZS + Oracle | 44.36 | 52.88 | 59.14 | 39.25 | 83.21 | 12.99 |
| Gorilla-RS + Oracle | 89.27 | 6.97 | 93.01 | 6.99 | 94.16 | 2.04 |

**关键洞察**：
1. 用 Retriever 时，ToolLLaMA 全面超越 Gorilla
2. 用 Oracle（Ground Truth API）时，ToolLLaMA 与 Gorilla 相当
3. Gorilla 的幻觉率更高（尤其是 ZS 设置）
4. ToolLLaMA 的泛化能力来自 ToolBench 的多样性训练

---

### 第七章：与其他方法对比

#### 工具学习数据集对比

```
时间线:
2022 ────── ReAct (Yao et al.)
           │
           └── 推理 + 行动框架，但数据有限

2023.03 ── API-Bank (Li et al.)
           │
           ├── 53 个 API
           └── 只有 274 条指令

2023.05 ── APIBench (Patil et al.)
           │
           ├── 1,645 个 API
           ├── 17,002 条指令
           └── 无多工具场景

2023.06 ── ToolAlpaca (Tang et al.)
           │
           ├── 400 个 API
           ├── 3,938 条指令
           └── 模拟响应，无真实调用

2023.07 ── ToolBench (Qin et al.) ← 本篇论文
           │
           ├── 16,464 个 API
           ├── 126,486 条解法路径
           ├── 真实 API 调用
           ├── 多工具场景
           └── DFSDT 推理
```

#### 详细对比表

| 特性 | ToolBench | APIBench | API-Bank | ToolAlpaca |
|------|-----------|----------|----------|------------|
| 真实 API | ✅ | ❌ | ✅ | ❌ |
| 真实 API 调用 | ✅ | ❌ | ✅ | ❌ |
| 多工具场景 | ✅ | ❌ | ❌ | ❌ |
| API 检索 | ✅ | ✅ | ❌ | ❌ |
| 多步推理 | ✅ | ❌ | ✅ | ✅ |
| API 数量 | 16,464 | 1,645 | 53 | 400 |
| 实例数量 | 126,486 | 17,002 | 274 | 3,938 |
| 真实 API 调用数 | 469,585 | 0 | 568 | 0 |
| 平均推理步数 | 4.0 | 1.0 | 2.1 | 1.0 |

#### 局限性分析

ToolLLM 并非完美，存在以下局限：

1. **依赖 ChatGPT 生成数据**
   - 如果 ChatGPT 在某些任务上也失败，数据质量会受影响
   - 虽然 DFSDT 提高了成功率，但仍有部分指令无法标注

2. **上下文长度限制**
   - LLaMA-2 原始上下文是 4096，通过位置插值扩展到 8192
   - 但对于非常长的 API 响应，仍可能截断

3. **推理成本**
   - DFSDT 比 ReAct 需要更多 token（探索多条路径）
   - 在实际部署时需要考虑延迟和成本

4. **评估的复杂性**
   - 工具学习评估比传统 NLP 任务更复杂
   - 同一指令可能有无数"正确"解法
   - 人类评估者之间一致性也不高

#### 改进方向

1. **Reflexion 结合**
   - 让模型从失败中学习
   - 跨 trial 改进策略

2. **更长上下文**
   - 使用 32K/128K 上下文模型
   - 处理更复杂的 API 响应

3. **多 Agent 协作**
   - 不同 Agent 负责不同工具
   - 分工合作完成复杂任务

4. **在线学习**
   - 从用户反馈中学习
   - 持续改进工具使用能力

---

### 第八章：如何应用

#### 推荐配置

**使用 ToolLLaMA**：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. 加载模型
model_name = "OpenBMB/ToolLLaMA-2-7b"  # 假设的模型名
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. 加载 API Retriever
retriever = load_retriever("api_retriever_checkpoint")
all_apis = load_api_docs("rapidapi_16k.json")

# 3. 推理函数
def tool_llama_inference(instruction):
    # 检索相关 API
    relevant_apis = retriever.retrieve(instruction, all_apis, top_k=5)

    # 构建 prompt
    prompt = build_prompt(instruction, relevant_apis)

    # 生成
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=2048,
        do_sample=False  # 贪心解码
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. 使用
instruction = "帮我查一下北京明天的天气"
response = tool_llama_inference(instruction)
print(response)
```

**使用 DFSDT 推理**：
```python
def dfsdt_inference(model, instruction, apis, max_depth=10):
    """使用 DFSDT 策略进行推理"""

    def dfs(history, depth):
        if depth > max_depth:
            return None

        # 生成动作
        prompt = build_step_prompt(instruction, history, apis)
        action = model.generate(prompt)

        # 解析动作
        thought, api_name, params = parse_action(action)

        # 执行 API 调用
        try:
            observation = call_api(api_name, params)
        except Exception as e:
            observation = {"error": str(e)}

        new_history = history + [(thought, api_name, params, observation)]

        # 检查是否完成
        if is_final_answer(action):
            return new_history
        elif should_give_up(observation, history):
            return None  # 回退
        else:
            # 继续深度探索
            result = dfs(new_history, depth + 1)
            if result:
                return result
            return None  # 回退

    return dfs([], 0)
```

#### 适用场景

| 场景 | 是否推荐 | 说明 |
|------|----------|------|
| **构建工具增强 LLM** | ✅ 强烈推荐 | ToolLLM 提供了完整的数据 + 训练方案 |
| **多工具复杂任务** | ✅ 强烈推荐 | DFSDT 在多工具场景优势明显 |
| **需要泛化到新 API** | ✅ 强烈推荐 | 学会了读文档用 API 的能力 |
| **简单单工具任务** | ⚠️ 可考虑 ReAct | DFSDT 优势不明显，成本更高 |
| **资源极度受限** | ⚠️ 考虑蒸馏 | 7B 模型仍较大，可蒸馏到更小模型 |
| **实时性要求极高** | ⚠️ 考虑 ReAct | DFSDT 推理延迟较高 |

#### 避坑指南

**常见错误 1：忽略 API 质量过滤**
```python
# ❌ 错误：直接用所有爬取的 API
all_apis = crawl_rapidapi()
train_model(all_apis)  # 很多 API 根本不能用

# ✅ 正确：严格过滤
all_apis = crawl_rapidapi()
filtered_apis = [api for api in all_apis if test_api(api) and evaluate_response(api)]
train_model(filtered_apis)
```

**常见错误 2：DFSDT 深度设置过大**
```python
# ❌ 错误：深度太大，成本爆炸
dfsdt_inference(instruction, max_depth=50)  # 可能调用数百次 API

# ✅ 正确：合理设置深度
dfsdt_inference(instruction, max_depth=10)  # 通常足够
```

**常见错误 3：Retriever 返回太多 API**
```python
# ❌ 错误：返回太多 API，超出上下文
relevant_apis = retriever.retrieve(instruction, top_k=50)  # 上下文爆炸

# ✅ 正确：返回适量的 API
relevant_apis = retriever.retrieve(instruction, top_k=5)  # 5 个足够
```

---

### 第九章：延伸思考

#### 深度问题

1. **为什么 ToolLLaMA 能泛化到未见过的 API？**
   - 提示：模型学到的是什么能力？是记忆特定 API，还是理解 API 文档的通用能力？

2. **DFSDT 和 Tree-of-Thought（ToT）有什么区别？**
   - 提示：ToT 针对的是封闭任务（如 Game of 24），DFSDT 针对的是开放任务（无限工具）

3. **为什么 GPT-4 + ReACT 不如 ToolLLaMA + DFSDT？**
   - 提示：数据质量 vs 模型大小，哪个更重要？

4. **如果让你设计 ToolLLM v2，你会改进什么？**
   - 提示：考虑多模态工具、在线学习、Agent 协作等方向

5. **ToolEval 的局限性是什么？**
   - 提示：ChatGPT 评估 vs 人类评估，一致性有多少？

6. **为什么 Vicuna 和 Alpaca 在工具使用上完全失败？**
   - 提示：指令微调数据的覆盖范围有多重要？

7. **DFSDT 的"回退"机制在什么情况下会失效？**
   - 提示：如果所有路径都失败，怎么办？

#### 实践挑战

1. **复现 ToolBench 构建**
   - 从 RapidAPI 爬取 API
   - 用 ChatGPT 生成指令和解法
   - 比较你的数据与原论文数据

2. **实现 DFSDT 算法**
   - 实现基础 DFS
   - 添加多样性提示
   - 添加回退机制
   - 在 ToolBench 上测试

3. **训练自己的 ToolLLaMA**
   - 使用 LLaMA-2-7B
   - 在 ToolBench 上微调
   - 对比不同推理策略的效果

4. **扩展 OOD 泛化实验**
   - 在更多领域测试 ToolLLaMA
   - 分析泛化失败的原因
   - 提出改进方案

---

## 总结

ToolLLM 通过**大规模真实 API 数据** + **DFSDT 推理算法** + **API Retriever**，成功让开源 LLM（LLaMA-2 7B）在工具使用能力上追平了 ChatGPT。

**核心贡献**：
1. **ToolBench 数据集**：16,464 个真实 API，126,486 条解法路径，三阶段自动构建
2. **DFSDT 算法**：支持回退和多路径探索，pass rate 从 35.3% 提升到 63.8%
3. **ToolEval 评估器**：Pass Rate + Win Rate，与人类评估 87%/80% 一致性
4. **API Retriever**：从 16K+ API 中精准检索，NDCG@5 达 84.9
5. **ToolLLaMA 模型**：开源 7B 模型，工具使用能力媲美 ChatGPT

**历史地位**：
- 首个大规模工具学习数据集（16K+ API）
- 证明了数据质量比模型大小更重要
- 为开源 LLM 的工具使用能力设定了新基准
- 论文引用量持续增长，成为工具学习领域的基础工作

**一句话总结**：ToolLLM 让开源 LLM 从"只会聊天"进化到"能使用 16,000+ 工具完成任务"， democratizing AI 工具使用能力。

---

**论文元信息**
- 标题：ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs
- 作者：Yujia Qin, Shihao Liang, Yining Ye, et al.（清华大学，中国人民大学，耶鲁大学，腾讯）
- arXiv: 2307.16789
- 发表年份：2023
- 代码/数据：https://github.com/OpenBMB/ToolBench
- 阅读时间：2026-03-15
- 方法论：高保真交互式精读协议
