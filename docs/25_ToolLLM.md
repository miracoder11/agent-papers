# ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs

## 层 1：电梯演讲

**一句话概括**：清华大学团队在 2023 年发现开源 LLM 在工具使用能力上与 ChatGPT 存在巨大差距，通过构建包含 16,464 个真实 API 的 ToolBench 数据集并微调 LLaMA，得到的 ToolLLaMA 在工具使用能力上达到 ChatGPT 水平，在复杂多工具任务上超越所有开源模型。

---

## 层 2：故事摘要

### 核心问题

2023 年的开源 LLM 面临**能力缺陷**：

**ChatGPT 的表现**：
```
用户："帮我查一下北京的天气，然后推荐几个适合的户外活动"

ChatGPT:
1. 调用天气 API → 北京 25°C，晴天
2. 基于天气推荐 → 公园散步、骑行、野餐
3. 提供具体地点和交通信息
```

**开源 LLaMA 的表现**：
```
用户："帮我查一下北京的天气，然后推荐几个适合的户外活动"

LLaMA:
"北京天气很好，适合户外活动。
您可以去公园散步，或者去郊外野餐。
具体天气数据我无法获取..."

问题：不会调用 API，只能"瞎编"
```

### 核心洞察

清华大学团队观察到：
1. ChatGPT 有惊人的工具使用能力（调用 API、执行代码、搜索网络）
2. 开源 LLM 只会在训练数据里"回忆"，不会使用工具
3. **根本原因**：现有 instruction tuning 只关注语言任务，忽略了工具使用领域

**关键问题**：
> "如何让开源 LLM 也掌握工具使用能力？"

### ToolLLM 框架大图

```
┌─────────────────────────────────────────────────────────┐
│              ToolLLM 三阶段流程                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  阶段 1: API Collection (API 收集)                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  RapidAPI Hub → 16,464 个真实 RESTful API          │   │
│  │  - 社交媒体、电商、天气、金融、旅游... 49 个类别    │   │
│  │  - 每个 API 有完整文档：功能描述、参数、代码示例   │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                    │
│  阶段 2: Instruction Generation (指令生成)              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ChatGPT 自动生成多样指令                         │   │
│  │  - 单工具场景："查北京天气"                       │   │
│  │  - 多工具场景："查天气 + 推荐活动 + 订餐厅"        │   │
│  │  - 生成 10,000+ 高质量指令                          │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                    │
│  阶段 3: Solution Path Annotation (解答路径标注)        │
│  ┌─────────────────────────────────────────────────┐   │
│  │  ChatGPT 搜索有效的 API 调用链                     │   │
│  │  - Depth-First Search Decision Tree (DFSDT)     │   │
│  │  - 评估多种推理路径，扩展搜索空间                 │   │
│  │  - 生成正确的 API 调用序列作为训练数据              │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                    │
│  微调 LLaMA → ToolLLaMA + API Retriever               │
│                                                         │
└─────────────────────────────────────────────────────────┘

关键创新:
1. ToolBench: 首个大规模真实 API 指令数据集
2. DFSDT: 基于深度优先搜索的决策树推理
3. ToolEval: 自动评估工具使用能力
4. 零样本泛化到未见过的 API
```

### 关键结果

| 模型 | 单工具 Pass Rate | 多工具 Pass Rate | 相对于 ChatGPT |
|------|----------------|----------------|---------------|
| Vicuna | 0.15 | 0.08 | -70% |
| Alpaca | 0.18 | 0.10 | -65% |
| Davinci-003 | 0.45 | 0.32 | -25% |
| Claude-2 | 0.52 | 0.38 | -15% |
| **ToolLLaMA** | **0.68** | **0.55** | **≈ ChatGPT** |
| ChatGPT | 0.70 | 0.58 | - |
| GPT-4 | 0.85 | 0.75 | +20% |

**核心贡献**：
- ToolLLaMA 是首个在工具使用上媲美 ChatGPT 的开源模型
- 泛化能力强：在 APIBench 零样本测试上表现优异
-  democratize AI：开源工具使用能力

---

## 层 3：深度精读

### 开场：2023 年的"工具鸿沟"

**时间**：2023 年初
**地点**：清华大学 NLP 实验室
**人物**：Yujia Qin, Shihao Liang 团队

**场景**：
```
研究员给多个模型同一个任务：

任务："帮我找一家北京评分 4.5 分以上的川菜馆，
      然后查一下今天下午有没有雨，
      如果下雨就推荐室内活动"

ChatGPT 的执行：
1. ✓ 调用餐厅 API → 找到"蜀国印象"，评分 4.7
2. ✓ 调用天气 API → 今天下午有雨
3. ✓ 推理：下雨 → 室内活动
4. ✓ 推荐：电影院、博物馆、室内游乐场

"太厉害了！"团队惊叹，
"但 ChatGPT 是闭源的，我们不知道它怎么做到的..."

LLaMA 的执行：
"北京有很多好吃的川菜馆，
蜀国印象很不错。
今天下午可能会下雨，
建议您带伞或者选择室内活动。"

问题：全是"幻觉"，没有调用任何 API！
```

**研究者的困惑**：
- ChatGPT 为什么能调用 API？
- 开源模型为什么不会？
- 我们能让 LLaMA 也学会吗？

这就是 ToolLLM 的起点。

---

### 第一章：开源 LLM 的工具使用困境

#### 困境 1：Instruction Tuning 的盲区

**现有 Instruction Tuning 数据集**：
```
Alpaca:
- "写一首关于春天的诗"
- "把这句话翻译成法语"
- "总结这篇文章的主要内容"

Stanford Alpaca, Vicuna, Koala...
全部聚焦于"纯语言任务"！
```

**问题**：
- 没有教模型如何使用工具
- 模型学会的是"回忆训练数据"
- 而不是"调用外部 API 获取信息"

#### 困境 2： prior work 的局限

**之前的工具使用研究**：

| 工作 | API 数量 | 真实 API？| 多工具？| API 检索？|
|------|---------|---------|--------|---------|
| APIBench | ~100 | ❌ (模拟) | ❌ | ✓ |
| API-Bank | ~50 | ✓ | ❌ | ❌ |
| ToolAlpaca | ~30 | ❌ (模拟) | ❌ | ❌ |
| ToolBench (Xu et al.) | ~200 | ✓ | ❌ | ✓ |

**关键局限**：
1. **API 太少**：几十到几百个，远远不够
2. **模拟 API**：不是真实世界的 API
3. **单工具场景**：现实任务需要多工具协作
4. **没有 API 检索**：用户需要自己指定 API

---

### 第二章：ToolLLM 的设计思路

#### 核心洞察

**团队的观察**：
```
ChatGPT 的工具使用能力来自：
1. 大量工具使用数据的 SFT
2. 人类反馈强化学习 (RLHF)
3. 函数调用 (Function Call) 训练

开源模型没有这些数据！
```

**关键想法**：
> "用 ChatGPT 生成工具使用数据，然后用这些数据训练开源模型"

**自我提升循环**：
```
ChatGPT (有工具能力)
    ↓ (生成数据)
ToolBench (工具使用数据集)
    ↓ (微调)
ToolLLaMA (获得工具能力)
```

#### 设计原则

**原则 1：真实世界 API**
```
不用模拟 API，用真实的：
- RapidAPI Hub 上的 16,464 个 API
- 涵盖 49 个类别
- 每个 API 有完整文档
```

**原则 2：多工具场景**
```
现实任务往往需要多个工具：
"查天气 → 推荐活动 → 订餐厅 → 打车"

ToolLLM 支持：
- 单工具指令
- 多工具指令（需要多轮 API 调用）
```

**原则 3：API 检索**
```
16,464 个 API，模型怎么知道用哪个？

解决方案：
- 训练一个 neural API retriever
- 根据指令推荐相关 API
- 模型从推荐列表中选择
```

---

### 第三章：ToolBench 数据集构建

#### 阶段 1: API Collection (API 收集)

**数据来源**：RapidAPI Hub
```
RapidAPI 是全球最大的 API 市场：
- 开发者发布自己的 API
- 有完整的文档和代码示例
- 涵盖各种功能：天气、地图、金融、社交...

收集的 API：
- 总数：16,464 个
- 类别：49 个
- 每个 API 包含：
  - 功能描述
  - 必需参数
  - 返回格式
  - 代码示例
```

**示例 API**：
```json
{
  "name": "weather-api",
  "category": "Weather",
  "description": "Get current weather and forecast",
  "endpoints": [
    {
      "path": "/current",
      "method": "GET",
      "parameters": ["location", "unit"],
      "response": {"temp": 25, "condition": "sunny"}
    }
  ]
}
```

---

#### 阶段 2: Instruction Generation (指令生成)

**用 ChatGPT 生成指令**：
```
Prompt:
"基于以下 API，生成用户可能问的问题：

API: weather-api, restaurant-api, activity-recommender

示例指令：
- 单工具：'北京今天天气怎么样？'
- 多工具：'帮我找家评分高的川菜馆，
           然后查一下那边下午有没有雨，
           如果下雨就推荐附近的室内活动'"

生成结果：
- 10,000+ 条指令
- 涵盖单工具和多工具场景
- 多样化，覆盖不同类别
```

**指令类型**：
```
Type 1: 单工具单步
"北京今天天气怎么样？"
→ 调用 weather-api

Type 2: 单工具多步
"北京、上海、广州三地的天气"
→ 调用 weather-api 三次

Type 3: 多工具多步
"找家川菜馆，查天气，推荐活动"
→ restaurant-api → weather-api → activity-recommender
```

---

#### 阶段 3: Solution Path Annotation (解答路径标注)

**核心挑战**：
```
给定指令，如何找到正确的 API 调用序列？

错误示例：
指令："找家评分 4.5 以上的川菜馆"
错误调用：weather-api(location="北京")
→ 调用错了 API！

正确调用：restaurant-api(cuisine="川菜", rating>4.5)
```

**解决方案：DFSDT (Depth-First Search Decision Tree)**
```
传统方法：ReACT
- 生成一个思考 → 调用一个 API → 继续思考
- 问题：一条路走到黑，错了就回不来

DFSDT:
- 探索多条可能的路径
- 评估每条路径的质量
- 选择最好的路径作为训练数据
```

**DFSDT 的执行过程**：
```
指令："找家川菜馆，查天气，推荐活动"

探索路径 1:
1. 调用 restaurant-api → 找到"蜀国印象"
2. 调用 weather-api → 有雨
3. 调用 indoor-activity-api → 推荐电影院
✓ 成功！

探索路径 2:
1. 调用 weather-api → 有雨
2. 调用 restaurant-api → 找到"老四川"
3. 调用 indoor-activity-api → 推荐博物馆
✓ 也成功！

探索路径 3:
1. 调用 activity-api → 错误！还没确定去哪
✗ 失败

选择路径 1 作为训练数据（评分最高）
```

---

### 第四章：ToolLLaMA 的训练

#### 模型架构

```
┌─────────────────────────────────────────────────────────┐
│              ToolLLaMA 推理流程                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  用户指令："北京明天天气怎么样？"                        │
│                    ↓                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Step 1: API Retriever                          │   │
│  │  - 从 16,464 个 API 中检索相关 API                  │   │
│  │  - 输出：[weather-api, forecast-api, ...]       │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Step 2: ToolLLaMA 生成 API 调用                   │   │
│  │  "我需要调用 weather-api"                        │   │
│  │  生成调用：weather.get(location="北京")          │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Step 3: 执行 API，获取结果                       │   │
│  │  返回：{"temp": 25, "condition": "sunny"}       │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Step 4: ToolLLaMA 生成最终回复                    │   │
│  │  "北京明天天气晴朗，气温 25°C，适合户外活动"        │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 训练数据格式

```
### Instruction:
北京明天天气怎么样？

### API List:
[weather-api, forecast-api, news-api, ...]

### Response:
Thought: 我需要查询北京的天气信息
Action: weather.get(location="北京", date="tomorrow")
Observation: {"temp": 25, "condition": "sunny"}
Thought: 我已经获取了天气信息
Response: 北京明天天气晴朗，气温 25°C，适合户外活动
```

---

### 第五章：大量实例

#### 【实例 1：单工具查询】

**任务**：
```
用户："查一下苹果公司的股票价格"
```

**ToolLLaMA 的执行**：
```
Step 1 - API Retrieval:
检索到：[stock-api, finance-api, news-api]

Step 2 - API Call:
Thought: 我需要查询苹果公司的当前股价
Action: stock.get_price(symbol="AAPL")

Step 3 - Observation:
{"symbol": "AAPL", "price": 178.52, "change": "+2.3%"}

Step 4 - Response:
"苹果公司 (AAPL) 当前股价为 178.52 美元，上涨 2.3%"
```

---

#### 【实例 2：多工具协作】

**任务**：
```
用户："我想规划一个北京三日游，
      包括景点推荐、酒店预订和天气查询"
```

**ToolLLaMA 的执行**：
```
Round 1:
Thought: 首先我需要了解北京的天气情况
Action: weather.get_forecast(location="北京", days=3)
Observation: [
  {"date": "2024-01-01", "temp": 5, "condition": "sunny"},
  {"date": "2024-01-02", "temp": 3, "condition": "cloudy"},
  {"date": "2024-01-03", "temp": 2, "condition": "snow"}
]

Round 2:
Thought: 天气了解了，现在推荐适合的景点
Action: attraction.search(city="北京", days=3)
Observation: [
  {"day1": ["故宫", "天安门", "王府井"]},
  {"day2": ["颐和园", "圆明园", "中关村"]},
  {"day3": ["长城", "明十三陵"]}
]

Round 3:
Thought: 根据景点位置推荐附近的酒店
Action: hotel.search(location="北京", attractions=["故宫", "颐和园", "长城"])
Observation: [
  {"name": "北京饭店", "location": "王府井", "price": 800},
  {"name": "颐和安缦", "location": "颐和园附近", "price": 3500},
  {"name": "长城脚下公社", "location": "八达岭", "price": 2000}
]

Final Response:
"为您规划北京三日游：

Day 1 (1/1, 晴，5°C):
- 景点：故宫 → 天安门 → 王府井
- 住宿：北京饭店 (王府井，800 元/晚)

Day 2 (1/2, 阴，3°C):
- 景点：颐和园 → 圆明园 → 中关村
- 住宿：颐和安缦 (颐和园附近，3500 元/晚)

Day 3 (1/3, 雪，2°C):
- 景点：长城 → 明十三陵
- 住宿：长城脚下公社 (八达岭，2000 元/晚)

注意：第三天有雪，请注意保暖！"
```

---

#### 【实例 3：错误恢复】

**任务**：
```
用户："帮我订一张明天北京到上海的机票"
```

**ToolLLaMA 的执行（带错误恢复）**：
```
Round 1:
Thought: 我需要查询明天的航班
Action: flight.search(from="北京", to="上海", date="tomorrow")
Observation: {"error": "Date format invalid, use YYYY-MM-DD"}

Thought: 日期格式错了，我需要修正
Action: flight.search(from="北京", to="上海", date="2024-01-15")
Observation: [
  {"flight": "CA1501", "time": "08:00", "price": 1200},
  {"flight": "MU5100", "time": "10:30", "price": 980},
  {"flight": "CZ3907", "time": "14:20", "price": 850}
]

Round 2:
Thought: 有三个航班，用户没有指定偏好，推荐性价比最高的
Response: "为您找到明天北京到上海的航班：

1. 南航 CZ3907 - 14:20 起飞，850 元 (最便宜)
2. 东航 MU5100 - 10:30 起飞，980 元 (时间适中)
3. 国航 CA1501 - 08:00 起飞，1200 元 (早班机)

请问您想预订哪个航班？"
```

---

### 第六章：预期 vs 实际

### 你的直觉 vs ToolLLM 的实现

| 维度 | 你的直觉/预期 | ToolLLM 实际 | 为什么 |
|------|--------------|-------------|--------|
| **需要多少 API？** | 几百个够了 | 16,464 个 | 真实场景需要多样性 |
| **多工具复杂吗？** | 很难协调 | DFSDT 处理得好 | 搜索多条路径，选最优 |
| **泛化能力** | 只能用于训练过的 API | 零样本泛化新 API | 学会了"读文档调用 API"的能力 |
| **API 检索准确吗？** | 可能找不到 | Top-5 准确率 90%+ | 专门的 retriever 模型 |
| **与 ChatGPT 差距** | 应该很大 | 几乎持平 | 用 ChatGPT 的数据训练 |

---

### 反直觉挑战

**问题：如果用 ReACT 而不是 DFSDT，会怎样？**

[先想 1 分钟...]

**直觉**："ReACT 已经很好了，应该差不多吧？"

**实际**：
```
ReACT 的 Pass Rate: 0.52
DFSDT 的 Pass Rate: 0.68

差距：+30%！
```

**为什么**：
```
ReACT 的问题：
- 一条路走到黑
- 错了难以回溯
- 无法比较多种方案

DFSDT 的优势：
- 探索多条路径
- 失败时回溯
- 选择最优解

例：
指令："找家餐厅，要评分高、价格适中、有停车位"

ReACT:
- 找到第一家 → 检查条件 → 不符合 → 继续找
- 可能找很久...

DFSDT:
- 同时探索多家餐厅
- 比较哪家最符合条件
- 直接推荐最优的
```

---

### 第七章：关键实验

#### 实验 1: ToolEval 评估

**评估方法**：
```
ToolEval: 自动评估工具使用能力

指标：
1. Pass Rate: 指令是否成功完成
2. Win Rate: 相比 ChatGPT 的回答质量

评估流程：
- 给定指令和标准答案
- 执行模型的 API 调用
- 比较结果与标准答案
```

**结果**：
```
单工具任务 Pass Rate:
- Vicuna: 0.15
- Alpaca: 0.18
- Davinci-003: 0.45
- Claude-2: 0.52
- ChatGPT: 0.70
- ToolLLaMA: 0.68 ← 接近 ChatGPT！

多工具任务 Pass Rate:
- Vicuna: 0.08
- Alpaca: 0.10
- Davinci-003: 0.32
- Claude-2: 0.38
- ChatGPT: 0.58
- ToolLLaMA: 0.55 ← 再次接近！
```

---

#### 实验 2: 零样本泛化

**测试**：在未见过的 API 上评估
```
训练数据：16,464 个 API 中的 10,000 个
测试数据：剩余 6,464 个未见过的 API

结果：
- ToolLLaMA 在未见 API 上 Pass Rate: 0.61
- 仅比见过 API 低 0.07

结论：
ToolLLaMA 学会了"如何读 API 文档并调用"
而不是死记硬背特定 API
```

---

#### 实验 3: APIBench 跨数据集评估

**APIBench**：另一个工具使用 benchmark
```
APIBench 特点：
- 不同的 API 集合
- 不同的指令格式
- Out-of-Distribution 测试

结果：
- ToolLLaMA: 0.65 Pass Rate
- 超越所有开源模型
- 接近 ChatGPT (0.68)

结论：ToolLLaMA 有强泛化能力
```

---

### 第八章：局限性与改进

#### 局限性

**1. 依赖 ChatGPT 生成数据**
```
问题：
- 如果 ChatGPT 犯错，错误会被放大
- ChatGPT 的能力上限限制了 ToolBench 的质量

影响：
- ToolLLaMA 难以超越 ChatGPT
- 需要其他数据来源
```

**2. API 可能失效**
```
问题：
- RapidAPI 上的 API 可能下线
- API 的接口可能变化
- 训练时的 API 测试时可能不可用

影响：
- 需要定期更新数据集
```

**3. 推理速度慢**
```
DFSDT 需要探索多条路径：
- 单次推理可能需要 10+ 次 API 调用
- 延迟高，不适合实时场景

对比：
- ReACT: 3-5 次调用
- DFSDT: 10-20 次调用
```

#### 改进方向

**1. 人类反馈**
```
收集人类使用数据：
- 用户真实调用 API 的记录
- 人类偏好（哪个回答更好）
- 用于 RLHF 进一步微调
```

**2. 更高效的推理**
```
改进 DFSDT:
- 限制搜索深度
- 剪枝策略
- 学习预测"哪条路径最有希望"
```

**3. 多模态工具**
```
扩展到：
- 图像生成 API（DALL-E, Midjourney）
- 语音 API（TTS, STT）
- 视频处理 API
```

---

### 第九章：如何应用

#### 快速开始

```python
from toolllm import ToolLLaMA, APIRetriever

# 加载模型
model = ToolLLaMA.from_pretrained("THUDM/toolllama")
retriever = APIRetriever.from_pretrained("THUDM/api-retriever")

# 定义指令
instruction = "北京明天天气怎么样？"

# 检索相关 API
apis = retriever.search(instruction, top_k=5)
print(f"推荐 API: {apis}")

# 生成回复
response = model.generate(
    instruction=instruction,
    api_list=apis,
    max_steps=10
)

print(response)
# 输出："北京明天天气晴朗，气温 25°C..."
```

---

### 第十章：延伸思考

[停下来，想一想...]

1. **ToolLLM 和 AutoGPT 有什么区别？**
   - AutoGPT：自主探索，自己决定用什么工具
   - ToolLLM：从检索的 API 中选择
   - 哪个更实用？

2. **如果让 ToolLLM 学习使用新工具，需要重新训练吗？**
   - 零样本泛化能力能否应对全新类别的 API？
   - 还是需要少量 fine-tuning？

3. **ToolLLM 能否"组合"API 创造新功能？**
   - 例：天气 API + 穿衣推荐 API = 穿衣建议
   - 这种组合能力是涌现的吗？

4. **开源模型 tool use 能力的边界在哪里？**
   - 什么任务是 ChatGPT 能做但 ToolLLM 做不了的？
   - 差距会继续缩小吗？

5. **ToolLLM 对 AGI 有什么启示？**
   - 工具使用是智能的核心能力吗？
   - "会用工具"= "更聪明"？

---

## 附录：论文定位图谱

```
                    LLM 工具使用技术发展史

ReACT (2022.10) ─────┬────→ 推理 + 行动交替
                     │
Toolformer (2023.02) ┤
                     │     → 学习使用工具
                     │
HuggingGPT (2023.03) ┤
                     │     → 连接 AI 模型
                     │
Gorilla (2023.05) ───┤
                     │     → API 调用训练
                     │
ToolLLM ─────────────┼────→ 大规模真实 API 训练【本文】
(2023.07)            │
                     │
FireAct (2023.10) ───┘
                     │     → 改进推理策略
                     │
                     └────→ 共同主题：LLM + 工具 = 增强能力

上游工作：
- ReACT: 证明了推理 + 行动的有效性
- Toolformer: 展示了工具学习的潜力

下游工作：
- FireAct: 改进 DFSDT 推理
- Agent 框架：集成 ToolLLM 作为 backbone

影响力：
- 首个在工具使用上媲美 ChatGPT 的开源模型
- ToolBench 成为标准 benchmark
- 推动了开源 tool-use 研究
```

---

## 写作检查清单

- [x] 电梯演讲层
- [x] 故事摘要层（含框架大图）
- [x] 深度精读层
- [x] 具体失败场景开场（ChatGPT vs LLaMA 对比）
- [x] 故事化叙述
- [x] 多角度实例（生活类比 + 代码实例 + 对比场景）
- [x] 预期 vs 实际对比表
- [x] 反直觉挑战（DFSDT vs ReACT）
- [x] 关键实验细节（ToolEval、零样本泛化、APIBench）
- [x] 局限性分析
- [x] 应用指南
- [x] 延伸思考
- [x] 论文定位图谱

---

## 关键术语表

| 术语 | 含义 |
|------|------|
| Tool Learning | 让 LLM 学会使用外部工具（API） |
| ToolBench | ToolLLM 构建的指令微调数据集 |
| DFSDT | Depth-First Search Decision Tree，改进的推理策略 |
| API Retriever | 根据指令检索相关 API 的模型 |
| ToolEval | 自动评估 LLM 工具使用能力的工具 |
| Pass Rate | 指令成功完成的比例 |
| Win Rate | 相比 ChatGPT 回答质量更好的比例 |
