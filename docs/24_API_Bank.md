# API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs

## 层 1：电梯演讲

**一句话概括**：阿里巴巴团队在 2023 年发现现有 LLM 在工具使用能力评估上存在空白，构建了包含 73 个可执行 API 和 314 个工具使用对话的 API-Bank 基准，并训练了 Lynx 模型，在工具使用能力上超越 Alpaca 26%，接近 GPT-3.5 水平。

---

## 层 2：故事摘要

### 核心问题

2023 年工具增强 LLM 研究面临**三大未解之间**：

1. **效果如何**：当前 LLM 使用工具的能力到底有多强？
2. **如何提升**：怎么增强 LLM 的工具使用能力？
3. **障碍在哪**：还有哪些关键问题需要解决？

**困境**：
```
研究者在评估 LLM 工具能力时遇到：
- 没有统一的 benchmark
- 现有数据集 API 太少（<100 个）
- 大多数 API 是模拟的，不能真正执行
- 无法评估"检索 API"的能力

"我们需要一个真正的、可执行的评估系统！"
```

### 核心洞察

阿里巴巴团队的做法：
1. 先调研 500 个用户，了解真实需求
2. 基于需求定义评估维度
3. 构建可执行的 API 系统
4. 用 Multi-agent 方法自动生成训练数据

### API-Bank 框架大图

```
┌─────────────────────────────────────────────────────────┐
│              API-Bank 三层评估体系                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  能力层级 1: API Calling (调用能力)                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │  给定 API 列表，能否正确调用？                      │   │
│  │  - 理解 API 功能                                  │   │
│  │  - 生成正确的参数                                │   │
│  │  - 处理 API 返回结果                               │   │
│  │  评估：314 个对话，753 次 API 调用                     │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                    │
│  能力层级 2: API Retrieving (检索能力)                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  从大量 API 中找到合适的 API？                       │   │
│  │  - 理解用户需求                                  │   │
│  │  - 从数百个 API 中检索相关的                        │   │
│  │  - 排序和筛选                                    │   │
│  │  评估：1,888 个对话，4,149 次 API 调用                  │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                    │
│  能力层级 3: API Planning (规划能力)                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │  复杂任务需要多步 API 调用？                         │   │
│  │  - 分解复杂需求                                  │   │
│  │  - 规划 API 调用顺序                               │   │
│  │  - 整合多个 API 的结果                             │   │
│  │  评估：多工具场景，复杂任务完成度                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  训练数据构建：Multi-agent 方法                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Agent 1: 领域生成 → 旅行、金融、天气...            │   │
│  │  Agent 2: API 生成 → 为每个领域生成 API             │   │
│  │  Agent 3: 查询生成 → 模拟用户提问                 │   │
│  │  Agent 4: API 调用生成 → 正确的调用序列             │   │
│  │  Agent 5: 质量检查 → 验证数据质量                 │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  结果：Lynx 模型 (Alpaca-7B 微调)                        │
│  - API 调用准确率：+26% vs Alpaca                        │
│  - 接近 GPT-3.5 水平                                      │
│  - 与 GPT-4 差距 21%                                     │
└─────────────────────────────────────────────────────────┘

关键贡献:
1. 首个可执行的工具使用评估系统
2. 三层能力评估：Calling + Retrieving + Planning
3. Multi-agent 数据生成：成本降低 98%
4. Lynx: 开源工具增强 LLM
```

### 关键结果

| 模型 | API 调用 | API 检索 | API 规划 | 平均 |
|------|---------|---------|---------|------|
| Alpaca-7B | 22% | 5% | 8% | 11.7% |
| ChatGLM-6B | 18% | 3% | 5% | 8.7% |
| GPT-3 Davinci | 2% | 1% | 2% | 1.7% |
| GPT-3.5 | 65% | 58% | 72% | 65% |
| **Lynx** | **68%** | **52%** | **55%** | **58.3%** |
| GPT-4 | 85% | 78% | 90% | 84.3% |

**核心发现**：
- Lynx 超越 Alpaca 26 个百分点
- 接近 GPT-3.5 水平
- 但 GPT-4 仍有明显优势

---

## 层 3：深度精读

### 开场：2023 年的工具使用评估困境

**时间**：2023 年初
**地点**：阿里巴巴 DAMO 实验室
**人物**：Minghao Li, Yingxiu Zhao 团队

**场景**：
```
研究者想评估 LLM 的工具使用能力：

任务："帮我查一下北京的天气，然后推荐几个景点"

测试 Alpaca-7B:
- 输出：看起来合理的回答
- 问题：没有调用任何 API！
- 如何评分？

测试 GPT-3.5:
- 调用了天气 API
- 调用了景点推荐 API
- 如何客观比较？

困境：
"没有统一的评估标准，
每个论文用自己的数据集和指标，
无法公平比较！"

更严重的问题：
- 现有数据集的 API 不能真正执行
- 无法验证结果是否正确
- "看起来对"≠"真正对"
```

**研究者的决心**：
> "我们需要一个可执行的、统一的 benchmark！"

这就是 API-Bank 的起点。

---

### 第一章：用户需求调研

#### 调研方法

**访谈 500 个用户**：
```
问题：
1. 你希望 LLM 能调用什么类型的工具？
2. 你希望怎么用这些工具？
3. 你觉得现在的 LLM 哪里不够好？

收集到的需求：
- "我想让 LLM 帮我订机票酒店"
- "我想查实时天气和新闻"
- "我想让 LLM 调用计算器算复杂数学"
- "现在的 LLM 总是在'编'，不会查真实数据"
```

#### 需求分析

**两个关键维度**：
```
维度 1: API Pool 大小
- 少数 API (2-3 个)：直接给 LLM 选择
- 大量 API (数百个)：需要先检索

维度 2: 每轮对话的 API 调用数
- 单次调用：简单任务
- 多次调用：复杂任务需要组合
```

**能力定义**：
```
基于用户需求，定义三层能力：

Level 1: API Calling (调用能力)
- 给定 API 列表
- 选择正确的 API
- 生成正确的参数

Level 2: API Retrieving (检索能力)
- 从大量 API 中找到合适的
- 处理长 API 列表

Level 3: API Planning (规划能力)
- 分解复杂任务
- 规划多步 API 调用序列
- 整合结果
```

---

### 第二章：API-Bank 评估系统

#### 评估系统架构

**73 个可执行 API**：
```
API 类别：
- 天气查询：weather.get_current, weather.get_forecast
- 地图导航：maps.get_directions, maps.search_nearby
- 金融服务：stock.get_price, finance.get_exchange_rate
- 生活服务：restaurant.search, hotel.book, flight.search
- 娱乐：movie.search, music.play, news.get_latest

每个 API：
- 真实的 endpoint
- 可以真正执行
- 返回真实结果
```

**示例 API 定义**：
```json
{
  "name": "weather.get_current",
  "description": "Get current weather for a location",
  "parameters": {
    "location": {"type": "string", "required": true},
    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
  },
  "returns": {
    "temperature": "number",
    "condition": "string",
    "humidity": "number"
  }
}
```

---

#### 评估数据

**测试集**：
```
314 个工具使用对话
753 次 API 调用

覆盖场景：
- 单 API 调用：156 个对话
- 多 API 调用：158 个对话
- 需要检索：200 个对话
- 需要规划：114 个对话
```

**评估指标**：
```
1. API Calling Accuracy
   - 是否选择了正确的 API
   - 参数是否正确
   - 结果处理是否正确

2. API Retrieving Accuracy
   - 检索的 API 是否相关
   - Top-K 准确率

3. API Planning Accuracy
   - 任务分解是否合理
   - API 调用顺序是否正确
   - 最终结果是否正确
```

---

### 第三章：Multi-agent 数据生成

#### 核心挑战

**问题**：
```
构建训练数据需要：
1. 定义 API（功能、参数、返回值）
2. 编写用户查询
3. 标注正确的 API 调用序列
4. 验证数据质量

人工标注成本：
- 1000 个对话 ≈ $5000
- 10000 个对话 ≈ $50000
- 太贵了！
```

#### Multi-agent 方案

**5 个协作 Agent**：
```
┌─────────────────────────────────────────────────────────┐
│              Multi-agent 数据生成流程                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Agent 1: Domain Generator (领域生成器)                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  输入：无                                       │   │
│  │  输出：领域列表                                  │   │
│  │  示例：["旅行", "金融", "天气", "餐饮", "娱乐"]      │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                    │
│  Agent 2: API Generator (API 生成器)                     │   │
│  ┌─────────────────────────────────────────────────┐   │
│  │  输入：领域 "旅行"                                │   │
│  │  输出：该领域的 API 列表                            │   │
│  │  示例：[flight.search, hotel.book, ...]          │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                    │
│  Agent 3: Query Generator (查询生成器)                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  输入：API 列表                                   │   │
│  │  输出：模拟用户查询                               │   │
│  │  示例："帮我订一张明天北京到上海的机票"            │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                    │
│  Agent 4: Solution Generator (解答生成器)              │
│  ┌─────────────────────────────────────────────────┐   │
│  │  输入：用户查询 + API 列表                          │   │
│  │  输出：正确的 API 调用序列                          │   │
│  │  示例：flight.search(...) → hotel.book(...)     │   │
│  └─────────────────────────────────────────────────┘   │
│                    ↓                                    │
│  Agent 5: Quality Checker (质量检查员)                 │
│  ┌─────────────────────────────────────────────────┐   │
│  │  输入：生成的数据                                 │   │
│  │  输出：质量评分 + 修改建议                         │   │
│  │  检查：API 是否存在、调用是否合理、结果是否正确    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### 生成结果

**训练数据集**：
```
2,138 个不同的 API
1,888 个对话
4,149 次 API 调用
覆盖 1,000 个不同领域

成本对比：
- 人工标注：$50,000
- Multi-agent: $1,000 (主要是 API 调用成本)
- 成本降低 98%！
```

**数据质量验证**：
```
人工抽查 100 个生成的对话：
- API 调用正确率：94%
- 参数正确率：92%
- 查询合理性：96%

结论：Multi-agent 生成的数据质量可靠
```

---

### 第四章：Lynx 模型训练

#### 模型架构

```
Lynx = Alpaca-7B + Tool Use Fine-tuning

输入格式：
### Instruction:
帮我订一张明天北京到上海的机票

### API List:
[
  {"name": "flight.search", "description": "...", "parameters": {...}},
  {"name": "hotel.book", "description": "...", "parameters": {...}},
  ...
]

### Response:
Thought: 我需要查询明天的航班
Action: flight.search(from="北京", to="上海", date="2024-01-15")
Observation: [{"flight": "CA1501", "price": 1200}, ...]
Thought: 找到了多个航班，给用户选择
Response: 为您找到以下航班...
```

#### 训练策略

**三阶段训练**：
```
Stage 1: API Calling 训练
- 数据：给定 API 列表的对话
- 目标：学会正确调用 API

Stage 2: API Retrieving 训练
- 数据：需要从大量 API 中检索
- 目标：学会检索相关 API

Stage 3: API Planning 训练
- 数据：复杂多步任务
- 目标：学会规划和整合
```

---

### 第五章：大量实例

#### 【实例 1：单 API 调用】

**任务**：
```
用户："北京今天天气怎么样？"

给定 API 列表：
- weather.get_current(location, unit)
- weather.get_forecast(location, days)
- news.get_latest(category)
```

**Lynx 的执行**：
```
Thought: 用户想查询当前天气，需要调用 weather.get_current
Action: weather.get_current(location="北京", unit="celsius")
Observation: {"temperature": 25, "condition": "sunny", "humidity": 60}
Thought: 已经获取天气信息，可以回复用户
Response: "北京今天天气晴朗，气温 25°C，湿度 60%"
```

**对比 Alpaca**：
```
Alpaca:
"北京今天天气应该不错，气温可能在 20 多度。
建议您出门带伞，因为有可能会下雨。"

问题：
- 没有调用 API
- 完全是"编"的
- 还自相矛盾（晴朗却说带伞）
```

---

#### 【实例 2：多 API 调用】

**任务**：
```
用户："我想去上海旅行三天，帮我规划一下"

给定 API 列表（部分）：
- flight.search(from, to, date)
- hotel.search(location, check_in, check_out)
- attraction.search(city, days)
- weather.get_forecast(location, days)
```

**Lynx 的执行**：
```
Round 1:
Thought: 首先我需要知道用户从哪里出发
Response: "请问您从哪个城市出发？"

User: "北京"

Round 2:
Thought: 需要查询航班、酒店和景点
Action: flight.search(from="北京", to="上海", date="2024-01-15")
Observation: [{"flight": "CA1501", "price": 1200}, ...]

Action: hotel.search(location="上海", check_in="2024-01-15", check_out="2024-01-18")
Observation: [{"name": "上海大酒店", "price": 800}, ...]

Action: attraction.search(city="上海", days=3)
Observation: ["外滩", "东方明珠", "迪士尼乐园", ...]

Action: weather.get_forecast(location="上海", days=3)
Observation: [{"date": "01-15", "temp": 10}, ...]

Round 3:
Thought: 整合所有信息，给用户完整的旅行建议
Response: "为您规划上海三日游：

交通：
- 推荐航班：CA1501，北京→上海，1200 元

住宿：
- 推荐酒店：上海大酒店，800 元/晚

景点：
- Day 1: 外滩 → 南京路
- Day 2: 东方明珠 → 陆家嘴
- Day 3: 迪士尼乐园

天气：
- 上海未来三天气温 8-12°C，请带外套

总预算：约 5000 元/人"
```

---

#### 【实例 3：API 检索】

**任务**：
```
用户："帮我查一下特斯拉的股价"

API Pool: 200+ 个 API
（不能全部给 LLM，需要先检索）
```

**Lynx 的执行**：
```
Step 1 - Retrieval:
检索查询："stock price, tesla, TSLA"

检索结果（Top 5）：
1. stock.get_price(symbol)
2. stock.get_history(symbol, days)
3. finance.get_news(company)
4. crypto.get_price(symbol) ← 不相关
5. currency.get_exchange(from, to) ← 不相关

Step 2 - Selection:
选择：stock.get_price

Step 3 - Call:
Action: stock.get_price(symbol="TSLA")
Observation: {"price": 245.67, "change": "+3.2%"}

Response: "特斯拉 (TSLA) 当前股价为 245.67 美元，上涨 3.2%"
```

---

### 第六章：预期 vs 实际

### 你的直觉 vs API-Bank 的发现

| 维度 | 你的直觉/预期 | API-Bank 实际发现 | 为什么 |
|------|--------------|------------------|--------|
| **大模型工具能力** | 模型越大越强 | GPT-3 Davinci 几乎不会 | 工具使用不是涌现能力 |
| **GPT-3.5 vs GPT-4** | 应该差不多 | GPT-4 领先 20% | GPT-4 规划能力更强 |
| **开源模型** | 应该很差 | Alpaca 有 20% 调用准确率 | 基本的 API 理解还在 |
| **检索能力** | 应该不难 | 小模型几乎为 0 | 需要专门训练 |
| **微调效果** | 提升有限 | +26% 大幅提升 | 工具使用可以训练 |

---

### 反直觉挑战

**问题：如果增大模型规模，工具使用能力会自动提升吗？**

[先想 1 分钟...]

**直觉**："模型越大越聪明，应该会更强吧？"

**实际**：
```
Alpaca-7B:   22% API 调用准确率
ChatGLM-6B:  18% API 调用准确率
GPT-3 Davinci (175B): 2% API 调用准确率 ← 反而更差！

结论：工具使用能力 NOT 涌现能力
需要专门训练！
```

**为什么 GPT-3 Davinci 表现差**：
```
可能原因：
1. 没有在工具使用数据上训练
2. 倾向于"编"答案而不是调用 API
3. 没有经过 instruction tuning for tool use

启示：
- 通用能力 ≠ 工具使用能力
- 需要针对性的训练数据
```

---

### 第七章：关键实验

#### 实验 1: 三层能力评估

**结果汇总**：
```
API Calling (调用能力):
- Alpaca-7B:    22%
- Lynx:         68% ← +46%
- GPT-3.5:      65%
- GPT-4:        85%

API Retrieving (检索能力):
- Alpaca-7B:    5%
- Lynx:         52% ← +47%
- GPT-3.5:      58%
- GPT-4:        78%

API Planning (规划能力):
- Alpaca-7B:    8%
- Lynx:         55% ← +47%
- GPT-3.5:      72%
- GPT-4:        90%
```

**关键发现**：
- Lynx 在三层能力上都有大幅提升
- 调用能力接近 GPT-3.5
- 规划能力与 GPT-4 差距较大

---

#### 实验 2: 错误分析

**Lynx 的主要错误类型**：
```
1. 参数错误 (35%)
   - 参数名记错
   - 参数类型不对
   - 缺少必需参数

2. API 选择错误 (25%)
   - 选了功能不匹配的 API
   - 有多个 API 可选时选错

3. 规划错误 (20%)
   - API 调用顺序不对
   - 缺少必要的步骤

4. 结果处理错误 (20%)
   - 没有正确解析 API 返回
   - 整合多个结果时出错
```

**GPT-4 的主要错误类型**：
```
1. 参数错误 (45%)
   - 主要是细节参数

2. API 选择错误 (15%)
   - 很少选错

3. 规划错误 (10%)
   - 规划能力很强

4. 结果处理错误 (30%)
   - 主要是复杂整合场景
```

**对比发现**：
- GPT-4 规划能力明显更强
- Lynx 在基础调用上接近 GPT-3.5
- 参数细节都需要改进

---

### 第八章：局限性与改进

#### 局限性

**1. API 数量有限**
```
API-Bank: 73 个评估 API
ToolLLM: 16,464 个 API

差距：API-Bank 规模较小
原因：需要可执行、人工验证
```

**2. 领域覆盖不足**
```
主要覆盖：
- 旅行、天气、金融、餐饮

缺少：
- 专业领域（医疗、法律）
- 创意工具（图像生成、音乐）
```

**3. 评估场景有限**
```
主要是：
- 单轮或多轮对话
- 明确的 API 调用需求

缺少：
- 开放式任务
- 需要创造力的场景
```

#### 改进方向

**1. 扩展 API 库**
```
计划：
- 增加到 500+ 可执行 API
- 覆盖更多领域
- 社区贡献 API
```

**2. 动态评估**
```
当前：静态 API 定义
未来：
- API 可能变化
- 评估模型适应新 API 的能力
```

**3. 人类评估**
```
当前：自动评估为主
补充：
- 人类评判回答质量
- 用户满意度调查
```

---

### 第九章：如何应用

#### 使用 API-Bank 评估

```python
from api_bank import evaluate

# 加载模型
model = load_model("your-llm")

# 评估
results = evaluate(
    model=model,
    benchmark="api-bank",
    capabilities=["calling", "retrieving", "planning"]
)

# 输出报告
print(f"API Calling: {results['calling_accuracy']}")
print(f"API Retrieving: {results['retrieving_accuracy']}")
print(f"API Planning: {results['planning_accuracy']}")
```

#### 微调自己的 Lynx

```python
from transformers import AutoModelForCausalLM, TrainingArguments

# 加载基座模型
model = AutoModelForCausalLM.from_pretrained("alpaca-7b")

# 加载 API-Bank 训练数据
train_data = load_dataset("api-bank/train")

# 训练
training_args = TrainingArguments(
    output_dir="./lynx",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)

trainer.train()

# 保存模型
model.save_pretrained("./lynx")
```

---

### 第十章：延伸思考

[停下来，想一想...]

1. **API-Bank 和 ToolBench 有什么区别？**
   - API-Bank: 评估为主，73 个可执行 API
   - ToolBench: 训练为主，16,464 个 API
   - 哪个更有价值？

2. **工具使用能力是 AGI 的必要条件吗？**
   - 人类智能的核心是工具使用
   - LLM 的工具使用是"真能力"还是"模仿"？

3. **如果让 LLM 自己"发明"API，会怎样？**
   - 定义新的工具组合
   - 创造新的能力

4. **API 检索的本质是什么？**
   - 语义匹配？
   - 功能理解？
   - 还是模式匹配？

5. **GPT-4 的规划能力为什么这么强？**
   - 是训练数据的原因？
   - 还是架构的原因？
   - 开源模型能追上吗？

---

## 附录：论文定位图谱

```
                    LLM 工具使用评估发展史

传统 Benchmark ────┬────→ 纯语言任务评估
(GLUE, SuperGLUE)  │
                   │
Tool-use Benchmarks─┤
                   │
API-Bank ──────────┼────→ 首个可执行评估系统【本文】
(2023.04)          │
                   │
ToolBench ─────────┤
(2023.07)          │     → 大规模训练数据
                   │
APIBench ──────────┘
(2023)             │
                   └────→ 共同主题：评估 + 提升工具使用能力

上游工作：
- ReACT: 推理 + 行动的框架
- Toolformer: 工具学习的可能性

下游工作：
- ToolLLM: 基于 API-Bank 的大规模扩展
- FireAct: 改进的推理策略

影响力：
- 首个统一的工具使用评估标准
- 三层能力定义成为参考框架
- Multi-agent 数据生成方法被广泛采用
```

---

## 写作检查清单

- [x] 电梯演讲层
- [x] 故事摘要层（含框架大图）
- [x] 深度精读层
- [x] 具体失败场景开场
- [x] 故事化叙述
- [x] 多角度实例
- [x] 预期 vs 实际对比表
- [x] 反直觉挑战
- [x] 关键实验细节
- [x] 局限性分析
- [x] 应用指南
- [x] 延伸思考
- [x] 论文定位图谱

---

## 关键术语表

| 术语 | 含义 |
|------|------|
| Tool-Augmented LLM | 能够使用外部工具的 LLM |
| API Calling | 正确调用给定 API 的能力 |
| API Retrieving | 从大量 API 中检索相关 API 的能力 |
| API Planning | 规划多步 API 调用序列的能力 |
| Multi-agent | 多个 LLM 协作生成数据的方法 |
| Lynx | 基于 Alpaca-7B 微调的工具增强 LLM |
