# Gorilla: Large Language Model Connected with Massive APIs

## 开场：一个失败的场景

**时间**：2023年5月的一个下午
**地点**：UC Berkeley 实验室
**人物**：Gorilla 研究团队正在测试最新的模型

**场景**：

研究员让模型调用一个天气 API：

```
User: "What's the weather in Tokyo?"
Model: get_weather(city="Tokyo", units="celsius")
```

很好！模型正确调用了 API。

"现在试试这个，" 研究员说：

```
User: "What's the weather in Tokyo next Monday?"
Model: get_weather_forecast(city="Tokyo", date="2023-05-15", units="celsius", include_humidity=True, include_wind=True, hourly=True)
```

研究员皱眉——问题来了：

1. **API 不存在**：没有 `get_weather_forecast` 这个 API
2. **参数错误**：真实 API 只需要 `city`，不需要那些参数
3. **幻觉**：模型编造了一个看起来很合理的 API 调用

"这就是 **API Hallucination**，" 研究员解释，"模型看起来很自信，但调用的 API 根本不存在。"

**更糟的是**：

- 模型可能调用 `get_wether(city="Tokyo")`（拼写错误）
- 模型可能调用 `weather("Tokyo", "metric", True, False)`（参数数量错误）
- 模型可能调用 `GET /api/v1/weather/{city}`（格式不匹配）

**研究者的困境**：

Toolformer 证明了 LLM 可以学会使用工具，但在真实 API 调用场景中存在严重问题：
- **API 幻觉**：编造不存在的 API
- **参数格式错误**：参数类型、数量、顺序错误
- **无法跟随 API 演化**：API 更新后模型无法适应

"如果我们能让模型**准确理解** API 文档并**正确调用** API，特别是在 API 不断演化的情况下，那会怎样？"

这个想法，最终变成了 Gorilla。

---

## 第一章：研究者的困境

### 当时学界卡在哪里？

**2023年5月的核心困境**：Toolformer 等开创性工作证明了 LLM 可以学会使用工具，但在实际 API 调用场景中存在严重问题——**API 幻觉**和**参数格式错误**。

**具体表现**：

1. **API 幻觉（API Hallucination）**
   - 编造不存在的 API 名称
   - 混淆不同 API 的功能
   - 创造不合理的参数

2. **参数格式错误**
   - 参数类型错误（字符串 vs 数字）
   - 必需参数缺失
   - 可选参数处理不当
   - 参数顺序错误

3. **API 演化问题**
   - API 更新后模型无法使用
   - 无法理解版本差异
   - 无法适应新的 API 设计

4. **大规模 API 场景**
   - 现实中有成千上万个 API
   - 上下文窗口有限，无法容纳所有 API 文档
   - 需要检索能力

### 旧方案的失败

**方案1：Toolformer 的 self-supervised 方法**

问题清单：
- 主要针对 text-to-text 的简单 API
- 没有专门解决 API 调用格式问题
- 无法处理复杂的嵌套 API 结构
- 对 API 幻觉问题关注不足

**具体失败案例**：

```
任务：调用 Spotify API 搜索歌曲
Toolformer输出：spotify_search(query="song name", artist="artist", limit=10, offset=0, market="US")
问题：真实API是search(type="track", q="song name")
→ Toolformer编造了参数名和格式
```

**方案2：传统的 Fine-tuning**

问题清单：
- 需要大量 API 调用标注数据
- API 更新后需要重新训练
- 无法泛化到未见过的 API
- 标注成本极高

**具体失败案例**：

研究团队尝试微调 GPT-4 使用 GitHub API：
- 收集了 10,000 个真实的 API 调用示例
- 花费数周标注
- 训练后，模型在训练过的 API 上表现很好
- 但：GitHub API v3 → v4 更新后，模型完全失效
- 需要重新收集数据、重新训练

**方案3：In-context Learning**

问题清单：
- 上下文窗口有限，无法放太多 API 文档
- 无法处理长尾 API
- API 文档理解能力有限
- 成本高（每次推理都要提供文档）

**具体失败案例**：

在 prompt 中提供 API 文档：
```
API: get_weather(city)
Description: Get weather for a city
Parameters:
- city (string, required): The city name

User: "What's the weather in Tokyo and Paris?"
模型可能：
1. 只调用一个城市（遗漏）
2. 编造格式：get_weather(cities=["Tokyo", "Paris"])
3. 正确但笨拙：两次分别调用
```

**研究者面对的核心问题**：

> 如何让 LLM **准确理解** API 文档并**正确调用** API，特别是在 API 不断演化的情况下？

---

## 第二章：试错的旅程

### 第一阶段：问题诊断

**2023年初，团队的分析**

"Toolformer 的方法很好，但不够用，" 首席研究员说。

**关键洞察**：

API 调用与一般工具使用不同，有独特的挑战：
1. **格式严格性**：参数格式必须完全正确
2. **API 量级**：现实中有成千上万个 API
3. **API 演化**：API 会更新、废弃，模型需要适应

**对 Toolformer 的批判性继承**：
- 继承：让模型自主学习的思想
- 批判：没有专门针对 API 调用格式优化
- 改进：需要专门的 API 调用训练方法

### 第二阶段：核心方法设计

**关键设计1：训练数据构造**

**创新点**：使用 self-instruction 生成训练数据

**流程**：
```
1. 收集真实 API 文档
   ├── 从 API hub（如 RapidAPI）
   ├── 从官方文档
   └── 标准化格式

2. 让模型生成可能的 query
   ├── 基于用户场景
   ├── 覆盖不同参数组合
   └── 包含边界情况

3. 生成对应的 API 调用
   ├── 根据 API 文档
   ├── 确保语法正确
   └── 验证调用的有效性

4. 验证调用的正确性
   ├── 语法检查（AST）
   ├── 语义验证
   └── 执行测试（如果可能）

5. 筛选高质量样本
   ├── 去除低质量样本
   ├── 平衡不同难度
   └── 确保多样性
```

**读者陷阱**：容易认为这是简单的数据增强。实际上关键是**基于真实 API 文档的 self-instruction**，保证了 API 调用的真实性。

**代码实例**：

```python
# Self-instruction 示例

# 1. API 文档
"""
API: get_movie_info(movie_id)
Description: Get detailed information about a movie
Parameters:
- movie_id (string, required): IMDb ID of the movie
"""

# 2. 生成的 query
queries = [
    "Tell me about the movie tt0111161",
    "What's the movie The Shawshank Redemption about?",
    "Get details for Shawshank Redemption",
    "Movie info for tt0111161 please"
]

# 3. 对应的 API 调用
api_calls = [
    'get_movie_info(movie_id="tt0111161")',
    'get_movie_info(movie_id="tt0111161")',
    'get_movie_info(movie_id="tt0111161")',  # 需要先知道 IMDB ID
    'get_movie_info(movie_id="tt0111161")'
]

# 注意：模型学会了将不同的 query 映射到正确的 API 调用
```

**关键设计2：Retrieval-aware Training**

**问题**：上下文窗口有限，无法塞入所有 API 文档

**解决**：训练时模拟检索场景

**方法**：
```
1. 每个样本随机提供 k 个 API 文档
   ├── 其中 1 个是正确的（ground truth API）
   ├── k-1 个是干扰项（其他 API）
   └── 模型需要从 k 个中选择正确的 API

2. 训练目标
   ├── 学会从多个 API 中识别正确的
   ├── 学会忽略无关 API
   └── 学会根据功能选择 API

3. 难度递进
   ├── k=2（二选一）
   ├── k=5（五选一）
   └── k=10 或更多（模拟真实场景）
```

**读者陷阱**：这不是简单的检索增强，而是**让模型学会在检索场景下工作**。训练和测试的 gap 因此大大缩小。

**对比实例**：

```python
# 传统方法 vs Retrieval-aware Training

# 传统方法
训练时：只提供目标 API 文档
"API: get_weather(city)"
测试时：从 1000 个 API 中检索
→ Gap！模型从未见过"选择"的场景

# Gorilla 的方法
训练时：提供 k 个 API（1 个正确，k-1 个干扰）
"Available APIs:
1. get_weather(city)
2. get_weather_forecast(city, date)
3. get_temperature(city)
4. search_movies(query)
5. get_movie_info(id)

Which API should I use for 'weather in Tokyo'?"
→ 模型学会选择 API
测试时：同样从多个 API 中选择
→ 无 gap！
```

**关键设计3：AST-based Evaluation**

**问题**：传统的 string matching 评估太严格

**例子**：
```python
# 预期输出
get_weather(city="Tokyo")

# 模型输出 1
get_weather(city = "Tokyo")  # 多了空格
→ String match: ✗ (不同)
→ AST match: ✓ (相同语义)

# 模型输出 2
get_weather("Tokyo")  # 位置参数
→ String match: ✗ (不同)
→ AST match: ✓ (相同语义)

# 模型输出 3
get_weather(city="Tokyo", units="celsius")  # 多了可选参数
→ String match: ✗ (不同)
→ AST match: ✓ (正确调用，多了额外参数)
```

**解决**：使用抽象语法树 (AST) 比较

**好处**：
- 忽略格式差异（空格、换行）
- 关注语义正确性
- 更符合实际应用场景

**顿悟时刻**：API 调用的"正确性"应该是语义层面的，不是字符层面的。AST 比较是更合理的评估方法。

### 第三阶段：实验验证与迭代

**实验1：基础 API 调用能力**

**设置**：
- 数据集：TASF（API 调用数据集）
- Baseline：GPT-4, Claude, Toolformer
- 评估：String match vs AST match

**结果**：

| 模型 | String Match | AST Match |
|------|-------------|-----------|
| GPT-4 | 67% | 73% |
| Claude | 65% | 71% |
| Toolformer | 45% | 52% |
| **Gorilla (7B)** | **71%** | **89%** |

**关键发现**：
- 专门针对 API 调用的 training 确实有效
- 即使是 7B 模型也能达到很高的 API 调用准确率
- AST evaluation 下的性能更高（更合理的评估）

**实验2：API 幻觉检测**

**设置**：
- 测试模型是否会调用不存在的 API
- 提供部分真实 API + 部分虚构 API

**结果**：

| 模型 | 幻觉率 |
|------|--------|
| GPT-4 | 23% |
| Claude | 19% |
| Toolformer | 31% |
| **Gorilla (7B)** | **7%** |

**关键发现**：
- Gorilla 显著降低 API 幻觉率
- 主要因为训练数据都是基于真实 API 的
- Self-instruction 从真实 API 文档生成数据，天然避免了 API 幻觉

**实验3：Retrieval 场景**

**设置**：
- 提供 128 个 API 文档（其中 1 个相关）
- 测试模型能否找到并正确调用相关 API

**结果**：

| k（API 数量） | Gorilla (无 RAT) | Gorilla (有 RAT) |
|--------------|----------------|-----------------|
| 5 | 67% | **82%** |
| 10 | 54% | **76%** |
| 50 | 23% | **61%** |
| 128 | 12% | **54%** |

**关键发现**：
- Retrieval-aware training (RAT) 显著提升性能
- 训练测试一致性真的很重要
- 模型确实学会了"从多个 API 中选择"

**实验4：API 演化适应**

**设置**：
- 模拟 API 更新场景
- 在 API v1 上训练，在 API v2 上测试

**示例**：
```
# API v1
get_weather(city)

# API v2
get_current_weather(location, units="metric")

# 变化：
1. API 名称改变
2. 参数名改变 (city → location)
3. 新增可选参数
```

**结果**：

| 方法 | v1→v2 适应率 | v2→v3 适应率 |
|------|------------|------------|
| Toolformer | 34% | 21% |
| 传统微调 | 28% | 15% |
| **Gorilla** | **67%** | **59%** |

**关键发现**：
- 只需要提供更新后的 API 文档
- 模型能适应 API 变化
- 学到的是"阅读 API 文档"的能力，不是记忆特定 API

---

## 第三章：核心概念 - 大量实例

### 概念1：API Hallucination

**这是一个新提出的核心问题，让我们深入理解它。**

#### 生活类比（3个）

**类比1：想用手机 App 但编造名字**

想象你想点外卖：
- 你记得有个"饿了么"App
- 但你错误地说"打开'饭来了'App"
- 手机：没有这个 App
- 你：困惑，但很自信地说"有这个 App！"

这就是 API Hallucination——模型自信地调用不存在的 API。

**类比2：记错了函数调用**

你是程序员，写过很多代码：
- 你记得有个 `get_data()` 函数
- 但实际库中只有 `fetch_data()` 和 `retrieve_data()`
- 你自信地写 `result = get_data(id=123)`
- 编译错误：`NameError: name 'get_data' is not defined`

模型也会犯同样的错误——"记住"了不存在的 API。

**类比3：点菜时的幻觉**

你去餐厅：
- 你记得菜单上有"宫保鸡丁"
- 但实际只有"辣子鸡"和"口水鸡"
- 你坚持点"宫保鸡丁"
- 服务员：我们没有这道菜

模型也会这样——坚持调用不存在的 API，即使文档中没写。

#### 代码实例（5个）

**实例1：编造 API 名称**

```python
# 用户请求
User: "Search for movies directed by Nolan"

# 模型幻觉
Model: search_director_movies(director="Christopher Nolan")
# 问题：真实 API 是 search_movies(query="Christopher Nolan director")
```

**实例2：编造参数**

```python
# 用户请求
User: "Get weather in Tokyo"

# 模型幻觉
Model: get_weather(city="Tokyo", date="today", detailed=True, hourly=True)
# 问题：真实 API 只有 get_weather(city)
#       其他参数都是模型编造的
```

**实例3：混淆 API**

```python
# 用户请求
User: "Play some music"

# 模型幻觉
Model: spotify_play(song="Shape of You")
# 问题：应该是 spotify_search 然后 spotify_play
#       或者直接 spotify_play_track(uri="...")
```

**实例4：格式幻觉**

```python
# 用户请求
User: "Get user profile"

# 模型幻觉
Model: GET /api/v1/users/{id}
# 问题：模型混淆了 REST API 格式和函数调用格式
#       应该是 get_user_profile(user_id="123")
```

**实例5：Gorilla 的正确行为**

```python
# 用户请求
User: "Search for movies directed by Nolan"

# Gorilla（经过训练）
Model: search_movies(query="Christopher Nolan director")
# 正确！使用真实 API，格式正确
```

#### 对比场景实例（3个）

**对比1：有幻觉 vs 无幻觉**

| 场景 | 有幻觉的模型 | Gorilla (无幻觉) |
|------|------------|----------------|
| "播放音乐" | `music_play(song="...")` ✗ | `spotify_play(track_uri="...")` ✓ |
| "获取天气" | `get_weather_data(city="...")` ✗ | `get_weather(city="...")` ✓ |
| "搜索电影" | `movie_search(query="...")` ✗ | `search_movies(query="...")` ✓ |

**对比2：幻觉的严重程度**

| 幻觉类型 | 影响 | 示例 |
|---------|------|------|
| API 名称错误 | 调用失败 | `get_wether` (拼写错误) |
| 参数名错误 | 调用失败 | `get_weather(location=...)` (应该是 city) |
| 参数类型错误 | 可能失败 | `get_weather(city=123)` (应该是字符串) |
| 编造参数 | 可能被忽略 | `get_weather(city="Tokyo", detailed=True)` |

**对比3：不同模型的幻觉率**

| 模型 | 幻觉率 | 主要问题 |
|------|--------|---------|
| GPT-4 | 23% | 参数格式错误 |
| Claude | 19% | API 名称混淆 |
| Toolformer | 31% | 编造复杂参数 |
| **Gorilla** | **7%** | 少数边界情况 |

#### 逐步演化实例（1个）

**演化：从幻觉到准确调用**

**版本1：无训练（严重幻觉）**
```python
User: "What's the weather in Tokyo?"
Model: get_current_weather_data(city_name="Tokyo", units="celsius", detailed=True)
# 问题：
# 1. API 名称错误（多了 current, data）
# 2. 参数名错误（city_name vs city）
# 3. 编造参数（detailed）
```

**版本2：传统微调（减少幻觉）**
```python
User: "What's the weather in Tokyo?"
Model: get_weather(city="Tokyo", units="celsius")
# 改进：
# 1. API 名称正确
# 2. 参数名正确
# 问题：units 参数不存在（可选参数幻觉）
```

**版本3：Gorilla（准确调用）**
```python
User: "What's the weather in Tokyo?"
Model: get_weather(city="Tokyo")
# 完全正确！
```

---

### 概念2：Retrieval-aware Training (RAT)

**这是 Gorilla 的第二个核心创新。**

#### 生活类比（2个）

**类比1：考试时的选择题**

**传统训练**：
- 老师教你"答案是 A"
- 考试时直接写 A

**Retrieval-aware Training**：
- 老师给你 A、B、C、D 四个选项
- 让你选择正确答案
- 考试时也是选择题

同样的问题，训练方式不同，适应能力也不同。

**类比2：超市找商品**

**传统方法**：
- 记住"牛奶在 A3 过道"
- 超市重新布局后，找不到

**RAT 方法**：
- 学会"看指示牌找商品"
- 超市重新布局后，仍然能找到

Gorilla 让模型学会"看文档找 API"，而不是"记住特定 API"。

#### 代码实例（4个）

**实例1：RAT 的训练格式**

```python
# 训练样本格式
"""
Available APIs:
1. get_weather(city) - Get current weather
2. get_weather_forecast(city, date) - Get weather forecast
3. get_temperature(city) - Get temperature only
4. search_movies(query) - Search for movies
5. get_movie_info(movie_id) - Get movie details

User query: "What's the weather in Tokyo?"

Think: The user wants current weather, not forecast or just temperature.
Choose: get_weather(city="Tokyo")
"""

# 模型学会：
# 1. 阅读所有可用 API
# 2. 理解每个 API 的功能
# 3. 选择最合适的 API
# 4. 正确调用选中的 API
```

**实例2：不同 k 值的训练**

```python
# k=2（简单）
Available APIs:
1. get_weather(city)
2. get_temperature(city)

User: "Weather in Tokyo?"
→ 模型学会区分功能差异

# k=5（中等）
Available APIs:
1. get_weather(city)
2. get_weather_forecast(city, date)
3. get_temperature(city)
4. get_humidity(city)
5. get_wind(city)

User: "Weather in Tokyo?"
→ 模型学会选择最合适的

# k=10+（困难）
Available APIs:
[10 个相似的 weather 相关 API]

User: "Weather in Tokyo?"
→ 模型学会细粒度的区分
```

**实例3：干扰 API 的设计**

```python
# 好的干扰 API设计
Available APIs:
1. get_weather(city)  # 正确
2. get_temperature(city)  # 相似功能
3. get_forecast(city)  # 相似名称
4. search_weather(query)  # 相似关键词
5. weather_alert(city)  # 相似领域

→ 干扰项与正确 API 相关但有区别
→ 训练模型学会精细区分

# 坏的干扰 API 设计
Available APIs:
1. get_weather(city)  # 正确
2. search_movies(query)  # 完全无关
3. get_user(id)  # 完全无关
4. send_email(to, subject)  # 完全无关

→ 干扰项太容易区分
→ 训练效果不佳
```

**实例4：测试时的检索场景**

```python
# 测试场景
# 系统有 1000+ API，只能检索 top-k

# Step 1: 检索
User: "What's the weather in Tokyo?"
检索系统 → 返回 top 5 最相关的 API：
1. get_weather(city)
2. get_weather_forecast(city, date)
3. get_temperature(city)
4. weather_alert(city)
5. get_climate(city)

# Step 2: 模型选择
模型（经过 RAT 训练）：
→ 理解用户想要"当前天气"
→ 排除 forecast, temperature_only, alert, climate
→ 选择 get_weather(city="Tokyo")

# 关键：模型在训练时见过类似的场景！
```

#### 对比场景（2个）

**对比1：有 RAT vs 无 RAT**

| 场景 | 无 RAT | 有 RAT (Gorilla) |
|------|--------|-----------------|
| 训练时 | 只见目标 API | 见目标 API + 干扰项 |
| 测试时 | 需要从 1000 个 API 选择 | 从检索的 top-k 选择 |
| Gap | 大（训练测试不一致） | 小（场景一致） |
| 性能 | 在大规模 API 下差 | 在大规模 API 下好 |

**对比2：不同 k 值的效果**

| k 值 | 训练难度 | 测试性能 | 适用场景 |
|-----|---------|---------|---------|
| k=2 | 简单 | 小规模 API 好 | API 数量少 |
| k=5 | 中等 | 中等规模 API 好 | 一般场景 |
| k=10+ | 困难 | 大规模 API 好 | 真实场景 |

#### 演化实例（1个）

**演化：从固定 API 到 Retrieval-aware**

**版本1：固定 API（传统方法）**
```python
# 训练
训练数据：每个样本只包含目标 API
"API: get_weather(city)"
"User: Weather in Tokyo?"
"Call: get_weather(city='Tokyo')"

# 问题
测试时：从 1000 个 API 中检索
→ 模型从未见过"选择"的场景
→ 性能下降
```

**版本2：简单检索（naive）**
```python
# 训练
训练数据：随机加入一些无关 API
"Available: get_weather(city), search_movies(query), get_user(id)"
"User: Weather in Tokyo?"
"Call: get_weather(city='Tokyo')"

# 问题
干扰项太容易区分
→ 没有真正学会"选择"
```

**版本3：Gorilla RAT（完整版）**
```python
# 训练
训练数据：精心设计的干扰项
"Available:
1. get_weather(city)  # 目标
2. get_temperature(city)  # 相似功能
3. get_forecast(city)  # 相似名称
4. weather_alert(city)  # 相关领域
5. get_climate_summary(city)  # 容易混淆"

"User: Weather in Tokyo?"
"Think: User wants current weather, not temperature, forecast, alert, or climate"
"Call: get_weather(city='Tokyo')"

# 成功
→ 模型学会精细区分
→ 测试时从检索结果中选择，表现良好
```

---

### 概念3：AST-based Evaluation

**这是 Gorilla 的第三个关键创新，改变了 API 调用的评估方式。**

#### 生活类比（2个）

**类比1：评分作文的宽容度**

**严格评分**（String match）：
- "The weather is sunny" ✓
- "The weather is sunny." ✗ (多了句号)
- "weather is sunny" ✗ (首字母大写问题)

**合理评分**（AST match）：
- "The weather is sunny" ✓
- "The weather is sunny." ✓ (意义相同)
- "weather is sunny" ✓ (小问题，不影响理解)

**类比2：编程代码的等价性**

```python
# 版本1
result = get_weather(city="Tokyo")

# 版本2
result = get_weather(city = "Tokyo")  # 多了空格

# 版本3
result = get_weather("Tokyo")  # 位置参数

# String match 认为：三者都不同
# AST match 认为：三者语义相同
```

#### 代码实例（4个）

**实例1：String Match vs AST Match**

```python
# 预期输出
get_weather(city="Tokyo", units="metric")

# 模型输出变体
output1 = "get_weather(city='Tokyo')"  # 少了可选参数
output2 = "get_weather(city = 'Tokyo', units='metric')"  # 空格差异
output3 = "get_weather('Tokyo', 'metric')"  # 位置参数
output4 = "get_weather(city='Tokyo', units='metric')"  # 单引号 vs 双引号

# String Match 结果
output1: ✗ (不匹配)
output2: ✗ (空格不同)
output3: ✗ (格式不同)
output4: ✗ (引号不同)

# AST Match 结果
output1: ✓ (语义正确，少可选参数可接受)
output2: ✓ (完全等价)
output3: ✓ (语义等价)
output4: ✓ (完全等价)
```

**实例2：AST 解析示例**

```python
import ast

# 解析 API 调用
code = 'get_weather(city="Tokyo", units="metric")'
tree = ast.parse(code)

# AST 结构
# Call(
#   func=Name(id='get_weather'),
#   args=[],
#   keywords=[
#     keyword(arg='city', value=Constant(value='Tokyo')),
#     keyword(arg='units', value=Constant(value='metric'))
#   ]
# )

# 比较
def api_calls_equal(call1, call2):
    tree1 = ast.parse(call1)
    tree2 = ast.parse(call2)
    # 比较：函数名、参数名、参数值
    # 忽略：空格、换行、引号类型等
    return ast.dump(tree1) == ast.dump(tree2)
```

**实例3：评估不同质量的输出**

```python
# 预期
get_weather(city="Tokyo")

# 不同质量等级
# Level 1: 完全正确（AST +）
get_weather(city="Tokyo")

# Level 2: 格式不同但语义正确（AST +）
get_weather(city = "Tokyo")
get_weather("Tokyo")  # 如果只有一个参数

# Level 3: 多了可选参数（AST +）
get_weather(city="Tokyo", units="metric")

# Level 4: 参数名错误（AST -）
get_weather(location="Tokyo")

# Level 5: API 名称错误（AST -）
get_current_weather(city="Tokyo")

# String Match: 只有 Level 1 算对
# AST Match: Level 1-3 都算对
```

**实例4：实际评估结果对比**

```python
# 研究中的实际结果

# 模型 A（保守，严格遵循格式）
String Match: 71% accuracy
AST Match: 73% accuracy
→ 差距小（输出格式规范）

# 模型 B（灵活，格式不严格）
String Match: 45% accuracy
AST Match: 67% accuracy
→ 差距大（AST 揭示真实能力）

# 模型 C（混乱，但语义正确）
String Match: 23% accuracy
AST Match: 58% accuracy
→ 差距巨大（String Match 严重低估）
```

#### 对比场景（2个）

**对比1：不同评估方法的差异**

| 评估方法 | 关注点 | 优点 | 缺点 | 适用场景 |
|---------|--------|------|------|---------|
| String Match | 字符完全匹配 | 简单、严格 | 太严格，误判多 | 格式要求严格的场景 |
| AST Match | 语义等价 | 合理、宽容 | 需要解析 | API 调用评估 |
| Execution | 实际执行 | 最准确 | 成本高、不安全 | 测试环境 |
| Human | 人工判断 | 最灵活 | 成本高、主观 | 小规模评估 |

**对比2：AST Match 的影响**

| 场景 | String Match 下排名 | AST Match 下排名 | 变化 |
|------|------------------|-----------------|------|
| GPT-4 | 1st | 1st | 不变 |
| Claude | 2nd | 2nd | 不变 |
| Gorilla 7B | 4th | 1st | ↑ 大幅提升 |
| Toolformer | 3rd | 4th | ↓ 相对下降 |

**关键insight**：AST Match 更能反映模型的真实 API 调用能力。

#### 演化实例（1个）

**演化：评估方法的进步**

**版本1：String Match（早期）**
```python
# 评估代码
def evaluate(prediction, ground_truth):
    return prediction == ground_truth

# 问题
evaluate('get_weather(city="Tokyo")', 'get_weather(city = "Tokyo")')
# → False（显然不合理）
```

**版本2：规范化后比较（改进）**
```python
# 评估代码
def normalize(code):
    return code.replace(' ', '').replace('"', "'")

def evaluate(prediction, ground_truth):
    return normalize(prediction) == normalize(ground_truth)

# 改进
evaluate('get_weather(city="Tokyo")', 'get_weather(city = "Tokyo")')
# → True

# 但仍有问题
evaluate('get_weather("Tokyo")', 'get_weather(city="Tokyo")')
# → False（位置参数 vs 关键字参数）
```

**版本3：AST Match（Gorilla）**
```python
# 评估代码
def evaluate_ast(prediction, ground_truth):
    tree1 = ast.parse(prediction)
    tree2 = ast.parse(ground_truth)
    return compare_ast_semantics(tree1, tree2)

# 完整解决
evaluate_ast('get_weather(city="Tokyo")', 'get_weather(city = "Tokyo")')
# → True

evaluate_ast('get_weather("Tokyo")', 'get_weather(city="Tokyo")')
# → True（语义等价）

evaluate_ast('get_weather(location="Tokyo")', 'get_weather(city="Tokyo")')
# → False（参数名不同，语义不同）
```

---

## 第四章：预期 vs 实际

### 你的直觉 vs Gorilla 的实现

| 维度 | 你的直觉/预期 | Gorilla 实际实现 | 为什么有差距？ |
|------|--------------|-----------------|---------------|
| 如何减少幻觉？ | 人工标注"正确调用" | 从真实 API 文档生成数据 | 真实文档天然防止幻觉 |
| 如何评估正确性？ | String match | AST-based evaluation | 语义正确性 > 字符匹配 |
| 如何处理大规模 API？ | 精简到常用 API | Retrieval-aware Training | 训练测试场景一致 |
| 如何适应 API 更新？ | 重新训练模型 | 只需更新文档 | 学会的是"读文档"不是"记 API" |
| 需要多少数据？ | 大量真实调用记录 | API 文档 + self-instruction | 文档比调用记录更容易获取 |
| 模型大小？ | 需要大模型（GPT-4） | 7B 模型就够了 | 专门训练 > 通用大模型 |

### 反直觉挑战

#### 挑战1：更大的模型一定更好吗？

**问题**：Gorilla (7B) vs GPT-4，谁在 API 调用上更好？

[先想1分钟...]

**直觉可能说**："GPT-4 更强大，应该更好。"

**实际**：Gorilla (7B) 在 API 调用上 **优于** GPT-4！

**为什么？**

```
# GPT-4（通用大模型）
- 在各种任务上训练
- API 调用只是众多能力之一
- 容易产生 API 幻觉（见过太多代码）
- AST Match: 73%

# Gorilla 7B（专门化小模型）
- 专门针对 API 调用训练
- 所有训练数据都是真实 API 调用
- 学会的是"读 API 文档"的模式
- AST Match: 89%
```

**关键insight**：**专门化 > 通用化**。让模型做一件事并做好，比让模型做所有事更好。

#### 挑战2：更多训练数据一定更好吗？

**问题**：如果用 100 万个 API 调用示例训练 Gorilla，会更好吗？

[先想1分钟...]

**直觉可能说**："当然，数据越多越好。"

**实际**：不一定！关键在于数据**质量**而不是数量。

**为什么？**

```
# 低质量大量数据（坏）
- 100 万个示例
- 但很多有格式错误
- 包含 API 幻觉
- 混乱的标注
→ 模型学会错误模式

# 高质量适量数据（好）
- 10 万个示例
- 全部从真实 API 文档生成
- AST 验证正确性
- 清晰的标注
→ 模型学会正确模式
```

**关键insight**：在 API 调用任务上，**数据质量 > 数据数量**。Self-instruction 从真实文档生成高质量数据。

#### 挑战3：AST evaluation 会"放水"吗？

**问题**：AST match 更宽容，会不会让错误通过？

[先想1分钟...]

**直觉可能说**："更宽容的评估 = 更容易通过 = 质量下降。"

**实际**：AST match 更准确，不会"放水"。

**为什么？**

```python
# 真正的错误（AST 也能检测）
get_weather(location="Tokyo")  # 参数名错误
→ AST match: ✗（正确拒绝）

get_current_weather(city="Tokyo")  # API 名错误
→ AST match: ✗（正确拒绝）

get_weather(city="Tokyo", invalid_param=True)  # 编造参数
→ AST match: ✗（正确拒绝）

# 格式差异（AST 正确接受）
get_weather(city = "Tokyo")  # 空格
→ AST match: ✓（正确接受）

get_weather("Tokyo")  # 位置参数
→ AST match: ✓（正确接受，单参数时可省略名）
```

**关键insight**：AST match 区分"格式差异"和"语义错误"，更符合实际需求。格式不重要，语义才重要。

### 预测-验证循环

#### 互动时刻1

**在继续阅读前，预测一下**：

如果让 Gorilla 处理这个请求，它会输出什么？

```
Available APIs:
1. get_weather(city)
2. get_weather_forecast(city, date)
3. get_temperature(city)
4. search_movies(query)
5. get_movie_info(movie_id)

User: "What's the weather in Tokyo?"
```

**你的预测**：
```
Gorilla 会选择：_____
```

[继续阅读看实际输出]

**实际输出**：
```
Think: The user wants current weather in Tokyo, not forecast or just temperature.
Selected API: get_weather
Call: get_weather(city="Tokyo")
```

**你的预测和实际有什么不同？**

- 你预测它会选择 `get_weather_forecast` 吗？
- 实际上它正确区分了"当前天气"和"预报"
- 为什么？因为训练数据中有大量类似的区分任务

**关键insight**：Gorilla 学会了理解用户意图的细微差别，不只是匹配关键词。

#### 互动时刻2

**预测一下**：

```
Available APIs:
1. spotify_search(query)
2. spotify_play(track_uri)
3. spotify_pause()
4. spotify_skip()

User: "Play some Beatles songs"
```

**你的预测**：
Gorilla 会怎么做？

[思考...]

**实际输出**：
```
Step 1: Search for Beatles
Call: spotify_search(query="Beatles")

Step 2: Play the first result
Call: spotify_play(track_uri="spotify:track:3...")

（如果需要多步，Gorilla 会分步执行）
```

**观察**：
- Gorilla 理解需要两步：搜索 → 播放
- 不是直接"播放 Beatles"（不可能，需要 URI）
- 展示了多步推理能力

**预测-验证**：
你预测它会尝试一步完成吗？实际上 Gorilla 学会了分解任务。

#### 互动时刻3

**最后一个挑战**：

```
# 场景：API 更新

# v1 API（训练时见过）
get_weather(city)

# v2 API（测试时遇到）
get_current_weather(location, units="metric")

User: "What's the weather in Tokyo?"
```

**预测**：Gorilla 会如何应对？

[思考...]

**答案**：Gorilla 会适应新 API！

```
# Gorilla 的输出
（阅读 v2 API 文档后）
Call: get_current_weather(location="Tokyo", units="metric")
```

**为什么？**
- Gorilla 学会的是"读 API 文档"的模式
- 不是记忆特定的 API 调用
- 给出新文档，它能适应

**对比**：
```
# 传统微调模型
Call: get_weather(city="Tokyo")
→ 错误！（API 已废弃）

# Gorilla
Call: get_current_weather(location="Tokyo", units="metric")
→ 正确！（适应新 API）
```

**关键insight**：Gorilla 学会的是**能力**（如何读 API 文档），不是**知识**（特定 API 的调用方式）。这使其能适应 API 演化。

---

## 第五章：与其他方法对比

### Gorilla vs 其他方法

| 维度 | Standard Fine-tuning | Toolformer | Gorilla |
|------|---------------------|-----------|---------|
| 训练数据 | 人工标注调用 | Self-supervised | API 文档 + self-instruction |
| API 幻觉 | 高 | 中等 | **低** |
| 参数准确性 | 中 | 中 | **高** |
| 检索能力 | 无 | 无 | **有 (RAT)** |
| API 适应性 | 低 | 中 | **高** |
| 评估方法 | String match | Task loss | **AST match** |

### Gorilla vs Toolformer

| 维度 | Toolformer | Gorilla |
|------|-----------|---------|
| 关注点 | 通用工具学习 | API 调用准确性 |
| 工具类型 | 任何 text-to-text | 主要是 API |
| 训练数据 | Self-supervised | API 文档 + self-instruction |
| 幻觉问题 | 关注不足 | **核心关注** |
| 评估方法 | Task performance | **AST-based correctness** |
| 检索场景 | 不支持 | **Retrieval-aware Training** |

**关系**：
- Toolformer 证明了 LLM 可以学会使用工具
- Gorilla 专注于解决 API 调用的具体问题
- Gorilla 继承了 Toolformer 的思想，但更专业化

### Gorilla vs TaskMatrix

| 维度 | Gorilla | TaskMatrix |
|------|---------|------------|
| 关注点 | API 调用准确性 | 任务分解和协调 |
| 工具类型 | API | 多模态工具 |
| 训练方式 | 专门微调 | In-context learning |
| 多步能力 | 有限 | **核心能力** |
| 规划 | 无 | **有** |

**互补性**：
- Gorilla 擅长"准确调用单个 API"
- TaskMatrix 擅长"分解任务并协调多步"
- 可以结合：用 Gorilla 的准确调用能力增强 TaskMatrix

### Gorilla vs ReAct

| 维度 | Gorilla | ReAct |
|------|---------|-------|
| 关注点 | API 调用准确性 | 推理-行动循环 |
| 训练 | 专门微调 | In-context learning |
| 工具使用 | 单步 API | 多步行动 |
| 推理 | 隐式 | 显式（Thought） |

**可以结合**：
- ReAct 提供"推理框架"
- Gorilla 提供"准确的 API 调用"
- 结合后：推理 + 准确行动

### 局限性分析

**Gorilla 的局限**：

1. **单步限制**
   - 主要处理单步 API 调用
   - 复杂多步任务需要额外推理

2. **API 文档依赖**
   - 依赖高质量 API 文档
   - 文档质量差 → 性能下降

3. **计算开销**
   - 检索 + 调用有成本
   - 大规模 API 场景下延迟较高

4. **错误处理**
   - 对 API 错误处理能力有限
   - 需要额外的错误恢复机制

### 改进方向

**基于 Gorilla 的改进**：

1. **多步推理**
   - 结合 ReAct/CoT 支持多步
   - 增加任务分解能力

2. **主动学习**
   - 从错误中学习
   - 持续改进 API 调用

3. **更好的检索**
   - 学习 API 表示
   - 改进检索质量

4. **多模态扩展**
   - 支持图像、音频 API
   - 多模态参数理解

---

## 第六章：如何应用

### 适用场景

**✅ 适合使用 Gorilla 的场景**：

1. **大规模 API 调用**
   - API 数量多（100+）
   - 需要检索能力

2. **API 频繁更新**
   - API 版本迭代快
   - 需要适应能力

3. **准确性要求高**
   - API 调用不能出错
   - 参数格式严格

4. **文档驱动**
   - 有良好的 API 文档
   - 可以从文档学习

**❌ 不适合的场景**：

1. **简单任务**
   - 只有几个 API
   - 不需要检索

2. **多步推理任务**
   - 需要复杂的任务分解
   - 需要长程规划

3. **无文档场景**
   - API 文档缺失
   - 文档质量差

### 设计 API 文档

**原则**：
1. 清晰的功能描述
2. 完整的参数说明
3. 示例调用
4. 返回值说明

**好的文档示例**：
```python
"""
API: get_movie_info

Description:
Get detailed information about a movie including title, year, director, cast, plot, and ratings.

Parameters:
- movie_id (string, required): The IMDb ID of the movie (e.g., "tt0111161")

Returns:
Dictionary with keys: title, year, director, cast, plot, rating

Example:
get_movie_info(movie_id="tt0111161")
→ {"title": "The Shawshank Redemption", "year": 1994, ...}
"""
```

**坏的文档示例**：
```python
"""
API: get_movie_info

Get movie details.

Parameters:
- id: movie ID
"""
```

### 实现指南

**Step 1：收集 API 文档**
```python
# 从 API hub 爬取
apis = scrape_from_rapidapi()

# 或从官方文档解析
apis = parse_official_docs()

# 标准化格式
apis = standardize_format(apis)
```

**Step 2：生成训练数据**
```python
# Self-instruction
for api in apis:
    # 生成 queries
    queries = generate_queries(api)

    # 生成 API 调用
    for query in queries:
        call = generate_api_call(api, query)
        verify_ast(call)  # 验证正确性
```

**Step 3：训练时加入检索**
```python
# 每个 sample
sample = {
    "available_apis": sample_k_apis(target_api, k=5),
    "query": query,
    "target_call": target_api_call
}

# 训练
model.train(sample)
```

**Step 4：评估使用 AST**
```python
def evaluate_gorilla(prediction, ground_truth):
    tree_pred = ast.parse(prediction)
    tree_gt = ast.parse(ground_truth)
    return compare_ast_semantics(tree_pred, tree_gt)
```

---

## 第七章：延伸思考

### 苏格拉底追问

#### 追问1：为什么专门化模型优于通用模型？

**停下来的点**：在看性能对比时

**引导思考**：
- GPT-4 有更多参数、更多训练数据
- 为什么 Gorilla (7B) 在 API 调用上更好？
- 这违反直觉吗？

**深入思考**：
- 通用模型在各种任务上训练
- API 调用只是众多能力之一
- 专门化模型专注一个任务
- 学得更深、更准确

**预期方向**：
未来可能有更多专门化模型，而不是"一个模型做所有事"。

#### 追问2：AST evaluation 的边界在哪里？

**停下来的点**：在看评估方法时

**引导思考**：
- AST match 忽略格式差异
- 但什么情况下格式很重要？
- AST 会"太宽容"吗？

**深入思考**：
```python
# 情况1：格式不重要（AST 正确接受）
get_weather(city="Tokyo")
get_weather(city = "Tokyo")
→ 都可以

# 情况2：格式可能重要（AST 可能太宽容）
get_data(query="SELECT * FROM users")
get_data(query="SELECT * FROM users WHERE active=1")
→ AST 认为都正确，但语义不同
```

**预期方向**：
需要结合语义理解和 AST 匹配。

#### 追问3：如何处理 API 调用失败？

**停下来的点**：在看错误处理时

**引导思考**：
- 如果 API 调用失败会怎样？
- 模型会重试吗？
- 会尝试其他 API 吗？

**深入思考**：
- 当前 Gorilla 主要关注"正确调用"
- 对调用失败的处理有限
- 需要额外的错误恢复机制

**预期方向**：
结合 Reflexion 等方法，从失败中学习。

#### 追问4：检索质量如何影响性能？

**停下来的点**：在看检索场景时

**引导思考**：
- 如果检索系统返回的都是无关 API？
- Gorilla 会尝试"强行调用"吗？
- 还是会拒绝？

**深入思考**：
- 检索质量至关重要
- 垃圾 in → 垃圾 out
- 需要好的检索系统

**预期方向**：
联合优化检索和调用，端到端训练。

#### 追问5：多模态 API 怎么办？

**停下来的点**：在考虑扩展时

**引导思考**：
- 图像生成 API？
- 音频处理 API？
- Gorilla 能处理吗？

**深入思考**：
- 当前主要关注文本 API
- 多模态需要新的表示
- 参数可能是图像、音频

**预期方向**：
扩展到多模态 API 调用，需要新的方法。

---

## 核心洞见摘录

> "We introduce Gorilla, a model fine-tuned on a large amount of API call data that can reliably call APIs."

> "Our key contribution is a retrieval-aware training approach that enables the model to learn how to select the correct API from a large set of candidate APIs."

> "We propose using AST (Abstract Syntax Tree) based evaluation to assess the correctness of API calls, which is more robust than string-based matching."

> "By training on data generated from real API documentation, we significantly reduce API hallucination."

> "Specialized models can outperform larger general-purpose models on specific tasks."

---

## 方法论总结

### Gorilla 的成功要素

1. **Self-instruction from API docs**
   - 从真实文档生成训练数据
   - 避免幻觉
   - 容易扩展

2. **Retrieval-aware Training**
   - 训练测试场景一致
   - 支持大规模 API
   - 学会选择能力

3. **AST-based evaluation**
   - 语义层面评估
   - 更合理
   - 揭示真实能力

4. **专门化训练**
   - 专注 API 调用
   - 深度优化
   - 优于通用模型

### 可复用模式

**模式1：从文档生成训练数据**
```
1. 收集真实文档
2. 生成多样 queries
3. 生成对应调用
4. 验证正确性
```

**模式2：检索感知训练**
```
1. 训练时提供多个选项
2. 学会选择正确的
3. 测试时同样选择
4. 训练测试一致
```

**模式3：语义评估**
```
1. 解析为 AST
2. 比较语义结构
3. 忽略格式差异
4. 关注正确性
```

### 关键insight

**专门化 > 通用化。让模型做一件事并做好，比让模型做所有事更好。**

---

## 认知链接

### 解决了什么问题？

1. **API 幻觉问题**
   - 通过真实 API 文档训练减少幻觉

2. **参数格式问题**
   - 专门针对 API 调用格式训练

3. **API 检索问题**
   - Retrieval-aware training 支持从大量 API 中选择

4. **API 演化问题**
   - 通过文档理解能力适应 API 更新

### 被什么论文改进？

1. **后续工作**
   - 结合 reasoning 支持多步 API 调用
   - 更好的 API 管理和检索

2. **API 管理**
   - 自动化 API 文档生成
   - API 版本管理

3. **错误处理**
   - 更 robust 的 API 错误处理机制

### 与其他论文的关系

1. **与 Toolformer 的关系**
   - Toolformer：通用工具学习，self-supervised
   - Gorilla：专注 API 调用，专门训练
   - Gorilla 继承了 Toolformer 的自主调用思想，但更专业化

2. **与 TaskMatrix 的关系**
   - Gorilla：单步 API 调用，专注于准确性
   - TaskMatrix：多步任务分解，支持复杂工作流
   - 可以结合：用 Gorilla 的准确调用能力

3. **与 ReAct 的关系**
   - Gorilla：专注于"正确调用单个 API"
   - ReAct：专注于"推理 + 行动循环"，支持多步决策
   - 可以结合：ReAct 提供推理框架，Gorilla 提供准确调用

### 局限性与未来方向

1. **单步限制**
   - 主要处理单步调用
   - 复杂任务需要多步

2. **API 文档依赖**
   - 依赖高质量 API 文档
   - 文档差 → 性能下降

3. **计算开销**
   - 检索和执行 API 有成本
   - 大规模场景下延迟高

4. **错误处理**
   - 对 API 错误处理能力有限
   - 需要额外机制

---

## 总结

Gorilla 的核心贡献不是"另一个 LLM"，而是**证明了专门化训练的价值**。

通过：
- 从真实 API 文档生成训练数据（避免幻觉）
- Retrieval-aware training（支持大规模 API）
- AST-based evaluation（更合理的评估）

Gorilla 在 API 调用任务上超越了更大的通用模型。

**关键启示**：
- 让模型专注一件事，并把它做好
- 数据质量 > 数据数量
- 训练测试场景一致性至关重要
- 专门化 > 通用化（在特定任务上）
