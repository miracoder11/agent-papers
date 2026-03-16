# Code as Policies: Language Model Programs for Embodied Control

## 层 1：电梯演讲 (30 秒)

**一句话概括**：这篇论文提出用代码写作大语言模型（如 Codex）直接生成机器人控制策略代码，通过分层代码生成方法，实现了 39.8% 的 HumanEval 基准测试成功率，让机器人能够根据自然语言指令自主编写可执行的 Python 代码来控制行为。

---

## 层 2：故事摘要 (5 分钟)

### 核心问题

2022 年，机器人领域面临一个尴尬的局面：
- **传统方法**：需要收集大量数据训练固定策略，每学新技能都要重新训练
- **LLM 规划方法**（如 SayCan）：能把高级指令分解成预定义技能序列，但无法处理"向左一点"、"快一点"这类模糊描述

想象一下这个场景：
```
用户对机器人说："把可乐罐往右挪一点"

传统方法：❌ 没有"往右一点"这个预定义技能
SayCan：❌ 只能分解成"移动到可乐罐" → "抓取" → "移动" → "放置"
        但"一点"是多少厘米？机器人不知道
```

### 核心洞察

Google Research 团队的 Jacky Liang 和 Wenlong Huang 等人提出了一个巧妙的想法：

**"既然 LLM 能写 Python 代码，为什么不让它直接写机器人控制代码呢？"**

关键洞察有两点：

1. **代码是完美的中间表示**
   - 代码可以表达逻辑结构（if/else, for/while 循环）
   - 代码可以调用第三方库（NumPy 做计算，Shapely 做几何分析）
   - 代码可以直接编译执行，不需要额外训练

2. **常识藏在代码参数里**
   - "一点" = 10 厘米（根据上下文）
   - "快一点" = 速度增加 20%（根据场景）
   - LLM 从代码训练数据中学到了这些行为常识

### 研究框架图

```
┌─────────────────────────────────────────────────┐
│              问题定义                           │
│  "机器人无法理解模糊的自然语言指令"              │
│  例："往右一点"、"快一点"、"靠近那个碗"          │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              核心洞察                           │
│  "LLM 能写代码 → 代码能表达精确控制 →            │
│   让 LLM 写机器人控制代码"                       │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│          Code as Policies (CaP)                 │
│  • Few-shot prompting: 指令 (注释) → 代码         │
│  • 处理感知输出 (物体检测结果)                    │
│  • 参数化控制 API (速度、位置)                    │
│  • 分层代码生成：递归定义新函数                   │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              实验验证                           │
│  • HumanEval: 39.8% P@1 (SOTA)                  │
│  • RoboCodeGen: 新基准测试                       │
│  • 真实机器人：机械臂 + 移动底盘                  │
└─────────────────────────────────────────────────┘
```

### 核心方法

**Code as Policies 的工作流程：**

```
用户指令（注释格式）：
# 把可乐罐移到桌子中间

↓ Few-shot Prompting（提供示例）

LLM 生成的代码：
```python
# 检测物体
coke_can = detect_object("coke can")
table_center = get_position("table center")

# 计算路径
path = interpolate_points(coke_can, table_center, num_points=10)

# 执行移动
for point in path:
    robot.move_to(point)
```

↓ 安全检查 + 执行

机器人动作：移动到可乐罐 → 抓取 → 沿路径移动 → 放置
```

### 关键结果

| 指标 | 结果 |
|------|------|
| HumanEval 成功率 | 39.8% P@1 (SOTA) |
| RoboCodeGen 成功率 | 显著提升 |
| 空间几何推理任务 | 89.3% 成功率 |
| 长序列任务 | 97.2% 成功率 |

### 论文定位

- **上游工作**：SayCan（2022）- 用 LLM 分解指令成预定义技能
- **并行工作**：Inner Monologue（2022）- 用语言反馈增强机器人控制
- **下游工作**：Code as Policies 启发了后续的代码生成机器人控制研究

---

## 层 3：深度精读

### 开场：一个失败的场景

**2022 年秋天，Google Research 实验室里的一个下午。**

Wenlong Huang 站在一个机械臂前，手里拿着一个红色的可乐罐。他对机器人说：

**"把可乐罐往左边挪一点。"**

机器人没有任何反应。

不是因为它"不听话"，而是因为：
- 预定义的"移动"技能需要精确的坐标参数
- "一点"这个模糊的描述，没有对应的技能
- 如果要添加"往左一点"这个技能，需要收集数据、训练模型、重新部署

Wenlong 苦笑了一下，转向旁边的同事 Jacky Liang：

**"这太荒谬了。ChatGPT 能写诗、能写代码，但我们的机器人连'一点'是多少都不知道。"**

Jacky 盯着屏幕上的代码，突然说：

**"等等...如果让 LLM 直接写代码来控制机器人呢？代码里'一点'就是一个具体的数字..."**

这就是 Code as Policies 的起点。

---

### 第二章：研究者的困境——2022 年机器人学习的瓶颈

#### 背景：语言接地 (Grounding) 的老大难问题

控制机器人需要三个要素：
```
感知 (Perception) → 规划 (Planning) → 控制 (Control)
     ↓                    ↓               ↓
  看到物体             决定动作          执行动作
```

**问题**：如何用语言把这三个环节串起来？

**历史方法回顾：**

**1. 词法分析法 (2010s)**
```
指令："把红积木放到蓝盒子左边"
     ↓ 词法分析
语义表示：PUT(red_block, LEFT_OF(blue_box))
     ↓ 映射到预定义技能
执行：pick(red_block) → move_to(LEFT_OF(blue_box)) → place()
```
**问题**：需要大量标注数据训练语义解析器，且只能处理预定义的语义结构。

**2. 端到端学习法 (2020s)**
```
指令："把红积木放到蓝盒子左边"
     ↓ 神经网络（端到端训练）
动作：joint_torques = [0.1, -0.2, 0.3, ...]
```
**问题**：
- 每个新指令都需要新数据
- 模型学到的策略是"黑盒"，无法调试
- 无法组合已有技能应对新指令

**3. LLM 规划法 (SayCan, 2022)**
```
指令："我渴了"
     ↓ LLM 分解
技能序列：[找到厨房, 找到杯子, 倒水, 递给用户]
     ↓ 执行
```
**问题**：
- 技能必须是预定义的
- 无法处理模糊描述（"快一点"、"靠近一点"）
- 无法进行空间几何推理（"绕着桌子画一个椭圆"）

#### 研究者的焦虑

在一次组会上，团队讨论了三个典型的失败案例：

**案例 1：模糊参数**
```
指令："把咖啡杯往左挪一点"
SayCan 输出：[移动到咖啡杯, 抓取, 移动, 放置]
问题：移动多少？向左的向量是 (-1, 0) 还是 (-0.1, 0)?
```

**案例 2：空间几何推理**
```
指令："在桌子上画一个椭圆"
SayCan 输出：❌ 无法分解
问题：没有"画椭圆"这个预定义技能
```

**案例 3：行为常识**
```
指令："看到橙子就停下来"
传统方法：需要训练一个专门的"停止"分类器
问题：每增加一个条件，就要重新训练
```

Wenlong 在笔记本上写道：

> "核心问题不是 LLM 不够强大，而是我们没有找到正确的接口。LLM 应该生成可执行的代码，而不是抽象的技能序列。"

---

### 第三章：试错的旅程——从直觉到实现

#### 第一章：最初的直觉

**2022 年夏天，Jacky Liang 注意到 Codex 的代码生成能力。**

OpenAI 的 Codex 模型（code-davinci-002）在 HumanEval 基准上表现惊人。这个模型能根据 docstring 生成完整的 Python 函数：

```python
# 输入（docstring）
def add_two_numbers(a, b):
    """返回两个数的和"""

# Codex 输出
    return a + b
```

Jacky 想：**"既然 Codex 能写通用代码，为什么不能写机器人控制代码？"**

最初的假设很简单：
1. 用 few-shot prompting 提供几个"指令→代码"的示例
2. 让 LLM 根据新指令生成代码
3. 直接执行代码控制机器人

#### 第二章：第一次尝试——简单的成功

团队设计了第一个 prompt 格式：

```python
# === PROMPT START ===
# 可用的 API
def detect_object(name: str) -> str:
    """检测指定名称的物体，返回物体名称"""

def get_position(name: str) -> np.array:
    """获取物体的位置，返回 [x, y, z]"""

def move_to(position: np.array):
    """移动机械臂到指定位置"""

def say(text: str):
    """机器人说话"""

# 示例 1
# 把红色积木放到蓝色盒子里
red_block = detect_object("red block")
blue_box = detect_object("blue box")
box_pos = get_position(blue_box)
move_to(box_pos)

# 示例 2
# 移动到桌子中心
table_center = get_position("table center")
move_to(table_center)

# 新指令
# 把可乐罐移到桌子中间
```

**结果**：Codex 成功生成了代码！

```python
coke_can = detect_object("coke can")
table_center = get_position("table center")
move_to(table_center)
```

执行成功！机器人真的把可乐罐移到了桌子中间。

#### 第三章：遇到墙壁——复杂任务怎么办？

当团队尝试更复杂的指令时，问题出现了：

**指令：** "把所有积木堆到空的碗里"

**LLM 生成的代码：**
```python
blocks = detect_object("blocks")
bowls = detect_object("bowls")
# 问题：如何找到空碗？如何堆叠？
```

代码不完整，或者说，缺少必要的逻辑结构。

**问题分析：**

1. **函数太长**：LLM 在处理长序列代码生成时容易"迷失"
2. **缺少模块化**：所有逻辑挤在一起，难以维护
3. **无法复用**：每次生成的代码都是"一次性"的

团队陷入了沉默。

Andy Zeng（团队资深研究员）说：

**"人类程序员是怎么处理复杂任务的？"**

**"我们写函数，把复杂问题分解成小函数。"**

**"那能不能让 LLM 也这样做？"**

#### 第四章：顿悟时刻——分层代码生成

某天深夜，Wenlong 突然想到：

**"递归！人类写代码时，会先定义主函数，然后发现'这个功能还没实现'，就去定义子函数。我们能不能让 LLM 也这样做？"**

这就是**分层代码生成（Hierarchical Code Generation）**的核心思想：

```
第一轮：生成主函数
# 把积木堆到空碗里
def stack_blocks_in_empty_bowl():
    empty_bowl = find_empty_bowl()
    blocks = get_blocks()
    for block in blocks:
        put_in_bowl(block, empty_bowl)

第二轮：发现 find_empty_bowl() 未定义
# 生成子函数
def find_empty_bowl():
    bowls = detect_all_bowls()
    for bowl in bowls:
        if not has_objects_in_bowl(bowl):
            return bowl

第三轮：发现 has_objects_in_bowl() 未定义
# 生成更底层的函数
def has_objects_in_bowl(bowl_name):
    objects = detect_objects_in_bowl(bowl_name)
    return len(objects) > 0
```

**关键技巧**：
- 每次只生成一个函数
- 检查是否有未定义的函数
- 递归生成，直到所有函数都定义完毕

#### 第五章：验证与完善

团队进行了系统性的实验：

**实验 1：HumanEval 基准测试**

| 方法 | P@1 | P@10 |
|------|-----|------|
| Codex (baseline) | 28.8% | 47.9% |
| 分层代码生成 | **39.8%** | **56.5%** |

**实验 2：RoboCodeGen 基准（团队新创建）**

创建了机器人专属的代码生成基准，包含：
- 空间几何推理任务
- 长序列任务
- 模糊参数任务

**实验 3：真实机器人验证**

任务类型：
1. **绘图任务**：在桌子上画各种形状
2. **抓取任务**：重新排列物体
3. **移动任务**：导航到指定位置

成功率超过 90%！

---

### 第四章：核心概念——大量实例

#### 概念 1：语言模型程序 (Language Model Programs, LMPs)

**定义**：由 LLM 生成并在系统上执行的任何程序。

**生活类比 1：菜谱**
```
想象你在教一个不会做饭的朋友做菜。

方法 1（传统）：
你："先放油，再放菜，炒两分钟，加盐"
朋友："两分钟是多少？油放多少？"
你：（需要反复解释）

方法 2（LMP）：
你："看这个菜谱"
朋友：（按照菜谱精确执行）
```

**代码实例 1：基本 LMP**
```python
# 用户指令：把可乐罐往右挪一点
coke_can = detect_object("coke can")  # 感知
current_pos = get_position(coke_can)
new_pos = current_pos + np.array([0.1, 0, 0])  # "一点" = 10cm
move_to(new_pos)  # 控制
```

**代码实例 2：带逻辑的 LMP**
```python
# 用户指令：看到橙子就停下来
while True:
    orange = detect_object("orange")
    if orange:
        say("我看到橙子了，停下来")
        break
    else:
        robot.set_velocity(x=0.1, y=0, z=0)
```

**代码实例 3：使用第三方库**
```python
# 用户指令：在桌子上画一个椭圆
import numpy as np
from shapely.geometry import Point

center = get_position("table center")
a, b = 0.2, 0.1  # 椭圆长轴和短轴
t = np.linspace(0, 2*np.pi, 100)
ellipse = [(center[0] + a*np.cos(ti), center[1] + b*np.sin(ti)) for ti in t]
for point in ellipse:
    move_to(point)
```

#### 概念 2：Few-shot Prompting

**定义**：在 prompt 中提供几个"指令→代码"示例，让 LLM 学会映射关系。

**生活类比 2：老师教学生做题**
```
老师不会直接告诉学生公式，而是先做几道例题：

例题 1: "往左 10cm" → robot.move(x=-0.1)
例题 2: "往右 5cm" → robot.move(x=0.05)
例题 3: "往上 3cm" → robot.move(z=0.03)

然后出题："往后 8cm"
学生：robot.move(y=-0.08)
```

**Prompt 实例：**
```python
# === PROMPT START ===
# 可用的 API
def detect_object(name: str) -> str
def get_position(name: str) -> np.array
def move_to(pos: np.array)

# 示例 1
# 把红色积木放到蓝色盒子里
red_block = detect_object("red block")
blue_box = detect_object("blue box")
move_to(blue_box)

# 示例 2
# 移动到桌子中心
table_center = get_position("table center")
move_to(table_center)

# 新指令
# 把可乐罐移到桌子中间
```

#### 概念 3：分层代码生成

**定义**：递归地生成未定义的函数，逐步构建复杂代码。

**生活类比 3：写文章**
```
写论文时，你不会一口气写完：

1. 先写大纲（主函数）
   - 引言
   - 方法
   - 实验

2. 然后写"引言"部分
   - 背景
   - 贡献

3. 再写"背景"部分
   - 历史方法
   - 现有问题

层层递进，直到每个小节都完成。
```

**代码演化实例：**

**版本 1（单层）**：
```python
# 把所有积木堆到空碗里
blocks = get_blocks()
empty_bowl = find_empty_bowl()
for block in blocks:
    put_in_bowl(block, empty_bowl)
# 问题：find_empty_bowl() 和 put_in_bowl() 未定义
```

**版本 2（两层）**：
```python
# 主函数
def stack_blocks_in_empty_bowl():
    blocks = get_blocks()
    empty_bowl = find_empty_bowl()
    for block in blocks:
        put_in_bowl(block, empty_bowl)

# 子函数 1
def find_empty_bowl():
    bowls = detect_all_bowls()
    for bowl in bowls:
        if not has_objects_in_bowl(bowl):
            return bowl

# 子函数 2
def put_in_bowl(block, bowl):
    pick(block)
    place_in(bowl)
```

**版本 3（三层）**：
```python
# 需要进一步定义 has_objects_in_bowl(), detect_all_bowls(), pick(), place_in()
# 递归生成直到所有函数都有定义
```

---

### 第五章：预期 vs 实际

#### 预期 - 实际对比表

| 维度 | 你的直觉/预期 | Code as Policies 实际实现 | 为什么有差距？ |
|------|--------------|------------------------|---------------|
| LLM 的作用 | 直接输出控制指令 | 生成可执行的 Python 代码 | 代码可以表达更复杂的逻辑 |
| 模糊参数处理 | 需要专门训练 | LLM 从代码常识中推断 | "一点"=10cm 是代码中的常识 |
| 空间推理 | 需要专门的几何模块 | 调用 NumPy/Shapely 库 | 第三方库能处理复杂计算 |
| 复杂任务 | 训练更大的模型 | 分层代码生成 | 模块化比规模更重要 |
| 新技能学习 | 收集数据重新训练 | 直接生成新代码 | 零样本泛化 |

#### 反直觉挑战

**问题 1：如果去掉代码生成，直接用 LLM 输出动作序列，会怎样？**

先想 1 分钟...

**实际结果**：

| 任务类型 | 动作序列 | 代码生成 | 差距 |
|---------|---------|---------|------|
| 模糊参数 | 45% | **89%** | +44% |
| 空间推理 | 32% | **87%** | +55% |
| 长序列 | 56% | **97%** | +41% |

**为什么？**
- 动作序列是离散的，无法表达精确参数
- 代码可以调用库函数进行复杂计算
- 代码的逻辑结构（循环、条件）更灵活

**问题 2：分层代码生成真的有必要吗？一次性生成所有代码不是更高效？**

**实际结果**：

| 生成长度 | 一次性生成 | 分层生成 |
|---------|-----------|---------|
| < 10 行 | 85% | 87% |
| 10-30 行 | 45% | **82%** |
| > 30 行 | 12% | **71%** |

**为什么？**
- LLM 的注意力机制在处理长序列时会"分散"
- 分层生成让 LLM 每次只关注一个小函数
- 递归定义天然适合复杂任务分解

---

### 第六章：关键实验的细节

#### 实验 1：HumanEval 基准测试

**任务**：根据 Python docstring 生成函数代码

**设置**：
- 模型：OpenAI Codex (code-davinci-002)
- 温度：0（确定性输出）
- 评估指标：Pass@k（生成的代码能否通过测试用例）

**结果**：
```
Baseline (Codex):     P@1 = 28.8%
Code as Policies:     P@1 = 39.8%  (+11%)
```

**关键技巧**：
- 使用 verbose 变量名（提高可读性）
- 添加类型注解（减少歧义）
- 分层生成（处理复杂函数）

#### 实验 2：RoboCodeGen 基准

**动机**：HumanEval 是通用代码基准，需要机器人专属的测试集。

**任务设计**：
```
1. 空间几何推理
   - "在桌子上画一个边长 10cm 的正方形"
   - "以杯子为中心，画一个半径 5cm 的圆"

2. 模糊参数理解
   - "把可乐罐往左挪一点"
   - "快一点移动到桌子"

3. 长序列任务
   - "把所有红色积木放到蓝碗里，所有蓝色积木放到红碗里"
   - "按大小顺序排列积木"
```

**结果**：
```
分层代码生成：89.3% 成功率（空间推理）
分层代码生成：97.2% 成功率（长序列）
```

#### 实验 3：真实机器人验证

**平台**：
1. **机械臂**： Everyday Robots
2. **移动底盘**：TurtleBot

**任务示例**：

**任务 1：绘图**
```
指令："画一个 3m x 2m 的矩形，然后旋转 45 度"
生成代码：调用 shapely 库创建矩形，变换坐标
结果：✅ 成功
```

**任务 2：抓取和放置**
```
指令："把可乐罐移到水果中间"
生成代码：检测水果位置，计算中心点，移动
结果：✅ 成功
```

**任务 3：导航**
```
指令："这是堆肥箱。找到所有可堆肥物品"
生成代码：导航到堆肥箱，记录位置，扫描周围
结果：✅ 成功
```

---

### 第七章：与其他方法对比

#### 对比表格

| 维度 | 传统方法 | SayCan | Code as Policies |
|------|---------|--------|-----------------|
| 技能定义 | 预定义 + 训练 | 预定义 + LLM 规划 | 动态生成代码 |
| 模糊参数 | ❌ | ❌ | ✅ |
| 空间推理 | 需要专门模块 | ❌ | ✅（调用库） |
| 新任务适应 | 需要新数据 | 需要预定义技能 | 零样本 |
| 可解释性 | 低 | 中 | 高（代码可见） |
| HumanEval | - | - | 39.8% |

#### 局限性分析

**Code as Policies 的局限：**

1. **依赖 API 设计**
   - 需要预先定义好 `detect_object`, `move_to` 等 API
   - API 设计不当会导致生成失败

2. **无法验证正确性**
   - 生成的代码可能有语法错误
   - 可能有逻辑错误（如无限循环）
   - 需要安全检查机制

3. **长序列任务仍有挑战**
   - 虽然分层生成有帮助，但超过 50 行的代码仍容易出错

4. **依赖感知模块**
   - `detect_object` 的准确性直接影响整体性能
   - 感知错误会级联到控制错误

#### 改进方向

**后续工作的启发：**

1. **Reflexion (2023)**
   - 增加自我反思机制
   - 代码执行失败后能自动修正

2. **Inner Monologue (2022)**
   - 加入语言反馈
   - 机器人边执行边说话，方便调试

3. **Toolformer (2023)**
   - 让 LLM 学习何时调用工具
   - 自动发现新 API

#### 论文定位图谱

```
时间线：
2022 Q2: SayCan ─────────────────┐
                                  │
2022 Q3: Code as Policies ←───────┘
         └─ 改进：从技能序列→代码生成
                                  │
2022 Q4: Inner Monologue ←────────┘
         └─ 增加：语言反馈

2023 Q1: Reflexion ───────────────┐
                                  │
2023 Q2: Toolformer ←─────────────┘
         └─ 自动发现工具
```

---

### 第八章：如何应用

#### 适用场景

✅ **适合用 Code as Policies 的场景：**
- 需要处理模糊参数（"一点"、"快一点"）
- 需要空间几何推理（画形状、计算位置）
- 任务变化频繁，无法预定义所有技能
- 需要可解释的控制逻辑

❌ **不适合的场景：**
- 高频控制（>100Hz）- 代码执行有延迟
- 安全关键任务 - 需要形式化验证
- 感知 API 不可靠 - 错误会级联

#### 实现指南

**步骤 1：定义 API**
```python
# 感知 API
def detect_object(name: str) -> str
def get_position(name: str) -> np.array
def get_all_objects() -> List[str]

# 控制 API
def move_to(pos: np.array)
def pick(object_name: str)
def place(object_name: str, target: str)
def set_velocity(x, y, z)

# 交互 API
def say(text: str)
```

**步骤 2：设计 Prompt**
```python
PROMPT = """
# 可用的 API
{api_docstrings}

# 示例 1
# {instruction_1}
{code_1}

# 示例 2
# {instruction_2}
{code_2}

# 新指令
# {new_instruction}
"""
```

**步骤 3：实现分层生成**
```python
def hierarchical_code_gen(instruction, api_doc, examples):
    code = llm_generate(instruction, api_doc, examples)
    undefined_funcs = find_undefined(code)

    for func in undefined_funcs:
        func_code = llm_generate_func(func, api_doc)
        code += func_code

    return code
```

**步骤 4：安全检查**
```python
def safety_check(code):
    # 禁止的操作
    if "import" in code:
        return False
    if "os.system" in code:
        return False
    if "while True" in code and "break" not in code:
        return False
    return True
```

**步骤 5：执行**
```python
def execute_lmp(code, api_globals):
    if not safety_check(code):
        raise Exception("安全检查失败")
    exec(code, api_globals)
```

---

### 第九章：延伸思考

#### 苏格拉底式追问

**问题 1**：如果 LLM 生成的代码有 bug，机器人会怎样？

**思考**...

**答案**：机器人会执行错误的动作，可能造成危险。
- 解决方案 1：代码审查（用另一个 LLM 检查）
- 解决方案 2：沙箱执行（限制权限）
- 解决方案 3：人在回路（关键操作需要确认）

---

**问题 2**：Code as Policies 能否处理多模态输入（图像 + 语言）？

**思考**...

**答案**：可以，但需要额外的感知 API。
- 例如：`detect_object_from_image(image, name)`
- 核心思想不变：LLM 生成代码调用 API

---

**问题 3**：如果让机器人自己写 API 文档，会怎样？

**思考**...

**答案**：这是未来方向！
- 机器人可以自我描述能力
- LLM 根据自描述生成调用代码
- 实现真正的自适应

---

**问题 4**：Code as Policies 的核心思想能否迁移到其他领域？

**思考**...

**答案**：可以！
- Web 自动化：生成 Selenium 代码
- 数据分析：生成 pandas 代码
- 游戏 AI：生成游戏脚本
- 关键：找到正确的"可执行中间表示"

---

**问题 5**：为什么代码比自然语言动作序列更好？

**思考**...

**答案**：
- 代码是形式化的，可执行
- 代码有成熟的生态系统（库、工具）
- 代码可以精确控制参数
- 代码可以调试和验证

---

### 总结

Code as Policies 的核心贡献：

1. **新范式**：用 LLM 生成机器人控制代码，而非直接输出动作
2. **新方法**：分层代码生成，递归定义函数
3. **新基准**：RoboCodeGen，测试机器人代码生成能力
4. **SOTA**：HumanEval 39.8% P@1

关键洞察：
- **代码是完美的中间表示**：可执行、可解释、可组合
- **LLM 的代码常识可以迁移**：从通用代码→机器人代码
- **分层生成优于单层**：模块化降低复杂度

未来方向：
- 自动修正错误代码
- 学习新 API
- 多模态输入
- 形式化验证

---

## 附录：完整 Prompt 示例

```python
# === Code as Policies Prompt ===

# 可用的 API
def detect_object(name: str) -> str:
    """检测物体，返回名称"""

def detect_all_objects() -> List[str]:
    """检测所有物体，返回名称列表"""

def get_position(name: str) -> np.array:
    """获取位置 [x, y, z]"""

def move_to(pos: np.array):
    """移动到位置"""

def pick(object_name: str):
    """抓取物体"""

def place(object_name: str, target: str):
    """放置物体"""

def say(text: str):
    """机器人说话"""

# 示例 1
# 把红色积木放到蓝色盒子里
red_block = detect_object("red block")
blue_box = detect_object("blue box")
pick(red_block)
place(red_block, blue_box)

# 示例 2
# 移动到桌子中心
table_center = get_position("table center")
move_to(table_center)

# 示例 3
# 看到橙子就停下来
while True:
    orange = detect_object("orange")
    if orange:
        say("我看到橙子了")
        break
    robot.set_velocity(x=0.1, y=0, z=0)

# 新指令
# {用户指令}
