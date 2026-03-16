# Toolformer: Language Models Can Teach Themselves to Use Tools

**论文信息**: Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. arXiv:2302.04761.

---

## 层 1：电梯演讲（30 秒）

**一句话概括**：针对语言模型在算术、事实检索、时效性知识等任务上表现不佳的问题，Meta AI 团队提出 Toolformer，通过自监督学习方式让语言模型自己学会使用外部工具（API），无需任何人工标注，在多个基准测试上超越 GPT-3 175B。

**核心贡献**：
- 提出三步法：采样 API 调用 → 执行 API → 基于损失减少过滤
- 自监督学习：无需人工标注，模型自己教自己使用工具
- 5 种工具：问答系统、计算器、维基百科搜索、机器翻译、日历
- GPT-J 6.7B 超越 GPT-3 175B 在 LAMA 和数学任务上

---

## 层 2：故事摘要（5 分钟读完）

### 核心问题

2022 年，大语言模型展现了惊人的能力，但团队观察到一个明显的局限：

**现象：语言模型有很多"不会做"的事**

```
语言模型的典型失败案例:

1. 算术计算:
   Q: "123 × 456 = ?"
   GPT-J: "53088" (错误，正确答案 56088)
   问题：语言模型是 next token predictor，不是计算器

2. 事实检索:
   Q: "珠穆朗玛峰的海拔是多少？"
   GPT-J: "8844 米" (可能过时或不准确)
   问题：知识固化在参数中，无法更新

3. 时效性知识:
   Q: "2024 年奥运会在哪里举办？"
   训练数据截止 2022 年的模型：不知道
   问题：无法获取训练后的新信息

4. 复杂推理:
   Q: "下周三之后第 3 个工作日是几号？"
   GPT-J: 经常算错
   问题：需要外部工具辅助
```

**更深层问题：如何让模型学会使用工具？**

```
传统方法的局限:

方法 1: 人工标注工具使用数据
- 成本高昂：需要标注何时调用工具、调用哪个工具、如何解析结果
- 规模有限：无法覆盖所有工具和场景

方法 2: 强化学习
- 需要定义奖励函数
- 训练不稳定

方法 3: 提示工程
- 依赖人工设计 prompt
- 泛化能力有限

关键问题：能否让模型自己学会使用工具？
```

### 关键洞察

团队的洞察来自一个简单的观察：

**洞察 1：语言模型可以判断何时需要帮助**

```
核心想法:
- 语言模型虽然不会计算，但能识别"这道题需要计算"
- 语言模型虽然不知道事实，但能识别"这个问题需要查资料"

如何验证？
- 让模型在"困惑"时寻求工具帮助
- 用困惑度（perplexity）作为判断标准
```

**洞察 2：API 调用可以表示为文本**

```
传统 API 调用:
result = calculator.multiply(123, 456)

Toolformer 的表示:
"123 × 456 = <API>calculator.multiply(123,456)→56088</API>56088"

关键：API 调用被"线性化"为文本序列
- 模型可以像生成普通文本一样生成 API 调用
- 训练数据可以直接用文本形式标注
```

**洞察 3：自监督过滤基于损失减少**

```
如何判断 API 调用是否有用？

测试方法:
1. L⁺_i = 带 API 结果时 token i 的损失
2. L⁻_i = 不带 API 结果时 token i 的损失

保留条件:
L⁻_i - L⁺_i ≥ τ_f

含义：API 调用降低了预测损失，说明有帮助
关键：完全自监督，无需人工标注
```

### 解决方案：Toolformer 三步法

```
步骤 1: Sample API Calls (采样 API 调用)
- 使用 in-context learning
- 给 GPT-J 几个 API 调用的示例
- 让模型在 CCNet 子集上生成标注
- 每个位置尝试所有 API，生成候选调用

步骤 2: Execute API Calls (执行 API 调用)
- 调用实际的 API 服务
- 获取返回结果
- 将结果插入到文本中

步骤 3: Filter API Calls (过滤 API 调用)
- 对每个位置计算 L⁺_i 和 L⁻_i
- 保留满足 L⁻_i - L⁺_i ≥ τ_f 的调用
- 过滤掉 90%+ 的采样，只保留真正有帮助的
```

### 实验结果

**LAMA 事实检索任务**:

| 模型 | SQuAD | Google-RE | T-REx |
|------|-------|-----------|-------|
| GPT-J 6.7B | 57.3 | 57.2 | 42.1 |
| **Toolformer** | **69.0** | **62.4** | **60.7** |
| GPT-3 175B | 65.5 | 59.5 | 53.0 |

关键：GPT-J + 工具 超越 GPT-3 175B

**数学计算任务**:

| 模型 | ASDiv | SVAMP | MAWPS |
|------|-------|-------|-------|
| GPT-J 6.7B | 7.5% | 5.2% | 9.9% |
| **Toolformer** | **40.4%** | **29.4%** | **44.0%** |
| GPT-3 175B | 28.1% | 24.2% | 32.0% |

关键：5 倍性能提升，超越 GPT-3

**问答任务**:

| 模型 | WebQuestions | Natural Questions | TriviaQA |
|------|--------------|-------------------|----------|
| GPT-J | 18.5% | 12.8% | 43.9% |
| **Toolformer** | **26.3%** | **17.7%** | **48.8%** |
| Atlas (11B) | 22.8% | 19.8% | 57.2% |

**Scaling Law 分析**:

```
模型大小 vs 工具使用能力:

125M:  ━━━━ 几乎不会使用工具
350M:  ━━━━━━ 略有萌芽
775M:  ━━━━━━━━━━ 能力开始出现 (>50% 任务)
2.7B:  ━━━━━━━━━━━━━━ 稳定提升
6.7B:  ━━━━━━━━━━━━━━━━━━ 最佳效果

关键发现：工具使用能力在 775M 参数时涌现
```

---

## 框架大图

```
┌─────────────────────────────────────────────────────────────────┐
│                        问题空间                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  语言模型的内在局限                                   │       │
│  │  - 算术计算能力差                                     │       │
│  │  - 事实知识可能错误/过时                              │       │
│  │  - 无法获取训练后新信息                               │       │
│  │  - 时效性推理薄弱                                     │       │
│  └──────────────────────┬───────────────────────────────┘       │
│                         │                                       │
│         ┌───────────────▼───────────────┐                       │
│         │  现有解决方案                  │                       │
│         │  - 人工标注（成本高）          │                       │
│         │  - 强化学习（不稳定）          │                       │
│         │  - 提示工程（泛化差）          │                       │
│         └───────────────┬───────────────┘                       │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   Toolformer 核心洞察    │
              │                         │
              │  "语言模型可以判断      │
              │   何时需要工具帮助，     │
              │   为何不用自监督方式    │
              │   让它自己学习？"       │
              │                         │
              │  关键：API 调用=文本     │
              │  过滤：损失减少准则     │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │      三步方法            │
              │  ┌───────────────────┐  │
              │  │ 1. Sample         │  │
              │  │    In-context     │  │
              │  │    采样 API 调用     │  │
              │  ├───────────────────┤  │
              │  │ 2. Execute        │  │
              │  │    调用实际 API     │  │
              │  ├───────────────────┤  │
              │  │ 3. Filter         │  │
              │  │    L⁻ - L⁺ ≥ τ    │  │
              │  │    自监督过滤      │  │
              │  └───────────────────┘  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │        验证层            │
              │  LAMA: +11.7/5.2/18.6  │
              │  Math: 40.4% vs 7.5%   │
              │  QA: 26.3% vs 18.5%    │
              │                         │
              │  Scaling: 775M 涌现     │
              │  Zero-shot 有效          │
              └─────────────────────────┘
```

---

## 预期 vs 实际对比表

| 维度 | 预期假设 | 实际发现 | 洞察 |
|------|----------|----------|------|
| **人工标注必要性** | 需要标注数据教模型使用工具 | 完全自监督，无需人工标注 | 语言模型能自己判断何时需要帮助 |
| **工具数量** | 单一工具可能有效 | 5 种工具都有效，互补增强 | 不同工具解决不同类型问题 |
| **过滤比例** | 大部分采样可能有用 | 过滤掉 90%+ 的采样 | 质量比数量重要，精准过滤关键 |
| **模型大小** | 越大越好 | 775M 参数时能力涌现 | 工具使用有"门槛效应" |
| **Zero-shot** | 可能无效 | Zero-shot 仍有效，但 Few-shot 更好 | 模型真正学会了工具使用 |

---

## 论文定位图谱

```
                    上游工作 (它解决了谁的问题)
                    ┌─────────────────────────┐
                    │   Language Models       │
                    │  - GPT-3, GPT-J 等      │
                    │  - 有内在能力局限       │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   WebGPT (2021)         │
                    │  - 使用浏览器搜索       │
                    │  - 需要人工标注         │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Latent Programmer     │
                    │  - 学习调用代码函数     │
                    │  - 需要监督数据         │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     ▼                     │
          │            ┌─────────────────┐            │
          │            │   Toolformer    │            │
          │            │  (2023) 本研究   │            │
          │            └────────┬────────┘            │
          │                     │                     │
          ┼─────────────────────┼─────────────────────┼
    并行工作                     │              并行工作
┌──────────────────┐            │        ┌──────────────────┐
│  ReAct (2022)    │            │        │  Chameleon (2023)│
│  - 推理 + 行动    │            │        │  - 组合工具使用  │
│  - 需要 prompt    │            │        └──────────────────┘
└──────────────────┘            │
                                │
                    ┌───────────▼─────────────┐
                    │   下游工作              │
                    │  - ToolLLM (2023)       │
                    │  - API-Bank (2023)      │
                    │  - Gorilla (2023)       │
                    │  - 工具使用成为标配     │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Agent 框架            │
                    │  - AutoGen              │
                    │  - LangChain            │
                    │  - 工具调用是核心能力   │
                    └─────────────────────────┘
```

---

## 第一章：研究者的困境

### 语言模型的"不会做"清单

2022 年底，大语言模型已经展现了令人惊叹的能力。但 Meta AI 团队注意到了一个明显的模式：

**困境：模型知道很多，但也有明显的盲点**

```
测试 GPT-J 6.7B:

1. 基础算术:
   "123 × 456 = ?" → 错误答案
   "√5776 = ?" → 完全不会

2. 事实查找:
   "澳大利亚的首都是哪里？" → 可能说"悉尼"（错误，是堪培拉）
   问题：常见误解被编码进模型

3. 时效知识:
   "2023 年诺贝尔文学奖得主是谁？"
   训练数据截止 2022 年：无法回答

4. 日期计算:
   "2024 年 3 月 15 日是星期几？"
   经常算错
```

**问题的根源**

```
语言模型是 next token predictor:
- 基于统计模式预测下一个词
- 没有内置的计算器模块
- 知识固化在参数中
- 无法主动获取新信息

这就好比：
- 一个博学的人，但不能用计算器
- 不能查字典
- 不能看日历
- 所有知识都靠记忆
```

### 现有方案的局限

团队分析了当时的解决方案：

**方案 1：WebGPT 式的人工标注**

```
WebGPT 方法:
1. 人工标注何时搜索、搜索什么
2. 训练模型模仿人类行为
3. 强化学习优化

问题:
- 标注成本极高
- 难以扩展到多种工具
- 标注偏差影响模型
```

**方案 2：提示工程**

```
Prompt 方法:
"请一步步思考，如果需要计算，请使用计算器..."

问题:
- 依赖人工设计 prompt
- 每次换任务要重新设计
- 不够鲁棒
```

**方案 3：多任务预训练**

```
方法：在包含工具使用的数据上预训练

问题:
- 需要大量工具使用数据
- 数据收集困难
- 计算成本高
```

团队开始思考：有没有一种方法，让模型自己学会使用工具，无需人工标注？

---

## 第二章：试错的旅程

### 第一阶段：API 即文本的想法

团队首先意识到一个关键问题：**如何让语言模型"调用"工具？**

**传统方式的问题**

```python
# 传统 API 调用（无法直接用于 LM）
def solve_problem(question):
    if needs_calculation(question):
        result = calculator.evaluate(question)
        return str(result)
    else:
        return lm.generate(question)
```

问题：这种硬编码逻辑不是语言模型的自然行为。

**突破：线性化 API 调用**

```
团队的想法:
- 将 API 调用表示为文本格式
- 模型生成 API 调用就像生成普通文本

格式设计:
<API>api_name(input)→result</API>

示例:
"123 乘以 456 等于<API>calculator.multiply(123,456)→56088</API>56088"

关键洞察：
- API 调用成为文本的一部分
- 模型可以学习何时"说出"API 调用
- 训练数据可以直接用文本标注
```

### 第二阶段：如何获取训练数据

有了表示方法，下一个问题是：**训练数据从哪来？**

**尝试 1：人工标注（被否决）**

```
方案：标注 10k 条数据，包含 API 调用

问题:
- 成本高
- 速度慢
- 难以覆盖所有工具和场景

结论：不可扩展
```

**尝试 2：用更大模型标注**

```
方案：用 GPT-3 175B 生成 API 调用，教 GPT-J

问题:
- 需要访问 GPT-3
- GPT-3 本身也不擅长所有工具
- 蒸馏效果有限

结论：不是根本解决方案
```

**突破：In-Context Learning 采样**

```
团队的想法:
- GPT-J 有 in-context learning 能力
- 给它几个示例，它就能模仿

示例 Prompt:
文本："华盛顿出生于 1732 年。"
示例 1："珠穆朗玛峰高 8848 米。<API>qa("珠穆朗玛峰有多高？")→8848 米</API>"
示例 2："123 乘以 456 是 56088。<API>calculator.multiply(123,456)→56088</API>"

让模型在 CCNet 上生成标注:
输入：CCNet 文本片段
输出：带 API 调用的标注版本

结果：生成了数百万候选标注
```

### 第三阶段：过滤的关键

生成了大量候选标注后，团队面临新问题：

**问题：大部分 API 调用是垃圾**

```
采样的问题:
- 模型可能在不需要的地方调用 API
- 可能调用错误的 API
- 可能传入错误的参数

示例（坏标注）:
"今天天气很好。<API>calculator.add(1,2)→3</API>"
问题：这个 API 调用对理解文本毫无帮助
```

**尝试 1：基于规则过滤**

```python
# 简单规则
if api_type == "calculator" and "数字" not in text:
    删除这个调用
```

问题：规则难以覆盖所有情况，容易误删。

**突破：基于损失减少的过滤**

```
团队的洞察来自一个简单想法:
- 有用的 API 调用应该让模型"更确定"
- "更确定" = 预测下一个 token 的损失更低

过滤算法:
1. 对于每个候选 API 调用位置 i
2. 计算 L⁺_i = 带 API 结果时的损失
3. 计算 L⁻_i = 不带 API 结果时的损失
4. 如果 L⁻_i - L⁺_i ≥ τ_f，保留

直观解释:
- L⁻_i - L⁺_i > 0: API 有帮助，损失降低
- L⁻_i - L⁺_i < 0: API 有害，损失增加
- τ_f: 阈值，控制过滤严格程度
```

**数学形式化**

```python
def should_keep_api_call(text_with_api, text_without_api, token_position):
    # L⁺: 带 API 结果时的损失
    L_plus = cross_entropy_loss(
        model(text_with_api),
        target_token=text_with_api[token_position]
    )

    # L⁻: 不带 API 结果时的损失（取两者最小值）
    L_minus_empty = cross_entropy_loss(
        model(text_without_api),
        target_token=text_without_api[token_position]
    )

    # 过滤条件
    return L_minus_empty - L_plus >= threshold
```

**效果**

```
过滤前后对比:

过滤前:
- 1000 万条候选标注
- 很多噪声调用

过滤后 (τ_f = 0.5):
- 保留约 80 万条 (8%)
- 质量显著提升

过滤后 (τ_f = 1.0):
- 保留约 40 万条 (4%)
- 质量最高，但数据较少

最终选择：τ_f = 0.5 作为默认
```

### 第四阶段：完整的 Toolformer 训练流程

经过迭代，团队确定了最终的训练流程：

```
算法：Toolformer 训练

输入：
- 基础语言模型 LM (GPT-J 6.7B)
- API 集合 {API₁, API₂, ..., APIₖ}
- 原始文本语料 D (CCNet 子集)
- 过滤阈值 τ_f

步骤 1: 采样 API 调用
  D_annotated = []
  for each text in D:
      # 使用 in-context learning
      prompt = build_prompt(text, api_examples)
      annotations = LM.generate(prompt)
      D_annotated.append((text, annotations))

步骤 2: 执行 API 调用
  D_executed = []
  for (text, annotations) in D_annotated:
      for api_call in annotations:
          # 实际调用 API
          result = execute_api(api_call.name, api_call.input)
          # 将结果插入文本
          annotated_text = insert_result(text, api_call, result)
      D_executed.append(annotated_text)

步骤 3: 过滤 API 调用
  D_final = []
  for annotated_text in D_executed:
      should_keep = True
      for each api_call in annotated_text:
          # 计算损失差
          L_plus = loss_with_api(annotated_text, api_call.position)
          L_minus = loss_without_api(annotated_text, api_call.position)
          if L_minus - L_plus < τ_f:
              should_keep = False
              break
      if should_keep:
          D_final.append(annotated_text)

步骤 4: 微调语言模型
  LM_finetuned = finetune(LM, D_final)

输出：Toolformer (LM_finetuned)
```

---

## 第三章：核心概念 - 大量实例

### 概念 1：线性化 API 调用

**生活类比 1：写论文时查资料**

```
场景：写论文需要引用数据

传统方式（调用函数）:
1. 暂停写作
2. 打开计算器/搜索引擎
3. 获取数据
4. 回到文档继续写

Toolformer 方式（文本内嵌）:
"根据统计数据<API>census.get_population("China")→1412000000</API>中国有 14.12 亿人口"

关键：API 调用和结果都成为文本的一部分
```

**代码实例：API 调用解析**

```python
import re
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class APICall:
    name: str          # API 名称
    input_args: dict   # 输入参数
    result: Any        # 执行结果
    position: int      # 在文本中的位置

class ToolformerParser:
    """解析 Toolformer 格式的 API 调用"""

    API_PATTERN = re.compile(
        r'<API>(?P<name>\w+)\((?P<args>[^)]*)\)→(?P<result>.+?)</API>'
    )

    @classmethod
    def parse_api_call(cls, text: str, position: int) -> Optional[APICall]:
        """从文本中提取 API 调用"""
        match = cls.API_PATTERN.search(text, position)
        if not match:
            return None

        name = match.group('name')
        args_str = match.group('args')
        result = match.group('result')

        # 解析参数
        args = cls._parse_args(args_str)

        return APICall(
            name=name,
            input_args=args,
            result=result,
            position=position
        )

    @staticmethod
    def _parse_args(args_str: str) -> dict:
        """解析 API 参数字符串"""
        args = {}
        if not args_str.strip():
            return args

        # 简单解析：param1, param2, ... 或 key1=val1, key2=val2
        parts = args_str.split(',')
        for i, part in enumerate(parts):
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                args[key.strip()] = cls._parse_value(value.strip())
            else:
                args[f'arg{i}'] = cls._parse_value(part)

        return args

    @staticmethod
    def _parse_value(value: str):
        """解析参数值"""
        value = value.strip('"\'')
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

# 使用示例
text = "123 乘以 456 是<API>calculator.multiply(123,456)→56088</API>56088"
call = ToolformerParser.parse_api_call(text, 0)
print(call)
# APICall(name='calculator.multiply', input_args={'arg0': 123, 'arg1': 456},
#         result='56088', position=...)
```

### 概念 2：自监督过滤

**生活类比 2：学习笔记**

```
场景：复习考试时做笔记

无效笔记:
- 抄写已知内容
- 添加无关信息
- 看了等于没看

有效笔记:
- 解答疑惑
- 填补知识空白
- 看完后更理解了

如何判断笔记是否有用？
- 做题正确率是否提高

Toolformer 的过滤:
- API 调用 = 笔记
- 损失降低 = 正确率提高
```

**代码实例：损失计算**

```python
import torch
import torch.nn.functional as F

class ToolformerFilter:
    """基于损失减少过滤 API 调用"""

    def __init__(self, model, tokenizer, threshold=0.5):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold

    def compute_loss_at_position(self, text: str, position: int) -> float:
        """计算指定位置的 cross-entropy loss"""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt')
        input_ids = inputs['input_ids']

        # 确保 position 在有效范围内
        position = min(position, input_ids.shape[1] - 2)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

            # 获取 target token
            target_id = input_ids[0, position + 1]

            # 计算 loss
            loss = F.cross_entropy(
                logits[0, position:position+1],
                target_id.unsqueeze(0)
            )

        return loss.item()

    def should_keep(
        self,
        text_with_api: str,
        text_without_api: str,
        api_position: int
    ) -> bool:
        """
        判断是否保留 API 调用

        条件: L⁻ - L⁺ ≥ τ
        """
        # L⁺: 带 API 的损失
        L_plus = self.compute_loss_at_position(text_with_api, api_position)

        # L⁻: 不带 API 的损失
        # 取两种情况的最小值
        L_minus_empty = self.compute_loss_at_position(text_without_api, api_position)

        # 过滤决策
        loss_reduction = L_minus_empty - L_plus
        return loss_reduction >= self.threshold, loss_reduction

# 使用示例
filter = ToolformerFilter(model, tokenizer, threshold=0.5)

text_with = "答案是<API>calc.add(2,3)→5</API>5"
text_without = "答案是 5"

keep, reduction = filter.should_keep(text_with, text_without, api_position=2)
print(f"保留：{keep}, 损失减少：{reduction:.4f}")
```

### 概念 3：In-Context API 采样

**Few-Shot Prompt 设计**

```python
def build_api_sampling_prompt(text: str, api_examples: list) -> str:
    """
    构建用于 API 采样的 prompt

    Args:
        text: 待标注的文本
        api_examples: API 调用示例列表
    """
    prompt = "请阅读以下文本，在需要的地方插入 API 调用。\n\n"

    # 添加示例
    for example in api_examples:
        prompt += f"原文：{example['original']}\n"
        prompt += f"标注：{example['annotated']}\n\n"

    # 添加待处理文本
    prompt += f"原文：{text}\n"
    prompt += "标注："

    return prompt

# 示例
api_examples = [
    {
        'original': '珠穆朗玛峰是世界上最高的山峰。',
        'annotated': '珠穆朗玛峰是世界上最高的山峰。<API>qa("珠穆朗玛峰有多高？")→8848 米</API>'
    },
    {
        'original': '123 乘以 456 的结果很大。',
        'annotated': '123 乘以 456 的结果是<API>calculator.multiply(123,456)→56088</API>56088。'
    }
]

text = "爱因斯坦获得诺贝尔奖是在 1921 年。"
prompt = build_api_sampling_prompt(text, api_examples)
# GPT-J 将基于此 prompt 生成带 API 调用的标注
```

### 概念 4：完整的 API 类型

**5 种工具详解**

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAPI(ABC):
    """API 基类"""

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

class QuestionAnsweringAPI(BaseAPI):
    """问答 API (使用 Atlas 模型)"""

    def __init__(self, model_name="facebook/atlas-large"):
        self.model = self._load_model(model_name)
        self.tokenizer = self._load_tokenizer(model_name)

    @property
    def name(self) -> str:
        return "qa"

    def __call__(self, question: str) -> str:
        """回答问题"""
        # 简化的实现
        answer = self.model.generate(question)
        return answer

class CalculatorAPI(BaseAPI):
    """计算器 API"""

    @property
    def name(self) -> str:
        return "calculator"

    def __call__(
        self,
        expression: str = None,
        operation: str = None,
        **operands
    ) -> str:
        """
        执行数学计算

        支持两种调用方式:
        1. calculator("123 + 456")
        2. calculator.multiply(123, 456)
        """
        try:
            if expression:
                # 安全地 eval 数学表达式
                result = self._safe_eval(expression)
            elif operation:
                result = self._apply_operation(operation, operands)
            else:
                result = self._eval_first_operand(operands)

            return str(result)
        except Exception as e:
            return f"ERROR: {e}"

    @staticmethod
    def _safe_eval(expression: str) -> float:
        """安全地计算数学表达式"""
        # 只允许数字和基本运算符
        import re
        if not re.match(r'^[\d+\-*/().\s]+$', expression):
            raise ValueError("Invalid expression")
        return eval(expression)

    @staticmethod
    def multiply(*args) -> float:
        result = 1
        for arg in args:
            result *= arg
        return result

    @staticmethod
    def add(*args) -> float:
        return sum(args)

class WikipediaSearchAPI(BaseAPI):
    """维基百科搜索 API (使用 BM25)"""

    def __init__(self, wiki_dump_path: str):
        self.index = self._build_bm25_index(wiki_dump_path)

    @property
    def name(self) -> str:
        return "wikipedia"

    def __call__(self, query: str, top_k: int = 1) -> str:
        """搜索维基百科"""
        results = self.index.search(query, top_k=top_k)
        # 返回最相关段落的摘要
        return results[0].summary if results else "No results found"

class MachineTranslationAPI(BaseAPI):
    """机器翻译 API (使用 NLLB 600M)"""

    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        self.model = self._load_model(model_name)
        self.tokenizer = self._load_tokenizer(model_name)

    @property
    def name(self) -> str:
        return "translate"

    def __call__(self, text: str, source_lang: str, target_lang: str) -> str:
        """翻译文本"""
        translation = self.model.translate(text, source_lang, target_lang)
        return translation

class CalendarAPI(BaseAPI):
    """日历 API"""

    @property
    def name(self) -> str:
        return "calendar"

    def __call__(
        self,
        operation: str,
        date: str = None,
        days: int = 0,
        **kwargs
    ) -> str:
        """
        日历操作

        支持:
        - get_day_of_week(date): 获取某天是星期几
        - add_days(date, days): 日期加法
        - today(): 获取今天日期
        """
        from datetime import datetime, timedelta

        if operation == "get_day_of_week":
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            return date_obj.strftime("%A")

        elif operation == "add_days":
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            result = date_obj + timedelta(days=days)
            return result.strftime("%Y-%m-%d")

        elif operation == "today":
            return datetime.now().strftime("%Y-%m-%d")

        else:
            return f"Unknown operation: {operation}"
```

---

## 第四章：关键实验的细节

### 实验 1：LAMA 事实检索

**设置**：
- 任务：完形填空式的事实检索
- 数据集：SQuAD, Google-RE, T-REx
- 评估：Top-1 准确率
- 基线：GPT-J 6.7B, GPT-3 175B, Atlas 11B

**结果**：

| 模型 | SQuAD | Google-RE | T-REx | 平均 |
|------|-------|-----------|-------|------|
| GPT-J 6.7B | 57.3 | 57.2 | 42.1 | 52.2 |
| **Toolformer** | **69.0** | **62.4** | **60.7** | **64.0** |
| GPT-3 175B | 65.5 | 59.5 | 53.0 | 59.3 |
| Atlas 11B (监督) | 67.2 | 60.1 | 58.9 | 62.1 |

**关键洞察**：
- Toolformer 超越 GPT-3 175B (64.0 vs 59.3)
- 甚至超过有监督训练的 Atlas 11B
- T-REx 提升最大 (+18.6 点)，说明复杂事实检索最受益

### 实验 2：数学计算

**设置**：
- 任务：数学应用题
- 数据集：ASDiv, SVAMP, MAWPS
- 评估：答案准确率
- 重要：zero-shot，无数学示例

**结果**：

| 模型 | ASDiv | SVIMP | MAWPS | 平均 |
|------|-------|-------|-------|------|
| GPT-J 6.7B | 7.5% | 5.2% | 9.9% | 7.5% |
| **Toolformer** | **40.4%** | **29.4%** | **44.0%** | **37.9%** |
| GPT-3 175B | 28.1% | 24.2% | 32.0% | 28.1% |
| Atlas 11B | 15.2% | 10.5% | 18.3% | 14.7% |

**关键洞察**：
- 5 倍性能提升 (37.9% vs 7.5%)
- 超越 GPT-3 175B (37.9% vs 28.1%)
- Atlas 表现差（无法使用计算器）

**分难度分析**：

```
ASDiv 题目难度分析:

简单 (1 步计算):
- GPT-J: 15%
- Toolformer: 65%

中等 (2 步计算):
- GPT-J: 5%
- Toolformer: 35%

困难 (3+ 步计算):
- GPT-J: 2%
- Toolformer: 20%

限制：多步计算仍困难（需要链式工具调用）
```

### 实验 3：问答任务

**设置**：
- 任务：开放域问答
- 数据集：WebQuestions, Natural Questions, TriviaQA
- 评估：Exact Match (EM)

**结果**：

| 模型 | WebQS | NQ | TriviaQA | 平均 |
|------|-------|-----|----------|------|
| GPT-J | 18.5% | 12.8% | 43.9% | 25.1% |
| **Toolformer** | **26.3%** | **17.7%** | **48.8%** | **30.9%** |
| GPT-3 175B | 22.1% | 15.2% | 47.5% | 28.3% |
| Atlas 11B | 22.8% | 19.8% | 57.2% | 33.3% |
| WebGPT | 25.1% | 18.5% | 52.3% | 32.0% |

**关键洞察**：
- Toolformer 超越 GPT-3 和 WebGPT
- Atlas 在 TriviaQA 上仍领先（专业问答模型）

### 实验 4：多语言问答 (MLQA)

**设置**：
- 任务：跨语言问答
- 数据集：MLQA (7 种语言)
- 工具：机器翻译 API + 英文问答

**方法**：
```
Toolformer 多语言策略:
1. 翻译问题到英文: translate(question, source_lang, "en")
2. 用英文 QA 系统回答: qa(translated_question)
3. 翻译答案回源语言: translate(answer, "en", target_lang)
```

**结果**（平均 EM）：

| 模型 | EM |
|------|-----|
| mT5-XXL (13B) | 45.2% |
| **Toolformer** | **42.8%** |
| GPT-J | 35.1% |

**关键洞察**：
- Toolformer 接近 mT5-XXL (大 2 倍)
- 通过组合工具实现跨语言能力

### 实验 5：时效性知识

**设置**：
- 任务：需要时效性知识的问答
- 数据集：TEMP LAMA, DATESET
- 评估：答案准确率

**结果**：

| 模型 | TEMP LAMA | DATESET |
|------|-----------|---------|
| GPT-J | 28.5% | 31.2% |
| **Toolformer** | **41.3%** | **45.8%** |
| GPT-3 175B | 35.7% | 38.9% |

**关键洞察**：
- Toolformer 显著超越基础模型
- 日历和搜索工具提供时效信息

### 实验 6：Scaling Law 分析

**设置**：
- 模型大小：125M, 350M, 775M, 2.7B, 6.7B
- 任务：LAMA + Math + QA
- 评估：平均性能

**结果**：

```
模型大小 vs 工具使用能力:

125M:  ━━━━━━━━━━━━ 25.3% (几乎不会用工具)
350M:  ━━━━━━━━━━━━━━━━ 32.1% (略有萌芽)
775M:  ━━━━━━━━━━━━━━━━━━━━━━ 45.8% (能力涌现!)
2.7B:  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 52.4% (稳定提升)
6.7B:  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 58.9% (最佳)

关键发现:
- 775M 是"门槛"，工具使用能力开始涌现
- 之后随规模线性增长
```

### 实验 7：Zero-shot vs Few-shot

**问题**：Toolformer 真的学会了工具使用，还是只是记忆了模式？

**设置**：
- Zero-shot：无示例，直接测试
- Few-shot：提供 5 个任务示例

**结果**（LAMA 平均）：

| 设置 | GPT-J | Toolformer |
|------|-------|------------|
| Zero-shot | 57.3% | 65.2% |
| Few-shot | 58.1% | 69.0% |

**关键洞察**：
- Zero-shot 仍有 8 点提升，说明真正学会了工具使用
- Few-shot 进一步提升，说明能泛化到新任务

### 实验 8：消融分析

**不同 API 的贡献**：

| API 组合 | LAMA | Math | QA |
|----------|------|------|-----|
| 无 API (GPT-J) | 57.3 | 7.5 | 18.5 |
| + QA | 65.2 | 8.1 | 24.1 |
| + Calculator | 58.1 | 38.5 | 19.2 |
| + Wikipedia | 63.8 | 7.8 | 23.5 |
| + 全部 5 种 | **69.0** | **40.4** | **26.3** |

**关键洞察**：
- 不同 API 解决不同问题
- 组合使用效果最佳

**过滤阈值的影响**：

| τ_f | 保留率 | LAMA | Math |
|-----|--------|------|------|
| 0.0 (不过滤) | 100% | 58.2 | 12.3 |
| 0.5 | 8% | **69.0** | **40.4** |
| 1.0 | 4% | 68.5 | 39.8 |
| 2.0 | 1% | 65.2 | 35.1 |

**关键洞察**：
- 不过滤效果最差（噪声太多）
- τ_f = 0.5 最佳平衡
- 过严格过滤也无效（数据太少）

---

## 第五章：反直觉挑战

**问题 1：为什么采样后需要过滤？**

直觉：GPT-J 生成的 API 调用应该大部分有用。

实际：过滤掉 90%+ 的采样。

原因：
```
In-context learning 的问题:
- 模型模仿示例格式，但不理解语义
- 可能在不需要的地方调用 API
- 可能调用错误的 API

示例（错误采样）:
"今天天气晴朗。<API>calculator.add(1,1)→2</API>"
问题：格式正确，但毫无意义

过滤的必要性:
- 基于损失减少的过滤确保质量
- 只保留真正有帮助的调用
```

**问题 2：为什么不用强化学习？**

直觉：RL 可以学习最优的工具使用策略。

实际：用简单的自监督过滤。

原因：
```
RL 的问题:
- 需要设计奖励函数
- 训练不稳定
- 样本效率低

Toolformer 的优势:
- 过滤标准明确（损失减少）
- 一次性数据处理，无需多轮训练
- 样本效率高（百万级数据，单次微调）
```

**问题 3：Toolformer 能链式调用工具吗？**

直觉：应该能组合多个工具。

实际：不能，这是主要局限。

原因：
```
当前限制:
- 每个位置独立决策
- 无法规划多步工具使用

示例（失败案例）:
问题："2020 年美国总统大选的获胜者年龄是多少？"
需要：1) 搜索大选结果 → 2) 搜索获胜者年龄

Toolformer: 只能调用一个工具
解决：需要更高级的规划能力（未来工作）
```

**问题 4：为什么不用更大的模型生成训练数据？**

直觉：用 GPT-3 175B 标注应该更好。

实际：用 GPT-J 自己标注自己。

原因：
```
考虑过但放弃:
- GPT-3 不一定更擅长工具使用
- 需要 API 访问，成本高
- 蒸馏可能丢失 GPT-J 的特性

自我标注的优势:
- 模型学习适合自己的工具使用模式
- 无需外部依赖
- 更可扩展
```

**问题 5：过滤会引入偏差吗？**

直觉：基于损失的过滤可能有偏差。

实际：是的，但偏差是"正确"的。

分析：
```
潜在偏差:
- 倾向于保留训练数据中的模式
- 可能对某些 API 类型有偏好

但偏差方向正确:
- 保留的是降低损失的调用
- 降低损失 = 提高预测准确性
- 这正是我们想要的

验证:
- Zero-shot 泛化良好
- 在不同任务上都有效
```

---

## 第六章：如何应用

### 场景 1：构建自己的 Toolformer

```python
from transformers import GPTJForCausalLM, AutoTokenizer
import torch

class SimpleToolformer:
    """简化的 Toolformer 实现"""

    def __init__(
        self,
        model_name="EleutherAI/gpt-j-6B",
        api_registry=None,
        filter_threshold=0.5
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = GPTJForCausalLM.from_pretrained(model_name)
        self.api_registry = api_registry or {}
        self.filter_threshold = filter_threshold

    def register_api(self, name: str, func: callable):
        """注册一个 API"""
        self.api_registry[name] = func

    def sample_api_calls(self, text: str, examples: list) -> str:
        """使用 in-context learning 采样 API 调用"""
        prompt = self._build_prompt(text, examples)
        inputs = self.tokenizer(prompt, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                max_length=len(inputs['input_ids'][0]) + 200,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def execute_api_calls(self, text: str) -> str:
        """执行文本中的所有 API 调用"""
        import re
        pattern = r'<API>(\w+)\(([^)]*)\)→</API>'

        def replace_api(match):
            api_name = match.group(1)
            args_str = match.group(2)

            if api_name not in self.api_registry:
                return match.group(0)

            # 解析并执行 API
            args = self._parse_args(args_str)
            result = self.api_registry[api_name](**args)

            return f'<API>{api_name}({args_str})→{result}</API>{result}'

        return re.sub(pattern, replace_api, text)

    def filter_by_loss(self, text: str) -> str:
        """基于损失减少过滤 API 调用"""
        # 简化实现：保留显著提升的调用
        # 完整实现需要计算每个位置的 loss
        pass

    def train(self, corpus: list, examples: list):
        """训练 Toolformer"""
        annotated_data = []

        for text in corpus:
            # 步骤 1: 采样
            annotated = self.sample_api_calls(text, examples)

            # 步骤 2: 执行
            executed = self.execute_api_calls(annotated)

            # 步骤 3: 过滤
            filtered = self.filter_by_loss(executed)

            if filtered:
                annotated_data.append(filtered)

        # 步骤 4: 微调
        self._finetune(annotated_data)

    def _build_prompt(self, text: str, examples: list) -> str:
        prompt = "请在文本中插入 API 调用：\n\n"
        for ex in examples:
            prompt += f"{ex['original']}\n{ex['annotated']}\n\n"
        prompt += f"{text}\n"
        return prompt

    def _parse_args(self, args_str: str) -> dict:
        # 解析参数字符串
        pass

    def _finetune(self, data: list):
        # 微调模型
        pass

# 使用示例
toolformer = SimpleToolformer()

# 注册 API
toolformer.register_api('add', lambda a, b: a + b)
toolformer.register_api('multiply', lambda a, b: a * b)

# 训练
corpus = ["123 加 456 是多少？", "计算 789 乘以 123"]
examples = [
    {'original': '2 加 2 等于？', 'annotated': '2 加 2 等于<API>add(2,2)→4</API>4'}
]

toolformer.train(corpus, examples)
```

### 场景 2：集成到现有 LLM 应用

```python
# 使用 Hugging Face transformers + 自定义工具
from transformers import pipeline

class ToolAugmentedLLM:
    """工具增强的 LLM"""

    def __init__(self, model_name="gpt2"):
        self.llm = pipeline("text-generation", model=model_name)
        self.tools = {
            'calculator': self._calculator,
            'search': self._search,
            'translate': self._translate,
        }

    def generate(self, prompt: str, max_tools: int = 3) -> str:
        """生成响应，可调用工具"""
        response = ""
        remaining = prompt
        tools_used = 0

        while tools_used < max_tools:
            # 检测是否需要工具
            tool_call = self._detect_tool_call(remaining)

            if not tool_call:
                # 无需工具，直接生成
                response += self.llm(remaining, max_length=100)[0]['generated_text']
                break

            # 执行工具调用
            result = self._execute_tool(tool_call)
            response += f"<tool>{tool_call['name']}({tool_call['args']})→{result}</tool>"
            remaining = f"已知：{result}。继续："
            tools_used += 1

        return response

    def _detect_tool_call(self, text: str) -> dict:
        # 检测文本中隐含的工具调用需求
        # 简化实现：关键词匹配
        if any(kw in text.lower() for kw in ['计算', '等于', '乘以']):
            return {'name': 'calculator', 'args': text}
        return None

    def _execute_tool(self, tool_call: dict):
        tool_func = self.tools.get(tool_call['name'])
        if tool_func:
            return tool_func(tool_call['args'])
        return "Unknown tool"

    def _calculator(self, args):
        # 简单计算器
        import re
        numbers = re.findall(r'\d+', args)
        if len(numbers) >= 2:
            return int(numbers[0]) * int(numbers[1])
        return "Cannot parse"

    def _search(self, args):
        return "Search result"

    def _translate(self, args):
        return "Translation"

# 使用
llm = ToolAugmentedLLM()
response = llm.generate("123 乘以 456 等于多少？")
print(response)
```

### 场景 3：选择合适的过滤阈值

| 应用场景 | 推荐 τ_f | 说明 |
|----------|----------|------|
| 高精度需求 | 1.0-2.0 | 严格过滤，只保留最确定的调用 |
| 通用场景 | 0.5 | 平衡质量和数量 |
| 探索模式 | 0.1-0.3 | 宽松过滤，发现潜在有用调用 |
| 低资源工具 | 0.3-0.5 | API 调用成本高时 |

### 场景 4：何时使用 Toolformer

**适用场景**：
- 需要外部知识（事实、时效信息）
- 需要计算能力（数学、日期）
- 有可用的 API 服务
- 训练数据充足（10 万 + 样本）

**不适用场景**：
- 需要链式工具调用（考虑 ReAct/Agent）
- 实时交互式工具使用
- 训练数据极少（<1 万样本）

---

## 第七章：延伸思考

1. **为什么工具使用能力在 775M 参数时涌现？** 这个规模有什么特殊性？与模型的"in-context learning"能力有关吗？

2. **Toolformer 的局限性是否可以通过架构改进解决？** 比如显式的工具调用头（tool calling head）或分离的工具规划模块？

3. **自监督过滤是否适用于其他任务？** 比如代码生成（基于编译通过率过滤）、创意写作（基于人类反馈过滤）？

4. **如何让模型学会链式工具调用？** 是否需要引入规划（planning）或强化学习？

5. **Toolformer 与 ReAct 的关系是什么？** 两者都结合推理和工具使用，但方法不同。能否融合？

6. **工具使用是否应该成为语言模型的预训练目标之一？** 而不是事后微调？

---

**论文元信息**
- 标题：Toolformer: Language Models Can Teach Themselves to Use Tools
- 作者：Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom
- 机构：Meta AI Research
- arXiv: 2302.04761
- 阅读时间：2026-03-15
- 方法论：高保真交互式精读协议
