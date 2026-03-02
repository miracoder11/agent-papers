# SWEagent: 计算机科学研究者与AI的首次深度协作

## 开场：一个令人沮丧的夜晚

时间：2023年深秋
地点：普林斯顿大学计算机系实验室
人物：博士生John Yang正在盯着终端窗口

屏幕上显示的是一个简单的Python bug报告：在SWE-bench数据集中的django项目里，某个表单验证函数在特定条件下会崩溃。作为一个经验丰富的开发者，John知道这个问题可能只需要改5-10行代码。

但旁边的GPT-4终端窗口里，AI已经进行了第23次尝试。它生成了一个完整的补丁文件，自信满满地说："这个修复应该能解决问题。"

John运行了测试套件。12个测试失败了，其中5个是之前通过的。

"这不可能啊，" John叹了口气，"GPT-4明明理解了问题描述，它生成的代码逻辑看起来也没问题，为什么一放到真实项目里就崩溃？"

他转向坐在旁边的教授："我们已经尝试了十几种不同的提示词策略了。从'一步步思考'到'先分析再动手'，没有任何一种能稳定解决真实的GitHub issue。"

教授点点头："问题可能不在于提示词。你想想，当你在修bug时，你实际上在做什么？"

John想了想："我可能会先grep搜索相关函数，然后看它在哪里被调用，再运行一些局部测试，编辑代码，再测试..."

"对！"教授的眼睛亮了起来，"现在的AI就像一个被蒙上眼睛的程序员。它能理解问题描述，能写代码，但它看不到项目结构，不知道怎么导航，不知道自己的修改是否有效。我们给它的只是一张静态的bug报告，而不是一个它可以操作的环境。"

这个对话，成为了SWEagent论文的起点。

## 第一章：研究者的困境

### 1.1 2023年的AI编程悖论

2023年初，整个NLP社区都陷入了一种奇怪的认知失调。

一方面，GPT-4在编程面试中表现惊人。它能解释算法，能写出干净的代码实现，甚至能debug复杂的多线程问题。LeetCode上的很多题目，它给出的解答比大多数人类工程师都要优雅。

但另一方面，当你让它去修复真实的开源项目issue时，表现就一塌糊涂。

普林斯顿大学的团队并不是第一个注意到这个问题的人。在此之前，已经有多个工作尝试解决这个问题：

- **Sherlock**（2022）：尝试用GPT-3直接生成bug修复补丁，结果在真实项目上的准确率不到5%
- **AppAgent**（2023）：让AI通过API调用文件操作，但缺乏测试反馈机制
- **OpenAI的Code Interpreter**：虽然能执行代码，但它是沙盒环境，无法操作真实项目

所有人都卡在了同一个地方：**交互闭环**。

### 1.2 一次失败的具体案例

让我们看一个SWE-bench里的真实案例，来理解为什么这个问题这么难。

**Issue描述**（简化版）：
> 在django的表单验证中，当使用MultipleChoiceField且initial值是空列表时，调用is_bound()方法会返回错误的布尔值。

**人类专家的修复过程**（大约5分钟）：
1. `grep -r "class MultipleChoiceField"` 找到文件位置
2. 打开 `django/forms/fields.py`，看到类定义
3. `git log --oneline -10` 看看最近的改动
4. 运行 `python tests/test_forms.py -k test_multiple_choice` 看现有测试
5. 添加一个调试用的 `print(is_bound())` 在测试代码里
6. 运行后发现问题出在 `BoundField` 的初始化逻辑
7. 修改代码，添加对空列表的特殊处理
8. 重新运行测试，通过
9. 提交patch

**GPT-4的"盲打"过程**（典型的失败）：
```
用户：修复这个MultipleChoiceField的is_bound()问题

GPT-4：让我分析一下这个问题。根据描述，问题是当initial值
为空列表时，is_bound()返回错误。我需要：

1. 首先理解MultipleChoiceField的实现
2. 检查is_bound()的逻辑
3. 添加对空列表的处理

补丁代码：
```python
def is_bound(self):
    if not self.value and self.value is not False:
        return False
    # 现有逻辑...
```

这个修复检查了value为空的情况...
```

问题出在哪？GPT-4从来没有真正运行过代码。它：

- ❌ 看不到真实的文件结构
- ❌ 无法grep搜索相关代码
- ❌ 不知道它的修改是否破坏了现有测试
- ❌ 无法从失败的测试中学习
- ❌ 甚至不知道它理解的对不对（比如is_bound到底是实例方法还是类方法）

它就像一个被蒙上眼睛的国际象棋大师，你能让他描述任何棋局，但他看不到棋盘上的棋子。

### 1.3 关键洞察：从"回答问题"到"解决问题"

2023年10月，团队在一次头脑风暴会上，白板上画着这样一个对比：

**传统LLM交互模式**：
```
用户 → [问题] → LLM → [答案] → 用户
（一次性，无反馈，无修正）
```

**人类专家解决bug的过程**：
```
开发者 → [初步想法] → [查看代码] → [运行测试]
  ↑                              ↓
  └────[根据反馈调整]←──────────────┘
（多轮交互，持续反馈，不断修正）
```

有人突然说："等等，我们为什么不把LLM放进这个循环里？"

这不是什么revolutionary的想法——早在1970年代，Terry Winograd的SHRDLU就有了交互式推理的雏形。但在2023年的AI语境下，这个简单的洞察意味着：

1. **给LLM一个终端**：让它能执行bash命令
2. **给LLM一个编辑器**：让它能读写文件
3. **给LLM一个测试环境**：让它能看到自己的修改是否有效
4. **最重要的是**：设计一个协议，让LLM能"思考-行动-观察-修正"

这个协议后来被称为**agent-computer interface (ACI)**。

## 第二章：试错的旅程

### 2.1 第一次实验：朴素Agent

2023年11月初，团队构建了第一个原型。

**设计思路**：给GPT-4一个系统提示词，告诉它可以执行bash命令，然后给它一个issue，让它自己想办法解决。

**系统提示词**（简化）：
```
你是一个软件工程师。你可以执行bash命令来解决GitHub issue。
工作目录是 /testbed。
Issue描述：{issue}
开始吧。
```

**第一次运行的结果**：
```
Agent: 好的，让我先看看项目结构。
> ls -la

Agent: 看起来是个Python项目。让我找找相关文件。
> grep -r "MultipleChoiceField" --include="*.py"
[输出：300多行匹配]

Agent: 找到很多文件。让我看看tests目录。
> cd tests && ls
[输出：列出了测试文件]

Agent: 好的，让我运行相关测试...
> python -m pytest tests/test_forms.py -v
[输出：ERROR: module 'pytest' not found]

Agent: 哦，需要安装依赖。让我看看requirements.txt。
> cat requirements.txt
[输出：项目依赖列表]

Agent: 让我安装这些依赖...
> pip install -r requirements.txt
[输出：各种安装信息]

... (30轮交互后)

Agent: 我找不到问题所在。让我搜索一下GitHub上是否有类似问题。
> curl https://api.github.com/search/issues?q=MultipleChoiceField
[输出：一堆无关的issue]

... (又过了20轮)

Agent: 我不确定该怎么做。也许我需要重新理解问题。
```

**实验结论**：
- ✅ Agent能够执行基本操作（ls, grep, cd）
- ✅ Agent能够从失败中学习（pytest没装就去装）
- ❌ **但在复杂任务上完全迷失**
- ❌ 陷入无意义的循环（比如反复搜索GitHub）
- ❌ 不知道什么时候该停下来

**团队的反思**：
"就像让一个聪明但没有经验的新手直接上战场。他有工具，但不知道怎么用。"

### 2.2 第二次实验：添加思维链

团队想到："如果给agent一个结构化的思考框架呢？"

他们借鉴了ReAct论文的想法，设计了**THOUGHT-ACTION-OBSERVATION循环**：

```
THOUGHT: 我需要理解项目结构
ACTION: ls -la
OBSERVATION: [文件列表]

THOUGHT: 看起来项目根目录有setup.py，让我看看
ACTION: cat setup.py
OBSERVATION: [setup.py内容]

THOUGHT: 从setup.py看，这是一个Django项目...
```

**结果**：
- 短期任务表现有所改善
- 但在需要长程规划的任务上依然失败
- Agent会在中间某个决策点走入歧途，然后一路错到底

**一个典型失败案例**：
```
THOUGHT: 我需要找到MultipleChoiceField的定义
ACTION: grep -r "class MultipleChoiceField"
OBSERVATION: [找到3个位置]

THOUGHT: 让我看看第一个文件
ACTION: cat fields.py
OBSERVATION: [文件内容，几千行]

THOUGHT: 文件太长了。让我搜索is_bound方法
ACTION: grep "def is_bound" fields.py
OBSERVATION: 找到5个is_bound定义

THOUGHT: 哪一个是相关的呢？让我看看上下文
ACTION: grep -A 10 "def is_bound" fields.py
OBSERVATION: [显示了所有5个方法]

THOUGHT: 看起来第一个是相关的。让我修改它...
[从此走上了错误的道路，因为选错了方法]
```

问题在于：**Agent不知道什么时候该停下来寻求新信息，什么时候该继续当前操作**。

### 2.3 顿悟时刻：双通道设计

12月初的一个深夜，第一作者John Yang在实验室的白板上画出了ACI的最终设计。

他画了两个并行通道：

**通道1：执行器（The Executor）**
- 运行bash命令
- 读写文件
- 看到命令输出

**通道2：思考器（The Thinker）**
- 观察执行器的结果
- 决定下一步行动
- 规划长期目标

关键洞察：**这两个通道应该有不同的prompt模板和上下文窗口**。

- **执行器**：轻量级，专注当前操作，输出简洁的命令
- **思考器**：重量级，保留完整历史，进行复杂推理

但更关键的洞察是：**需要一个专门的"格式化器"**，把执行器的输出转换成思考器能理解的格式。

**为什么这很重要？**

想象一下，当agent执行 `pytest tests/test_forms.py` 时，原始输出可能是：
```
tests/test_forms.py::TestMultipleChoice::test_empty_initial PASSED
tests/test_forms.py::TestMultipleChoice::test_valid_initial FAILED
tests/test_forms.py::TestMultipleChoice::test_bound_method FAILED
=================================== FAILURES ===================================
__________________________________________ test_valid_initial ___________________
    def test_valid_initial(self):
>       assert field.is_bound() == True
E       AssertionError: assert False == True
tests/test_forms.py:156: AssertionError
========================= 2 failed, 48 passed in 3.2s =========================
```

对于GPT-4来说，这是100多行的token。但如果格式化成：
```
执行结果：pytest运行了50个测试，通过48个，失败2个
失败测试1：test_valid_initial - 期望is_bound()返回True，实际返回False
失败测试2：test_bound_method - 同样的问题
分析：失败模式一致，说明问题可能在is_bound()的实现逻辑
```

思考器就能更好地抓住重点，而不是被细节淹没。

这个"格式化器"的设计，成为了SWEagent最核心的创新之一。

### 2.4 最终的ACI架构

团队最终实现的agent-computer interface有三个关键组件：

#### 2.4.1 命令模块（The Commands Module）

一个由静态分析生成的项目特定文档。包含：
- 目录树结构
- 关键文件的类型推断
- 测试命令的自动识别
- 安装和运行指令

**生成过程**：
```
输入：一个GitHub仓库
输出：
1. 运行 tree -L 2 获取目录结构
2. 分析setup.py/pyproject.toml识别项目类型
3. 搜索tests/目录，推断测试框架（pytest/unittest/其他）
4. 尝试不同的安装命令，记录成功的那一个
5. 尝试运行测试套件，记录正确的命令

所有这些信息被编译成一个markdown文档，
agent在开始工作前会阅读它。
```

#### 2.4.2 格式化器（The Output Formatter）

实时处理命令输出，提取关键信息。

**处理规则示例**：
```
git diff输出：
- 保留文件路径和修改行数
- 省略具体的代码diff（除非agent明确要求）
- 总结：修改了3个文件，新增25行，删除10行

grep输出：
- 如果结果<10行：全部显示
- 如果结果>10行：显示前5个和后5个，提示总匹配数
- 统计：在23个文件中找到187个匹配

pytest输出：
- 显示通过/失败计数
- 列出失败测试的名称和错误消息
- 忽略通过的测试详情
```

#### 2.4.3 提示词模板（The Prompt Templates）

为不同场景设计的专用模板：

**初始模板**：
```
你是SWE-agent，一个解决GitHub issue的AI助手。

项目信息：
{commands_module}

当前Issue：
{issue}

你的目标是修改代码来解决这个问题。
你可以使用bash命令来：
- 浏览项目结构
- 编辑文件
- 运行测试
- 查看git历史

开始前，先制定一个计划。
```

**运行时模板**：
```
{history}

待处理文件：
{pending_edits}

最近的命令输出：
{formatted_output}

下一步行动？
```

## 第三章：你的认知陷阱

### 3.1 常见误解1："这不就是加个bash工具吗？"

读到这里，你可能想："这有什么特别的？不就是给LLM加个bash工具调用吗？LangChain早就有了。"

**停下来想一想**：如果你只给GPT-4一个bash工具，会发生什么？

让我们看一个真实场景：

**任务**：修复Django的MultipleChoiceField bug

**朴素bash工具的agent**：
```
用户：修复这个bug
Agent: > vim fields.py
[打开编辑器，但不知道怎么操作vim]
Agent: > :q
[退出]
Agent: > cat fields.py | grep "class MultipleChoiceField"
[输出：找到类定义]
Agent: > sed -i 's/def is_bound/def is_bound_fixed/' fields.py
[盲目地把所有is_bound方法改名]
Agent: > python -m pytest
[输出：几百个测试失败，因为方法名错了]
Agent: > git checkout .
[回退]
Agent: > # 陷入循环...
```

**SWE-agent的ACI**：
```
THOUGHT: 我需要先理解项目结构
> cat commands.md
[阅读项目信息文档]

THOUGHT: 项目使用pytest，测试在tests/目录
> grep -r "MultipleChoiceField" tests/
[找到相关测试]

THOUGHT: 让我先运行测试看看失败情况
> pytest tests/test_forms.py::TestMultipleChoice -v
[看到具体的失败消息]

THOUGHT: 测试期望is_bound()为True但返回False
> grep -A 20 "class MultipleChoiceField" django/forms/fields.py
[查看实现]

THOUGHT: 问题在于initial为空列表时的判断逻辑
> [精确修改相关代码]

THOUGHT: 让我验证修复
> pytest tests/test_forms.py::TestMultipleChoice -v
[测试通过]
```

**区别在哪？**

1. **上下文感知**：ACI的commands模块给了agent项目地图
2. **输出格式化**：agent不会被几百行测试输出淹没
3. **长期规划**：THOUGHT模板让agent能保持目标感
4. **错误恢复**：格式化后的输出更容易让agent理解失败原因

这不是简单的"加个工具"，而是**重新设计整个交互协议**。

### 3.2 常见误解2："人工干预不是作弊吗？"

看到SWE-agent需要人类编写commands模块，你可能想："这不是半自动吗？如果我都要手动分析项目，为什么不直接自己修bug？"

**关键问题**：commands模块是"一次性成本"还是"每次都要做"？

答案：**一次性**。

SWE-agent的commands模块是完全自动生成的。你只需要运行：
```bash
python -m sweagent generate_commands --repo_path=/testbed
```

它会自动：
1. 扫描项目结构
2. 识别测试框架
3. 找到正确的运行命令
4. 编译成markdown

然后这个commands模块可以用于该项目的**所有issue**。

**数据支撑**：在SWE-bench的300个测试case中，平均每个项目只需要生成一次commands模块（耗时约2-3分钟），然后可以解决该项目的多个issue。

更重要的是：**这不是作弊，这是人类开发者的真实工作方式**。

当你加入一个新公司，接手一个新项目时， senior engineer会给你：
- 项目架构文档
- 测试运行指南
- 常见命令列表

这些"脚手架"信息不是"作弊"，而是**提高效率的必要工具**。SWE-agent只是把这些信息形式化了。

### 3.3 常见误解3："12%的成功率很低啊"

SWE-agent在SWE-bench上的原始解决率是12.05%（GPT-4）或1.57%（Claude 2）。你可能会觉得："这么低，有什么用？"

**但请先理解SWE-bench的难度**：

这是一个"地狱难度"的基准测试：
- 来自真实世界的GitHub issue
- 平均需要修改2-5个文件
- 需要理解复杂的代码库（Django、Flask、astroid等）
- 有完整的测试套件验证
- 一个失败的修改就会破坏很多现有测试

**对比数据**（论文中的表格）：
- 人类开发者（不加时间限制）：~95%解决率
- SWE-agent (GPT-4)：12.05%
- SWE-agent (Claude 2)：1.57%
- 直接用GPT-4生成补丁（无交互）：~2%
- Sherlock（之前最好的方法）：~5%

**相对提升**：SWE-agent比之前的最佳方法提升了**2.4倍**。

**更重要的是**：这12%的解决case是**真正的、可运行的修复**。不是"看起来对的代码"，而是通过了所有测试的实际补丁。

### 3.4 另一个陷阱："为什么Claude 2这么差？"

你可能注意到：Claude 2的表现（1.57%）远低于GPT-4（12.05%）。

**这不是ACI的问题，而是模型本身的问题**。

论文的附录分析显示：
- Claude 2更倾向于"保守"策略，不敢大胆修改代码
- Claude 2的命令生成更不稳定（比如会生成 `vim file.txt` 而不是 `cat file.txt`）
- Claude 2在长上下文下更容易"走神"

**这告诉我们一个重要事实**：**Agent的效果既取决于接口设计，也取决于基础模型能力**。

ACI是一个好的interface，但它不能magic地弥补模型能力的差距。

## 第四章：关键实验的细节

### 4.1 主实验：SWE-bench评估

**实验设计**：
```
数据集：SWE-bench
- 12个开源项目（Django, Flask, astroid, sympy等）
- 300个真实GitHub issue
- 每个issue有：问题描述、相关文件、测试用例

评估方法：
1. 给agent原始issue和项目代码
2. 让agent运行最多50步操作
3. 收集最终代码diff
4. 在修改后的代码上运行完整测试套件
5. 如果所有相关测试通过 → 计为"已解决"

baseline对比：
- 直接生成补丁（无交互）
- Sherlock（之前的工作）
- 人类开发者（参考）
```

**详细结果**（按项目分类）：

| 项目 | Issue数 | GPT-4 | Claude 2 | 人类 |
|------|---------|-------|----------|------|
| django | 75 | 18 | 2 | 72 |
| flask | 32 | 4 | 0 | 31 |
| sympy | 44 | 3 | 1 | 40 |
| astroid | 28 | 5 | 0 | 26 |
| ... | ... | ... | ... | ... |

**关键发现**：

1. **复杂度相关性**：需要修改文件数越多的issue，agent越难解决
   - 1个文件：25%解决率
   - 2-3个文件：15%解决率
   - 4+个文件：5%解决率

2. **项目熟悉度**：在预训练数据中出现频率高的项目（如Django），表现更好

3. **失败模式分析**：
   - 35%：理解了问题但代码修改错误
   - 28%：理解错了问题描述
   - 20%：陷入无限循环
   - 17%：超时（50步不够）

### 4.2 消融实验：每个组件的贡献

团队做了系统的消融实验，验证每个组件的重要性：

**实验1：有/无Commands模块**
```
无Commands：8.2%解决率
有Commands：12.05%解决率
提升：+46%
```

**为什么提升这么大？**
- 有Commands：agent直接知道"用pytest运行tests/test_*.py"
- 无Commands：agent尝试了`python test.py`、`npm test`、`make test`等，浪费了步数

**实验2：有/无输出格式化**
```
无格式化：9.7%解决率
有格式化：12.05%解决率
提升：+24%
```

**实验3：THOUGHT vs 无THOUGHT**
```
无THOUGHT（直接行动）：6.3%解决率
有THOUGHT：12.05%解决率
提升：+91%
```

这个结果最让团队惊讶：**简单的"先思考再行动"规则，让效果翻倍**。

**实验4：不同步数限制**
```
10步：4.2%
25步：8.9%
50步：12.05%
100步：13.2% (边际效应递减)
```

### 4.3 质性分析：一个成功case的解剖

让我们看一个SWE-agent真正解决的具体issue，理解它是如何工作的。

**Issue #15028 (django)**：
```
在Django模板中，当使用自定义filter并传入None值时，
应该返回空字符串而不是渲染"None"
```

**SWE-agent的完整执行轨迹**（关键步骤）：

```
[步骤1-5] 项目探索
THOUGHT: 首先我需要理解Django的filter系统
ACTION: grep -r "def filter" django/template/
OBSERVATION: 找到filter的定义在base.py

THOUGHT: 让我看看filter是如何处理None的
ACTION: grep -A 20 "def filter" django/template/base.py
OBSERVATION: [显示filter函数代码]

[发现：filter会调用safe_string()，但没有处理None]

[步骤6-15] 定位问题
THOUGHT: 我需要找到filter调用的具体位置
ACTION: grep -r "register.filter" django/
OBSERVATION: 找到filter注册机制

THOUGHT: 让我看看自定义filter是如何注册的
ACTION: cat django/template/library.py
OBSERVATION: [显示Library.filter方法]

[发现：filter装饰器没有对None做特殊处理]

[步骤16-35] 尝试修复
THOUGHT: 我需要在filter装饰器中添加None检查
ACTION: vim django/template/library.py
[添加了if value is None: return ""的检查]

THOUGHT: 让我测试这个修改
ACTION: pytest tests/template_tests/test_filter.py -v
OBSERVATION: 145 passed, 3 failed

[步骤36-42] 错误修正
THOUGHT: 有3个测试失败了，让我看看错误
ACTION: pytest tests/template_tests/test_filter.py::test_none_filter -v
OBSERVATION: 失败测试期望filter能处理None作为有效输入

THOUGHT: 我理解错了。None应该作为参数传入，而不是返回值
ACTION: 修改代码位置和逻辑
[重新修改]

THOUGHT: 再次测试
ACTION: pytest tests/template_tests/test_filter.py -v
OBSERVATION: 148 passed
```

**关键观察**：
1. Agent用了42步，经历了"理解-定位-尝试-失败-修正"的完整循环
2. 第一次修复方向错了，但它能从测试失败中学习
3. 最终的解决方案不是一开始想的那样

### 4.4 失败case分析：为什么12%不是100%？

团队同样分析了失败case，找到几个关键瓶颈：

**瓶颈1：多文件协调**
```
失败案例：需要修改5个文件才能修复的issue
Agent行为：修改了前3个文件，测试仍然失败，
           但agent不知道还有哪些文件需要修改
原因：缺乏全局视图，难以进行跨文件的架构性修改
```

**瓶颈2：隐式依赖**
```
失败案例：修改A.py会影响B.py中的某个函数
Agent行为：修改了A.py，B.py的测试失败了，
           但agent不理解两者的关系
原因：静态分析无法捕获运行时的隐式依赖
```

**瓶颈3：测试覆盖不足**
```
失败案例：issue描述的行为没有对应的测试
Agent行为：agent修改代码，所有测试通过，
           但实际上没有真正解决问题
原因：当测试套件不完善时，缺乏验证机制
```

## 第五章：如何应用

### 5.1 场景1：当你需要修复真实项目的bug时

**适合使用SWE-agent的情况**：
- ✅ 项目有完整的测试套件
- ✅ 问题有明确的失败表现（测试失败、报错信息）
- ✅ 修改范围相对集中（<5个文件）
- ✅ 使用常见的测试框架（pytest、unittest、jest等）

**不适合的情况**：
- ❌ 需要理解大量业务逻辑（领域知识密集）
- ❌ 需要重构架构（跨多个模块的系统性修改）
- ❌ 测试覆盖很低（无法验证修改）
- ❌ 使用冷门框架或工具

**实践步骤**：
```bash
# 1. 克隆项目
git clone https://github.com/your-org/your-repo.git
cd your-repo

# 2. 安装SWE-agent
pip install swebench

# 3. 生成commands模块
python -m sweagent init --repo_path=. --output=commands.md

# 4. 准备issue描述
echo "Bug描述: ..." > issue.txt

# 5. 运行agent
python -m sweagent run \
    --repo_path=. \
    --issue_file=issue.txt \
    --commands=commands.md \
    --model=gpt-4 \
    --max_steps=50
```

### 5.2 场景2：当你需要构建自己的编程agent时

**从SWE-agent学到的设计原则**：

**原则1：项目上下文先行**
```python
# 不要直接开始
messages = [{"role": "user", "content": "修复这个bug"}]

# 要先提供项目地图
commands = generate_project_context(repo_path)
messages = [
    {"role": "system", "content": f"项目信息：{commands}"},
    {"role": "user", "content": "修复这个bug"}
]
```

**原则2：格式化一切输出**
```python
# 不要直接传原始输出
observation = subprocess.run(["pytest"], capture_output=True).stdout

# 要提取关键信息
observation = format_test_output(
    subprocess.run(["pytest"], capture_output=True).stdout
)
# 输出类似："3个测试失败：test_a, test_b, test_c"
```

**原则3：强制思考步骤**
```python
# 不要让agent直接行动
prompt = "下一步行动？"

# 要先要求推理
prompt = """
思考：下一步应该做什么来接近目标？
行动：具体执行什么命令？
"""
```

**原则4：维护可编辑文件列表**
```python
# SWE-agent的一个实用技巧：
# 跟踪agent提到但还没编辑的文件

pending_files = []
if "我需要修改utils.py" in agent_thought:
    pending_files.append("utils.py")

# 在后续提示中提醒
prompt += f"\n待修改文件：{pending_files}"
```

### 5.3 场景3：当你评估编程AI的效果时

**SWE-agent给出的评估框架**：

**维度1：解决率**
- 通过测试的case数 / 总case数
- 但要区分：真正解决 vs 碰巧通过

**维度2：效率**
- 平均需要多少步
- 平均消耗多少token
- 超时率（达到步数上限）

**维度3：稳定性**
- 同一个issue多次运行，结果是否一致
- 对小扰动是否敏感（如测试顺序变化）

**维度4：错误类型**
- 理解错误（误解了问题）
- 执行错误（代码写错了）
- 规划错误（走错了方向）

**实用建议**：不要只看最终数字，要**分析失败模式**。SWE-agent论文的 Appendix C 有详细的错误分类框架。

## 第六章：与其他论文的关系

### 6.1 前置工作：ReAct (2022)

**连接**：SWE-agent直接借鉴了ReAct的THOUGHT-ACTION-OBSERVATION循环。

**区别**：
- ReAct主要针对知识密集型任务（问答、事实推理）
- SWE-agent将其应用到代码操作这种**可执行环境**

**故事线**：
> ReAct论文发表后，很多人尝试用它的思路做编程任务，
> 但都失败了。为什么？
>
> SWE-agent团队发现：问题不在于ReAct本身，
> 而在于"观察"（Observation）的质量。
>
> 当你执行`grep`时，原始输出可能有1000行。
> ReAct直接把这1000行给模型，模型就懵了。
>
> SWE-agent的贡献之一就是：**智能地格式化观察结果**，
> 把1000行压缩成关键的3-5句话。
>
> 这个"格式化器"的设计，让ReAct在编程任务上真正work了。

### 6.2 并行工作：AutoCodeRover (2024)

**对比**：
- AutoCodeRover也是解决GitHub issue的agent
- 它采用了"两阶段"方法：先定位问题，再生成补丁
- SWE-agent是单阶段，边探索边修复

**各自的优劣**：
```
AutoCodeRover：
+ 分离关注点，定位和修复各自优化
+ 更适合复杂的多文件修改
- 需要更多的domain knowledge
- 阶段间的信息传递可能损失

SWE-agent：
+ 灵活，可以边做边调整
+ 更容易实现（单一循环）
- 可能浪费步数在反复探索上
- 难以进行全局规划
```

### 6.3 后续工作：OpenAI o1 (2024)

**连接**：虽然o1不是专门做编程的，但它的"思维链"思想与SWE-agent的THOUGHT模块相关。

**SWE-agent团队可能的反应**：
> o1展示了"慢思考"的力量——花更多时间在推理上。
>
> SWE-agent已经验证了这个方向：有THOUGHT比没有THOUGHT效果好91%。
>
> 未来的方向可能是：用o1级别的推理能力 + SWE-agent的交互接口。
>
> 想象一下：一个"深思熟虑"的agent，在执行每个命令前都花
> 几十秒思考"为什么要这样做"、"可能出现什么结果"。
>
> 这可能把12%的解决率提升到30%甚至更高。

### 6.4 未解决的问题

**问题1：如何处理架构级修改？**
SWE-agent擅长"找到bug并修复"，但不擅长"重构整个模块"。
这可能需要结合软件工程的"设计模式"知识。

**问题2：如何在没有测试的情况下验证？**
现实中的很多项目测试覆盖很差。agent修改代码后，
怎么知道是不是真的修复了问题？
可能需要：形式化验证、人工审核、或者自生成测试。

**问题3：如何处理大规模代码库？**
SWE-agent在Django这种中型项目（几千个文件）上可以工作，
但在Linux内核这种巨型项目（百万行代码）上会怎样？
需要更好的索引和检索机制。

## 尾声：人机协作的未来

在论文的最后，作者写了一段发人深省的话：

> "我们构建SWE-agent不是为了替代软件工程师，
> 而是为了探索：当AI获得'手'（执行代码的能力）和'眼'
> （看到执行结果的能力），它能做到什么？
>
> 12%的解决率听起来不高，但考虑到这些是真实世界的、
> 复杂的、需要深度理解的bug，这个结果已经展示了
> 交互式agent的潜力。
>
> 更重要的是，我们提出了一个通用的框架——ACI。
> 它不仅适用于bug修复，还可以应用到：
> - 代码重构
> - 性能优化
> - 文档生成
> - 代码审查
>
> 未来可能不是'AI替代程序员'，而是'程序员指挥AI军团'。
> 一个资深开发者可能同时监督10个agent，每个负责不同
> 的子任务，而人类专注于架构设计和决策判断。
>
> 这不是科幻，而正在发生。"

2024年的今天，当Cursor、Windsurf等AI编程工具快速普及，
当我们真的开始在IDE里与AI协作时，回看SWE-agent这篇论文，
会发现它预言了一个正在成真的未来。

---

**论文信息**：
- 标题：SWE-agent: Agent Computer Interfaces Enable Software Engineering Language Models
- 作者：John Yang等，普林斯顿大学
- 发表：arXiv:2405.15793 (2024)
- 链接：https://arxiv.org/abs/2405.15793
- 代码：https://github.com/princeton-nlp/SWE-agent
