# 从 30 行代码到自治团队：Claude Code 的演化之路

> 预计时长: 10-15 分钟 | 目标听众: 对 AI Agent 感兴趣的开发者

---

## 开场 (1-2 分钟)

### 钩子：一个神秘现象

你们有没有遇到过这种情况：

用 Claude Code 写代码，一开始特别顺，越用越觉得它"懂我"。但过了半小时，它开始"犯蠢"——明明刚才说过的事，它忘了；明明知道的项目约定，它不遵守。

然后你开始写更长的 CLAUDE.md，加更多的规则，接更多的 MCP 工具……结果呢？它反而更不稳定了。

**为什么？**

答案是：**这不是 Prompt 的问题，而是这套系统的设计就是这样的。**

今天我想和大家分享的，就是 Claude Code 底层是怎么"长"出来的——从一个 30 行的简单循环，一步步演化成能跑多智能体协作的复杂系统。

### 今天会讲什么

1. **起点**：30 行代码的 Agent Loop —— "原来智能体这么简单？"
2. **演化**：四个关键机制 —— 每一个都解决一个具体问题
3. **陷阱**：上下文工程的隐形开销 —— "我的 200K 去哪了？"
4. **终局**：自治智能体 —— "队友自己看看板，有活就认领"

---

## 第一部分：起点——30 行代码的魔法 (2-3 分钟)

### 一个简单的问题

大语言模型能推理代码，但它碰不到真实世界——不能读文件、跑测试、看报错。

没有循环，每次工具调用你都得手动把结果粘回去。**你自己就是那个循环。**

### 解决方案：一个 while True

```python
def agent_loop(query):
    messages = [{"role": "user", "content": query}]
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM,
            messages=messages, tools=TOOLS,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_bash(block.input["command"])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```

**不到 30 行，这就是整个智能体。**

一个循环 + 一个退出条件 + 一个工具执行器，完了。后面所有复杂的机制——Subagent、Skills、Hooks、Agent Teams——都是在这个循环上叠加的，**循环本身始终不变**。

> [来源: s01-the-agent-loop.md]

### 格言

> **"One loop & Bash is all you need"** —— 一个工具 + 一个循环 = 一个智能体

---

## 第二部分：演化——四个关键机制 (5-6 分钟)

这个简单的循环很快就会遇到问题。让我们看看它是怎么"长"出更多能力的。

### 机制 1：Subagent——上下文隔离

**问题**：智能体工作越久，messages 数组越胖。每次读文件、跑命令的输出都永久留在上下文里。

> "这个项目用什么测试框架？" 可能要读 5 个文件，但父智能体只需要一个词："pytest"。

**解决方案**：派一个独立的 Claude 去干脏活。

```
Parent agent                     Subagent
+------------------+             +------------------+
| messages=[...]   |             | messages=[]      | <-- fresh
|                  |  dispatch   |                  |
| tool: task       | ----------> | while tool_use:  |
|                  |             |   call tools     |
|                  |  summary    |   append results |
| result = "pytest"| <---------- | return last text |
+------------------+             +------------------+

Parent context stays clean. Subagent context is discarded.
```

子智能体可能跑了 30+ 次工具调用，但**整个消息历史直接丢弃**。父智能体收到的只是一段摘要文本。

> [来源: s04-subagent.md]

**关键洞察**：Subagent 的最大价值不是"并行"，而是**隔离**。扫代码库、跑测试、做审查这类会产生大量输出的事，塞进主线程很快就把有效上下文挤没了。

### 机制 2：Skills——按需加载的知识

**问题**：你希望智能体遵循特定工作流——git 约定、测试模式、代码审查清单。全塞进系统提示太浪费。

> 10 个技能，每个 2000 token，就是 20,000 token，大部分跟当前任务毫无关系。

**解决方案**：两层注入。

```
System prompt (Layer 1):
+--------------------------------------+
| Skills available:                    |
|   - git: Git workflow helpers        |  ~100 tokens/skill
|   - test: Testing best practices     |
+--------------------------------------+

When model calls load_skill("git"):
+--------------------------------------+
| tool_result (Layer 2):               |
| <skill name="git">                   |
|   Full git workflow instructions... |  ~2000 tokens
| </skill>                             |
+--------------------------------------+
```

模型知道有哪些技能（便宜），需要时再加载完整内容（贵）。

> [来源: s05-skill-loading.md]

### 机制 3：Context Compact——三层压缩

**问题**：读一个 1000 行的文件就吃掉 ~4000 token；读 30 个文件、跑 20 条命令，轻松突破 100K token。

**解决方案**：三层压缩，激进程度递增。

```
[Layer 1: micro_compact]  ← 每次 LLM 调用前
  旧 tool_result → "[Previous: used {tool_name}]"

[Layer 2: auto_compact]   ← token 超过阈值时
  保存完整对话到磁盘
  LLM 做摘要
  替换所有消息为 [summary]

[Layer 3: compact tool]   ← 手动触发
  同样的摘要机制
```

完整历史通过 transcript 保存在磁盘上。**信息没有真正丢失，只是移出了活跃上下文。**

> [来源: s06-context-compact.md]

**陷阱提醒**：默认压缩算法按"可重新读取"判断，早期的 Tool Output 和文件内容会被优先删掉，**顺带把架构决策和约束理由也一起扔了**。两小时后再改，可能根本不记得两小时前定了什么。

### 机制 4：Agent Teams——持久化队友

**问题**：Subagent 是一次性的——生成、干活、返回摘要、消亡。没有身份，没有跨调用的记忆。

真正的团队协作需要三样东西：
1. 能跨多轮对话存活的持久智能体
2. 身份和生命周期管理
3. 智能体之间的通信通道

**解决方案**：JSONL 邮箱 + 状态机。

```
Communication:
  .team/
    config.json           <- team roster + statuses
    inbox/
      alice.jsonl         <- append-only, drain-on-read
      bob.jsonl
      lead.jsonl
```

> [来源: s09-agent-teams.md]

### 演化总结

| 机制 | 问题 | 格言 |
|------|------|------|
| Subagent | 上下文污染 | *"大任务拆小，每个小任务干净的上下文"* |
| Skills | 知识塞太满 | *"用到什么知识，临时加载什么知识"* |
| Compact | 上下文会满 | *"上下文总会满，要有办法腾地方"* |
| Teams | 一个人干不完 | *"任务太大一个人干不完，要能分给队友"* |

---

## 第三部分：陷阱——上下文的隐形开销 (2-3 分钟)

### 一个真实的数字

Claude Code 的 200K 上下文并非全部可用：

```
200K 总上下文
├── 固定开销 (~15-20K)
│   ├── 系统指令: ~2K
│   ├── Skill 描述符: ~1-5K
│   ├── MCP Server 工具定义: ~10-20K  ← 最大隐形杀手
│   └── LSP 状态: ~2-5K
│
├── 半固定 (~5-10K)
│   ├── CLAUDE.md: ~2-5K
│   └── Memory: ~1-2K
│
└── 动态可用 (~160-180K)
    ├── 对话历史
    ├── 文件内容
    └── 工具调用结果
```

> [来源: 你不知道的 Claude Code]

### 算一笔账

一个典型 MCP Server（如 GitHub）包含 20-30 个工具定义，每个约 200 tokens，合计 **4,000-6,000 tokens**。

接 5 个 Server，光这部分固定开销就到了 **25,000 tokens（12.5%）**。

我第一次算出这个数字的时候，真没想到有这么多。在要读大量代码的场景，这 12.5% 真的很关键。

### 推荐的上下文分层

```
始终常驻    → CLAUDE.md：项目契约 / 构建命令 / 禁止事项
按路径加载  → rules：语言 / 目录 / 文件类型特定规则
按需加载    → Skills：工作流 / 领域知识
隔离加载    → Subagents：大量探索 / 并行研究
不进上下文  → Hooks：确定性脚本 / 审计 / 阻断
```

**说白了，偶尔用的东西就不要每次都加载进来。**

---

## 第四部分：终局——自治智能体 (2 分钟)

### 最后一步：自组织

s09-s10 中，队友只在被明确指派时才动。领导得给每个队友写 prompt，任务看板上 10 个未认领的任务得手动分配。这扩展不了。

**真正的自治**：队友自己扫描任务看板，认领没人做的任务，做完再找下一个。

```
Teammate lifecycle with idle cycle:

+-------+
| spawn |
+---+---+
    |
    v
+-------+                +-------+
| WORK  | <------------> |  LLM  |
+---+---+                +-------+
    |
    v
+--------+
|  IDLE  |  poll every 5s for up to 60s
+---+----+
    |
    +---> check inbox --> message? ------> WORK
    |
    +---> scan .tasks/ --> unclaimed? ---> claim -> WORK
    |
    +---> 60s timeout ------------------> SHUTDOWN
```

> [来源: s11-autonomous-agents.md]

### 一个有趣的细节：身份重注入

上下文压缩后，智能体可能忘了自己是谁。

```python
if len(messages) <= 3:  # 说明发生了压缩
    messages.insert(0, {"role": "user",
        "content": f"<identity>You are '{name}', role: {role}...</identity>"})
```

**智能体也有"失忆症"——每次 compact 就像喝断片，醒来只记得摘要。**

---

## 结尾 (1 分钟)

### 核心要点回顾

1. **起点**：30 行代码，一个 while True 循环
2. **演化**：每个机制解决一个具体问题，循环本身不变
3. **陷阱**：MCP 工具定义占用 12.5% 上下文，是最大隐形杀手
4. **终局**：自治智能体 = 自组织 + 身份重注入

### 一句话总结

> **"模型就是智能体。我们的工作就是给它工具，然后让开。"**

### 行动建议

如果你听完想自己试试：

1. 去 [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) 跑一遍 12 个课程
2. 用 `/context` 观察你的上下文消耗
3. 检查你的 MCP Server 数量——每多一个，就多一份固定开销

### 参考资料

- [learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) - 从零构建 nano Claude Code-like agent
- [你不知道的 Claude Code：架构、治理与工程实践](https://x.com/HiTw93/article/2032091246588518683) - 深度实践文章
- ReAct, Toolformer, Gorilla 等 Agent 论文（见 papers 目录）

---

## Q&A 预设问题

**Q: Subagent 和直接在主对话里干有什么区别？**

A: 核心区别是上下文隔离。Subagent 跑完 30 次工具调用，主对话只拿到一段摘要。如果直接在主对话里干，这 30 次调用的结果全留在上下文里，很快就把有效空间挤满了。

**Q: 什么时候该用 Subagent，什么时候该用 Skill？**

A: 简单说：
- Skill 是"知识"，告诉 Claude 怎么做某类事
- Subagent 是"执行者"，派它去干会产生大量输出的事

扫代码库、跑测试、做审查 → Subagent
Git 约定、测试模式、代码审查清单 → Skill

**Q: CLAUDE.md 应该写多长？**

A: Anthropic 官方自己的 CLAUDE.md 大约只有 2.5K tokens。建议保持短、硬、可执行——优先写命令、约束、架构边界。大段背景介绍和完整 API 文档应该放到别处。

**Q: 上下文压缩后，智能体忘了重要决策怎么办？**

A: 两个方案：
1. 在 CLAUDE.md 里写 Compact Instructions，明确压缩时必须保留什么
2. 开新会话前，让 Claude 写一份 HANDOFF.md，把进度、尝试过什么、哪些走通了、哪些是死路写清楚