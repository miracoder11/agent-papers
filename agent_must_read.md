# Agent 开发必读论文清单

> 精选最重要论文，按优先级排序

---

## 🔥 一级必读 (理解 Agent 的基础)

这 3 篇是 Agent 的基石，必须先读才能理解后面的一切。

| # | 论文 | arXiv | 核心 | 读它因为 |
|---|------|-------|------|---------|
| 1 | **ReAct: Synergizing Reasoning and Acting in Language Models** | [2210.03629](https://arxiv.org/pdf/2210.03629.pdf) | Thought + Action | 所有 Agent 框架的基础范式 |
| 2 | **Reflexion: Language Agents with Verbal Reinforcement Learning** | [2303.11366](https://arxiv.org/pdf/2303.11366.pdf) | 自我反思 | 让 Agent 从失败中学习，不重复犯错 |
| 3 | **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** | [2201.11903](https://arxiv.org/pdf/2201.11903.pdf) | 思维链 | 理解 LLM 如何进行复杂推理 |

**下载这 3 篇**:
```bash
curl -O https://arxiv.org/pdf/2210.03629.pdf # ReAct
curl -O https://arxiv.org/pdf/2303.11366.pdf # Reflexion
curl -O https://arxiv.org/pdf/2201.11903.pdf # CoT
```

---

## ⭐ 二级必读 (多 Agent 框架)

想做多 Agent 系统，这 4 篇必读。

| # | 论文 | arXiv | 核心 | 读它因为 |
|---|------|-------|------|---------|
| 4 | **MetaGPT: Meta Programming for Multi-Agent Collaborative Framework** | [2308.00352](https://arxiv.org/pdf/2308.00352.pdf) | SOP + 角色定义 | 最成熟的多 Agent 框架，可落地 |
| 5 | **AutoGen: Enabling Next-Gen LLM Applications** | [2308.08155](https://arxiv.org/pdf/2308.08155.pdf) | 对话编排 | 微软出品，Agent 对话范式 |
| 6 | **CAMEL: Communicative Agents for "Mind" Exploration** | [2303.04637](https://arxiv.org/pdf/2303.04637.pdf) | 通信协议 | 最早的多 Agent 通信研究 |
| 7 | **Voyager: An Open-Ended Embodied Agent with Large Language Models** | [2305.16291](https://arxiv.org/pdf/2305.16291.pdf) | 技能库 | Agent 如何积累和复用技能 |

---

## 🛠️ 三级必读 (工具使用)

Agent 必须能调用外部工具。

| # | 论文 | arXiv | 核心 | 读它因为 |
|---|------|-------|------|---------|
| 8 | **Toolformer: Language Models Can Teach Themselves to Use Tools** | [2303.04607](https://arxiv.org/pdf/2303.04607.pdf) | 自主工具学习 | 模型自己学会何时调用什么工具 |
| 9 | **Gorilla: Fine-tuned LLaMA for Tool Use** | [2305.15334](https://arxiv.org/pdf/2305.15334.pdf) | API 调用 | 解决幻觉问题，精确调用 API |
| 10 | **TaskMatrix: When LLM Meets Function Calling** | [2303.16354](https://arxiv.org/pdf/2303.16354.pdf) | 工具编排 | 微软的工具调用框架 |

---

## 🧠 四级必读 (规划与记忆)

Agent 的高级能力。

| # | 论文 | arXiv | 核心 | 读它因为 |
|---|------|-------|------|---------|
| 11 | **Tree of Thoughts (ToT): Deliberate Problem Solving** | [2305.08291](https://arxiv.org/pdf/2305.08291.pdf) | 树搜索 | 复杂问题的系统化求解 |
| 12 | **Generative Agents: Interactive Simulacra of Human Behavior** | [2304.03442](https://arxiv.org/pdf/2304.03442.pdf) | 长期记忆 | 如何让 Agent 记住历史并形成人格 |
| 13 | **AgentInstruct: Towards Instruction Tuning for Agents** | [2312.06692](https://arxiv.org/pdf/2312.06692.pdf) | 指令微调 | 如何训练 Agent 专用模型 |

---

## 📊 五级选读 (评估与最新进展)

了解如何评估 Agent，以及 2024-2025 最新方向。

| # | 论文 | arXiv | 核心 | 读它因为 |
|---|------|-------|------|---------|
| 14 | **AgentBench: Benchmarking LLMs as Agents** | [2308.03688](https://arxiv.org/pdf/2308.03688.pdf) | 评估基准 | 如何衡量 Agent 能力 |
| 15 | **SWE-agent: Solving GitHub Issues with LLM Agents** | [2405.15793](https://arxiv.org/pdf/2405.15793.pdf) | 代码 Agent | 软件工程 Agent 的实践 |
| 16 | **OpenHands: Autonomous Software Development** | [官网](https://github.com/OpenDevin/OpenDevin) | 开源 Devin | 全栈开发 Agent |
| 17 | **LangGraph: Building Stateful Agents** | [官网](https://github.com/langchain-ai/langgraph) | 状态图 | Agent 状态管理最佳实践 |

---

## 📖 建议阅读顺序

```
Week 1: ReAct → CoT → Reflexion
        (理解 Agent 的核心机制)

Week 2: MetaGPT → AutoGen → CAMEL
        (多 Agent 系统设计)

Week 3: Toolformer → Gorilla → Voyager
        (工具使用 + 技能积累)

Week 4: 根据兴趣选读其他论文
```

---

## 🔗 对应代码实现 (边读边看)

| 论文 | 对应代码 |
|------|---------|
| ReAct | [LangChain Agents](https://github.com/langchain-ai/langchain) |
| MetaGPT | [MetaGPT](https://github.com/geekan/MetaGPT) |
| AutoGen | [AutoGen](https://github.com/microsoft/autogen) |
| CAMEL | [CAMEL](https://github.com/camel-ai/camel) |
| SWE-agent | [SWE-agent](https://github.com/princeton-nlp/SWE-agent) |
| LangGraph | [LangGraph](https://github.com/langchain-ai/langgraph) |

---

## 🚀 一键下载必读 10 篇

```bash
# 创建目录
mkdir -p agent_papers

# 下载核心 10 篇
cd agent_papers

# 一级必读
curl -O https://arxiv.org/pdf/2210.03629.pdf # ReAct
curl -O https://arxiv.org/pdf/2303.11366.pdf # Reflexion
curl -O https://arxiv.org/pdf/2201.11903.pdf # CoT

# 二级必读
curl -O https://arxiv.org/pdf/2308.00352.pdf # MetaGPT
curl -O https://arxiv.org/pdf/2308.08155.pdf # AutoGen
curl -O https://arxiv.org/pdf/2303.04637.pdf # CAMEL
curl -O https://arxiv.org/pdf/2305.16291.pdf # Voyager

# 三级必读
curl -O https://arxiv.org/pdf/2303.04607.pdf # Toolformer
curl -O https://arxiv.org/pdf/2305.15334.pdf # Gorilla

# 四级必读
curl -O https://arxiv.org/pdf/2305.08291.pdf # ToT
```

---

**最后更新**: 2025-02-18

**总结**: 前 10 篇是核心中的核心，读完这 10 篇你就能理解 Agent 的完整图景了。
