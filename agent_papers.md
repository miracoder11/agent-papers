# Agent 开发必读论文

> **专注 Agent 开发，只列最重要的 15 篇**

---

## 🔥 第一优先级 - Agent 核心范式 (必读 3 篇)

这些是理解 Agent 的基础，必须先读。

| # | 论文 | arXiv | 年份 | 为什么重要 |
|---|------|-------|------|-----------|
| 1 | **ReAct: Synergizing Reasoning and Acting in Language Models** | [2210.03629](https://arxiv.org/pdf/2210.03629.pdf) | 2023 | Agent 的基石，推理+行动范式 |
| 2 | **Reflexion: Language Agents with Verbal Reinforcement Learning** | [2303.11366](https://arxiv.org/pdf/2303.11366.pdf) | 2023 | 自我反思机制，让 Agent 从错误中学习 |
| 3 | **Tree of Thoughts: Deliberate Problem Solving with Large Language Models** | [2305.08291](https://arxiv.org/pdf/2305.08291.pdf) | 2023 | 树状搜索推理，解决复杂问题 |

---

## ⭐ 第二优先级 - 多 Agent 系统 (必读 4 篇)

多 Agent 协作是实际应用的核心。

| # | 论文 | arXiv | 年份 | 为什么重要 |
|---|------|-------|------|-----------|
| 4 | **MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework** | [2308.00352](https://arxiv.org/pdf/2308.00352.pdf) | 2023 | SOP 工作流，角色定义，生产级框架 |
| 5 | **AutoGen: Enabling Next-Gen LLM Applications** | [2308.08155](https://arxiv.org/pdf/2308.08155.pdf) | 2023 | 微软出品，对话式 Agent 框架 |
| 6 | **CAMEL: Communicative Agents for "Mind" Exploration** | [2303.04637](https://arxiv.org/pdf/2303.04637.pdf) | 2023 | 多 Agent 通信协议，早期经典 |
| 7 | **AgentVerse: Facilitating Multi-Agent Collaboration** | [2308.10848](https://arxiv.org/pdf/2308.10848.pdf) | 2023 | 协作平台，人类-Agent 混合 |

---

## 🛠️ 第三优先级 - 工具使用 (必读 3 篇)

Agent 调用工具的能力是关键。

| # | 论文 | arXiv | 年份 | 为什么重要 |
|---|------|-------|------|-----------|
| 8 | **Toolformer: Language Models Can Teach Themselves to Use Tools** | [2303.04607](https://arxiv.org/pdf/2303.04607.pdf) | 2023 | 自主学习何时调用哪个工具 |
| 9 | **Gorilla: Fine-tuned LLaMA for Tool Use** | [2305.15334](https://arxiv.org/pdf/2305.15334.pdf) | 2023 | 精确 API 调用，减少幻觉 |
| 10 | **TaskMatrix: When LLM Meets Function Calling** | [2303.16354](https://arxiv.org/pdf/2303.16354.pdf) | 2023 | 微软出品，工具调用框架 |

---

## 🎯 第四优先级 - 规划与记忆 (必读 3 篇)

Agent 的规划能力和长期记忆。

| # | 论文 | arXiv | 年份 | 为什么重要 |
|---|------|-------|------|-----------|
| 11 | **Voyager: An Open-Ended Embodied Agent with Large Language Models** | [2305.16291](https://arxiv.org/pdf/2305.16291.pdf) | 2023 | 技能库积累，具身 Agent 代表 |
| 12 | **Generative Agents: Interactive Simulacra of Human Behavior** | [2304.03442](https://arxiv.org/pdf/2304.03442.pdf) | 2023 | 长期记忆，社交行为模拟 |
| 13 | **Chain-of-Hindsight: Aligning Language Models with Feedback** | [2309.01762](https://arxiv.org/pdf/2309.01762.pdf) | 2023 | 从反馈中学习 |

---

## 📊 第五优先级 - 评估与基准 (必读 2 篇)

如何评估 Agent 的能力。

| # | 论文 | arXiv | 年份 | 为什么重要 |
|---|------|-------|------|-----------|
| 14 | **AgentBench: Benchmarking LLMs as Agents** | [2308.03688](https://arxiv.org/pdf/2308.03688.pdf) | 2023 | Agent 综合评估基准 |
| 15 | **ToolEval: Tool-Augmented LLMs Evaluation** | [2306.05527](https://arxiv.org/pdf/2306.05527.pdf) | 2023 | 工具使用能力评估 |

---

## 📖 阅读顺序建议

```
第1周: ReAct → Reflexion → Tree of Thoughts
       (理解 Agent 的基本范式)

第2周: MetaGPT → AutoGen
       (多 Agent 系统核心)

第3周: Toolformer → Gorilla → Voyager
       (工具使用 + 技能积累)

第4周: 其余论文
       (根据兴趣选读)
```

---

## 🔗 代码实现 (边读边看)

| 框架 | GitHub | 对应论文 |
|------|--------|---------|
| [LangChain](https://github.com/langchain-ai/langchain) | Agent 通用框架 | ReAct |
| [MetaGPT](https://github.com/geekan/MetaGPT) | 多 Agent 框架 | MetaGPT |
| [AutoGen](https://github.com/microsoft/autogen) | 对话式 Agent | AutoGen |
| [CrewAI](https://github.com/joaomdmoura/crewAI) | 角色扮演 Agent | - |
| [LlamaIndex](https://github.com/run-llama/llama_index) | RAG Agent | RAG 相关 |

---

**最后更新**: 2025-02-18
