# 大模型算法必读论文

> **专注大模型算法，只列最重要的 15 篇**

---

## 🔥 第一优先级 - 架构基础 (必读 3 篇)

理解现代大模型的架构基石。

| # | 论文 | arXiv | 年份 | 为什么重要 |
|---|------|-------|------|-----------|
| 1 | **Attention Is All You Need** | [1706.03762](https://arxiv.org/pdf/1706.03762.pdf) | 2017 | Transformer 架构，一切的基础 |
| 2 | **Language Models are Few-Shot Learners (GPT-3)** | [2005.14165](https://arxiv.org/pdf/2005.14165.pdf) | 2020 | 大规模涌现能力，In-context Learning |
| 3 | **LLaMA 2: Open Foundation and Fine-Tuned Chat Models** | [2307.09288](https://arxiv.org/pdf/2307.09288.pdf) | 2023 | 开源最佳实践，完整的训练方法论 |

---

## ⭐ 第二优先级 - 训练与缩放 (必读 4 篇)

如何高效训练大模型。

| # | 论文 | arXiv | 年份 | 为什么重要 |
|---|------|-------|------|-----------|
| 4 | **Training Compute-Optimal Large Language Models (Chinchilla)** | [2203.15556](https://arxiv.org/pdf/2203.15556.pdf) | 2022 | 计算最优缩放，模型大小与数据量均衡 |
| 5 | **Scaling Laws for Neural Language Models** | [2001.08361](https://arxiv.org/pdf/2001.08361.pdf) | 2020 | 缩放定律基础，预测性能 |
| 6 | **Llama 3 Model Card** | 官网 | 2024 | 最新开源 SOTA 训练实践 |
| 7 | **Mixtral of Experts** | [2401.04088](https://arxiv.org/pdf/2401.04088.pdf) | 2023 | 稀疏 MoE 架构，高效推理 |

---

## 🧠 第三优先级 - 推理与微调 (必读 3 篇)

提升模型推理能力和微调效果。

| # | 论文 | arXiv | 年份 | 为什么重要 |
|---|------|-------|------|-----------|
| 8 | **Chain-of-Thought Prompting Elicits Reasoning** | [2201.11903](https://arxiv.org/pdf/2201.11903.pdf) | 2023 | 思维链，激发推理能力 |
| 9 | **Training Language Models to Follow Instructions (InstructGPT)** | [2203.02155](https://arxiv.org/pdf/2203.02155.pdf) | 2022 | RLHF 基石，对齐人类偏好 |
| 10 | **LoRA: Low-Rank Adaptation of Large Language Models** | [2106.09685](https://arxiv.org/pdf/2106.09685.pdf) | 2021 | 高效微调，参数量大幅减少 |

---

## 🔧 第四优先级 - 对齐方法 (必读 3 篇)

让模型输出更安全、更有用。

| # | 论文 | arXiv | 年份 | 为什么重要 |
|---|------|-------|------|-----------|
| 11 | **Constitutional AI: Harmlessness from AI Feedback** | [2212.08073](https://arxiv.org/pdf/2212.08073.pdf) | 2022 | RLAIF，用 AI 反馈替代人类反馈 |
| 12 | **Direct Preference Optimization (DPO)** | [2305.18290](https://arxiv.org/pdf/2305.18290.pdf) | 2023 | 无需奖励模型，直接优化 |
| 13 | **Principle-Driven Self-Alignment** | [2305.14530](https://arxiv.org/pdf/2305.14530.pdf) | 2023 | 自对齐，减少人类标注 |

---

## 📚 第五优先级 - 检索增强 (必读 2 篇)

RAG 是大模型落地的重要技术。

| # | 论文 | arXiv | 年份 | 为什么重要 |
|---|------|-------|------|-----------|
| 14 | **Retrieval-Augmented Generation for Knowledge-Intensive NLP** | [2005.11401](https://arxiv.org/pdf/2005.11401.pdf) | 2020 | RAG 原始论文，检索+生成 |
| 15 | **Dense Passage Retrieval for Open-Domain QA** | [2004.04906](https://arxiv.org/pdf/2004.04906.pdf) | 2020 | DPR 检索器，提升检索质量 |

---

## 📖 阅读顺序建议

```
第1周: Attention → GPT-3 → LLaMA 2
       (理解架构和大模型能力)

第2周: Chinchilla → Scaling Laws
       (理解如何高效训练)

第3周: CoT → InstructGPT → LoRA
       (推理、对齐、微调)

第4周: 其余论文
       (根据兴趣选读)
```

---

## 🔗 重要资源

| 资源 | URL |
|------|-----|
| **Papers With Code** | https://paperswithcode.com |
| **HuggingFace Models** | https://huggingface.co/models |
| **Transformer 详解** | https://jalammar.github.io/illustrated-transformer/ |
| **LLM 训练教程** | https://github.com/rasbt/LLMs-from-scratch |

---

## 🆕 2024-2025 值得关注

虽然不在必读列表，但这些代表了最新方向：

| 论文 | 年份 | 核心贡献 |
|------|------|---------|
| **DeepSeek-R1** | 2025 | 纯强化学习训练推理能力 |
| **OpenAI o1** | 2024 | 隐式思维链，长推理链 |
| **Llama 3.1** | 2024 | 405B 参数，开源最强 |
| **Qwen2.5** | 2024 | 中文能力领先 |

---

**最后更新**: 2025-02-18
